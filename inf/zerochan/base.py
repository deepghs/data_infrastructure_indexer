"""Session helper for zerochan.net.

Zerochan gates the first request behind a bot check: it answers ``503`` with a small HTML page
carrying an ``<img>`` whose load sets an ``xbotcheck`` cookie, after which normal responses
resume. Two things about that broke the previous implementation, and neither is visible from a
stack trace.

The challenge image path is generated per response. It used to be ``/xbotcheck-image.svg``,
which the old code requested by name; today it is ``/totally-innocent-logo-image.svg?<nonce>``
with a fresh nonce every time, so a hard-coded path can never clear the check again.

Worse, ``get_requests_session()`` mounts a ``Retry`` that treats 503 as retryable. The challenge
page therefore never reaches the caller: urllib3 burns its retries on it and raises
``RetryError``, so the very HTML carrying the way through gets swallowed. Reaching for
``curl_cffi`` here is not only about TLS fingerprints - it is a transport that hands back the
503 body instead of retrying it away.

Rate limiting is the other constant. Measured from one address, concurrency makes throughput
*worse*: 1 worker sustained 1.79 req/s, 4 workers 1.12 req/s, 8 workers 0.43 req/s with 19 of
48 requests timing out. Routing 16 workers through separate Bright Data exit addresses reached
1.00 req/s, still no better than staying serial, so the meter is not purely per-address. The
only control that helps is spacing requests out; see :data:`DEFAULT_MIN_INTERVAL`.
"""
import os
import re
import time
from functools import lru_cache
from typing import List, Optional, Tuple

from curl_cffi import requests as cffi_requests
from ditk import logging

from inf.utils.impersonate import build_ladder

__site_url__ = 'https://www.zerochan.net'

#: Kept under its original name because ``tag.py`` builds URLs from it.
_ROOT = __site_url__

DEFAULT_TIMEOUT = 60.0

#: Seconds between requests on one session. Concurrency does not help here (see the module
#: docstring), so pacing is the only throughput control that matters.
DEFAULT_MIN_INTERVAL = 0.5

#: Fingerprints verified against zerochan.net. Chrome clears the bot check, so the list stays
#: broad rather than resting on one target that could quietly stop working.
PREFERRED_IMPERSONATES: List[str] = [
    'chrome131', 'chrome124', 'chrome123', 'chrome119', 'chrome116', 'chrome110',
    'firefox135', 'firefox133', 'safari17_0',
]

IMPERSONATE_LADDER: List[str] = build_ladder(PREFERRED_IMPERSONATES, site='zerochan')

#: The bot check page embeds exactly one image, and loading it is what sets the cookie. Matched
#: by shape rather than by name, because the name is not stable.
_CHALLENGE_IMG = re.compile(r'<img\s+src="(?P<path>/[^"]+)"')

#: Cookie the bot check sets once cleared.
_BOTCHECK_COOKIE = 'xbotcheck'


class ZerochanFuckedUp(Exception):
    """Raised when the site cannot be coaxed into serving us."""


def _solve_bot_check(session, response) -> bool:
    """
    Clear a 503 bot check by loading the image it points at.

    :param session: Session to clear the check on; the cookie lands in its jar.
    :param response: The 503 response carrying the challenge page.
    :returns: Whether a challenge was found and cleared.
    :rtype: bool
    """
    matched = _CHALLENGE_IMG.search(response.text or '')
    if not matched:
        return False
    path = matched.group('path').replace('&amp;', '&')
    session.get(f'{__site_url__}{path}')
    return _BOTCHECK_COOKIE in session.cookies


def _login(session, username: str, password: str) -> bool:
    """
    Log in so the session sees whatever the account is entitled to.

    Implemented here rather than borrowed from waifuc's ``ZerochanSource._auth``, which passes
    ``follow_redirects`` - an httpx keyword - to a requests-style session.

    :returns: Whether the login appears to have taken.
    :rtype: bool
    """
    resp = session.post(
        f'{__site_url__}/login',
        data={'ref': '/', 'name': username, 'password': password, 'login': 'Login'},
        headers={
            'Referer': f'{__site_url__}/login?ref=%2F',
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        allow_redirects=False,
    )
    ok = resp.status_code in (200, 301, 302, 303) and 'z_id' in session.cookies
    if ok:
        logging.info(f'Logged into zerochan as {username!r}.')
    else:
        logging.warning(f'Zerochan login for {username!r} did not take '
                        f'(HTTP {resp.status_code}); continuing anonymously.')
    return ok


def get_zerochan_session(auth: Optional[Tuple[str, str]] = None,
                         impersonate: Optional[str] = None,
                         timeout: float = DEFAULT_TIMEOUT):
    """
    Build a session that has already cleared the bot check.

    :param auth: ``(username, password)`` to log in with, or None to stay anonymous.
    :type auth: Optional[Tuple[str, str]]
    :param impersonate: Fingerprint to use; walks :data:`IMPERSONATE_LADDER` by default.
    :type impersonate: Optional[str]
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :returns: A ready-to-use session.
    :raises ZerochanFuckedUp: When the bot check cannot be cleared with any fingerprint.
    """
    ladder = [impersonate] if impersonate else list(IMPERSONATE_LADDER)
    last = 'no fingerprint attempted'
    for chosen in ladder:
        try:
            session = cffi_requests.Session(impersonate=chosen, timeout=timeout)
        except Exception as err:
            if 'not supported' not in str(err):
                raise
            logging.warning(f'Impersonation target {chosen!r} rejected by curl_cffi, skipping.')
            continue
        session.headers.update({
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
        })
        try:
            resp = session.get(f'{__site_url__}/?json=1')
            if resp.status_code == 503:
                if not _solve_bot_check(session, resp):
                    last = f'{chosen}: 503 with no challenge image in the body'
                    continue
                resp = session.get(f'{__site_url__}/?json=1')
            if resp.status_code != 200:
                last = f'{chosen}: HTTP {resp.status_code} after the bot check'
                logging.warning(f'Zerochan refused fingerprint {chosen!r} - {last}.')
                continue
        except Exception as err:
            last = f'{chosen}: {type(err).__name__}: {err}'
            logging.warning(f'Zerochan session attempt failed - {last}.')
            continue

        logging.info(f'Zerochan session ready with fingerprint {chosen!r}.')
        if auth and auth[0] and auth[1]:
            _login(session, *auth)
        return session
    raise ZerochanFuckedUp(f'Could not get a usable zerochan session; last attempt - {last}')


class PacedSession:
    """
    Wrap a session so no two requests leave closer than ``min_interval`` apart.

    Zerochan meters by source, so spacing is the useful control rather than worker count. Doing
    it here keeps the pacing next to the transport instead of scattering ``sleep`` calls through
    callers, and lets the bot check be re-cleared in place: the cookie expires, so a long run
    meets the challenge more than once and should recover without dropping its login.
    """

    def __init__(self, session, min_interval: float = DEFAULT_MIN_INTERVAL):
        self._session = session
        self._min_interval = min_interval
        self._next_at = 0.0
        self.rechecks = 0

    def __getattr__(self, item):
        return getattr(self._session, item)

    def request(self, method, url, **kwargs):
        wait = self._next_at - time.time()
        if wait > 0:
            time.sleep(wait)
        resp = self._session.request(method, url, **kwargs)
        self._next_at = time.time() + self._min_interval
        if resp.status_code == 503 and _solve_bot_check(self._session, resp):
            self.rechecks += 1
            logging.info(f'Zerochan bot check re-cleared mid-run (#{self.rechecks}).')
            resp = self._session.request(method, url, **kwargs)
            self._next_at = time.time() + self._min_interval
        return resp

    def get(self, url, **kwargs):
        return self.request('GET', url, **kwargs)

    def post(self, url, **kwargs):
        return self.request('POST', url, **kwargs)


@lru_cache()
def get_session():
    """
    The process-wide session: bot check cleared, logged in, and paced.

    :returns: A ready-to-use paced session.
    """
    return PacedSession(get_zerochan_session((
        os.environ.get('ZEROCHAN_USERNAME'),
        os.environ.get('ZEROCHAN_PASSWORD'),
    )))


__all__ = [
    '__site_url__', 'ZerochanFuckedUp', 'PacedSession', 'PREFERRED_IMPERSONATES',
    'IMPERSONATE_LADDER', 'get_zerochan_session', 'get_session',
]
