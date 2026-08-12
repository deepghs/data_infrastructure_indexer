"""Session helper for booru.allthefallen.moe.

The site gates every request behind a proof-of-work challenge of its own making. This is not
Cloudflare - the responses come from nginx with no ``cf-ray`` header - so neither
``cloudscraper`` nor a browser TLS fingerprint helps, which is why the pyskeb implementation
(``ATFBooruSource`` plus ``_prune_session()``) now receives an HTML page where it expects JSON
and fails on the parse.

The challenge arrives as a 200 with an HTML body titled ``... | Verification`` carrying
everything needed to answer it::

    challenge_id, challenge_generated, challenge_cookie_expires
    powSeed, powPrefix = "0".repeat(5), delay = 5, lifetime = 120

Answering it is cheap and needs no JavaScript engine: find a nonce where
``sha1(f"{powSeed}:{nonce}")`` starts with five hex zeros, wait out ``delay`` seconds, then POST
the four challenge fields plus the nonce and hash back to the site root. That yields an
``atf-anti-bot`` cookie.

The arithmetic is not the cost here. Five hex zeros is about a million hashes, measured at 0.04
seconds; the mandatory wait dominates. Since ``challenge_cookie_expires`` sits a week past
``challenge_generated``, one handshake covers a very long run, so the wait amortises to nothing.
"""
import hashlib
import re
import time
from typing import List, Optional

from curl_cffi import requests as cffi_requests
from ditk import logging

from inf.utils.impersonate import build_ladder

__site_url__ = 'https://booru.allthefallen.moe'

DEFAULT_TIMEOUT = 60.0

#: Cookie the challenge grants on success.
_ANTI_BOT_COOKIE = 'atf-anti-bot'

#: Fingerprints verified against the site. Kept broad, though the challenge - not the
#: handshake - is what actually guards the door here.
PREFERRED_IMPERSONATES: List[str] = [
    'chrome131', 'chrome124', 'chrome119', 'chrome116', 'chrome110',
    'firefox135', 'firefox133', 'safari17_0',
]

IMPERSONATE_LADDER: List[str] = build_ladder(PREFERRED_IMPERSONATES, site='atfbooru')

#: Constants the challenge page declares. Read by shape rather than position, since the page is
#: generated per request.
_CONST = re.compile(r'const (?P<name>\w+) = "(?P<value>[^"]*)"')
_POW_PREFIX = re.compile(r'const powPrefix = "0"\.repeat\((?P<zeros>\d+)\)')
_DELAY = re.compile(r'const delay = (?P<delay>\d+)')

#: Marker that a response is the challenge rather than the thing that was asked for.
_CHALLENGE_MARKER = 'Verification'


class ATFBooruError(Exception):
    """Raised when the challenge cannot be cleared."""


def is_challenge(response) -> bool:
    """
    Whether a response is the verification page rather than real content.

    The site answers 200 with HTML, so status alone says nothing.

    :returns: True when the body is the challenge page.
    :rtype: bool
    """
    ctype = (response.headers.get('content-type') or '').lower()
    if 'json' in ctype:
        return False
    text = response.text or ''
    return _CHALLENGE_MARKER in text[:4000] and 'powSeed' in text


def solve_pow(seed: str, zeros: int) -> tuple:
    """
    Find a nonce whose sha1 digest starts with ``zeros`` hex zeros.

    :param seed: ``powSeed`` from the challenge page.
    :type seed: str
    :param zeros: Number of leading hex zeros required.
    :type zeros: int
    :returns: ``(nonce, hexdigest)``.
    :rtype: tuple
    """
    prefix = '0' * zeros
    nonce = 0
    while True:
        digest = hashlib.sha1(f'{seed}:{nonce}'.encode()).hexdigest()
        if digest.startswith(prefix):
            return nonce, digest
        nonce += 1


def clear_challenge(session, response, honour_delay: bool = True) -> bool:
    """
    Answer the verification challenge on ``session``, leaving it able to fetch content.

    :param session: Session to clear; the cookie lands in its jar.
    :param response: The response carrying the challenge page.
    :param honour_delay: Wait out the page's ``delay`` before answering. The server times the
        exchange, so answering instantly is a good way to be refused.
    :type honour_delay: bool
    :returns: Whether the challenge was answered and accepted.
    :rtype: bool
    """
    text = response.text or ''
    consts = {m.group('name'): m.group('value') for m in _CONST.finditer(text)}
    seed = consts.get('powSeed')
    challenge_id = consts.get('challenge_id')
    if not seed or not challenge_id:
        logging.warning('Challenge page carried no powSeed or challenge_id.')
        return False

    zeros_match = _POW_PREFIX.search(text)
    zeros = int(zeros_match.group('zeros')) if zeros_match else 5
    delay_match = _DELAY.search(text)
    delay = float(delay_match.group('delay')) if delay_match else 5.0

    started = time.time()
    nonce, digest = solve_pow(seed, zeros)
    solve_seconds = time.time() - started
    logging.info(f'ATFBooru challenge {challenge_id}: solved {zeros} zeros with nonce {nonce:,} '
                 f'in {solve_seconds:.2f}s.')

    if honour_delay:
        remaining = delay - solve_seconds
        if remaining > 0:
            # A small margin on top: the server compares against its own clock.
            time.sleep(remaining + 0.4)

    answer = session.post(
        f'{__site_url__}/',
        json={
            'challenge_id': challenge_id,
            'challenge_generated': consts.get('challenge_generated'),
            'challenge_cookie_expires': consts.get('challenge_cookie_expires'),
            'pow_nonce': str(nonce),
            'pow_hash': digest,
        },
        headers={
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
            'X-Verification-Challenge': '1',
            'Referer': f'{__site_url__}/',
        },
    )
    if answer.status_code != 200:
        logging.warning(f'ATFBooru challenge answer refused with HTTP {answer.status_code}.')
        return False
    if _ANTI_BOT_COOKIE not in session.cookies:
        logging.warning('ATFBooru accepted the answer but set no cookie.')
        return False
    logging.info('ATFBooru challenge cleared.')
    return True


class ChallengeSession:
    """
    Wrap a session so the verification challenge is answered whenever it appears.

    The cookie expires, and a long run will meet the challenge again, so recovery has to happen
    in place rather than by rebuilding the session and losing any other state.
    """

    def __init__(self, session, max_attempts: int = 3):
        self._session = session
        self._max_attempts = max_attempts
        self.challenges = 0

    def __getattr__(self, item):
        return getattr(self._session, item)

    def request(self, method, url, **kwargs):
        response = self._session.request(method, url, **kwargs)
        for _ in range(self._max_attempts):
            if not is_challenge(response):
                return response
            self.challenges += 1
            logging.info(f'ATFBooru challenge encountered (#{self.challenges}), answering ...')
            if not clear_challenge(self._session, response):
                break
            response = self._session.request(method, url, **kwargs)
        if is_challenge(response):
            raise ATFBooruError(f'Could not clear the challenge for {url!r} after '
                                f'{self._max_attempts} attempts.')
        return response

    def get(self, url, **kwargs):
        return self.request('GET', url, **kwargs)

    def post(self, url, **kwargs):
        return self.request('POST', url, **kwargs)


def get_atfbooru_session(impersonate: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT,
                         username: Optional[str] = None, api_key: Optional[str] = None):
    """
    Build a session that has already cleared the challenge.

    :param impersonate: Fingerprint to use; walks :data:`IMPERSONATE_LADDER` by default.
    :type impersonate: Optional[str]
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :param username: Site login, sent as a query parameter by the API when supplied.
    :type username: Optional[str]
    :param api_key: Matching API key.
    :type api_key: Optional[str]
    :returns: A ready-to-use :class:`ChallengeSession`.
    :raises ATFBooruError: When no fingerprint can get through.
    """
    ladder = [impersonate] if impersonate else list(IMPERSONATE_LADDER)
    last = 'no fingerprint attempted'
    for chosen in ladder:
        try:
            raw = cffi_requests.Session(impersonate=chosen, timeout=timeout)
        except Exception as err:
            if 'not supported' not in str(err):
                raise
            logging.warning(f'Impersonation target {chosen!r} rejected by curl_cffi, skipping.')
            continue
        if username and api_key:
            raw.params = {'login': username, 'api_key': api_key}
        session = ChallengeSession(raw)
        try:
            resp = session.get(f'{__site_url__}/posts.json', params={'limit': '1'})
            resp.raise_for_status()
            resp.json()
        except Exception as err:
            last = f'{chosen}: {type(err).__name__}: {err}'
            logging.warning(f'ATFBooru session attempt failed - {last}.')
            continue
        logging.info(f'ATFBooru session ready with fingerprint {chosen!r}.')
        return session
    raise ATFBooruError(f'Could not get a usable session; last attempt - {last}')


__all__ = ['__site_url__', 'ATFBooruError', 'ChallengeSession', 'IMPERSONATE_LADDER',
           'PREFERRED_IMPERSONATES', 'clear_challenge', 'get_atfbooru_session', 'is_challenge',
           'solve_pow']
