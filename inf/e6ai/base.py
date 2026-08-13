"""Session helper for e6ai.net.

An e621-derived site, and like the rest of that family it cares about the User-Agent: a bare
client-library fingerprint is what these sites refuse first. A browser TLS fingerprint plus a
browser UA answers 200, verified 2026-08-13.

Credentials are optional here. ``/posts.json`` answers anonymously and a sample of the 320 newest
posts came back with every ``file.url`` populated, so nothing is being withheld the way atfbooru
withholds banned files. ``E6AI_USERNAME`` / ``E6AI_APIKEY`` are wired through as HTTP Basic auth,
which is what the e621 API family accepts, for when that stops being true.
"""
import base64
from typing import Optional

from curl_cffi import requests as cffi_requests
from ditk import logging

from inf.utils.impersonate import build_ladder

__site_url__ = 'https://e6ai.net'

#: Per-request timeout.
DEFAULT_TIMEOUT = 60.0

#: Fingerprints to try, in order.
IMPERSONATE_LADDER = build_ladder(['chrome131', 'chrome124', 'safari17_0', 'firefox133'],
                                  site='e6ai')


class E6AIError(Exception):
    """Raised when no session can be established."""


def get_e6ai_session(impersonate: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT,
                     username: Optional[str] = None, api_key: Optional[str] = None):
    """
    Build a session that the site actually answers, verifying it before handing it back.

    :param impersonate: Fingerprint to use; walks :data:`IMPERSONATE_LADDER` by default.
    :type impersonate: Optional[str]
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :param username: Site login, sent as HTTP Basic auth when supplied.
    :type username: Optional[str]
    :param api_key: Matching API key.
    :type api_key: Optional[str]
    :returns: A session with a verified fingerprint.
    :raises E6AIError: When no fingerprint can get through.
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
        if username and api_key:
            token = base64.b64encode(f'{username}:{api_key}'.encode()).decode()
            session.headers.update({'Authorization': f'Basic {token}'})
        # Both headers are what the prototype sent. They cost nothing and a site that has
        # decided to be suspicious of the caller may want them.
        session.headers.update({
            'Content-Type': 'application/json; charset=utf-8',
            'Referer': f'{__site_url__}/',
        })
        try:
            resp = session.get(f'{__site_url__}/posts.json', params={'limit': '1'})
            if resp.status_code != 200:
                # The body is the only thing that distinguishes a blocked address from a
                # challenge or a missing header, so it goes in the log rather than a bare code.
                body = ' '.join(resp.text.split())[:240]
                last = f'{chosen}: HTTP {resp.status_code} - {body!r}'
                logging.warning(f'E6AI session attempt failed - {last}.')
                continue
            resp.json()
        except Exception as err:
            last = f'{chosen}: {type(err).__name__}: {err}'
            logging.warning(f'E6AI session attempt failed - {last}.')
            continue
        logging.info(f'E6AI session ready with fingerprint {chosen!r}'
                     f'{" (authenticated)" if username and api_key else ""}.')
        return session
    raise E6AIError(f'Could not get a usable session; last attempt - {last}')
