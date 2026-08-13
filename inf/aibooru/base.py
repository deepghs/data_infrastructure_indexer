"""Session helper for aibooru.online.

Nothing exotic guards this site - no proof of work, no Cloudflare challenge. A browser TLS
fingerprint is still worth using, since a plain client library fingerprint is the cheapest thing
for a site to start refusing later, and the ladder costs nothing when the first rung works.

Credentials are optional. Measured 2026-08-13: ``/posts.json`` and ``/counts/posts.json`` both
answer 200 anonymously, and ``counts`` reports the same 172,895 posts with or without a login, so
nothing is being withheld from an anonymous caller the way atfbooru withholds most of its
database. ``AIBOORU_USERNAME`` / ``AIBOORU_APIKEY`` are wired through for when that changes.
"""
from typing import Optional

from curl_cffi import requests as cffi_requests
from ditk import logging

from inf.utils.impersonate import build_ladder

__site_url__ = 'https://aibooru.online'

#: Per-request timeout. The API answers in well under a second; this is for a stalled connection.
DEFAULT_TIMEOUT = 60.0

#: Fingerprints to try, in order. Chrome works today, so it leads.
IMPERSONATE_LADDER = build_ladder(['chrome131', 'chrome124', 'safari17_0', 'firefox133'],
                                  site='aibooru')


class AIBooruError(Exception):
    """Raised when no session can be established."""


def get_aibooru_session(impersonate: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT,
                        username: Optional[str] = None, api_key: Optional[str] = None):
    """
    Build a session that the site actually answers, verifying it before handing it back.

    :param impersonate: Fingerprint to use; walks :data:`IMPERSONATE_LADDER` by default.
    :type impersonate: Optional[str]
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :param username: Site login, sent as a query parameter when supplied.
    :type username: Optional[str]
    :param api_key: Matching API key.
    :type api_key: Optional[str]
    :returns: A session with a verified fingerprint.
    :raises AIBooruError: When no fingerprint can get through.
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
            session.params = {'login': username, 'api_key': api_key}
        try:
            resp = session.get(f'{__site_url__}/posts.json', params={'limit': '1'})
            resp.raise_for_status()
            resp.json()
        except Exception as err:
            last = f'{chosen}: {type(err).__name__}: {err}'
            logging.warning(f'AIBooru session attempt failed - {last}.')
            continue
        logging.info(f'AIBooru session ready with fingerprint {chosen!r}'
                     f'{" (authenticated)" if username and api_key else ""}.')
        return session
    raise AIBooruError(f'Could not get a usable session; last attempt - {last}')
