"""Session helper for aibooru.online.

Nothing exotic guards this site - no proof of work, no Cloudflare challenge. A browser TLS
fingerprint is still worth using, since a plain client library fingerprint is the cheapest thing
for a site to start refusing later, and the ladder costs nothing when the first rung works.

Credentials are optional. Measured 2026-08-13: ``/posts.json`` and ``/counts/posts.json`` both
answer 200 anonymously, and ``counts`` reports the same 172,895 posts with or without a login, so
nothing is being withheld from an anonymous caller the way atfbooru withholds most of its
database. ``AIBOORU_USERNAME`` / ``AIBOORU_APIKEY`` are wired through for when that changes.


Getting in from CI
==================

The site sits behind Cloudflare, and a datacentre address gets the interstitial rather than the
API: every fingerprint in the ladder came back with ``403 <title>Just a moment...</title>`` from a
GitHub runner while all of them work from a residential one. No TLS fingerprint fixes that - the
challenge is about where the request comes from.

So the proxy is a fallback, not the default. A direct attempt is made first and only if the whole
ladder fails is the Bright Data pool tried, since it costs money per request. The runner's address
has to be allowlisted on the zone before it can use the pool, which
:func:`inf.utils.brightdata.ensure_proxy_access` does; a proxy session id pins the exit address so
consecutive requests do not each land on a different one.
"""
from typing import Optional

from curl_cffi import requests as cffi_requests
from ditk import logging

from inf.utils.brightdata import BrightDataError, ensure_proxy_access, with_session
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
                        username: Optional[str] = None, api_key: Optional[str] = None,
                        proxy_pool: Optional[str] = None, proxy_session: Optional[str] = None,
                        brd_api_key: Optional[str] = None, brd_zone: Optional[str] = None):
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
    :param proxy_pool: Bright Data proxy URL, used only if every direct attempt fails.
    :type proxy_pool: Optional[str]
    :param proxy_session: Session id pinning the proxy's exit address, so consecutive requests do
        not each land on a different one.
    :type proxy_session: Optional[str]
    :param brd_api_key: Bright Data API key, needed to allowlist this host on the zone.
    :type brd_api_key: Optional[str]
    :param brd_zone: Zone to allowlist into.
    :type brd_zone: Optional[str]
    :returns: A session with a verified fingerprint.
    :raises AIBooruError: When no fingerprint can get through.
    """
    ladder = [impersonate] if impersonate else list(IMPERSONATE_LADDER)

    session = _walk_ladder(ladder, timeout, username, api_key, None)
    if session is not None:
        return session
    if not proxy_pool:
        raise AIBooruError(f'Could not get a usable session directly and no proxy is configured; '
                         f'last attempt - {_LAST_FAILURE[0]}')

    logging.info('Direct access refused on every fingerprint; falling back to the proxy pool.')
    if brd_api_key:
        try:
            if not ensure_proxy_access(proxy_pool, api_key=brd_api_key, zone=brd_zone):
                raise AIBooruError('Proxy pool is not usable from this host.')
        except BrightDataError as err:
            raise AIBooruError(f'Proxy pool unusable - {err}') from err
    routed = with_session(proxy_pool, proxy_session) if proxy_session else proxy_pool
    session = _walk_ladder(ladder, timeout, username, api_key, routed)
    if session is not None:
        return session
    raise AIBooruError(f'Could not get a usable session, direct or proxied; '
                     f'last attempt - {_LAST_FAILURE[0]}')


#: Why the most recent route failed, for the error raised once every route is exhausted.
_LAST_FAILURE = ['no fingerprint attempted']


def _walk_ladder(ladder, timeout, username, api_key, proxy):
    """
    Try each fingerprint over one route, returning the first session the site answers.

    :param ladder: Fingerprints to try, in order.
    :param timeout: Per-request timeout in seconds.
    :param username: Site login, or None.
    :param api_key: Matching API key, or None.
    :param proxy: Proxy URL to route through, or None for a direct attempt.
    :returns: A working session, or None when every fingerprint failed.
    """
    route = 'proxied' if proxy else 'direct'
    last = 'no fingerprint attempted'
    for chosen in ladder:
        try:
            session = cffi_requests.Session(impersonate=chosen, timeout=timeout)
            if proxy:
                session.proxies = {'http': proxy, 'https': proxy}
        except Exception as err:
            if 'not supported' not in str(err):
                raise
            logging.warning(f'Impersonation target {chosen!r} rejected by curl_cffi, skipping.')
            continue
        if username and api_key:
            session.params = {'login': username, 'api_key': api_key}
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
                logging.warning(f'AIBooru session attempt failed ({route}) - {last}.')
                continue
            resp.json()
        except Exception as err:
            last = f'{chosen}: {type(err).__name__}: {err}'
            logging.warning(f'AIBooru session attempt failed ({route}) - {last}.')
            continue
        logging.info(f'AIBooru session ready with fingerprint {chosen!r} ({route})'
                     f'{" (authenticated)" if username and api_key else ""}.')
        return session
    _LAST_FAILURE[0] = f'{route} - {last}'
    return None
