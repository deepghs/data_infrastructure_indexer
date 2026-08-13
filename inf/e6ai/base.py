"""Session helper for e6ai.net.

An e621-derived site, and like the rest of that family it cares about the User-Agent: a bare
client-library fingerprint is what these sites refuse first. A browser TLS fingerprint plus a
browser UA answers 200, verified 2026-08-13.

Credentials are optional here. ``/posts.json`` answers anonymously and a sample of the 320 newest
posts came back with every ``file.url`` populated, so nothing is being withheld the way atfbooru
withholds banned files. ``E6AI_USERNAME`` / ``E6AI_APIKEY`` are wired through as HTTP Basic auth,
which is what the e621 API family accepts, for when that stops being true.


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
import base64
from typing import Optional

from curl_cffi import requests as cffi_requests
from ditk import logging

from inf.utils.brightdata import BrightDataError, ensure_proxy_access, with_session
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
                     username: Optional[str] = None, api_key: Optional[str] = None,
                     proxy_pool: Optional[str] = None, proxy_session: Optional[str] = None,
                     brd_api_key: Optional[str] = None, brd_zone: Optional[str] = None):
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
    :raises E6AIError: When no fingerprint can get through.
    """
    ladder = [impersonate] if impersonate else list(IMPERSONATE_LADDER)

    session = _walk_ladder(ladder, timeout, username, api_key, None)
    if session is not None:
        return session
    if not proxy_pool:
        raise E6AIError(f'Could not get a usable session directly and no proxy is configured; '
                         f'last attempt - {_LAST_FAILURE[0]}')

    logging.info('Direct access refused on every fingerprint; falling back to the proxy pool.')
    if brd_api_key:
        try:
            if not ensure_proxy_access(proxy_pool, api_key=brd_api_key, zone=brd_zone):
                raise E6AIError('Proxy pool is not usable from this host.')
        except BrightDataError as err:
            raise E6AIError(f'Proxy pool unusable - {err}') from err
    routed = with_session(proxy_pool, proxy_session) if proxy_session else proxy_pool
    session = _walk_ladder(ladder, timeout, username, api_key, routed)
    if session is not None:
        return session
    raise E6AIError(f'Could not get a usable session, direct or proxied; '
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
                # Cloudflare's mitigations look alike from the status code and differ entirely in
                # what answers them, so record what identifies them: cf-mitigated names the
                # mitigation, cf-ray proves it is Cloudflare at all, and the body carries the
                # challenge platform's script path when it is a JS challenge.
                body = ' '.join(resp.text.split())
                marks = {key: resp.headers.get(key) for key in
                         ('server', 'cf-ray', 'cf-mitigated', 'cf-chl-out', 'retry-after')
                         if resp.headers.get(key)}
                hints = [name for name in ('challenge-platform', 'turnstile', 'cf_chl_opt',
                                           'jschl', '__cf_bm', 'cf-please-wait')
                         if name in body]
                last = (f'{chosen}: HTTP {resp.status_code} {marks} hints={hints} '
                        f'body={body[:400]!r}')
                logging.warning(f'E6AI session attempt failed ({route}) - {last}.')
                continue
            resp.json()
        except Exception as err:
            last = f'{chosen}: {type(err).__name__}: {err}'
            logging.warning(f'E6AI session attempt failed ({route}) - {last}.')
            continue
        logging.info(f'E6AI session ready with fingerprint {chosen!r} ({route})'
                     f'{" (authenticated)" if username and api_key else ""}.')
        return session
    _LAST_FAILURE[0] = f'{route} - {last}'
    return None
