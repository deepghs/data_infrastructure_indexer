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
    # A residential pool hands out a different exit per session id, so a few attempts may find
    # one the site does not challenge. A datacentre pool fails identically every time - and that
    # sameness is itself the answer about which kind of zone this is. One fingerprint per exit
    # keeps the request count (and the bill) down while the exits are being sampled.
    for attempt in range(_PROXY_EXIT_ATTEMPTS):
        label = f'{proxy_session or "s"}x{attempt}'
        routed = with_session(proxy_pool, label)
        logging.info(f'Trying proxy exit {label!r} '
                     f'({attempt + 1}/{_PROXY_EXIT_ATTEMPTS}), exit address '
                     f'{_exit_address(routed, timeout)}.')
        session = _walk_ladder(ladder[:1], timeout, username, api_key, routed)
        if session is not None:
            return session
    # Every exit refused with one fingerprint; give the full ladder one last go on the last exit
    # in case the fingerprint rather than the address was the problem.
    session = _walk_ladder(ladder, timeout, username, api_key, routed)
    if session is not None:
        return session
    raise AIBooruError(f'Could not get a usable session, direct or proxied; '
                     f'last attempt - {_LAST_FAILURE[0]}')


#: Why the most recent route failed, for the error raised once every route is exhausted.
_LAST_FAILURE = ['no fingerprint attempted']

#: Proxy exits to sample before concluding the pool cannot reach the site. Each costs one request.
_PROXY_EXIT_ATTEMPTS = 4


def _exit_address(proxy: str, timeout: float) -> str:
    """
    The address the proxy is exiting from, for the log.

    Knowing whether the exits differ between attempts is what distinguishes a rotating residential
    pool from a fixed datacentre one, which decides whether retrying is worth anything at all.

    :param proxy: Proxy URL to route through.
    :type proxy: str
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :returns: The address, or a short reason it could not be read.
    :rtype: str
    """
    try:
        probe = cffi_requests.Session(impersonate='chrome131', timeout=timeout)
        probe.proxies = {'http': proxy, 'https': proxy}
        return probe.get('https://api.ipify.org').text.strip()[:40]
    except Exception as err:
        return f'unknown ({type(err).__name__})'


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
