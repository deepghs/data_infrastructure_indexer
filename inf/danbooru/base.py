"""Shared Danbooru session helpers.

``cdn.donmai.us`` and ``danbooru.donmai.us`` sit behind a Cloudflare bot classifier that
rejects plain HTTP/1.1 clients with a 403 challenge page regardless of credentials. Two
things get through, and both are required:

* an **HTTP/2** connection, so the TLS/ALPN handshake matches a real browser, and
* a **warm-up request** against ``posts.json``, which hands back the ``_danbooru2_session``
  cookie that subsequent CDN requests are checked against.

A session built this way downloads originals, including posts flagged deleted, at their exact
announced size. This mirrors what the long-running Danbooru sync job does.
"""
import time
from typing import Optional

import httpx
from ditk import logging

from inf.utils.session import get_random_ua

__site_url__ = 'https://danbooru.donmai.us'

DEFAULT_TIMEOUT = 15.0


def get_danbooru_session(max_retries: int = 10, timeout: float = DEFAULT_TIMEOUT,
                         proxy_pool: Optional[str] = None, retry_wait_time: float = 5.0,
                         username: Optional[str] = None, apitoken: Optional[str] = None) -> httpx.Client:
    """
    Build an HTTP/2 client that has already cleared Danbooru's Cloudflare check.

    Each attempt uses a fresh client and a fresh user agent, because a rejected fingerprint
    stays rejected for the life of that connection. The client is returned only after the
    warm-up call succeeds, so callers never have to handle the challenge themselves.

    :param max_retries: Number of warm-up attempts before giving up.
    :type max_retries: int
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :param proxy_pool: Proxy URL for every request made through this client.
    :type proxy_pool: Optional[str]
    :param retry_wait_time: Base seconds to wait between attempts; grows linearly.
    :type retry_wait_time: float
    :param username: Danbooru account name, used for the warm-up call when paired with a token.
    :type username: Optional[str]
    :param apitoken: Danbooru API key, used for the warm-up call when paired with a name.
    :type apitoken: Optional[str]
    :returns: A warmed-up client carrying the session cookie.
    :rtype: httpx.Client
    :raises RuntimeError: When no attempt manages to clear the check.
    """
    auth = (username, apitoken) if username and apitoken else None
    for attempt in range(1, max_retries + 1):
        kwargs = dict(http2=True, timeout=timeout, follow_redirects=True)
        if proxy_pool:
            kwargs['proxy'] = proxy_pool
        session = httpx.Client(**kwargs)
        session.headers.update({'User-Agent': get_random_ua()})

        try:
            resp = session.get(
                f'{__site_url__}/posts.json',
                params={'format': 'json', 'tags': '1girl', 'limit': 1},
                auth=auth,
            )
        except httpx.HTTPError as err:
            logging.warning(f'Danbooru warm-up attempt {attempt}/{max_retries} failed - {err!r}.')
            session.close()
            time.sleep(retry_wait_time * attempt)
            continue

        if resp.status_code // 100 == 2:
            logging.info(f'Danbooru session established on attempt {attempt}, '
                         f'cookies: {list(session.cookies.keys())!r}.')
            return session

        logging.warning(f'Danbooru warm-up attempt {attempt}/{max_retries} rejected with '
                        f'HTTP {resp.status_code}, retry with a new fingerprint.')
        session.close()
        time.sleep(retry_wait_time * attempt)

    raise RuntimeError(f'Unable to establish a Danbooru session after {max_retries} attempt(s).')


__all__ = ['get_danbooru_session', '__site_url__']
