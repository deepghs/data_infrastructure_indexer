"""Session helper for anime-pictures.net.

The original implementation in pyskeb reached for ``cloudscraper`` and, failing that, a proxy
pool. Neither is needed any more: a ``curl_cffi`` session with a browser fingerprint is admitted
on the handshake alone, verified against both the site root and ``api.anime-pictures.net``.
That removes a dependency whose whole job was to solve a challenge we no longer see.

The API lives on its own host. ``anime-pictures.net/api/v3`` answers too, but the site's own
pages call ``api.anime-pictures.net``, so that is what this uses.
"""
import random
from typing import List, Optional

from curl_cffi import requests as cffi_requests
from ditk import logging

from inf.utils.impersonate import build_ladder

__site_url__ = 'https://anime-pictures.net'

#: The API is served from a separate host, which is what the site's own pages call.
__api_url__ = 'https://api.anime-pictures.net'

DEFAULT_TIMEOUT = 60.0

#: Fingerprints verified against api.anime-pictures.net.
PREFERRED_IMPERSONATES: List[str] = [
    'chrome131', 'chrome124', 'chrome123', 'chrome119', 'chrome116', 'chrome110',
    'firefox135', 'firefox133', 'safari17_0',
]

IMPERSONATE_LADDER: List[str] = build_ladder(PREFERRED_IMPERSONATES, site='anime-pictures')


class AnimePicturesError(Exception):
    """Raised when no session can be established."""


def get_anime_pictures_session(impersonate: Optional[str] = None,
                               timeout: float = DEFAULT_TIMEOUT,
                               proxy_pool: Optional[str] = None):
    """
    Build a session the API will accept, verifying it before handing it back.

    :param impersonate: Fingerprint to use; walks :data:`IMPERSONATE_LADDER` by default.
    :type impersonate: Optional[str]
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :param proxy_pool: Optional proxy URL. Not needed in practice; kept for the case where the
        site starts refusing cloud address space the way konachan does.
    :type proxy_pool: Optional[str]
    :returns: A ready-to-use session.
    :raises AnimePicturesError: When every fingerprint is refused.
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
        if proxy_pool:
            session.proxies.update({'http': proxy_pool, 'https': proxy_pool})
        session.headers.update({'Referer': f'{__site_url__}/'})
        try:
            resp = session.get(f'{__api_url__}/api/v3/posts',
                               params={'page': '0', 'order_by': 'date', 'ldate': '0', 'lang': 'en'})
            if resp.status_code != 200:
                last = f'{chosen}: HTTP {resp.status_code}'
                logging.warning(f'anime-pictures refused fingerprint {chosen!r} - {last}.')
                continue
            resp.json()
        except Exception as err:
            last = f'{chosen}: {type(err).__name__}: {err}'
            logging.warning(f'anime-pictures session attempt failed - {last}.')
            continue
        logging.info(f'anime-pictures session ready with fingerprint {chosen!r}.')
        return session
    raise AnimePicturesError(f'Could not get a usable session; last attempt - {last}')


def pick_impersonate() -> str:
    """
    Pick a fingerprint at random, so a target that quietly stops working strands nothing.

    :returns: A fingerprint name.
    :rtype: str
    """
    return random.choice(IMPERSONATE_LADDER)


__all__ = ['__site_url__', '__api_url__', 'AnimePicturesError', 'PREFERRED_IMPERSONATES',
           'IMPERSONATE_LADDER', 'get_anime_pictures_session', 'pick_impersonate']
