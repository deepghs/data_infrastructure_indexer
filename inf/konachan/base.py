"""Session helper for konachan.com.

Konachan refuses ordinary clients from cloud address space on the TLS handshake. The index job
had been answering that by rotating User-Agent strings in a tight loop, which cannot work -
the rejection happens before a header is read - and one cancelled run burned 33,348 attempts
that way.

Measured on a GitHub runner, one egress address, both API endpoints, on 2026-08-09:

    requests (HTTP/1.1)                     403
    httpx (HTTP/2)                          403
    curl_cffi chrome110/116/124/131         403
    curl_cffi edge101                       403
    curl_cffi chrome120                     200 on tag.json, 403 on post.json
    curl_cffi safari15_5 / safari17_0       200 on both
    curl_cffi firefox133                    200 on both

So the ladder below is Safari and Firefox only. Chrome is excluded deliberately: every Chrome
target tested was refused, and chrome120 passing one endpoint but not the other makes it worse
than a clean failure - it would let a run start and then die partway through.
"""
import random
from typing import List, Optional

from curl_cffi import requests as cffi_requests
from ditk import logging

from inf.utils.impersonate import build_ladder

__site_url__ = 'https://konachan.com'

DEFAULT_TIMEOUT = 60.0

#: Fingerprints verified against konachan.com from a GitHub runner. Ordered newest-first within
#: each family. Chrome and Edge are absent on purpose - see the module docstring.
PREFERRED_IMPERSONATES: List[str] = [
    'safari18_0', 'safari17_2_1', 'safari17_0', 'safari15_5', 'safari15_3',
    'firefox135', 'firefox133',
]

#: Fingerprints actually usable here, resolved once against the installed build.
IMPERSONATE_LADDER: List[str] = build_ladder(PREFERRED_IMPERSONATES, site='konachan')


def get_konachan_session(impersonate: Optional[str] = None,
                         timeout: float = DEFAULT_TIMEOUT) -> cffi_requests.Session:
    """
    Build a browser-impersonating session for konachan.com.

    :param impersonate: Fingerprint to use. A random entry of :data:`IMPERSONATE_LADDER` when
        omitted, so a fingerprint that quietly stops working does not strand every run.
    :type impersonate: Optional[str]
    :param timeout: Default per-request timeout in seconds.
    :type timeout: float
    :returns: A ready-to-use session.
    :rtype: curl_cffi.requests.Session
    :raises RuntimeError: When the installed build offers no usable target.
    """
    for _ in range(len(IMPERSONATE_LADDER) + 1):
        chosen = impersonate or random.choice(IMPERSONATE_LADDER)
        try:
            session = cffi_requests.Session(impersonate=chosen, timeout=timeout)
        except Exception as err:
            if 'not supported' not in str(err):
                raise
            # The static filter should have caught this, but a build that reports one set and
            # accepts another would otherwise fail every request as a network error.
            logging.warning(f'Impersonation target {chosen!r} rejected by curl_cffi, dropping it.')
            if chosen in IMPERSONATE_LADDER and len(IMPERSONATE_LADDER) > 1:
                IMPERSONATE_LADDER.remove(chosen)
            impersonate = None
            continue
        session.headers.update({'Referer': f'{__site_url__}/'})
        return session
    raise RuntimeError('No usable impersonation target is available in this curl_cffi build.')


__all__ = ['__site_url__', 'PREFERRED_IMPERSONATES', 'IMPERSONATE_LADDER', 'get_konachan_session']
