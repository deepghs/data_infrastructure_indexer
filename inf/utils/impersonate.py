"""Browser-impersonation helpers shared by the sites that need a TLS fingerprint.

Some boorus reject a request on its TLS handshake alone, before any header is read. The
rejection is indistinguishable from a normal 403, so it looks like an address ban and invites
the wrong fix. Measured on a GitHub runner against konachan, from one egress address in one
job: plain ``requests`` and HTTP/2 ``httpx`` both got 403, Chrome fingerprints got 403, and
Safari and Firefox fingerprints got 200. The address was fine; the handshake was not.

Which fingerprints exist is a property of the installed ``curl_cffi`` build, and the build is a
property of the interpreter - Python 3.8 caps it at 0.9.0, which knows 29 targets, while 0.16.0
knows 53. Naming a target the local build has never heard of raises at session construction,
and downstream that is easily mistaken for a network fault, so the wanted list is filtered
against what the build reports before anything uses it.
"""
import typing
from typing import List, Optional

from ditk import logging


def supported_impersonates() -> Optional[set]:
    """
    Ask the installed ``curl_cffi`` which impersonation targets it actually has.

    :returns: Supported target names, or None when the build cannot be interrogated.
    :rtype: Optional[set]
    """
    try:
        from curl_cffi.requests.impersonate import BrowserTypeLiteral
        return set(typing.get_args(BrowserTypeLiteral))
    except Exception:
        pass
    try:
        from curl_cffi.requests import BrowserType
        return {entry.value for entry in BrowserType}
    except Exception:
        return None


def build_ladder(preferred: List[str], site: str = '') -> List[str]:
    """
    Narrow a wanted fingerprint list to what the installed build supports.

    :param preferred: Fingerprints to use, best first.
    :type preferred: List[str]
    :param site: Site name, used only to make the log line say which list was filtered.
    :type site: str
    :returns: Usable fingerprint names, never empty unless the build reports nothing.
    :rtype: List[str]
    """
    label = f' for {site}' if site else ''
    supported = supported_impersonates()
    if not supported:
        return list(preferred)
    usable = [imp for imp in preferred if imp in supported]
    dropped = [imp for imp in preferred if imp not in supported]
    if dropped:
        logging.info(f'Impersonation targets unavailable in this curl_cffi build{label}, '
                     f'dropped: {", ".join(dropped)}.')
    if not usable:
        # Nothing preferred is available. Rather than fail, take what the build does have.
        usable = sorted(supported - {'chrome', 'edge', 'firefox', 'safari'})
        logging.warning(f'No preferred impersonation target is available{label}; '
                        f'falling back to {len(usable)} build-provided targets.')
    return usable


__all__ = ['supported_impersonates', 'build_ladder']
