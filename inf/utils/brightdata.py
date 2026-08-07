"""Bright Data proxy access helpers.

A Bright Data zone can restrict which client addresses may use it. The restriction is
all-or-nothing: once the allowlist has a single entry, every other address is refused with
``407 Auth Failed (code: ip_forbidden)``. That makes the zone unusable from CI, where the
runner's egress address is different on every job and is not known in advance.

The refusal itself carries the answer. Bright Data puts the address it saw into the
``x-brd-err-msg`` header:

    client_10030: The IP address from which you are sending this request: 203.0.113.7 is not
    whitelisted in this zone's settings.

So access can be established without knowing the address beforehand: probe the proxy, read the
address out of the refusal, add it, probe again. Asking an external service such as
``api.ipify.org`` would not do — it reports the egress of a different route and can disagree
with what Bright Data sees.

This module deliberately talks to exactly one account-management endpoint, the allowlist add.
The same API can create zones, rotate credentials and change plans, all of which either cost
money or break other users of the account.
"""
import re
from typing import Optional

import requests
from ditk import logging

from .session import get_requests_session

#: Bright Data account-management API root.
API_ROOT = 'https://api.brightdata.com'

#: Cheap endpoint Bright Data provides for connectivity checks. Plain HTTP on purpose: an
#: ``ip_forbidden`` refusal then arrives as a readable response rather than a failed CONNECT.
PROBE_URL = 'http://lumtest.com/myip.json'

#: Error code meaning the client address is not on the zone allowlist.
IP_FORBIDDEN_CODE = 'client_10030'

_IP_IN_MESSAGE = re.compile(r'request:\s*(?P<ip>\d{1,3}(?:\.\d{1,3}){3})')


class BrightDataError(Exception):
    """Raised when the proxy cannot be made usable."""


def probe_proxy(proxy_url: str, timeout: float = 30.0):
    """
    Try the proxy once and report whether it works, plus the address it refused.

    :param proxy_url: Full proxy URL including credentials.
    :type proxy_url: str
    :param timeout: Request timeout in seconds.
    :type timeout: float
    :returns: Tuple of (usable, blocked address or None, detail string).
    :rtype: Tuple[bool, Optional[str], str]
    """
    session = get_requests_session(max_retries=1, timeout=timeout)
    session.trust_env = False
    proxies = {'http': proxy_url, 'https': proxy_url}
    try:
        resp = session.get(PROBE_URL, proxies=proxies, timeout=timeout)
    except requests.RequestException as err:
        return False, None, f'{type(err).__name__}: {err}'

    message = resp.headers.get('x-brd-err-msg') or resp.text or ''
    code = resp.headers.get('x-brd-err-code') or ''
    if resp.status_code == 200 and 'ip_forbidden' not in message:
        return True, None, resp.text[:200]

    blocked = None
    matched = _IP_IN_MESSAGE.search(message)
    if matched:
        blocked = matched.group('ip')
    return False, blocked, f'HTTP {resp.status_code} {code} {message[:200]}'


def allowlist_ip(api_key: str, ip: str, zone: Optional[str] = None, timeout: float = 30.0):
    """
    Add one address to a zone's allowlist.

    This is the only account-management call this codebase makes. The same API can create
    zones, rotate credentials and change plans; none of that belongs in an automated job.

    :param api_key: Bright Data API key, used as a bearer token.
    :type api_key: str
    :param ip: Address to allow.
    :type ip: str
    :param zone: Zone name. Applies to every zone on the account when omitted.
    :type zone: Optional[str]
    :raises BrightDataError: When the API refuses the addition.
    """
    payload = {'ip': ip}
    if zone:
        payload['zone'] = zone
    resp = requests.post(
        f'{API_ROOT}/zone/whitelist',
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        json=payload,
        timeout=timeout,
    )
    if resp.status_code not in (200, 201):
        raise BrightDataError(f'Allowlisting {ip} in zone {zone!r} failed with '
                              f'HTTP {resp.status_code}: {resp.text[:300]}')
    logging.info(f'Allowlisted {ip} for Bright Data zone {zone or "(all zones)"!r}.')


def ensure_proxy_access(proxy_url: str, api_key: Optional[str] = None, zone: Optional[str] = None,
                        timeout: float = 30.0) -> bool:
    """
    Make the proxy usable from this host, allowlisting the address if that is what is missing.

    :param proxy_url: Full proxy URL including credentials.
    :type proxy_url: str
    :param api_key: Bright Data API key. Without one the address can be reported but not added.
    :type api_key: Optional[str]
    :param zone: Zone to allowlist into.
    :type zone: Optional[str]
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :returns: Whether the proxy is usable.
    :rtype: bool
    :raises BrightDataError: When the address is refused and cannot be added.
    """
    usable, blocked, detail = probe_proxy(proxy_url, timeout=timeout)
    if usable:
        logging.info(f'Bright Data proxy already usable from this host: {detail}')
        return True

    if not blocked:
        logging.warning(f'Bright Data proxy unusable and no client address reported - {detail}')
        return False

    logging.info(f'Bright Data refused this host as {blocked}; adding it to the allowlist.')
    if not api_key:
        raise BrightDataError(
            f'This host appears to Bright Data as {blocked} and is not allowlisted, but no API '
            f'key was supplied. Set BRD_API_KEY, or add {blocked} manually under the zone\'s '
            f'Security settings.'
        )

    allowlist_ip(api_key=api_key, ip=blocked, zone=zone, timeout=timeout)
    usable, _, detail = probe_proxy(proxy_url, timeout=timeout)
    if not usable:
        raise BrightDataError(f'Allowlisted {blocked} but the proxy still refuses us - {detail}')
    logging.info(f'Bright Data proxy usable after allowlisting {blocked}: {detail}')
    return True


__all__ = ['BrightDataError', 'probe_proxy', 'allowlist_ip', 'ensure_proxy_access']
