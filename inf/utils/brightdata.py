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
import time
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


_PROXY_URL = re.compile(r'^(?P<scheme>\w+)://(?P<user>[^:@/]+):(?P<password>[^@/]+)@(?P<host>.+)$')


class BrightDataError(Exception):
    """Raised when the proxy cannot be made usable."""


def with_session(proxy_url: str, session_id: str) -> str:
    """
    Pin a proxy URL to one egress address by tagging the username with a session id.

    Without a tag Bright Data draws a different exit address for every single request, so a
    per-address rate limit downstream is never actually escaped: each request is a fresh
    lottery, and a rejection teaches you nothing reusable. Tagging ``-session-<id>`` holds one
    exit address for as long as that id is used, which turns the proxy into a pool of
    addressable identities. Verified: four requests on one tag all exited from the same
    address, and a different tag exited from a different one.

    :param proxy_url: Base proxy URL of the form ``scheme://user:password@host:port``.
    :type proxy_url: str
    :param session_id: Identifier to pin on. Letters and digits only, by Bright Data's rules.
    :type session_id: str
    :returns: Proxy URL bound to that session.
    :rtype: str
    :raises BrightDataError: When the URL cannot be parsed.
    """
    matched = _PROXY_URL.match(proxy_url)
    if not matched:
        raise BrightDataError(f'Cannot add a session id to proxy URL {proxy_url.split("@")[-1]!r}: '
                              f'expected scheme://user:password@host:port.')
    user = matched.group('user')
    if '-session-' in user:
        user = user.split('-session-')[0]
    safe = re.sub(r'[^0-9a-zA-Z]', '', session_id) or 'x'
    return (f'{matched.group("scheme")}://{user}-session-{safe}:'
            f'{matched.group("password")}@{matched.group("host")}')


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
    # trust_env stays on deliberately. If the host reaches the internet through a local proxy,
    # Bright Data sees that proxy's egress rather than this machine, and the probe has to travel
    # the same path the real downloads will or it allowlists an address nothing else uses.
    session = get_requests_session(max_retries=1, timeout=timeout)
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
    # The reference documents 201, but the live API answers 204 for an accepted addition.
    if resp.status_code // 100 != 2:
        raise BrightDataError(f'Allowlisting {ip} in zone {zone!r} failed with '
                              f'HTTP {resp.status_code}: {resp.text[:300]}')
    logging.info(f'Allowlisted {ip} for Bright Data zone {zone or "(all zones)"!r}.')


def ensure_proxy_access(proxy_url: str, api_key: Optional[str] = None, zone: Optional[str] = None,
                        timeout: float = 30.0, propagation_timeout: float = 180.0,
                        propagation_poll: float = 15.0) -> bool:
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
    :param propagation_timeout: How long to keep re-probing after a successful addition. The
        proxy edge does not honour a new entry immediately.
    :type propagation_timeout: float
    :param propagation_poll: Seconds between those probes.
    :type propagation_poll: float
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

    # An accepted addition is not immediately in force at the proxy edge; measured propagation
    # took upwards of a minute. Re-probing straight away reports a refusal that has already been
    # resolved, so wait it out instead of concluding the proxy is unusable.
    deadline = time.time() + propagation_timeout
    attempt = 0
    while True:
        attempt += 1
        usable, _, detail = probe_proxy(proxy_url, timeout=timeout)
        if usable:
            logging.info(f'Bright Data proxy usable after allowlisting {blocked} '
                         f'(took {attempt} probe(s)): {detail}')
            return True
        if time.time() >= deadline:
            raise BrightDataError(f'Allowlisted {blocked} but the proxy still refuses us after '
                                  f'{propagation_timeout:.0f}s - {detail}')
        logging.info(f'Allowlist not in force yet at the edge, probe {attempt}; waiting ...')
        time.sleep(propagation_poll)


__all__ = ['BrightDataError', 'probe_proxy', 'allowlist_ip', 'ensure_proxy_access']
