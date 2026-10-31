"""Shared pieces for the download hosts that asiaon posts link to."""
import re
import threading
import time

import pyrfc6266
import requests
from pyparsing import ParseException

from inf.utils.transfer import DownloadError, get_download_session, safe_local_name, stream_to_file

MODERN_UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
             'Chrome/129.0.0.0 Safari/537.36')

#: (connect, read) seconds. The read timeout applies to each wait for data, not to a whole transfer.
TIMEOUT = (15, 60)


def get_plain_session() -> requests.Session:
    """A plain requests session with a fixed browser user agent, for the hosts that need no challenge solving."""
    session = requests.Session()
    session.headers['User-Agent'] = MODERN_UA
    return session


def get(session, url: str, **kwargs):
    kwargs.setdefault('timeout', TIMEOUT)
    return session.get(url, **kwargs)


class Pacer:
    """Keeps a minimum gap between requests to one host, across threads."""

    def __init__(self, gap: float):
        self.gap, self._last, self._lock = gap, 0.0, threading.Lock()

    def wait(self):
        with self._lock:
            delay = self._last + self.gap - time.time()
            if delay > 0:
                time.sleep(delay)
            self._last = time.time()


def filename_from_response(resp, default: str = 'package.zip') -> str:
    """The name a download response announces, or ``default``. Only its extension is ever used."""
    disposition = resp.headers.get('Content-Disposition') or ''
    try:
        name = pyrfc6266.parse_filename(disposition)
    except ParseException:
        name = None
    if not name:
        match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', disposition)
        name = match.group(1) if match else None
    return name or default


def check_file_response(resp, host: str, what: str, expected_ctype_not: str = 'text/html') -> None:
    """Raise unless ``resp`` is a successful non-HTML answer, so an error page is never saved as an archive."""
    ctype = (resp.headers.get('Content-Type') or '').lower()
    if resp.status_code != 200:
        raise DownloadError(f'{host}: {what} answered HTTP {resp.status_code}', retryable=resp.status_code >= 500,
                            refused=resp.status_code in (403, 429))
    if ctype.startswith(expected_ctype_not):
        raise DownloadError(f'{host}: {what} answered an HTML page instead of a file', retryable=True)


__all__ = ['DownloadError', 'get_download_session', 'safe_local_name', 'stream_to_file', 'MODERN_UA', 'TIMEOUT',
           'get_plain_session', 'get', 'Pacer', 'filename_from_response', 'check_file_response']
