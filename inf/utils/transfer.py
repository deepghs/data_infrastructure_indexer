"""Pieces shared by the jobs that download packages from file hosts: errors, sessions, safe names, a guarded stream."""
import os
import time
from collections import deque
from typing import Callable, Optional

from ditk import logging

from inf.utils.session import get_cloudscraper_session


class DownloadError(Exception):
    """
    A package could not be fetched from a host, so the next host in the order is tried.

    ``retryable`` marks a failure that is likely to pass on a second attempt, such as a body cut
    short, as opposed to a removed share or a transfer too slow to be worth repeating. ``gone``
    marks a link the host says no longer exists, which is an expected end for an old post rather
    than a fault, so it never counts as one; ``kind`` then says how it is gone (``deleted``,
    ``takedown`` by the host, ``private`` or ``expired``) for the record kept of dead links.
    ``blocked`` marks a host that is refusing this run as a whole, so asking it again for the next
    post would only waste another transfer. ``slow`` marks a transfer abandoned for crawling along:
    one can be a bad file, so the caller counts them per host. ``refused`` marks an HTTP 403 or 429: one can
    be a single file the host will not hand out, but several in a row mean the host is turning this client away,
    so the caller counts those per host as well.
    """

    def __init__(self, message: str, retryable: bool = False, gone: bool = False, blocked: bool = False,
                 slow: bool = False, kind: str = 'deleted', refused: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.gone = gone
        self.blocked = blocked
        self.slow = slow
        self.refused = refused


def get_download_session(timeout=(15, 60)):
    """A cloudscraper session whose requests time out, see :func:`inf.utils.session.get_cloudscraper_session`."""
    return get_cloudscraper_session(timeout=timeout)


#: Archive types the unpacker is willing to be handed. The remote name decides the extension, and
#: the extension decides which tool runs, so anything else is stored without one and rejected.
_ARCHIVE_EXTENSIONS = {'rar', 'zip', '7z'}


def safe_local_name(remote_name: str, default: str = 'package') -> str:
    """
    A local file name for a remote one.

    Remote names are only ever used for display; the archive is stored under a fixed stem so a
    hostile or odd name cannot reach outside the download directory, and only a known archive
    extension is carried over.
    """
    _, ext = os.path.splitext(os.path.basename(remote_name or ''))
    ext = ext.lstrip('.').lower()
    return f'{default}.{ext}' if ext in _ARCHIVE_EXTENSIONS else default


#: A transfer averaging less than this over ``_STALL_WINDOW`` is abandoned, since at that speed a
#: multi-gigabyte package would not finish inside the run.
_MIN_SPEED = 1 * 1024 ** 2
_STALL_WINDOW = 5 * 60

#: Seconds between progress lines, so a slow transfer is visible in the log instead of looking hung.
_REPORT_EVERY = 60


def stream_to_file(response, dst_file: str, expected_size: Optional[int] = None,
                   chunk_size: int = 1 << 20, clock: Callable[[], float] = time.time) -> int:
    """
    Write a streaming response to ``dst_file`` and return the bytes written.

    The transfer is abandoned when the average speed over the last five minutes falls below
    1 MiB/s, and a body that ends short of ``Content-Length`` is an error rather than a
    truncated archive passed on to the unpacker.
    """
    if expected_size is None and response.headers.get('Content-Length'):
        expected_size = int(response.headers['Content-Length'])

    window = deque()  # (timestamp, size) of chunks inside the stall window
    window_bytes, written = 0, 0
    started = last_report = clock()
    os.makedirs(os.path.dirname(dst_file) or '.', exist_ok=True)
    try:
        with open(dst_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                now = clock()
                if now - last_report >= _REPORT_EVERY:
                    last_report = now
                    total = f' of {expected_size / 1024 ** 2:.0f} MiB' if expected_size else ''
                    logging.info(f'Downloading {os.path.basename(dst_file)!r}: {written / 1024 ** 2:.0f} MiB{total}, '
                                 f'{written / max(now - started, 1e-6) / 1024 ** 2:.1f} MiB/s on average.')
                window.append((now, len(chunk)))
                window_bytes += len(chunk)
                while window and window[0][0] < now - _STALL_WINDOW:
                    window_bytes -= window.popleft()[1]
                if window and now - window[0][0] >= _STALL_WINDOW * 0.99 and window_bytes / _STALL_WINDOW < _MIN_SPEED:
                    raise DownloadError(f'Too slow, {window_bytes / _STALL_WINDOW / 1024:.0f} KiB/s over '
                                        f'{_STALL_WINDOW}s, abandoned.', slow=True)
        if expected_size is not None and written != expected_size:
            raise DownloadError(f'Incomplete download, got {written} of {expected_size} bytes.', retryable=True)
    except Exception:
        if os.path.exists(dst_file):
            os.remove(dst_file)
        raise
    finally:
        response.close()
    logging.info(f'Downloaded {written} bytes to {os.path.basename(dst_file)!r}.')
    return written
