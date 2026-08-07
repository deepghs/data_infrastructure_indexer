import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Callable, Iterable, List, Optional, Sequence

import requests
from ditk import logging
from tqdm import tqdm

from .session import get_requests_session

DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB


class DownloadSizeMismatch(Exception):
    """Raised when a completed download does not match the size announced by the index."""

    def __init__(self, url: str, expected_size: int, actual_size: int):
        Exception.__init__(self, f'Size mismatch for {url!r} - expected {expected_size}, got {actual_size}.')
        self.url = url
        self.expected_size = expected_size
        self.actual_size = actual_size


def download_file(url: str, dst_file: str, session: Optional[requests.Session] = None,
                  expected_size: Optional[int] = None, chunk_size: int = DEFAULT_CHUNK_SIZE,
                  timeout: Optional[float] = None, **kwargs) -> int:
    """
    Stream a remote file to ``dst_file`` and return the number of bytes written.

    The body is written to a sibling ``.tmp`` path first and renamed only after the transfer
    finishes, so a killed job never leaves a truncated file that a later run would mistake
    for a complete one.

    :param url: Source URL to fetch.
    :type url: str
    :param dst_file: Local destination path. Parent directories are created when missing.
    :type dst_file: str
    :param session: Session to reuse. A fresh retrying session is built when omitted.
    :type session: Optional[requests.Session]
    :param expected_size: When given, the transferred size must match it exactly.
    :type expected_size: Optional[int]
    :param chunk_size: Streaming chunk size in bytes.
    :type chunk_size: int
    :param timeout: Per-request timeout override.
    :type timeout: Optional[float]
    :returns: Number of bytes written.
    :rtype: int
    :raises DownloadSizeMismatch: When ``expected_size`` is given and does not match.
    """
    session = session or get_requests_session()
    if os.path.dirname(dst_file):
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    tmp_file = f'{dst_file}.tmp'

    try:
        if timeout is not None:
            kwargs['timeout'] = timeout
        resp = session.get(url, stream=True, **kwargs)
        resp.raise_for_status()

        size = 0
        with open(tmp_file, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    size += len(chunk)

        if expected_size is not None and size != expected_size:
            raise DownloadSizeMismatch(url, expected_size, size)

        os.replace(tmp_file, dst_file)
        return size

    except Exception:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise


def parallel_call(items: Sequence[Any], fn: Callable[[Any], Any], max_workers: int = 8,
                  desc: Optional[str] = None, silent: bool = False,
                  raise_error: bool = False) -> List[Optional[BaseException]]:
    """
    Run ``fn`` over ``items`` in a thread pool and return one entry per item.

    A failing item yields its exception instead of aborting the pool, because a single dead
    upstream URL must not discard the work already done by the other workers.

    :param items: Work items, passed to ``fn`` one at a time.
    :type items: Sequence[Any]
    :param fn: Callable applied to each item.
    :type fn: Callable[[Any], Any]
    :param max_workers: Thread pool size.
    :type max_workers: int
    :param desc: Progress bar description.
    :type desc: Optional[str]
    :param silent: Suppress the progress bar when True.
    :type silent: bool
    :param raise_error: Re-raise the first captured exception once every item has been attempted.
    :type raise_error: bool
    :returns: Per-item exception, or None when the item succeeded.
    :rtype: List[Optional[BaseException]]
    """
    items = list(items)
    errors: List[Optional[BaseException]] = [None] * len(items)
    if not items:
        return errors

    lock = Lock()
    pg = tqdm(total=len(items), desc=desc, disable=silent)

    def _run(index: int, item: Any):
        try:
            fn(item)
        except BaseException as err:  # noqa: B036 - recorded per item, re-raised by the caller if wanted
            errors[index] = err
        finally:
            with lock:
                pg.update()

    with ThreadPoolExecutor(max_workers=max_workers) as tp:
        for i, item in enumerate(items):
            tp.submit(_run, i, item)
    pg.close()

    if raise_error:
        for err in errors:
            if err is not None:
                raise err
    return errors


def iter_chunks(items: Iterable[Any], chunk_size: int):
    """
    Yield ``items`` in lists of at most ``chunk_size`` entries.

    :param items: Any iterable.
    :type items: Iterable[Any]
    :param chunk_size: Maximum length of each yielded list.
    :type chunk_size: int
    """
    if chunk_size <= 0:
        raise ValueError(f'Chunk size should be positive, but {chunk_size!r} found.')
    buffer = []
    for item in items:
        buffer.append(item)
        if len(buffer) >= chunk_size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def get_free_disk_bytes(path: str) -> int:
    """
    Return the free space in bytes on the filesystem holding ``path``.

    Used to stop a volume early when the runner is close to filling its disk.

    :param path: Any existing path on the filesystem of interest.
    :type path: str
    :returns: Free bytes available to the current user.
    :rtype: int
    """
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def log_disk_usage(path: str, prefix: str = 'Disk'):
    """
    Log free/total space for the filesystem holding ``path``.

    :param path: Any existing path on the filesystem of interest.
    :type path: str
    :param prefix: Log line prefix.
    :type prefix: str
    """
    st = os.statvfs(path)
    free = st.f_bavail * st.f_frsize
    total = st.f_blocks * st.f_frsize
    logging.info(f'{prefix} {path!r}: {free / 1024 ** 3:.2f} GB free of {total / 1024 ** 3:.2f} GB.')


__all__ = [
    'DownloadSizeMismatch',
    'download_file',
    'parallel_call',
    'iter_chunks',
    'get_free_disk_bytes',
    'log_disk_usage',
]
