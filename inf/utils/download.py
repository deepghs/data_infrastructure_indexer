import os
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5
from threading import Lock
from typing import Any, Callable, Iterable, List, Optional, Sequence

import httpx
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


class DownloadDigestMismatch(Exception):
    """Raised when a completed download does not match the md5 announced by the index."""

    def __init__(self, url: str, expected_md5: str, actual_md5: str):
        Exception.__init__(self, f'MD5 mismatch for {url!r} - expected {expected_md5!r}, got {actual_md5!r}.')
        self.url = url
        self.expected_md5 = expected_md5
        self.actual_md5 = actual_md5


class AdaptiveRateLimiter:
    """
    Pace requests towards a metered endpoint, converging on the rate it will actually serve.

    A site that answers 429 is stating a rate, not refusing the work. Guessing that rate through
    a worker count does not work: too few workers leave throughput on the table, too many turn
    most requests into rejections. Measured on ``cdn.donmai.us`` from a GitHub runner, eight
    workers delivered 1.7 images/s with 0.84 rejections each, while twenty-four delivered 6.0/s
    but rejected nine of every ten requests and gave up on 58% of the batch.

    So the rate is discovered instead: additive increase while requests are served,
    multiplicative decrease when one is throttled. The decrease is rate-limited itself, because
    a burst of 429s from concurrent workers describes one overload, not twenty.

    :param initial: Requests per second to start from.
    :type initial: float
    :param minimum: Floor, so a bad patch cannot stall the run entirely.
    :type minimum: float
    :param maximum: Ceiling, so the limiter cannot run away on an unmetered stretch.
    :type maximum: float
    :param increase: Requests per second added after each ``probe_every`` successes.
    :type increase: float
    :param decrease: Multiplier applied on throttling.
    :type decrease: float
    :param probe_every: Successes between increases.
    :type probe_every: int
    :param decrease_cooldown: Minimum seconds between two decreases.
    :type decrease_cooldown: float
    """

    def __init__(self, initial: float = 4.0, minimum: float = 0.5, maximum: float = 64.0,
                 increase: float = 0.5, decrease: float = 0.7, probe_every: int = 25,
                 decrease_cooldown: float = 2.0):
        self._rate = float(initial)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._increase = float(increase)
        self._decrease = float(decrease)
        self._probe_every = max(int(probe_every), 1)
        self._decrease_cooldown = float(decrease_cooldown)

        self._lock = Lock()
        self._next_slot = 0.0
        self._successes = 0
        self._last_decrease = 0.0
        self._throttles = 0

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def throttles(self) -> int:
        return self._throttles

    def acquire(self):
        """Block until this caller's turn, keeping the fleet at the current rate."""
        with self._lock:
            now = time.time()
            start = max(now, self._next_slot)
            self._next_slot = start + 1.0 / max(self._rate, 1e-6)
        delay = start - time.time()
        if delay > 0:
            time.sleep(delay)

    def report_success(self):
        """Record a served request, probing upwards every so often."""
        with self._lock:
            self._successes += 1
            if self._successes >= self._probe_every:
                self._successes = 0
                self._rate = min(self._maximum, self._rate + self._increase)

    def report_throttled(self):
        """Record a 429, backing the rate off unless we just did."""
        with self._lock:
            self._throttles += 1
            now = time.time()
            if now - self._last_decrease < self._decrease_cooldown:
                return
            self._last_decrease = now
            self._successes = 0
            self._rate = max(self._minimum, self._rate * self._decrease)


def download_file(url: str, dst_file: str, session=None, expected_size: Optional[int] = None,
                  expected_md5: Optional[str] = None, chunk_size: int = DEFAULT_CHUNK_SIZE,
                  timeout: Optional[float] = None, **kwargs) -> int:
    """
    Stream a remote file to ``dst_file`` and return the number of bytes written.

    Accepts either a ``requests.Session`` or an ``httpx.Client``; sites behind a bot
    classifier generally need the HTTP/2 client, while everything else is happy with
    ``requests``. Both raise an error object exposing ``.response.status_code``, so callers
    can classify failures the same way regardless of which one they passed.

    The body is written to a sibling ``.tmp`` path first and renamed only after the transfer
    finishes, so a killed job never leaves a truncated file that a later run would mistake
    for a complete one.

    :param url: Source URL to fetch.
    :type url: str
    :param dst_file: Local destination path. Parent directories are created when missing.
    :type dst_file: str
    :param session: ``requests.Session`` or ``httpx.Client`` to reuse. A fresh retrying
        ``requests`` session is built when omitted.
    :param expected_size: When given, the transferred size must match it exactly.
    :type expected_size: Optional[int]
    :param expected_md5: When given, the digest is computed from the stream and must match.
        Hashing inline avoids reading the whole file back off disk a second time.
    :type expected_md5: Optional[str]
    :param chunk_size: Streaming chunk size in bytes.
    :type chunk_size: int
    :param timeout: Per-request timeout override.
    :type timeout: Optional[float]
    :returns: Number of bytes written.
    :rtype: int
    :raises DownloadSizeMismatch: When ``expected_size`` is given and does not match.
    :raises DownloadDigestMismatch: When ``expected_md5`` is given and does not match.
    """
    session = session or get_requests_session()
    if os.path.dirname(dst_file):
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    tmp_file = f'{dst_file}.tmp'
    if timeout is not None:
        kwargs['timeout'] = timeout
    hash_obj = md5() if expected_md5 else None

    try:
        size = 0
        if isinstance(session, httpx.Client):
            with session.stream('GET', url, **kwargs) as resp:
                resp.raise_for_status()
                with open(tmp_file, 'wb') as f:
                    for chunk in resp.iter_bytes(chunk_size):
                        if chunk:
                            f.write(chunk)
                            size += len(chunk)
                            if hash_obj is not None:
                                hash_obj.update(chunk)
        else:
            resp = session.get(url, stream=True, **kwargs)
            resp.raise_for_status()
            with open(tmp_file, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        size += len(chunk)
                        if hash_obj is not None:
                            hash_obj.update(chunk)

        if expected_size is not None and size != expected_size:
            raise DownloadSizeMismatch(url, expected_size, size)
        if hash_obj is not None:
            actual = hash_obj.hexdigest()
            if actual != expected_md5:
                raise DownloadDigestMismatch(url, expected_md5, actual)

        os.replace(tmp_file, dst_file)
        return size

    except Exception:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise


def parallel_call(items: Sequence[Any], fn: Callable[[Any], Any], max_workers: int = 8,
                  desc: Optional[str] = None, silent: bool = False, raise_error: bool = False,
                  postfix: Optional[Callable[[], dict]] = None) -> List[Optional[BaseException]]:
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
    :param postfix: Called on each completion to label the bar with live counters. Carrying
        outcome counts on the bar keeps them visible without one log line per item.
    :type postfix: Optional[Callable[[], dict]]
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
                if postfix is not None:
                    pg.set_postfix(postfix(), refresh=False)
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
    'AdaptiveRateLimiter',
    'DownloadSizeMismatch',
    'DownloadDigestMismatch',
    'download_file',
    'parallel_call',
    'iter_chunks',
    'get_free_disk_bytes',
    'log_disk_usage',
]
