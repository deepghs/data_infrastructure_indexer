"""Staging download stage for Fancaps.

Pull the original image bytes of the screenshots listed in ``deepghs/fancaps_index`` and pack them into
hfutils-indexed tar volumes inside a private staging repository (default ``deepghs/fancaps_newest_dl``).
A later repack stage consumes those volumes instead of hitting the CDN again.

Layout published to the staging repository
==========================================

::

    images/
      0/001.tar        hfutils-indexed tar volume
      0/001.json       sibling sidecar -- per-entry {offset,size,sha256}
      ...
    table.parquet      one row per image written into a volume
    meta.json          {max_volume_id, bad_image_ids}
    glob_exist_ids.json
                       read-only baseline: every id ``fancaps_full`` already holds, bad ids included
    README.md          statistics, coverage and usage

The repository therefore holds a *difference*, not a corpus: the images the index knows about that
``fancaps_full`` does not already store. Every commit carries one whole volume with its sidecar, the
table, the meta file and the README, so the repository is consistent at every commit. Candidates are
downloaded from the lowest id to the highest.

What measuring the image CDN showed, and what the code does about it
====================================================================

* The host is ``cdni.fancaps.net/file/fancaps-animeimages/{id}.jpg``. No ``Referer`` is needed.
* Cloudflare Polish is on: a cache miss returns the original, later hits return a smaller re-encode.
  A random ``?cb=`` query forces a miss every time, so the bytes stored here are the originals
  ``fancaps_full`` also holds. A response that announces a re-encode (``cf-polished`` with an
  ``orig_size`` different from the body) is retried under another query. Browser fingerprints of the
  Chrome family are served WebP instead of JPEG, so the default fingerprint is Safari, and anything that
  does not start with the JPEG magic is never stored.
* The limit is Cloudflare error 1015 **per egress address**: about 40 requests in ten seconds, then a
  ``Retry-After`` of an hour that blocks every fingerprint at once. Three requests a second were served
  for as long as it was measured; four a second tripped it within fifteen seconds. Retries count.
  That is why this job paces by **egress address** with a fixed ceiling instead of the adaptive limiter
  the other staging jobs use: the penalty for one request too many is an hour, so there is no rate to
  probe for. Several workers share one address's pacer, and the pacer is keyed by the address the
  egress really exits from, not by the proxy session that was asked for, because two sessions can land
  on one address.
* On the first 429/403 an address is retired for an hour and its slot is re-opened on another
  address; when too many addresses are retired at once the run stops instead of burning the pool.
* The index carries no md5 and no size. Integrity is therefore the announced ``Content-Length``, the
  JPEG magic, and a successful decode; the table records the md5 of the bytes that were stored.
"""
import datetime
import faulthandler
import hashlib
import io
import json
import math
import os
import re
import secrets
import signal
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image
from curl_cffi import requests as cffi_requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.index import tar_create_index_for_directory
from hfutils.operate import get_hf_client
from hfutils.utils import hf_normpath, number_to_tag
from pyrate_limiter import Rate, Limiter, Duration
from tqdm import tqdm

from inf.utils.brightdata import BrightDataError, with_session
from inf.utils.download import parallel_call, get_free_disk_bytes, log_disk_usage
from inf.utils.duration import duration_type
from inf.utils.safe import configure_hf_http_backend, safe_hf_hub_download, \
    safe_upload_directory_as_directory

Image.MAX_IMAGE_PIXELS = 32768 ** 2

#: Image CDN url of one id; ``cb`` is a random query that forces a cache miss, see the module docstring.
CDN_URL = 'https://cdni.fancaps.net/file/fancaps-animeimages/{id}.jpg'
#: Reports the address the proxy really exits from. Plain HTTP on purpose, like the Bright Data helpers.
EXIT_PROBE_URL = 'http://lumtest.com/myip.json'

#: Columns this job adds to every row. Every column of the index is carried next to them unchanged and is not
#: listed anywhere: the index is read whole, so a column it gains later is carried too and nothing it holds is
#: ever dropped. The index's own ``id`` is the one name both sides share.
_STAGING_COLUMNS = ['id', 'filename', 'volume_file', 'file_size', 'mimetype', 'file_ext', 'width', 'height', 'md5',
                    'file_url']


def index_columns_of(names: List[str]) -> List[str]:
    """The index's columns other than ``id``, in the index's order; one that clashes with a staging name is an error."""
    clash = [x for x in names if x != 'id' and x in _STAGING_COLUMNS]
    if clash:
        raise ValueError(f'The index has column(s) {clash} that this job already writes itself; one of the two '
                         f'has to be renamed before they can share a table.')
    return [x for x in names if x != 'id']


def table_columns_of(names: List[str]) -> List[str]:
    """The staging table's columns: ours first, then every column of the index."""
    return _STAGING_COLUMNS + index_columns_of(names)


#: Browser fingerprints; Chrome and Edge are served WebP, so Safari and Firefox come first.
_FINGERPRINTS = ['safari17_0', 'firefox135', 'safari15_5']

#: Mean bytes of one image, measured on staged volumes (118 to 130 KB), used only to plan how many images a
#: volume should try to hold; the run rescales it from every finished volume and the hard ceiling decides
#: what really fits.
_BYTES_PER_IMAGE = 125 * 1024

#: The hour Cloudflare blocks an address for, as measured.
DEFAULT_BAN_SECONDS = 3600.0


def estimate_bytes(row: Optional[dict] = None) -> int:
    """A rough size of one image for volume planning."""
    return _BYTES_PER_IMAGE


def _volume_paths(volume_id: int) -> Tuple[str, str]:
    rel_tar = f'images/{volume_id // 1000}/{volume_id % 1000:03d}.tar'
    return rel_tar, f'{os.path.splitext(rel_tar)[0]}.json'


def image_url(image_id: int, cache_buster: Optional[str] = None) -> str:
    """The url one download uses: a random query makes the CDN answer with the original every time."""
    return f'{CDN_URL.format(id=image_id)}?cb={cache_buster or secrets.token_hex(8)}'


# ---------------------------------------------------------------------------------------------------------
# Reading a response
# ---------------------------------------------------------------------------------------------------------

class FetchError(Exception):
    """A download that did not produce a usable image; ``kind`` says what to do about it."""

    def __init__(self, kind: str, message: str, status: Optional[int] = None, retry_after: Optional[float] = None):
        Exception.__init__(self, message)
        self.kind = kind
        self.status = status
        self.retry_after = retry_after


_ORIG_SIZE = re.compile(r'orig_size=(\d+)')


def _retry_after(headers: dict) -> Optional[float]:
    try:
        return float(headers.get('retry-after'))
    except (TypeError, ValueError):
        return None


def judge_response(status: int, headers: dict, body: bytes) -> None:
    """
    Raise the :class:`FetchError` a response deserves, or return when it is a usable original.

    Kinds: ``'blocked'`` (429, or a Cloudflare 403: the address is limited, retire it), ``'missing'`` (404/410, a
    suspicion that is looked at again slowly before an id is recorded bad), ``'bad_body'`` (a 200 that is
    not the file: wrong type, truncated, or a re-encode) and ``'network'`` (everything else).
    """
    headers = {str(k).lower(): v for k, v in (headers or {}).items()}
    if status == 429 or (status == 403 and (headers.get('cf-mitigated') or b'1015' in (body or b'')[:4096]
                                            or b'cloudflare' in (body or b'')[:4096].lower())):
        raise FetchError('blocked', f'HTTP {status}', status=status, retry_after=_retry_after(headers))
    if status in (404, 410):
        raise FetchError('missing', f'HTTP {status}', status=status)
    if status != 200:
        raise FetchError('network', f'HTTP {status}', status=status)
    if not body:
        raise FetchError('bad_body', 'empty body (a zero-length 200)', status=status)
    if body[:2] != b'\xff\xd8':
        raise FetchError('bad_body', f'not a JPEG (starts with {body[:4]!r}, type {headers.get("content-type")!r})',
                         status=status)
    announced = headers.get('content-length')
    if announced is not None and not headers.get('content-encoding'):
        try:
            if int(announced) != len(body):
                raise FetchError('bad_body', f'truncated: announced {announced} bytes, got {len(body)}', status=status)
        except ValueError:
            pass
    polished = str(headers.get('cf-polished') or '')
    matched = _ORIG_SIZE.search(polished)
    if matched and int(matched.group(1)) != len(body):
        raise FetchError('bad_body', f're-encoded by the CDN ({polished!r}, got {len(body)} bytes)', status=status)


def decode_size(data: bytes) -> Tuple[int, int]:
    """Width and height of an image, verifying that it decodes."""
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        image.verify()
    return width, height


# ---------------------------------------------------------------------------------------------------------
# Egress: addresses, pacing, retirement
# ---------------------------------------------------------------------------------------------------------

class EgressExhausted(Exception):
    """Every egress slot is retired, or too many were retired at once to keep going."""


class IpPacer:
    """
    Keeps the requests of one egress address at least ``1 / rate`` seconds apart, whoever sends them.

    ``rate`` is a number or a callable returning one; the pool passes a callable so that a rate it lowers
    or restores takes effect on the next request of every address.
    """

    def __init__(self, rate):
        self._rate = rate if callable(rate) else (lambda: float(rate))
        self._lock = Lock()
        self._next = 0.0

    def acquire(self):
        with self._lock:
            now = time.time()
            start = max(now, self._next)
            self._next = start + 1.0 / max(float(self._rate()), 1e-6)
        delay = start - time.time()
        if delay > 0:
            time.sleep(delay)


def _make_session(proxy_url: Optional[str], fingerprint: str):
    kwargs = {}
    if proxy_url:
        kwargs['proxies'] = {'http': proxy_url, 'https': proxy_url}
    return cffi_requests.Session(impersonate=fingerprint, **kwargs)


def _probe_exit_ip(session, timeout: float = 30.0) -> Optional[str]:
    """The address a session exits from, or None. A failed probe is routine (the proxy now and then answers it
    with something else than json) and the callers treat it as 'no information', so it is only logged at debug."""
    try:
        return str(session.get(EXIT_PROBE_URL, timeout=timeout).json().get('ip') or '') or None
    except Exception as err:
        logging.debug(f'Exit address probe failed - {err!r}')
        return None


class Lane:
    """
    One slot of the egress: sessions, the address they exit from, and what is allowed through them.

    The slot outlives its address. When the address is retired the slot is re-opened on another one, and
    the workers that hold a token of the slot simply wait for that.
    """

    def __init__(self, pool: 'EgressPool', index: int):
        self.pool = pool
        self.index = index
        self.generation = 0
        self.sessions: Optional[Queue] = None
        self.exit_ip: Optional[str] = None
        self.dead = False
        self.proxy_slot = True
        self.requests = 0
        self.last_used = 0.0
        self.fail_streak = 0
        self.ready = Event()
        self._reopen_lock = Lock()

    def fetch(self, url: str, timeout: float = 60.0) -> Tuple[int, dict, bytes]:
        """
        One paced request; raises :class:`EgressExhausted` when the slot is gone for good.

        A request starts over whenever the slot moved to another address while it waited, so that it is never
        sent, or paced, on an address the slot no longer has.
        """
        pool = self.pool
        while True:
            if not self.ready.wait(timeout=pool.reopen_timeout):
                raise EgressExhausted(f'lane {self.index} did not come back')
            if self.dead:
                raise EgressExhausted(f'lane {self.index} is retired')
            sessions, generation = self.sessions, self.generation
            # A curl session is not safe to use from two threads at once, so each request leases one exclusively.
            # Every session of a slot carries the same pinned proxy session id, hence the same exit address. The
            # wait has a timeout because the queue this thread waits on can be emptied for good when the slot is
            # re-opened: it then has to look at the slot again instead of waiting for a session nobody returns.
            try:
                session = sessions.get(timeout=pool.session_wait)
            except Empty:
                continue
            moved = False
            try:
                if generation != self.generation or self.dead:
                    moved = True
                else:
                    now = time.time()
                    due = pool.proxied and (
                            (pool.reprobe_every and (self.requests + 1) % pool.reprobe_every == 0)
                            or now - self.last_used > pool.idle_recheck)
                    if due:
                        # A proxy session that sat idle (an upload, a pause) may well have been given a new
                        # address; find out before a request goes out on the old identity.
                        pool.verify_exit(self, session, generation)
                        moved = generation != self.generation or self.dead
                    if not moved:
                        pool.pacer(self.exit_ip).acquire()
                        self.requests += 1
                        response = session.get(url, timeout=timeout)
                        self.last_used = time.time()
                        return response.status_code, dict(response.headers), response.content
            finally:
                if generation == self.generation and not self.dead:
                    sessions.put(session)
                else:
                    EgressPool._close_session(session)
            if not moved:  # pragma: no cover - the return above leaves the loop
                continue

    def retire(self, generation: int, ban_seconds: float, reason: str):
        """Ban the address the slot had at ``generation`` and re-open the slot elsewhere (once per generation)."""
        self.pool.retire(self, generation, ban_seconds, reason)

    def note(self, ok: bool, generation: int):
        """Record whether the slot's address served; an address that keeps failing is swapped for another."""
        if ok:
            self.fail_streak = 0
            return
        self.fail_streak += 1
        if self.fail_streak >= self.pool.max_lane_failures and self.pool.proxied:
            self.fail_streak = 0
            self.pool.reroll(self, generation, f'{self.pool.max_lane_failures} failures in a row')


class EgressPool:
    """
    The set of egress slots a run downloads through. Nothing in it is tuned to one particular proxy pool:
    how many slots there are, how fast an address may be driven and which addresses are good are all found
    out while running, so replacing the whole set of proxy addresses needs no change here.

    :param proxy_url: Proxy URL. Slots are pinned to their own session (Bright Data style username tag) and
        so to their own address; a proxy that cannot tag sessions still works, as one address.
    :param lanes: Number of proxy slots; zero or less means *discover*: open slots in parallel up to
        ``max_lanes`` until the pool has no address left to give, and in that case keep only ``1 - reserve``
        of what was found so that retired slots have addresses to move to.
    :param direct: Also (or only) use this host's own address as a slot.
    :param rate_per_ip: Ceiling of requests per second one address carries, across every worker using it. The
        pool lowers the effective rate when addresses get limited anyway (see ``rate_backoff``) and raises it
        back, never above this, after ``rate_recover_every`` seconds without a limit.
    :param sessions_per_lane: Sessions kept per slot, one per worker that can be in flight on it.
    :param reprobe_every: Requests of a slot between two checks of the address it really exits from. The proxy can
        move a pinned session to another address underneath, and a slot that moved onto an address another slot
        uses doubles that address's rate; the check bounds how long that can last.
    :param open_wait: Seconds a slot waits for a usable address when re-opening, instead of dying at once.
    :param max_retired: Addresses retired within ``retire_window`` seconds after which the pool refuses to go on
        (the circuit breaker, which stays open for the rest of the run).
    :param max_lane_failures: Consecutive failures after which a slot is moved to another address.
    :param idle_recheck: Seconds of silence on a slot after which its address is checked before the next request,
        because a pinned proxy session that sat idle (for example while a volume uploads) may have been moved.
    :param session_wait: Seconds a request waits for a free session before it looks at the slot again.
    :param session_factory: ``(proxy_url, fingerprint) -> session``; replaceable for tests.
    :param probe: ``session -> ip``; replaceable for tests.
    """

    def __init__(self, proxy_url: Optional[str] = None, lanes: int = 0, direct: bool = False,
                 rate_per_ip: float = 2.5, max_retired: Optional[int] = None,
                 fingerprints: Optional[List[str]] = None, reprobe_every: int = 30, reopen_timeout: float = 900.0,
                 open_wait: float = 900.0, max_address_tries: int = 12, retire_window: float = 900.0,
                 sessions_per_lane: int = 3, max_lanes: int = 64, reserve: float = 0.2, open_parallelism: int = 8,
                 min_rate: float = 1.0, rate_backoff: float = 0.85, rate_recover_every: float = 600.0,
                 max_lane_failures: int = 5, idle_recheck: float = 15.0, session_wait: float = 5.0,
                 session_factory: Callable = _make_session, probe: Callable = _probe_exit_ip):
        if not proxy_url and not direct:
            raise ValueError('The pool needs a proxy url, the direct route, or both.')
        self.proxy_url = proxy_url
        self.proxied = bool(proxy_url)
        self.auto = bool(proxy_url) and int(lanes) <= 0
        self.n_proxy_lanes = max(int(lanes), 0) if proxy_url else 0
        self.max_lanes = max(int(max_lanes), 1)
        self.reserve = min(max(float(reserve), 0.0), 0.9)
        self.open_parallelism = max(int(open_parallelism), 1)
        self.direct = direct
        self.rate_cap = float(rate_per_ip)
        self.rate_per_ip = float(rate_per_ip)
        self.min_rate = min(float(min_rate), self.rate_cap)
        self.rate_backoff = rate_backoff
        self.rate_recover_every = rate_recover_every
        self._last_block = 0.0
        self._last_rate_change = 0.0
        self.fingerprints = list(fingerprints or _FINGERPRINTS)
        self.reprobe_every = reprobe_every
        self.reopen_timeout = reopen_timeout
        self.open_wait = open_wait
        self.max_address_tries = max_address_tries
        self.max_lane_failures = max_lane_failures
        self.idle_recheck = idle_recheck
        self.session_wait = session_wait
        self.sessions_per_lane = max(int(sessions_per_lane), 1)
        self._session_factory = session_factory
        self._probe = probe
        self._lock = Lock()
        self._pacers: Dict[str, IpPacer] = {}
        self._banned: Dict[str, float] = {}
        self._in_use: Dict[str, int] = {}
        self._salt = secrets.token_hex(3)
        self._counter = 0
        self._warned_identity = False
        self.lanes: List[Lane] = []
        self.retired_total = 0
        self.retire_window = retire_window
        self._retire_times: List[float] = []
        self._tripped = False
        self._max_retired_arg = max_retired
        self.max_retired = max_retired if max_retired is not None else 2

    # -- addresses --------------------------------------------------------------------------------------

    def pacer(self, ip: Optional[str]) -> IpPacer:
        key = ip or 'unknown'
        with self._lock:
            self._maybe_recover_rate()
            if key not in self._pacers:
                self._pacers[key] = IpPacer(lambda: self.rate_per_ip)
            return self._pacers[key]

    def _note_block(self):
        """An address was limited at the current rate: ease off every address, at most once a minute."""
        with self._lock:
            now = time.time()
            self._last_block = now
            if now - self._last_rate_change >= 60.0:
                lowered = max(self.min_rate, self.rate_per_ip * self.rate_backoff)
                if lowered < self.rate_per_ip:
                    logging.warning(f'An address was limited at {self.rate_per_ip:.2f} req/s, '
                                    f'lowering the rate to {lowered:.2f}.')
                    self.rate_per_ip = lowered
                self._last_rate_change = now

    def _maybe_recover_rate(self):
        """Called under the lock. A quiet stretch earns some of the rate back, up to the configured ceiling."""
        if self.rate_per_ip >= self.rate_cap:
            return
        now = time.time()
        if now - max(self._last_block, self._last_rate_change) >= self.rate_recover_every:
            self.rate_per_ip = min(self.rate_cap, self.rate_per_ip * 1.1)
            self._last_rate_change = now
            logging.info(f'No address was limited for {self.rate_recover_every:.0f}s, '
                         f'raising the rate to {self.rate_per_ip:.2f} req/s.')

    def _try_claim(self, lane: Lane, ip: str) -> bool:
        """Atomically take an address for a slot, unless it is banned or another slot already holds it."""
        with self._lock:
            if self._banned.get(ip, 0.0) > time.time():
                return False
            if self._in_use.get(ip, 0) > 0 and lane.exit_ip != ip:
                return False
            if lane.exit_ip and lane.exit_ip != ip and self._in_use.get(lane.exit_ip, 0):
                self._in_use[lane.exit_ip] -= 1
            if lane.exit_ip != ip:
                self._in_use[ip] = self._in_use.get(ip, 0) + 1
            lane.exit_ip = ip
            lane.last_used = time.time()  # the address was just seen, there is nothing to re-check yet
            return True

    def _release(self, lane: Lane):
        with self._lock:
            if lane.exit_ip and self._in_use.get(lane.exit_ip, 0):
                self._in_use[lane.exit_ip] -= 1
        self._close_sessions(lane.sessions)
        lane.sessions = None

    def _tagged(self, tag: str) -> str:
        """The proxy url pinned to one session; a proxy that cannot be tagged is used as it is."""
        try:
            return with_session(self.proxy_url, tag)
        except BrightDataError:
            if not self._warned_identity:
                self._warned_identity = True
                logging.warning('The proxy url cannot carry a session tag, every slot will use it as it is.')
            return self.proxy_url

    def _open(self, lane: Lane, proxy_slot: bool, wait: bool = False) -> bool:
        """Give the slot sessions on an address nobody else uses and nobody banned; False when none is found."""
        lane.proxy_slot = proxy_slot
        if not proxy_slot:
            if not self._try_claim(lane, 'direct'):
                return False
            lane.sessions = self._build_sessions(None, self.fingerprints[0])
            return True
        deadline = time.time() + (self.open_wait if wait else 0.0)
        while True:
            probe_failures = 0
            url = None
            for _ in range(self.max_address_tries):
                with self._lock:
                    self._counter += 1
                    tag = f'fc{self._salt}l{lane.index}n{self._counter}'
                url = self._tagged(tag)
                fingerprint = self.fingerprints[lane.index % len(self.fingerprints)]
                first = self._session_factory(url, fingerprint)
                ip = self._probe(first)
                if ip is None:
                    probe_failures += 1
                elif self._try_claim(lane, ip):
                    lane.sessions = self._build_sessions(url, fingerprint, first=first)
                    return True
                self._close_session(first)
            if probe_failures and probe_failures == self.max_address_tries:
                # The address cannot be read off this proxy at all, so the slot is told apart by its session
                # id and paced on its own; collisions between slots cannot be seen then, only limited.
                if not self._warned_identity:
                    self._warned_identity = True
                    logging.warning('The exit address cannot be probed through this proxy; slots are paced '
                                    'per session instead of per address.')
                identity = f'session-{tag}'
                first = self._session_factory(url, self.fingerprints[lane.index % len(self.fingerprints)])
                if self._try_claim(lane, identity):
                    lane.sessions = self._build_sessions(url, self.fingerprints[lane.index % len(self.fingerprints)],
                                                         first=first)
                    return True
                self._close_session(first)
            if time.time() >= deadline or self.tripped:
                return False
            logging.warning(f'Lane {lane.index} found no usable address, waiting for one ...')
            time.sleep(min(30.0, max(deadline - time.time(), 0.1)))

    def _build_sessions(self, proxy_url: Optional[str], fingerprint: str, first=None) -> Queue:
        sessions: Queue = Queue()
        for index in range(self.sessions_per_lane):
            sessions.put(first if (index == 0 and first is not None)
                         else self._session_factory(proxy_url, fingerprint))
        return sessions

    @staticmethod
    def _close_session(session):
        try:
            session.close()
        except Exception:  # pragma: no cover - best effort
            pass

    @staticmethod
    def _close_sessions(sessions: Optional[Queue]):
        while sessions is not None and not sessions.empty():
            try:
                EgressPool._close_session(sessions.get_nowait())
            except Empty:  # pragma: no cover
                break

    def _open_all(self, lanes: List[Lane], proxy_slot: bool) -> List[Lane]:
        """Open the given slots in parallel; the ones that found an address are returned."""
        if not lanes:
            return []
        with ThreadPoolExecutor(max_workers=min(self.open_parallelism, len(lanes))) as tp:
            results = list(tp.map(lambda x: self._open(x, proxy_slot), lanes))
        opened = [x for x, ok in zip(lanes, results) if ok]
        for lane in opened:
            lane.ready.set()
        return opened

    def open(self) -> 'EgressPool':
        """
        Open the slots; at least one has to remain. With ``lanes`` explicit, that many proxy slots are opened (the
        ones that find no free address are dropped). Otherwise the pool is discovered: batches of slots are opened
        until ``max_lanes`` is reached or a whole batch finds nothing new, which means every address the proxy
        hands out is in use.
        """
        found: List[Lane] = []
        index = 0
        if self.auto:
            saturated = False
            while len(found) < self.max_lanes and not saturated:
                batch = [Lane(self, index + k) for k in range(min(self.open_parallelism, self.max_lanes - len(found)))]
                index += len(batch)
                opened = self._open_all(batch, proxy_slot=True)
                found.extend(opened)
                saturated = not opened
                logging.info(f'Discovering the pool: {len(found)} distinct address(es) so far.')
            if saturated and len(found) > 1:
                keep = max(1, int(len(found) * (1.0 - self.reserve)))
                for lane in found[keep:]:
                    self._release(lane)
                logging.info(f'The pool has no more addresses to give: using {keep} of the {len(found)} found, '
                             f'the rest stay free for slots that have to move.')
                found = found[:keep]
        elif self.n_proxy_lanes:
            found = self._open_all([Lane(self, k) for k in range(self.n_proxy_lanes)], proxy_slot=True)
            index = self.n_proxy_lanes
        if self.direct:
            found.extend(self._open_all([Lane(self, index)], proxy_slot=False))
        for number, lane in enumerate(found):
            lane.index = number
            where = 'the direct route' if not lane.proxy_slot else f'proxy address {lane.exit_ip}'
            logging.debug(f'Lane {number} ready on {where}.')
        self.lanes = found
        if not self.lanes:
            raise EgressExhausted('No egress slot could be opened.')
        # The breaker opens when this many addresses were limited within the window: half of the slots.
        self.max_retired = self._max_retired_arg if self._max_retired_arg is not None \
            else max(2, len(self.lanes) // 2 + 1)
        logging.info(f'Egress ready: {plural_word(len(self.lanes), "lane")} on '
                     f'{len({x.exit_ip for x in self.lanes})} distinct address(es), '
                     f'{self.rate_per_ip:.2f} req/s per address, breaker at {self.max_retired} retired addresses '
                     f'within {self.retire_window:.0f}s.')
        return self

    @property
    def total_rate(self) -> float:
        return self.rate_per_ip * len([x for x in self.lanes if not x.dead])

    # -- retirement -------------------------------------------------------------------------------------

    @property
    def tripped(self) -> bool:
        """True once the circuit breaker opened or every slot is dead."""
        return self._tripped or not any(not x.dead for x in self.lanes)

    def retire(self, lane: Lane, generation: int, ban_seconds: float, reason: str):
        """The CDN limited this address: ban it, count it towards the breaker, and re-open the slot elsewhere."""
        self._replace(lane, generation, reason, ban_seconds=ban_seconds)

    def reroll(self, lane: Lane, generation: int, reason: str):
        """Move the slot to another address without blaming the old one."""
        self._replace(lane, generation, reason, ban_seconds=None)

    def _replace(self, lane: Lane, generation: int, reason: str, ban_seconds: Optional[float]):
        """Re-open a slot on another address; ``ban_seconds=None`` re-rolls it without blaming the old address."""
        with lane._reopen_lock:
            if generation != lane.generation or lane.dead:
                return  # another worker already handled this address
            lane.ready.clear()
            ip = lane.exit_ip or 'unknown'
            if ban_seconds is not None:
                with self._lock:
                    self._banned[ip] = time.time() + max(ban_seconds, 1.0)
                    self.retired_total += 1
                    now = time.time()
                    self._retire_times = [t for t in self._retire_times if now - t < self.retire_window] + [now]
                    if len(self._retire_times) >= self.max_retired:
                        self._tripped = True
                logging.warning(f'RETIRE address {ip} of lane {lane.index} for {ban_seconds:.0f}s - {reason} '
                                f'({len(self._retire_times)}/{self.max_retired} retired within '
                                f'{self.retire_window:.0f}s{", BREAKER OPEN" if self._tripped else ""}).')
                self._note_block()
            else:
                logging.warning(f'Re-rolling lane {lane.index} off address {ip} - {reason}.')
            lane.generation += 1
            self._close_sessions(lane.sessions)
            reopened = False
            if not self.tripped:
                reopened = self._open(lane, proxy_slot=lane.proxy_slot, wait=True)
            if reopened:
                logging.info(f'Lane {lane.index} re-opened on {lane.exit_ip}.')
            else:
                lane.dead = True
                logging.warning(f'Lane {lane.index} is retired for good.')
            lane.ready.set()

    def verify_exit(self, lane: Lane, session, generation: int):
        """The proxy may move a session to another address underneath; if it did, follow it or leave it."""
        ip = self._probe(session)
        if ip and ip != lane.exit_ip and generation == lane.generation:
            logging.warning(f'Lane {lane.index} moved from {lane.exit_ip} to {ip} by itself.')
            if not self._try_claim(lane, ip):
                # Nobody limited the old address; the slot just must not share the new one.
                self.reroll(lane, generation, f'moved onto address {ip} that is banned or used by another lane')

    def close(self):
        for lane in self.lanes:
            self._close_sessions(lane.sessions)


class StallWatchdog:
    """
    Raises the alarm when a run stops making progress, with every thread's stack in the log.

    A volume can only end when every worker has finished; if one is stuck for any reason, the run would sit
    there silently for good. The watchdog cannot free it, but it makes the stall visible and shows where it is.
    """

    def __init__(self, progress: Callable[[], int], stall_seconds: float = 300.0, interval: float = 30.0,
                 on_stall: Optional[Callable[[float], None]] = None):
        self._progress = progress
        self.stall_seconds = stall_seconds
        self.interval = interval
        self._on_stall = on_stall or self._dump
        self._stop = Event()
        self._thread = Thread(target=self._run, name='stall-watchdog', daemon=True)
        self.alarms = 0

    @staticmethod
    def _dump(stalled_for: float):
        logging.error(f'STALL: no progress for {stalled_for:.0f}s, thread stacks follow.')
        faulthandler.dump_traceback(all_threads=True)

    def _run(self):
        last, since = self._progress(), time.time()
        while not self._stop.wait(self.interval):
            current = self._progress()
            if current != last:
                last, since = current, time.time()
            elif time.time() - since >= self.stall_seconds:
                self.alarms += 1
                self._on_stall(time.time() - since)
                since = time.time()  # the next alarm is a full period away

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


# ---------------------------------------------------------------------------------------------------------
# State and candidates
# ---------------------------------------------------------------------------------------------------------

def _load_baseline(hf_client, baseline_repository: str) -> np.ndarray:
    """Every id ``baseline_repository`` holds, bad ids included, as a sorted int64 array."""
    path = safe_hf_hub_download(hf_client, repo_id=baseline_repository, repo_type='dataset', filename='meta.json')
    with open(path, 'r') as f:
        meta = json.load(f)
    parts = [np.asarray(meta.get(key) or [], dtype=np.int64) for key in ('exist_ids', 'bad_ids')]
    del meta
    return np.unique(np.concatenate(parts))


def _parse_id_list(path: str) -> np.ndarray:
    """A json list of ints as a sorted int64 array, parsed straight from the text without building python ints."""
    with open(path, 'r') as f:
        text = f.read().strip().strip('[]').strip()
    return np.unique(np.fromstring(text, dtype=np.int64, sep=',')) if text else np.zeros(0, dtype=np.int64)


def _load_state(hf_client, repository: str, baseline_repository: str, glob_exist_ids_file: str):
    """
    The staging repository's state: ``(table, extra covered ids, bad ids, max volume id, baseline,
    baseline_is_new)``. The baseline is read from the repository when it holds one, and built from
    ``baseline_repository`` otherwise; in that case the first commit has to carry it.

    The stored table stays an arrow table, never a list of dicts: with millions of rows, turning it into python
    objects needs more memory than a 16 GB runner has, while the arrow columns take a few GB.
    """
    table: Optional[pa.Table] = None
    bad_image_ids: set = set()
    max_volume_id = 0

    baseline_is_new = False
    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename=glob_exist_ids_file):
        path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset',
                                    filename=glob_exist_ids_file)
        baseline = _parse_id_list(path)
    else:
        baseline = _load_baseline(hf_client, baseline_repository)
        baseline_is_new = True
        logging.info(f'No {glob_exist_ids_file!r} in {repository!r}, built one from {baseline_repository!r}.')
    logging.info(f'Baseline loaded, {plural_word(len(baseline), "id")}.')

    extra_parts = []
    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='table.parquet'):
        path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset', filename='table.parquet')
        table = pq.read_table(path)
        extra_parts.append(table.column('id').to_numpy().astype(np.int64))
        logging.info(f'Existing table loaded, {plural_word(table.num_rows, "row")}.')

    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='meta.json'):
        path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset', filename='meta.json')
        with open(path, 'r') as f:
            meta = json.load(f)
        bad_image_ids.update(int(x) for x in (meta.get('bad_image_ids') or []))
        max_volume_id = meta.get('max_volume_id') or 0
        extra_parts.append(np.fromiter(bad_image_ids, dtype=np.int64, count=len(bad_image_ids)))
        logging.info(f'Meta loaded, max_volume_id={max_volume_id}, {plural_word(len(bad_image_ids), "bad id")}.')

    extra_covered = np.unique(np.concatenate(extra_parts)) if extra_parts else np.zeros(0, dtype=np.int64)
    return table, extra_covered, bad_image_ids, max_volume_id, baseline, baseline_is_new


def _last_publish_time(hf_client, repository: str) -> Optional[datetime.datetime]:
    """When the newest volume was committed to the staging repository, or None when it has none yet."""
    for commit in hf_client.list_repo_commits(repo_id=repository, repo_type='dataset'):
        if commit.title.startswith('Sync volume'):
            return commit.created_at
    return None


def _append_rows(staged: Optional[pa.Table], new_records: List[dict], table_columns: List[str]) -> pa.Table:
    """The stored table with a volume's rows added, sorted by id like every table this job has written."""
    if staged is None:
        df = pd.DataFrame(new_records)[table_columns].sort_values(by=['id'], ascending=[True])
        return pa.Table.from_pandas(df, preserve_index=False)
    new = pa.Table.from_pylist(new_records, schema=staged.schema)
    combined = pa.concat_tables([staged, new])
    return combined.take(pc.sort_indices(combined, sort_keys=[('id', 'ascending')]))


def _iter_index_batches(src_repository: str, src_revision: str) -> Iterator[pa.RecordBatch]:
    """
    The index's rows, every column of them, batch by batch.

    Every column is read, which is nearly the whole file, so the file is downloaded once and read from local disk
    rather than pulled through a swarm of range requests.
    """
    hf_client = get_hf_client()
    path = safe_hf_hub_download(hf_client, repo_id=src_repository, repo_type='dataset', filename='fancaps.parquet',
                                revision=None if (not src_revision or src_revision == 'main') else src_revision)
    pf = pq.ParquetFile(path)
    with tqdm(total=pf.metadata.num_rows, desc='Scanning fancaps.parquet') as pg:
        for batch in pf.iter_batches(batch_size=500000):
            pg.update(batch.num_rows)
            yield batch


def scan_candidates(batches: Iterator[pa.RecordBatch], baseline: np.ndarray, extra_covered) -> pa.Table:
    """
    The images still to download as one columnar table, lowest id first.

    Held as arrow columns and sliced per volume: the first backlog is millions of rows, far too many for a
    list of dicts.
    """
    extra = np.asarray(extra_covered if isinstance(extra_covered, np.ndarray)
                       else np.fromiter(extra_covered, dtype=np.int64, count=len(extra_covered)), dtype=np.int64)
    parts = []
    for batch in batches:
        ids = batch.column('id').to_numpy(zero_copy_only=False).astype(np.int64)
        pos = np.searchsorted(baseline, ids)
        pos[pos >= len(baseline)] = max(len(baseline) - 1, 0)
        in_base = (baseline[pos] == ids) if len(baseline) else np.zeros(len(ids), dtype=bool)
        keep = ~in_base
        if len(extra):
            keep &= ~np.isin(ids, extra)
        if keep.any():
            parts.append(pa.Table.from_batches([batch]).filter(pa.array(keep)))
    if not parts:
        return pa.Table.from_pylist([], schema=pa.schema([('id', pa.int64())]))
    table = pa.concat_tables(parts)
    # The index can list an id twice (an episode re-crawled); a download is per id.
    table = table.take(pc.sort_indices(table, sort_keys=[('id', 'ascending')]))
    ids = table.column('id').to_numpy(zero_copy_only=False)
    first = np.ones(len(ids), dtype=bool)
    first[1:] = ids[1:] != ids[:-1]
    return table.filter(pa.array(first))


def initial_size_scale(sizes, window: int = 20000) -> float:
    """
    How far the planning estimate is off for the images being stored now, learned from the newest stored sizes.

    Image sizes drift with the id range (older and newer episodes differ a lot), so a run that resumes starts
    from what the repository already shows rather than from the built-in guess. ``sizes`` are file sizes in bytes,
    oldest first.
    """
    sizes = [x for x in list(sizes)[-window:] if x]
    if not sizes:
        return 1.0
    return min(max(sum(sizes) / len(sizes) / estimate_bytes(), 0.3), 3.0)


def take_volume(count: int, max_volume_files: int, max_volume_bytes: int, scale: float = 1.0,
                margin: float = 1.0) -> int:
    """
    How many of the next ``count`` candidates one volume is planned to hold, by estimated bytes times ``scale``.

    ``margin`` plans for more than the budget holds: the run seals a volume when the bytes really written reach the
    budget, so a plan that is too generous costs nothing (what is not downloaded moves to the next volume), while one
    that is too small leaves the volume under-filled whenever image sizes change faster than the estimate follows.
    """
    by_bytes = int(max_volume_bytes * max(margin, 1.0) // max(estimate_bytes() * scale, 1.0))
    return max(min(count, max_volume_files, by_bytes), 1 if count else 0)


def _rows(table: pa.Table) -> List[dict]:
    return [{**row, 'id': int(row['id'])} for row in table.to_pylist()]


def _size_text(num_bytes: int) -> str:
    """Bytes as the largest unit that keeps the number readable, GB below a terabyte."""
    return f'{num_bytes / 1024 ** 4:.3f} TB' if num_bytes >= 1024 ** 4 else f'{num_bytes / 1024 ** 3:.2f} GB'


def write_readme(md_file: str, table: pa.Table, bad_image_ids: set, max_volume_id: int, src_repository: str,
                 repository: str, baseline_repository: str, baseline: np.ndarray, glob_exist_ids_file: str):
    n_rows = table.num_rows
    total_bytes = int(pc.sum(table.column('file_size')).as_py() or 0) if n_rows else 0
    columns = table.column_names
    current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    baseline_max = int(baseline.max()) if len(baseline) else 0
    with open(md_file, 'w') as f:
        print('---', file=f)
        print('license: other', file=f)
        print('task_categories:', file=f)
        print('- image-classification', file=f)
        print('- zero-shot-image-classification', file=f)
        print('- text-to-image', file=f)
        print('language:', file=f)
        print('- en', file=f)
        print('- ja', file=f)
        print('tags:', file=f)
        print('- art', file=f)
        print('- anime', file=f)
        print('size_categories:', file=f)
        print(f'- {number_to_tag(n_rows)}', file=f)
        print('annotations_creators:', file=f)
        print('- no-annotation', file=f)
        print('source_datasets:', file=f)
        print('- fancaps', file=f)
        print('---', file=f)
        print('', file=f)

        print('# Fancaps Staging Download Dataset', file=f)
        print('', file=f)
        print(f'Raw original screenshots of [Fancaps](https://fancaps.net/) anime episodes listed in '
              f'`{src_repository}` that **`{baseline_repository}` does not already store**. This repository is a '
              f'staging area for the repack stage, and it is deliberately incomplete on its own.', file=f)
        print('', file=f)

        print('# Coverage: what is here, what is not', file=f)
        print('', file=f)
        print(f'This repository is the **complement of one collection**, `{baseline_repository}`. Before anything is '
              f'downloaded, every id that collection already holds is removed from the candidate list, so what lands '
              f'here is strictly what it is missing. Candidates are downloaded from the lowest id to the highest, so '
              f'the stored ids grow monotonically with the volume number.', file=f)
        print('', file=f)
        print(f'The exclusion set ships here as `{glob_exist_ids_file}` ({len(baseline):,} ids, the ids of '
              f'`{baseline_repository}` plus its handful of permanently-unavailable ones, highest {baseline_max:,}). '
              f'It is read-only: this job reads it to decide what to skip and never adds to it.', file=f)
        print('', file=f)
        print('So for the complete, up-to-date set of Fancaps anime screenshots:', file=f)
        print('', file=f)
        print('```text', file=f)
        print(f'{baseline_repository}   ({len(baseline):,} ids, up to {baseline_max:,})', file=f)
        print(f'  + {repository}   ({n_rows:,} images, what that collection lacks)', file=f)
        print('```', file=f)
        print('', file=f)
        print(f'The two are disjoint and neither is redundant. Taking only this repository leaves out the '
              f'{len(baseline):,} ids `{baseline_repository}` holds. Image ids are globally unique, so a union '
              f'needs no deduplication.', file=f)
        print('', file=f)

        print('# How to fetch one image', file=f)
        print('', file=f)
        print('Both collections are hfutils-indexed tar stores, so a single image is a range request rather than a '
              'whole-tar download. They differ in how the tar holding an image is found:', file=f)
        print('', file=f)
        print('| collection | tar holding image `id` | index | entry name |', file=f)
        print('|---|---|---|---|', file=f)
        print(f'| `{repository}` (this one) | `volume_file` from `table.parquet` | beside the tars | '
              f'`{{id}}.jpg` |', file=f)
        print(f'| `{baseline_repository}` | `images/{{id % 1000:04d}}.tar` | beside the tars | `{{id}}.jpg` |', file=f)
        print('', file=f)
        print('```python', file=f)
        print('import pandas as pd', file=f)
        print('from huggingface_hub import hf_hub_download', file=f)
        print('from hfutils.index import hf_tar_file_download', file=f)
        print('', file=f)
        print('', file=f)
        print('def fetch(image_id: int, dst: str) -> str:', file=f)
        print('    """Download one original into dst; return which collection served it."""', file=f)
        print(f'    table = pd.read_parquet(hf_hub_download(repo_id="{repository}", repo_type="dataset",', file=f)
        print('                                            filename="table.parquet"))', file=f)
        print('    hit = table[table["id"] == image_id]', file=f)
        print('    if len(hit):', file=f)
        print('        row = hit.iloc[0]', file=f)
        print(f'        hf_tar_file_download(repo_id="{repository}", repo_type="dataset",', file=f)
        print('                             archive_in_repo=row["volume_file"],', file=f)
        print('                             file_in_archive=row["filename"], local_file=dst)', file=f)
        print('        return "staging"', file=f)
        print(f'    hf_tar_file_download(repo_id="{baseline_repository}", repo_type="dataset",', file=f)
        print('                         archive_in_repo=f"images/{image_id % 1000:04d}.tar",', file=f)
        print('                         file_in_archive=f"{image_id}.jpg", local_file=dst)', file=f)
        print(f'    return "{baseline_repository.split("/")[-1]}"', file=f)
        print('```', file=f)
        print('', file=f)
        print('This repository and its baseline are private and live on a self-hosted endpoint: set `HF_ENDPOINT` '
              'to it and `HF_TOKEN` to a token with access.', file=f)
        print('', file=f)

        print('# Layout of this repository', file=f)
        print('', file=f)
        print('```text', file=f)
        print('images/0/001.tar     hfutils-indexed tar volume', file=f)
        print('images/0/001.json    sidecar: {filesize, hash, files: {name: {offset, size, sha256}}}', file=f)
        print('images/0/002.tar', file=f)
        print('...', file=f)
        print('table.parquet        one row per stored image', file=f)
        print('meta.json            {max_volume_id, bad_image_ids}', file=f)
        print(f'{glob_exist_ids_file}  read-only exclusion baseline (see above)', file=f)
        print('```', file=f)
        print('', file=f)
        print('Volumes are numbered from 1 and laid out as `images/{n // 1000}/{n % 1000:03d}.tar`. They are '
              'append-only: an existing volume is never rewritten, so any tar and sidecar you have already downloaded '
              'stays valid forever and only new volumes need fetching on an update.', file=f)
        print('', file=f)
        print('`table.parquet` columns: ' + ', '.join(f'`{c}`' for c in columns) + '. `volume_file` and '
              '`filename` are the two you need to pull the bytes. `md5` is the md5 of the stored bytes and '
              '`file_size` their length; the index lists neither, so integrity at download time is the announced '
              'content length, the JPEG signature and a successful decode.', file=f)
        print('', file=f)
        print(f'Every column of the index `{src_repository}` is carried into this table unchanged, for the same '
              f'image id: nothing the index records about an image (its page and thumbnail urls, its show and '
              f'episode ids, names and urls, and any column the index gains later) is left out. `file_url` is the '
              f'CDN address the bytes came from, which is not the index\'s `url`.', file=f)
        print('', file=f)
        print('`meta.json` carries `bad_image_ids`: images the index lists but the CDN answers 404 or 410 for, or '
              'serves bytes that never decode. They will never appear here and are not worth retrying.', file=f)
        print('', file=f)

        print('# Information', file=f)
        print('', file=f)
        shows = ''
        if n_rows and {'episode_id', 'bangumi_id'} <= set(columns):
            shows = (f' from {pc.count_distinct(table.column("episode_id")).as_py():,} episodes of '
                     f'{pc.count_distinct(table.column("bangumi_id")).as_py():,} shows')
        print(f'There are {plural_word(n_rows, "image")} in total across '
              f'{plural_word(max_volume_id, "volume")}, {_size_text(total_bytes)} of original bytes{shows}. '
              f'The maximum id is {int(pc.max(table.column("id")).as_py()) if n_rows else 0}. '
              f'{plural_word(len(bad_image_ids), "image")} are recorded as permanently unavailable. '
              f'Last updated at `{current_time}`.', file=f)
        print('', file=f)

        if n_rows:
            df_shown = table.slice(max(n_rows - 50, 0)).to_pandas().sort_values(by=['id'], ascending=[False])
            df_shown = df_shown[[c for c in ('id', 'volume_file', 'filename', 'width', 'height', 'file_size',
                                             'bangumi_id', 'bangumi_name', 'episode_id', 'episode_name')
                                 if c in df_shown.columns]]
            print(f'These are the {plural_word(len(df_shown), "most recent image")} (highest ids first):', file=f)
            print('', file=f)
            print(df_shown.to_markdown(index=False), file=f)
            print('', file=f)


# ---------------------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------------------

def sync(repository: str, src_repository: str, baseline_repository: str, src_revision: str = 'main',
         max_time_limit: Optional[float] = 5 * 60 * 60, max_volume_files: int = 100000,
         max_volume_bytes: int = int(7.5 * 1024 ** 3), max_volume_hard_bytes: int = 9 * 1024 ** 3,
         workers_per_lane: int = 3, min_free_disk: int = 24 * 1024 ** 3, upload_time_span: float = 30,
         glob_exist_ids_file: str = 'glob_exist_ids.json', max_volumes: Optional[int] = None,
         proxy_pool: Optional[str] = None, proxy_lanes: int = 0, max_lanes: int = 64, direct: Optional[bool] = None,
         rate_per_ip: float = 2.5, max_retired: Optional[int] = None, plan_margin: float = 1.2,
         min_publish_bytes: int = 0, max_publish_interval: Optional[float] = None, force: bool = False,
         retries: int = 6, retry_wait: float = 2.0,
         max_failed_ratio: float = 0.3, min_attempts_for_abort: int = 20, missing_retries: int = 2,
         diagnose_attempts: int = 2, diagnose_wait: float = 2.0, dry_run: bool = False,
         pool: Optional[EgressPool] = None):
    """
    Download the missing Fancaps originals into append-only tar volumes of the staging repository.

    :param repository: Staging dataset repository.
    :param src_repository: The index repository.
    :param baseline_repository: The repository whose ids are not downloaded again.
    :param workers_per_lane: Download threads per egress address; they share its pacer.
    :param proxy_pool: Bright Data proxy URL. Without it the run uses this host's own address.
    :param proxy_lanes: Proxy slots, each pinned to its own egress address; zero discovers how many the pool can
        give, up to ``max_lanes``.
    :param max_lanes: Ceiling of discovered slots.
    :param direct: Use this host's own address as a slot as well; by default only when there is no proxy.
    :param rate_per_ip: Requests per second one address may carry, the ceiling measured safe is 3.
    :param max_retired: Addresses retired by the CDN after which the run stops.
    :param min_publish_bytes: Publish nothing unless at least this many bytes (estimated) are pending, and never
        publish a trailing volume below it. Zero turns the bar off. This is what keeps the volumes of a scheduled job
        from being small when the site was quiet for a few days.
    :param max_publish_interval: Seconds after the newest volume past which the bar is waived, so that data cannot lag
        behind forever when the site is quiet; None never waives it.
    :param force: Waive the bar for this run.
    :param plan_margin: A volume is planned for this many times its byte budget and sealed when the bytes really
        written reach the budget; see :func:`take_volume`.
    :param retries: Attempts per image before it counts as failed.
    :param missing_retries: Answers of 404/410 (each on a fresh turn, without waiting) before an image is suspect.
    :param diagnose_attempts: Slow looks, in parallel across images, at a suspect before it is recorded bad.
    :param dry_run: Download and build, but upload nothing.
    :param pool: A ready egress pool; for tests.
    """
    start_time = time.time()
    configure_hf_http_backend()
    delete_detached_cache()
    hf_client = get_hf_client()

    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        logging.info(f'Staging repository {repository!r} created.')

    staged, extra_covered, bad_image_ids, max_volume_id, baseline, baseline_is_new = _load_state(
        hf_client=hf_client, repository=repository, baseline_repository=baseline_repository,
        glob_exist_ids_file=glob_exist_ids_file)
    logging.info(f'Current max volume id: {max_volume_id}.')

    candidates = scan_candidates(_iter_index_batches(src_repository, src_revision), baseline, extra_covered)
    if not candidates.num_rows:
        logging.info('No candidate to download, quit.')
        return
    index_columns = index_columns_of(candidates.schema.names)
    table_columns = table_columns_of(candidates.schema.names)
    logging.info(f'The index has {len(index_columns) + 1} columns, all carried into the table: '
                 f'{", ".join(["id"] + index_columns)}.')
    missing_columns = [c for c in table_columns if staged is not None and c not in staged.column_names]
    if missing_columns:
        raise RuntimeError(f'The existing table of {repository!r} lacks column(s) {missing_columns} that the index '
                           f'has; refusing to append rows that would carry them while older rows do not.')
    ids_col = candidates.column('id')
    logging.info(f'{plural_word(candidates.num_rows, "candidate")} to download, ids {ids_col[0].as_py()} - '
                 f'{ids_col[candidates.num_rows - 1].as_py()}, about '
                 f'{candidates.num_rows * estimate_bytes() / 1024 ** 4:.2f} TB estimated.')

    # real bytes / estimated bytes: started from the newest stored rows, then learned from the volumes this run finishes
    size_scale = initial_size_scale(
        staged.column('file_size').slice(max(staged.num_rows - 20000, 0)).to_pylist() if staged is not None else [])
    if staged is not None:
        logging.info(f'Planning with a size scale of {size_scale:.2f} from the newest stored rows.')

    # The bar below which a run is not worth a volume of its own.
    waived = force or min_publish_bytes <= 0
    if not waived:
        pending_estimate = candidates.num_rows * estimate_bytes() * size_scale
        if pending_estimate < min_publish_bytes:
            last = _last_publish_time(hf_client, repository)
            age = None if last is None else (datetime.datetime.now(datetime.timezone.utc) - last).total_seconds()
            if max_publish_interval is not None and (age is None or age >= max_publish_interval):
                logging.info(f'Only about {pending_estimate / 1024 ** 3:.2f} GB is pending, below the '
                             f'{min_publish_bytes / 1024 ** 3:.2f} GB bar, but the newest volume is '
                             f'{"missing" if age is None else f"{age / 86400:.1f} days old"}: publishing anyway.')
                waived = True
            else:
                logging.info(f'Only about {pending_estimate / 1024 ** 3:.2f} GB is pending, below the '
                             f'{min_publish_bytes / 1024 ** 3:.2f} GB bar and the newest volume is '
                             f'{"missing" if age is None else f"{age / 86400:.1f} days old"}: nothing to publish '
                             f'this time.')
                return

    if pool is None:
        pool = EgressPool(proxy_url=proxy_pool, lanes=proxy_lanes if proxy_pool else 0, max_lanes=max_lanes,
                          direct=(not proxy_pool) if direct is None else direct, rate_per_ip=rate_per_ip,
                          max_retired=max_retired, sessions_per_lane=workers_per_lane).open()
    slots: Queue = Queue()
    for lane in pool.lanes:
        for _ in range(max(workers_per_lane, 1)):
            slots.put(lane)
    n_workers = slots.qsize()
    logging.info(f'{n_workers} download workers, about {pool.total_rate:.1f} images/s at most.')

    volumes_done = 0
    pending = candidates
    parked: List[dict] = []  # suspects the CDN would not settle this run; asked again next run
    aborted = False
    while pending.num_rows:
        if max_time_limit is not None and start_time + max_time_limit < time.time():
            logging.info('Max time limit exceeded, stop scheduling new volumes.')
            break
        if max_volumes is not None and volumes_done >= max_volumes:
            logging.info(f'Reached --max-volumes {max_volumes}, stop.')
            break
        if pool.tripped:
            logging.error('The egress is retired, stop.')
            aborted = True
            break
        log_disk_usage(os.getcwd(), prefix='Disk before volume')
        free = get_free_disk_bytes(os.getcwd())
        if free < min_free_disk:
            logging.warning(f'Only {free / 1024 ** 3:.2f} GB free, below the {min_free_disk / 1024 ** 3:.2f} GB '
                            f'floor - stop cleanly.')
            break

        if not waived and pending.num_rows * estimate_bytes() * size_scale < min_publish_bytes:
            logging.info(f'The remaining {plural_word(pending.num_rows, "image")} (about '
                         f'{pending.num_rows * estimate_bytes() * size_scale / 1024 ** 3:.2f} GB) are below the '
                         f'{min_publish_bytes / 1024 ** 3:.2f} GB bar, left for a later run.')
            break
        end = take_volume(pending.num_rows, max_volume_files, max_volume_bytes, scale=size_scale, margin=plan_margin)
        plan, rest = _rows(pending.slice(0, end)), pending.slice(end)
        max_volume_id += 1
        rel_tar, rel_index = _volume_paths(max_volume_id)
        logging.info(f'Building volume #{max_volume_id} ({rel_tar}), {plural_word(len(plan), "image")}, '
                     f'ids {plan[0]["id"]} - {plan[-1]["id"]}, about '
                     f'{len(plan) * estimate_bytes() * size_scale / 1024 ** 3:.2f} GB planned '
                     f'(size scale {size_scale:.2f}) ...')

        with TemporaryDirectory() as td:
            upload_dir = os.path.join(td, 'upload')
            tar_file = os.path.join(upload_dir, rel_tar)
            os.makedirs(os.path.dirname(tar_file), exist_ok=True)

            new_records: List[dict] = []
            volume_bad: List[int] = []
            suspects: List[dict] = []
            deferred: List[dict] = []
            volume_bytes = [0]
            sealed = [None]
            control = [None]  # one item of this volume that downloaded fine, for the diagnosis
            lock = Lock()
            stats = {'ok': 0, 'gone': 0, 'unusable': 0, 'failed': 0, 'suspect': 0, 'retry': 0, 'deferred': 0,
                     'blocked': 0}
            started = time.time()

            with tarfile.open(tar_file, 'w:') as tar:
                def _fetch(item, lane) -> Tuple[bytes, int, int]:
                    """One attempt: the bytes with their size, judged and decoded."""
                    generation = lane.generation
                    try:
                        status, headers, body = lane.fetch(image_url(item['id']), timeout=60)
                    except EgressExhausted:
                        raise
                    except Exception as net_err:
                        lane.note(False, generation)
                        raise FetchError('network', repr(net_err)) from net_err
                    try:
                        judge_response(status, headers, body)
                    except FetchError as err:
                        if err.kind == 'blocked':
                            lane.retire(generation, err.retry_after or DEFAULT_BAN_SECONDS, str(err))
                        else:
                            # 404 and 410 are the CDN answering properly; anything else is the address failing.
                            lane.note(err.kind == 'missing', generation)
                        raise
                    try:
                        width, height = decode_size(body)
                    except Exception as decode_err:
                        lane.note(False, generation)
                        raise FetchError('bad_body', f'does not decode: {decode_err!r}', status=status) from decode_err
                    lane.note(True, generation)
                    return body, width, height

                def _store(item, body, width, height):
                    """Put a downloaded image in the tar and its row in the records, under the lock."""
                    filename = f'{item["id"]}.jpg'
                    info = tarfile.TarInfo(filename)
                    info.size = len(body)
                    info.mtime = int(time.time())
                    info.mode = 0o644
                    with lock:
                        tar.addfile(info, io.BytesIO(body))
                        volume_bytes[0] += len(body)
                        stats['ok'] += 1
                        if control[0] is None:
                            control[0] = item
                        new_records.append({
                            'id': item['id'], 'filename': filename, 'volume_file': hf_normpath(rel_tar),
                            'file_size': len(body), 'mimetype': 'image/jpeg', 'file_ext': 'jpg',
                            'width': width, 'height': height, 'md5': hashlib.md5(body).hexdigest(),
                            'file_url': CDN_URL.format(id=item['id']),
                            **{column: item.get(column) for column in index_columns},
                        })
                        if not sealed[0]:
                            if volume_bytes[0] >= max_volume_bytes:
                                sealed[0] = f'the byte budget of {max_volume_bytes / 1024 ** 3:.2f} GB is reached'
                                logging.info(f'Sealing volume #{max_volume_id} - {sealed[0]}; the rest of the plan '
                                             f'moves to the next volume.')
                            elif volume_bytes[0] >= max_volume_hard_bytes:
                                sealed[0] = (f'tar reached {volume_bytes[0] / 1024 ** 3:.2f} GB, over the '
                                             f'{max_volume_hard_bytes / 1024 ** 3:.2f} GB ceiling')
                            elif get_free_disk_bytes(td) < min_free_disk:
                                sealed[0] = f'free disk fell below {min_free_disk / 1024 ** 3:.2f} GB'
                            elif max_time_limit is not None and time.time() > start_time + max_time_limit:
                                sealed[0] = 'run deadline reached mid-volume'
                            if sealed[0] and 'byte budget' not in sealed[0]:
                                logging.warning(f'Sealing volume #{max_volume_id} early - {sealed[0]}; remaining '
                                                f'images move to the next volume.')

                def _lease():
                    """A slot token, or None once the egress is gone; never blocks forever on dead slots."""
                    while True:
                        if pool.tripped:
                            return None
                        try:
                            return slots.get(timeout=2.0)
                        except Empty:
                            continue

                def _defer(item):
                    with lock:
                        deferred.append(item)
                        stats['deferred'] += 1

                def _fn_download(item):
                    with lock:
                        halted = bool(sealed[0])
                    if halted or pool.tripped:
                        return _defer(item)
                    attempt, requeues, missing_hits, decode_failures = 0, 0, 0, 0
                    while attempt < retries:
                        lane = _lease()
                        if lane is None:
                            return _defer(item)
                        failure, result = None, None
                        try:
                            if lane.dead:
                                continue  # a retired slot's token is simply not given back
                            result = _fetch(item, lane)
                        except EgressExhausted:
                            if pool.tripped:
                                return _defer(item)
                            continue
                        except FetchError as err:
                            failure = err
                        finally:
                            if not lane.dead:
                                slots.put(lane)
                        if failure is None:
                            return _store(item, *result)
                        if failure.kind == 'blocked':
                            with lock:
                                stats['blocked'] += 1
                            requeues += 1
                            if pool.tripped or requeues > 40:
                                return _defer(item)
                            continue  # the address was the problem, not the image: not an attempt
                        if failure.kind == 'missing':
                            # A deleted episode is a run of consecutive 404s; waiting on each of them would
                            # idle every worker. The second look is on another turn at once, and the slow
                            # looks that decide are taken after the volume.
                            missing_hits += 1
                            if missing_hits >= missing_retries:
                                with lock:
                                    suspects.append(item)
                                    stats['suspect'] += 1
                                logging.debug(f'SUSPECT image {item["id"]}: {failure}')
                                return
                            continue
                        attempt += 1
                        if failure.kind == 'bad_body' and ('decode' in str(failure) or 'empty' in str(failure)):
                            decode_failures += 1
                        if attempt >= retries:
                            with lock:
                                if failure.kind == 'bad_body' and decode_failures >= max(retries // 2, 2):
                                    volume_bad.append(item['id'])
                                    stats['unusable'] += 1
                                else:
                                    stats['failed'] += 1
                            logging.warning(f'{"UNUSABLE" if decode_failures >= max(retries // 2, 2) else "FAILED"} '
                                            f'image {item["id"]} after {attempt} attempts ({failure.kind}, '
                                            f'HTTP {failure.status}): {failure}')
                            return
                        with lock:
                            stats['retry'] += 1
                        time.sleep(retry_wait * attempt * (0.5 + secrets.randbelow(1000) / 1000.0))

                batch = plan
                with StallWatchdog(lambda: stats['ok'] + stats['suspect'] + stats['unusable'] + stats['failed']
                                   + stats['deferred'] + stats['blocked'] + stats['retry']):
                    while True:
                        errors = parallel_call(batch, _fn_download, max_workers=n_workers,
                                               desc=f'Volume #{max_volume_id}', postfix=lambda: dict(stats))
                        for item, err in zip(batch, errors):
                            if err is not None:
                                logging.error(f'Unexpected error on image {item["id"]} - {err!r}')
                                stats['failed'] += 1
                        # The plan was a guess at how many images fill the budget. When the bytes really written
                        # fall short of it (image sizes changed faster than the estimate followed), more images
                        # are taken from what is left, sized by this volume's own average, until the budget is met.
                        with lock:
                            written, stored, stopped = volume_bytes[0], len(new_records), bool(sealed[0])
                        if (stopped or pool.tripped or not stored or rest.num_rows == 0
                                or len(plan) >= max_volume_files or written >= max_volume_bytes * 0.97
                                or (max_time_limit is not None and time.time() > start_time + max_time_limit)):
                            break
                        average = written / stored
                        more = min(int((max_volume_bytes - written) / max(average, 1.0) * 1.1) + 1, rest.num_rows,
                                   max_volume_files - len(plan))
                        logging.info(f'Volume #{max_volume_id} holds {written / 1024 ** 3:.2f} GB of its '
                                     f'{max_volume_bytes / 1024 ** 3:.2f} GB budget, topping it up with '
                                     f'{plural_word(more, "image")} (about {average / 1024:.0f} KB each so far) ...')
                        batch, rest = _rows(rest.slice(0, more)), rest.slice(more)
                        plan.extend(batch)

                # The slow looks at every suspect (images that answered 404 on every quick attempt). A control
                # download from this volume comes first: only an image that is still missing while the CDN
                # demonstrably serves is recorded as bad, and when the control fails the suspects wait for
                # another run instead of being blacklisted for the CDN's mood. The looks run in parallel, paced
                # like any other request, and the control result is reused for a short while.
                control_state = {'ok_at': 0.0}

                def _look(item):
                    lane = _lease()
                    if lane is None:
                        raise EgressExhausted('egress is gone')
                    try:
                        if lane.dead:
                            raise EgressExhausted(f'lane {lane.index} is retired')
                        return _fetch(item, lane)
                    finally:
                        if not lane.dead:
                            slots.put(lane)

                def _control_healthy() -> bool:
                    with lock:
                        if time.time() - control_state['ok_at'] < 30.0:
                            return True
                        probe = control[0]
                    if probe is None:
                        return False
                    try:
                        _look(probe)
                    except Exception as err:
                        logging.warning(f'Control download failed ({err!r}); the CDN is not serving now.')
                        return False
                    with lock:
                        control_state['ok_at'] = time.time()
                    return True

                def _diagnose(item):
                    if not _control_healthy():
                        with lock:
                            parked.append(item)
                            stats['suspect'] -= 1
                        return
                    for _ in range(diagnose_attempts):
                        time.sleep(diagnose_wait)
                        try:
                            body, width, height = _look(item)
                        except FetchError as err:
                            if err.kind != 'missing':
                                break
                        except EgressExhausted:
                            break
                        else:
                            _store(item, body, width, height)
                            with lock:
                                stats['suspect'] -= 1
                            return
                    else:
                        with lock:
                            volume_bad.append(item['id'])
                            stats['suspect'] -= 1
                            stats['gone'] += 1
                        logging.debug(f'GONE image {item["id"]}: 404 on every attempt and every slow look.')
                        return
                    with lock:
                        parked.append(item)
                        stats['suspect'] -= 1

                if suspects:
                    logging.info(f'Diagnosing {plural_word(len(suspects), "suspect")} ...')
                    parallel_call(list(suspects), _diagnose, max_workers=n_workers, desc='Diagnosing',
                                  postfix=lambda: dict(stats), silent=len(suspects) < 50)
                    if parked:
                        logging.warning(f'{plural_word(len(parked), "image")} left for the next run, none recorded as '
                                        f'bad.')
                    if stats['gone']:
                        logging.info(f'{plural_word(stats["gone"], "image")} confirmed gone (404 on every look).')

            elapsed = max(time.time() - started, 1e-6)
            deferred.sort(key=lambda x: x['id'])
            attempted = len(plan) - stats['deferred']
            logging.info(f'Egress: {pool.retired_total} address(es) retired in this run, '
                         f'{stats["blocked"]} blocked responses.')
            logging.info(f'Volume #{max_volume_id} downloaded: {stats["ok"]} ok, {stats["gone"]} gone, '
                         f'{stats["unusable"]} unusable, {stats["failed"]} failed, {len(parked)} parked, '
                         f'{stats["deferred"]} deferred, {stats["retry"]} retries, '
                         f'{volume_bytes[0] / 1024 ** 3:.2f} GB in {elapsed:.0f}s '
                         f'({stats["ok"] / elapsed:.1f} images/s, {volume_bytes[0] / 1024 ** 2 / elapsed:.1f} MB/s).')
            estimated = len(new_records) * estimate_bytes()
            if new_records and estimated > 0:
                size_scale = min(max(0.5 * size_scale + 0.5 * volume_bytes[0] / estimated, 0.3), 3.0)
            if deferred:
                rest = pa.concat_tables([pa.Table.from_pylist(deferred, schema=rest.schema), rest])
            pending = rest
            if stats['failed'] and attempted >= min_attempts_for_abort \
                    and stats['failed'] >= attempted * max_failed_ratio:
                logging.error(f'ABORT: {stats["failed"]} of {attempted} images failed in volume #{max_volume_id}, '
                              f'at or above the {max_failed_ratio:.0%} threshold - not publishing this volume.')
                aborted = True
                break

            if not new_records:
                logging.warning(f'Volume #{max_volume_id} produced no file, discarded.')
                max_volume_id -= 1
                if volume_bad:
                    bad_image_ids.update(volume_bad)
                if pool.tripped:
                    aborted = True
                    break
                continue

            tar_create_index_for_directory(os.path.join(upload_dir, 'images'), silent=True)
            if not os.path.exists(os.path.join(upload_dir, rel_index)):
                raise RuntimeError(f'Sidecar {rel_index!r} was not produced for volume #{max_volume_id}, refusing to '
                                   f'publish an unindexed tar.')

            staged = _append_rows(staged, new_records, table_columns)
            bad_image_ids.update(volume_bad)

            pq.write_table(staged, os.path.join(upload_dir, 'table.parquet'))
            with open(os.path.join(upload_dir, 'meta.json'), 'w') as f:
                json.dump({'max_volume_id': max_volume_id, 'bad_image_ids': sorted(bad_image_ids)}, f)
            if baseline_is_new:
                with open(os.path.join(upload_dir, glob_exist_ids_file), 'w') as f:
                    json.dump(baseline.tolist(), f, separators=(',', ':'))
            write_readme(os.path.join(upload_dir, 'README.md'), table=staged, bad_image_ids=bad_image_ids,
                         max_volume_id=max_volume_id, src_repository=src_repository, repository=repository,
                         baseline_repository=baseline_repository, baseline=baseline,
                         glob_exist_ids_file=glob_exist_ids_file)

            actual_bytes = os.path.getsize(tar_file)
            if dry_run:
                logging.info(f'[dry-run] Would upload volume #{max_volume_id}, '
                             f'{plural_word(len(new_records), "image")}, {actual_bytes / 1024 ** 3:.2f} GB.')
            else:
                limiter.try_acquire('hf_upload')
                logging.info(f'UPLOAD volume #{max_volume_id} starting - {plural_word(len(new_records), "image")}, '
                             f'{actual_bytes / 1024 ** 3:.2f} GB.')
                upload_started = time.time()
                safe_upload_directory_as_directory(
                    local_directory=upload_dir,
                    repo_id=repository,
                    repo_type='dataset',
                    path_in_repo='.',
                    message=f'Sync volume #{max_volume_id}, with {plural_word(len(new_records), "image")}',
                )
                upload_elapsed = time.time() - upload_started
                logging.info(f'UPLOAD volume #{max_volume_id} done in {upload_elapsed:.0f}s '
                             f'({actual_bytes / 1024 ** 2 / max(upload_elapsed, 1e-6):.1f} MB/s).')
                baseline_is_new = False
            volumes_done += 1
            if pool.tripped:
                logging.error('The egress is retired after this volume, stop.')
                aborted = True

        delete_detached_cache()
        log_disk_usage(os.getcwd(), prefix='Disk after volume')
        if aborted:
            break

    pool.close()
    logging.info(f'Done, {plural_word(volumes_done, "volume")} published in this run, '
                 f'{plural_word(staged.num_rows if staged is not None else 0, "image")} stored in total, '
                 f'{plural_word(len(parked), "image")} parked for the next run'
                 f'{", run stopped early (failure threshold or retired egress)" if aborted else ""}.')


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Download the Fancaps originals that the index lists and the baseline repository lacks into hfutils-indexed '
         'tar volumes of a private staging repository, lowest id first, one consistent commit per volume.',
)
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_FC_N_DL', default='deepghs/fancaps_newest_dl',
              show_default=True, help='Staging dataset repository.')
@click.option('-i', '--src-repository', type=str, envvar='REMOTE_REPOSITORY_FC', default='deepghs/fancaps_index',
              show_default=True, help='Index dataset repository to read images from.')
@click.option('-b', '--baseline-repository', type=str, envvar='REMOTE_REPOSITORY_FC_FULL',
              default='deepghs/fancaps_full', show_default=True,
              help='Repository whose ids (and bad ids) are not downloaded again.')
@click.option('--src-revision', type=str, default='main', show_default=True, help='Revision of the index to read.')
@click.option('-t', '--max-time-limit', type=duration_type(), default='5h', show_default=True,
              help='Stop scheduling work after this long.')
@click.option('-f', '--max-volume-files', type=int, default=100000, show_default=True,
              help='Entries per volume. Far above what the byte budget allows, so that the bytes decide.')
@click.option('-B', '--max-volume-bytes', type=int, default=int(7.5 * 1024 ** 3), show_default=True,
              help='Approximate byte budget of a volume: the aim is 5 to 10 GB per tar. A last volume that cannot '
                   'be filled is published with whatever there is.')
@click.option('-H', '--max-volume-hard-bytes', type=int, default=9 * 1024 ** 3, show_default=True,
              help='Ceiling on the payload bytes written into one tar; crossing it seals the volume. Kept below '
                   '10 GB (decimal) once the tar headers and the requests still in flight are added.')
@click.option('-p', '--proxy-pool', type=str, envvar='PP_FC', default=None,
              help='Bright Data proxy URL (scheme://user:password@host:port). Every slot gets its own pinned session. '
                   'Without it the run uses this host\'s own address.')
@click.option('-l', '--proxy-lanes', type=int, default=0, show_default=True,
              help='Proxy slots, each on its own egress address. Zero discovers how many addresses the pool can give '
                   '(up to --max-lanes) and keeps a fifth of them free when the pool runs out, so a replaced pool '
                   'needs no change here.')
@click.option('--max-lanes', type=int, default=64, show_default=True,
              help='Ceiling on the slots discovered when --proxy-lanes is zero.')
@click.option('--direct/--no-direct', default=None,
              help='Use this host\'s own address as a slot too. Default: only when there is no proxy.')
@click.option('-w', '--workers-per-lane', type=int, default=3, show_default=True,
              help='Download threads per egress address; they share its pacer.')
@click.option('-R', '--rate-per-ip', type=float, default=2.5, show_default=True,
              help='Requests per second one egress address may carry. Four a second trips Cloudflare error 1015, '
                   'which blocks the address for an hour; three were served for as long as it was measured.')
@click.option('--plan-margin', type=float, default=1.2, show_default=True,
              help='Plan this many times the byte budget per volume. The volume is sealed when the bytes written reach '
                   'the budget and topped up when they fall short, so its size does not depend on how well the image '
                   'size was guessed.')
@click.option('--min-publish-bytes', type=int, default=0, show_default=True,
              help='Publish nothing unless this many bytes (estimated) are pending, and never a trailing volume '
                   'below it. Keeps scheduled volumes from being small. Zero turns the bar off.')
@click.option('--max-publish-interval', type=duration_type(allow_none=True), default=None,
              help='Waive the bar once the newest volume is this old (for example 25d), so that a quiet site cannot '
                   'leave data unpublished forever. Default: never.')
@click.option('--force', is_flag=True, default=False, help='Waive the bar for this run.')
@click.option('--max-retired', type=int, default=None,
              help='Addresses retired by the CDN after which the run stops (default: half the slots plus one).')
@click.option('-d', '--min-free-disk', type=int, default=24 * 1024 ** 3, show_default=True,
              help='Stop before a new volume when free disk falls below this many bytes.')
@click.option('-u', '--upload-time-span', type=duration_type(), default=30, show_default=True,
              help='Minimum interval between upload commits.')
@click.option('-V', '--max-volumes', type=int, default=None, help='Stop after this many volumes (smoke runs).')
@click.option('-c', '--retries', type=int, default=6, show_default=True, help='Attempts per image.')
@click.option('-C', '--retry-wait', type=duration_type(), default=2.0, show_default=True,
              help='Base backoff between attempts.')
@click.option('--max-failed-ratio', type=float, default=0.3, show_default=True,
              help='A volume failing at this fraction is not published.')
@click.option('--missing-retries', type=int, default=2, show_default=True,
              help='Answers of 404/410 before an image becomes a suspect; they are taken back to back.')
@click.option('--diagnose-attempts', type=int, default=2, show_default=True,
              help='Slow looks at a suspect, before it is recorded bad.')
@click.option('--diagnose-wait', type=duration_type(), default=2.0, show_default=True,
              help='Seconds before each slow look.')
@click.option('--dry-run', is_flag=True, default=False, help='Download and build volumes, upload nothing.')
def cli(repository: str, src_repository: str, baseline_repository: str, src_revision: str,
        max_time_limit: Optional[float], max_volume_files: int, max_volume_bytes: int, max_volume_hard_bytes: int,
        proxy_pool: Optional[str], proxy_lanes: int, max_lanes: int, direct: Optional[bool], workers_per_lane: int,
        rate_per_ip: float, max_retired: Optional[int], plan_margin: float, min_publish_bytes: int,
        max_publish_interval: Optional[float], force: bool, min_free_disk: int, upload_time_span: float,
        max_volumes: Optional[int], retries: int, retry_wait: float, max_failed_ratio: float,
        missing_retries: int, diagnose_attempts: int, diagnose_wait: float, dry_run: bool):
    logging.try_init_root(logging.INFO)
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)  # `kill -USR1 <pid>` prints every thread's stack
    sync(repository=repository, src_repository=src_repository, baseline_repository=baseline_repository,
         src_revision=src_revision, max_time_limit=max_time_limit, max_volume_files=max_volume_files,
         max_volume_bytes=max_volume_bytes, max_volume_hard_bytes=max_volume_hard_bytes,
         workers_per_lane=workers_per_lane, min_free_disk=min_free_disk, upload_time_span=upload_time_span,
         max_volumes=max_volumes, proxy_pool=proxy_pool or None, proxy_lanes=proxy_lanes, max_lanes=max_lanes,
         direct=direct,
         rate_per_ip=rate_per_ip, max_retired=max_retired, plan_margin=plan_margin,
         min_publish_bytes=min_publish_bytes, max_publish_interval=max_publish_interval, force=force,
         retries=retries, retry_wait=retry_wait,
         max_failed_ratio=max_failed_ratio, missing_retries=missing_retries, diagnose_attempts=diagnose_attempts,
         diagnose_wait=diagnose_wait, dry_run=dry_run)


if __name__ == '__main__':
    cli()
