"""Shared Danbooru session helpers.

``cdn.donmai.us`` and ``danbooru.donmai.us`` sit behind a Cloudflare bot classifier that scores
the TLS handshake itself. Credentials, user agents and warm-up cookies make no difference to
it: what matters is whether the JA3/JA4 fingerprint looks like a real browser. ``curl_cffi``
reproduces browser handshakes byte for byte, which is enough to be served normally.

Sweeping every ``curl_cffi`` target against ``cdn.donmai.us`` on 2026-08-07, from an address
Cloudflare was actively challenging, 20 of 23 were served normally. Only ``chrome142``,
``chrome120``, ``edge99`` and the Safari family were rejected, and every Safari target failed.

The working set drifts and is not monotonic in version -- ``chrome120`` fails while the older
``chrome110`` succeeds -- so nothing here pins a value. :class:`DanbooruSessionPool` draws from
:data:`IMPERSONATE_LADDER` weighted by how each fingerprint has actually fared during the run,
so a target that stops working is demoted without anyone editing this file.
"""
import random
import threading
from typing import Dict, List, Optional, Tuple

from curl_cffi import requests as cffi_requests
from ditk import logging

__site_url__ = 'https://danbooru.donmai.us'

DEFAULT_TIMEOUT = 60.0

#: Browser fingerprints to draw from, all verified against cdn.donmai.us on 2026-08-07. Kept
#: wide on purpose: the pool weights them by observed success, so a few that stop working cost
#: little, while a narrow list leaves nowhere to go when one does.
IMPERSONATE_LADDER: List[str] = [
    'chrome146', 'chrome145', 'chrome136', 'chrome133a', 'chrome131', 'chrome124', 'chrome123',
    'chrome119', 'chrome116', 'chrome110', 'chrome107', 'chrome104', 'chrome101', 'chrome100',
    'edge101', 'firefox147', 'firefox144', 'firefox135', 'firefox133', 'chrome131_android',
]


def get_danbooru_session(impersonate: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT,
                         proxy_pool: Optional[str] = None) -> cffi_requests.Session:
    """
    Build a browser-impersonating session for donmai.us.

    No warm-up request is made. Cloudflare admits these sessions on the handshake alone, and a
    warm-up would only spend a request against the more tightly rate-limited API host.

    :param impersonate: Fingerprint to use. A random entry of :data:`IMPERSONATE_LADDER` when
        omitted.
    :type impersonate: Optional[str]
    :param timeout: Default per-request timeout in seconds.
    :type timeout: float
    :param proxy_pool: Proxy URL applied to every request.
    :type proxy_pool: Optional[str]
    :returns: A ready-to-use session.
    :rtype: curl_cffi.requests.Session
    """
    impersonate = impersonate or random.choice(IMPERSONATE_LADDER)
    kwargs = dict(impersonate=impersonate, timeout=timeout)
    if proxy_pool:
        kwargs['proxies'] = {'http': proxy_pool, 'https': proxy_pool}
    session = cffi_requests.Session(**kwargs)
    session.headers.update({'Referer': f'{__site_url__}/'})
    return session


class DanbooruSessionPool:
    """
    A pool of impersonating sessions, leased one at a time and replaced when rejected.

    Two properties drive the design:

    * ``curl_cffi.requests.Session`` is **not** thread-safe, unlike ``requests.Session``. A slot
      is therefore leased exclusively for the duration of one download and returned afterwards,
      rather than shared across workers.
    * Cloudflare scores a fingerprint per connection. Spreading work over many independent
      sessions keeps a rejection local to the slot that earned it, and retiring that slot draws
      a different fingerprint for its replacement.

    Draws are weighted by each fingerprint's observed success rate this run, Laplace-smoothed so
    an untried target still gets picked and a currently-good one is never locked in. Uniform
    draws would keep re-rolling fingerprints already known to be rejected.

    Slots fill lazily, so a pool far larger than the worker count costs nothing.

    Use :meth:`lease` rather than the raw acquire/release pair::

        with pool.lease() as (slot, generation, session):
            ...
    """

    def __init__(self, size: int = 32, **kwargs):
        if size < 1:
            raise ValueError(f'Session pool size should be positive, but {size!r} found.')
        self._kwargs = kwargs
        self._size = size
        self._cv = threading.Condition()
        self._sessions: List[Optional[cffi_requests.Session]] = [None] * size
        self._generations: List[int] = [0] * size
        self._available: List[int] = list(range(size))
        self._impersonates: List[Optional[str]] = [None] * size
        self._ok: Dict[str, int] = {imp: 0 for imp in IMPERSONATE_LADDER}
        self._bad: Dict[str, int] = {imp: 0 for imp in IMPERSONATE_LADDER}

    def _pick_impersonate(self) -> str:
        """
        Draw a fingerprint weighted by its success rate so far, with Laplace smoothing.

        :returns: Fingerprint name.
        :rtype: str
        """
        weights = [(self._ok[imp] + 1) / (self._ok[imp] + self._bad[imp] + 2)
                   for imp in IMPERSONATE_LADDER]
        total = sum(weights)
        threshold = random.random() * total
        for imp, weight in zip(IMPERSONATE_LADDER, weights):
            threshold -= weight
            if threshold <= 0:
                return imp
        return IMPERSONATE_LADDER[-1]

    def report_success(self, index: int):
        """
        Credit the fingerprint in ``index`` with a successful transfer.

        :param index: Slot index from :meth:`acquire`.
        :type index: int
        """
        with self._cv:
            imp = self._impersonates[index]
            if imp in self._ok:
                self._ok[imp] += 1

    def impersonate_stats(self) -> Dict[str, Tuple[int, int]]:
        """
        Return per-fingerprint (success, rejection) counts.

        :rtype: Dict[str, Tuple[int, int]]
        """
        with self._cv:
            return {imp: (self._ok[imp], self._bad[imp]) for imp in IMPERSONATE_LADDER
                    if self._ok[imp] or self._bad[imp]}

    def acquire(self) -> Tuple[int, int, cffi_requests.Session]:
        """
        Lease a random free slot, building its session when the slot is empty.

        Blocks while every slot is leased, which only happens when workers outnumber slots.

        :returns: Tuple of (slot index, slot generation, session).
        :rtype: Tuple[int, int, curl_cffi.requests.Session]
        """
        with self._cv:
            while not self._available:
                self._cv.wait()
            index = self._available.pop(random.randrange(len(self._available)))
            session = self._sessions[index]
            generation = self._generations[index]

        if session is None:
            with self._cv:
                impersonate = self._pick_impersonate()
            session = get_danbooru_session(impersonate=impersonate, **self._kwargs)
            with self._cv:
                self._sessions[index] = session
                self._impersonates[index] = impersonate
                generation = self._generations[index]
        return index, generation, session

    def release(self, index: int):
        """
        Return a leased slot to the pool.

        :param index: Slot index from :meth:`acquire`.
        :type index: int
        """
        with self._cv:
            if index not in self._available:
                self._available.append(index)
            self._cv.notify()

    def retire(self, index: int, seen_generation: int):
        """
        Discard the session in ``index`` so the next lease draws a new fingerprint.

        :param index: Slot the caller was using.
        :type index: int
        :param seen_generation: Generation the caller saw, so a slot rejected twice is only
            replaced once.
        :type seen_generation: int
        """
        with self._cv:
            if self._generations[index] != seen_generation or self._sessions[index] is None:
                return
            session = self._sessions[index]
            imp = self._impersonates[index]
            if imp in self._bad:
                self._bad[imp] += 1
            self._sessions[index] = None
            self._impersonates[index] = None
            self._generations[index] += 1
            # Debug, not info: a rejection here is routine and self-healing, and one line per
            # event floods the log badly enough to make a working run look broken. The caller
            # reports the aggregate instead.
            logging.debug(f'Retiring session slot #{index} after an upstream rejection.')
        # Closing outside the lock: this slot is leased by the caller, so nobody else holds it.
        try:
            session.close()
        except Exception:  # pragma: no cover - a dead session must not abort the run
            pass

    def lease(self):
        """
        Context manager wrapping :meth:`acquire` and :meth:`release`.

        :returns: Context manager yielding (slot index, slot generation, session).
        """
        return _SessionLease(self)

    def close(self):
        with self._cv:
            sessions = [s for s in self._sessions if s is not None]
            self._sessions = [None] * self._size
            self._impersonates = [None] * self._size
        for session in sessions:
            try:
                session.close()
            except Exception:  # pragma: no cover
                pass


class _SessionLease:
    def __init__(self, pool: DanbooruSessionPool):
        self._pool = pool
        self._index: Optional[int] = None

    def __enter__(self):
        index, generation, session = self._pool.acquire()
        self._index = index
        return index, generation, session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._index is not None:
            self._pool.release(self._index)
            self._index = None
        return False


__all__ = ['get_danbooru_session', 'DanbooruSessionPool', 'IMPERSONATE_LADDER', '__site_url__']
