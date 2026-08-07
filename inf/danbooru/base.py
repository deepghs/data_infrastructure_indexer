"""Shared Danbooru session helpers.

``cdn.donmai.us`` and ``danbooru.donmai.us`` sit behind a Cloudflare bot classifier that scores
the TLS handshake itself. Credentials, user agents and warm-up cookies make no difference to
it: what matters is whether the JA3/JA4 fingerprint looks like a real browser. ``curl_cffi``
reproduces browser handshakes byte for byte, which is enough to be served normally.

Sweeping every ``curl_cffi`` target against ``cdn.donmai.us`` on 2026-08-07, from an address
Cloudflare was actively challenging, 20 of 23 were served normally. Only ``chrome142``,
``chrome120``, ``edge99`` and the Safari family were rejected, and every Safari target failed.
The working set drifts and is not monotonic in version -- ``chrome120`` fails while the older
``chrome110`` succeeds -- so nothing here pins a value.

What reuse does and does not buy
================================

It is tempting to assume the classifier judges a connection, so that riding an accepted one
avoids further verdicts. Simulation says otherwise. Run 31185766987 delivered 1974 images with
0.84 rejections each and 4.7 seconds of worker time per 2 MB file. A model where an established
connection is never re-judged reproduces 0.12 rejections per image, seven times too few; the
observed figure only appears when requests are rejected at a roughly fixed rate whether or not
the connection is already up. **Rejection here is per request.**

Reuse is still worth having, for the handshake and TCP slow start it avoids rather than for any
verdict it dodges. So the pool keeps connections warm where it is free to do so, and does not
pretend that is the main lever:

* Slots go back to the same worker whenever free, so a hot connection stays hot.
* A slot is retired only after consecutive failures. One rejection carries no information, and
  discarding a working connection costs a handshake and a slow-start ramp.
* Fingerprints are scored by a decaying average rather than a lifetime tally, so a target that
  stops working is demoted within a few draws. Draws are epsilon-greedy: mostly from what is
  working now, occasionally from everything, so a recovered target can come back.
"""
import random
import threading
import typing
from typing import Dict, List, Optional, Tuple

from curl_cffi import requests as cffi_requests
from ditk import logging

from inf.utils.brightdata import with_session

__site_url__ = 'https://danbooru.donmai.us'

DEFAULT_TIMEOUT = 60.0

#: Fingerprints we would like to use, all verified against cdn.donmai.us on 2026-08-07. Kept
#: wide on purpose: the pool weights them by recent success, so a few that stop working cost
#: little, while a narrow list leaves nowhere to go when one does.
PREFERRED_IMPERSONATES: List[str] = [
    'chrome146', 'chrome145', 'chrome136', 'chrome133a', 'chrome131', 'chrome124', 'chrome123',
    'chrome119', 'chrome116', 'chrome110', 'chrome107', 'chrome104', 'chrome101', 'chrome100',
    'edge101', 'firefox147', 'firefox144', 'firefox135', 'firefox133', 'chrome131_android',
]


def _supported_impersonates() -> Optional[set]:
    """
    Ask the installed ``curl_cffi`` which impersonation targets it actually has.

    The available set is a property of the installed build, and the build is a property of the
    interpreter: Python 3.8 caps ``curl_cffi`` at 0.9.0, which knows 29 targets, while 0.16.0
    knows 53. Naming a target the local build has never heard of raises ``ImpersonateError`` at
    session construction, which looks like a network fault to everything downstream.

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


def _build_ladder() -> List[str]:
    """
    Narrow :data:`PREFERRED_IMPERSONATES` to what this build supports.

    :returns: Usable fingerprint names.
    :rtype: List[str]
    """
    supported = _supported_impersonates()
    if not supported:
        return list(PREFERRED_IMPERSONATES)
    usable = [imp for imp in PREFERRED_IMPERSONATES if imp in supported]
    dropped = [imp for imp in PREFERRED_IMPERSONATES if imp not in supported]
    if dropped:
        logging.info(f'Impersonation targets unavailable in this curl_cffi build, dropped: '
                     f'{", ".join(dropped)}.')
    if not usable:
        # Nothing preferred is available. Rather than fail, take what the build does have.
        usable = sorted(supported - {'chrome', 'edge', 'firefox', 'safari'})
        logging.warning(f'No preferred impersonation target is available; '
                        f'falling back to {len(usable)} build-provided targets.')
    return usable


#: Fingerprints actually usable here, resolved once against the installed build.
IMPERSONATE_LADDER: List[str] = _build_ladder()

#: Weight of the newest outcome when updating a fingerprint's score. High enough that a target
#: which stops working is demoted within a handful of draws.
SCORE_ALPHA = 0.2

#: Score below which a fingerprint is skipped during greedy draws.
SCORE_FLOOR = 0.35

#: Share of draws made uniformly rather than greedily, so a recovered fingerprint can return and
#: one unlucky early failure does not exile a good one for the rest of the run.
EXPLORE_RATE = 0.05


def get_danbooru_session(impersonate: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT,
                         proxy_pool: Optional[str] = None,
                         proxy_session: Optional[str] = None) -> cffi_requests.Session:
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
    :param proxy_session: Bright Data session id to pin the proxy's exit address on. Without one
        the proxy draws a new address per request, which never escapes a per-address rate limit.
    :type proxy_session: Optional[str]
    :returns: A ready-to-use session.
    :rtype: curl_cffi.requests.Session
    """
    if proxy_pool and proxy_session:
        proxy_pool = with_session(proxy_pool, proxy_session)
    for _ in range(len(IMPERSONATE_LADDER) + 1):
        chosen = impersonate or random.choice(IMPERSONATE_LADDER)
        kwargs = dict(impersonate=chosen, timeout=timeout)
        if proxy_pool:
            kwargs['proxies'] = {'http': proxy_pool, 'https': proxy_pool}
        try:
            session = cffi_requests.Session(**kwargs)
        except Exception as err:
            if 'not supported' not in str(err):
                raise
            # The static filter should have caught this, but a build that reports one set and
            # accepts another would otherwise fail every download as a network error.
            logging.warning(f'Impersonation target {chosen!r} rejected by curl_cffi, dropping it.')
            if chosen in IMPERSONATE_LADDER and len(IMPERSONATE_LADDER) > 1:
                IMPERSONATE_LADDER.remove(chosen)
            impersonate = None
            continue
        session.headers.update({'Referer': f'{__site_url__}/'})
        return session
    raise RuntimeError('No usable impersonation target is available in this curl_cffi build.')


class DanbooruSessionPool:
    """
    A pool of impersonating sessions that keeps workers on connections which already work.

    ``curl_cffi.requests.Session`` is not thread-safe, unlike ``requests.Session``, so a slot is
    leased exclusively for one download and returned afterwards. Within that constraint the pool
    avoids paying for a fresh connection on every file through affinity, patience about
    retirement, and recency-weighted fingerprint scoring. See the module docstring for the
    measurements behind each.

    Size the pool at or a little above the worker count; a much larger pool only dilutes reuse.

    Use :meth:`lease`::

        with pool.lease() as (slot, generation, session):
            ...
    """

    def __init__(self, size: int = 8, retire_after: int = 2, **kwargs):
        if size < 1:
            raise ValueError(f'Session pool size should be positive, but {size!r} found.')
        self._kwargs = kwargs
        self._size = size
        self._retire_after = max(retire_after, 1)
        self._cv = threading.Condition()
        self._affinity = threading.local()

        self._sessions: List[Optional[cffi_requests.Session]] = [None] * size
        self._generations: List[int] = [0] * size
        self._impersonates: List[Optional[str]] = [None] * size
        self._streaks: List[int] = [0] * size
        self._available: List[int] = list(range(size))

        self._scores: Dict[str, float] = {imp: 0.5 for imp in IMPERSONATE_LADDER}
        self._ok: Dict[str, int] = {imp: 0 for imp in IMPERSONATE_LADDER}
        self._bad: Dict[str, int] = {imp: 0 for imp in IMPERSONATE_LADDER}
        self._reuse_hits = 0
        self._reuse_misses = 0

    def _pick_impersonate(self) -> str:
        """
        Draw a fingerprint, mostly from those currently succeeding. The caller holds the lock.

        :returns: Fingerprint name.
        :rtype: str
        """
        if random.random() < EXPLORE_RATE:
            return random.choice(IMPERSONATE_LADDER)
        viable = [imp for imp in IMPERSONATE_LADDER if self._scores[imp] >= SCORE_FLOOR]
        if not viable:
            # Everything is failing. Fall back to the least bad rather than stall, so the run
            # keeps attempting while conditions change.
            viable = sorted(IMPERSONATE_LADDER, key=lambda i: -self._scores[i])[:4]
        # Squaring sharpens the preference: a clear winner is drawn far more often than a
        # marginal one, without ever dropping to a hard argmax.
        weights = [self._scores[imp] ** 2 for imp in viable]
        total = sum(weights) or 1.0
        threshold = random.random() * total
        for imp, weight in zip(viable, weights):
            threshold -= weight
            if threshold <= 0:
                return imp
        return viable[-1]

    def acquire(self) -> Tuple[int, int, cffi_requests.Session]:
        """
        Lease a slot, preferring the one this thread used last.

        :returns: Tuple of (slot index, slot generation, session).
        :rtype: Tuple[int, int, curl_cffi.requests.Session]
        """
        preferred = getattr(self._affinity, 'slot', None)
        with self._cv:
            while not self._available:
                self._cv.wait()
            if preferred is not None and preferred in self._available:
                index = preferred
                self._available.remove(index)
                self._reuse_hits += 1
            else:
                index = self._available.pop(random.randrange(len(self._available)))
                if preferred is not None:
                    self._reuse_misses += 1
            session = self._sessions[index]
            impersonate = None if session is not None else self._pick_impersonate()
            generation = self._generations[index]
        self._affinity.slot = index

        if session is not None:
            with self._cv:
                return index, self._generations[index], session

        # Built outside the lock; constructing a session must not stall the other workers. The
        # proxy session id carries the generation, so retiring a slot lands on a different exit
        # address rather than the one that just refused us.
        fresh = get_danbooru_session(impersonate=impersonate,
                                     proxy_session=f's{index}g{generation}', **self._kwargs)
        with self._cv:
            if self._sessions[index] is None:
                self._sessions[index] = fresh
                self._impersonates[index] = impersonate
                self._streaks[index] = 0
            return index, self._generations[index], self._sessions[index]

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

    def report_success(self, index: int):
        """
        Credit the slot's fingerprint and clear its failure streak.

        Call this while still holding the lease. Once the slot is released another worker can
        retire it, and the attribution is lost.

        :param index: Slot index from :meth:`acquire`.
        :type index: int
        """
        with self._cv:
            self._streaks[index] = 0
            imp = self._impersonates[index]
            if imp in self._scores:
                self._scores[imp] += SCORE_ALPHA * (1.0 - self._scores[imp])
                self._ok[imp] += 1

    def report_failure(self, index: int, seen_generation: int) -> bool:
        """
        Record a rejection, retiring the slot once its failures stop looking like noise.

        :param index: Slot the caller was using.
        :type index: int
        :param seen_generation: Generation the caller saw, so one bad connection is not retired
            twice by two workers.
        :type seen_generation: int
        :returns: Whether the session was retired.
        :rtype: bool
        """
        with self._cv:
            if self._generations[index] != seen_generation or self._sessions[index] is None:
                return False
            imp = self._impersonates[index]
            if imp in self._scores:
                self._scores[imp] += SCORE_ALPHA * (0.0 - self._scores[imp])
                self._bad[imp] += 1
            self._streaks[index] += 1
            if self._streaks[index] < self._retire_after:
                return False
            session = self._sessions[index]
            self._sessions[index] = None
            self._impersonates[index] = None
            self._streaks[index] = 0
            self._generations[index] += 1
            logging.debug(f'Retiring session slot #{index} ({imp}) after '
                          f'{self._retire_after} consecutive rejections.')
        try:
            session.close()
        except Exception:  # pragma: no cover - a dead session must not abort the run
            pass
        return True

    def stats(self) -> dict:
        """
        Snapshot for logging: connection reuse rate and per-fingerprint outcomes.

        :rtype: dict
        """
        with self._cv:
            total = self._reuse_hits + self._reuse_misses
            live = [(imp, self._ok[imp], self._bad[imp], round(self._scores[imp], 2))
                    for imp in IMPERSONATE_LADDER if self._ok[imp] or self._bad[imp]]
            return {
                'reuse_rate': (self._reuse_hits / total) if total else 0.0,
                'fingerprints': sorted(live, key=lambda x: -(x[1] + x[2])),
                'total_ok': sum(self._ok.values()),
                'total_bad': sum(self._bad.values()),
            }

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
