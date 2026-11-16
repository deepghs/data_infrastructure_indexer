"""Read-only adapter for the anime section of fancaps.net.

Nothing here touches Hugging Face or parquet. The module owns three things: how a request gets past
Cloudflare, what a *valid* response looks like, and how each of the four page types is parsed.

Pages of the anime section
==========================

``showList.php?<c>=1``
    All shows whose name starts with ``c`` (``0`` for digits), on one page.
``showimages.php?<id>-<slug>&page=N``
    The episode list of a show, ten per page in the site's own order (by episode number, **not** by
    upload time: a late upload appears in the middle). Past the last page the site answers a valid
    page that simply has no episode rows.
``episodeimages.php?<episode_id>-<slug>/<title>&page=N``
    The screenshots of one episode, about 36 per page. The page repeats a ``.topImages`` block of
    featured pictures on every page, so that block has to be removed before reading, otherwise the
    pages after the last one never look empty.

What measuring on GitHub runners showed
=======================================

The site sits behind Cloudflare. Over eight jobs on GitHub-hosted runners, each from a different
egress address, ``cloudscraper`` was challenged (HTTP 403, ``cf-mitigated: challenge``) in six, and in
the two jobs where its first request passed the following ones were refused. ``curl_cffi`` with the
``chrome124`` and ``chrome131`` fingerprints was challenged in every job that tried it. The Safari,
Firefox and Edge fingerprints of ``curl_cffi`` and plain ``requests`` were let in every time and
served 22 to 54 requests per second to sixteen threads reading whole episodes cold, every response
complete. The transport therefore keeps an ordered *ladder* of strategies, starts on the first one
and, when a response is a challenge, moves every thread to the next one.

A response is accepted only when it is complete: HTTP 200, the ``Counting`` widget present, the body
ends with ``</html>`` and the layout marker of the page type is there. Anything else (a challenge, a
truncated body, an empty body) is retried and never read as "the page is empty". 404 and 410 are
permanent and are not retried.
"""
import math
import random
import re
import threading
import time
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlsplit

from ditk import logging
from pyquery import PyQuery as pq

from inf.utils.impersonate import supported_impersonates

BASE = 'https://fancaps.net/anime/'
#: Present on every valid page of the site (the global image counter widget).
MARK = 'Counting'
#: Row title prefix on episode lists.
EPISODE_PREFIX = 'Images From '
CDN_URL = 'https://cdni.fancaps.net/file/fancaps-animeimages/{id}.jpg'
#: Episode list pages hold this many rows; the last page holds fewer.
LIST_PAGE_SIZE = 10
#: A hard stop for one episode, far beyond the largest known one (102 pages).
MAX_EPISODE_PAGES = 400
MAX_LIST_PAGES = 400

#: Strategies in the order they are tried. ``curl:<target>`` is a ``curl_cffi`` browser fingerprint.
DEFAULT_LADDER = ['curl:safari17_0', 'curl:firefox135', 'curl:edge101', 'requests', 'curl:chrome120',
                  'cloudscraper']
_UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
_ID_NAME = re.compile(r'^(\d+)-([^&=]*)')


class SiteError(Exception):
    """A response that could not be used after every retry."""


class SiteGone(SiteError):
    """HTTP 404 / 410: permanent, never retried."""


class SiteBlocked(SiteError):
    """Every strategy of the ladder was challenged, several times over."""


class Page(NamedTuple):
    text: str
    url: str


class Episode(NamedTuple):
    id: int
    name: str
    url: str


class Show(NamedTuple):
    id: int
    name: str
    url: str
    slug: str


def is_valid(text: str, must: Sequence[str] = ()) -> bool:
    """Whether a 200 body is a complete page of the site."""
    return MARK in text and text.rstrip().endswith('</html>') and all(m in text for m in must)


def build_ladder(wanted: Optional[Sequence[str]] = None) -> List[str]:
    """Drop the ``curl:`` targets the installed ``curl_cffi`` build does not know."""
    supported = supported_impersonates()
    ladder = []
    for name in (wanted or DEFAULT_LADDER):
        if name.startswith('curl:') and supported is not None and name[5:] not in supported:
            logging.info(f'Impersonation target {name[5:]!r} is not in this curl_cffi build, dropped.')
            continue
        ladder.append(name)
    return ladder


class Transport:
    """
    One HTTP session per thread and strategy, and a ladder to move along when the site refuses.

    :param ladder: Strategies, best first. (default: :data:`DEFAULT_LADDER`)
    :param proxy: Optional proxy url used by every strategy.
    :param timeout: Seconds per request.
    """

    def __init__(self, ladder: Optional[Sequence[str]] = None, proxy: Optional[str] = None, timeout: float = 60.0):
        self.ladder = build_ladder(ladder)
        if not self.ladder:
            raise ValueError('No usable request strategy.')
        self.proxy = proxy or None
        self.timeout = timeout
        self._lock = threading.Lock()
        self._local = threading.local()
        self._index = 0
        self.generation = 0
        self.rotations = 0

    @property
    def strategy(self) -> str:
        return self.ladder[self._index]

    def _build(self, strategy: str):
        kind, _, arg = strategy.partition(':')
        proxies = {'http': self.proxy, 'https': self.proxy} if self.proxy else None
        if kind == 'curl':
            from curl_cffi import requests as cffi_requests
            return cffi_requests.Session(impersonate=arg, proxies=proxies)
        if kind == 'cloudscraper':
            import cloudscraper
            session = cloudscraper.create_scraper()
        elif kind == 'requests':
            import requests
            session = requests.Session()
        else:
            raise ValueError(f'Unknown strategy {strategy!r}.')
        session.headers.update({'User-Agent': _UA})
        if proxies:
            session.proxies.update(proxies)
        return session

    def _session(self):
        local = self._local
        if getattr(local, 'generation', None) != self.generation:
            local.session = self._build(self.strategy)
            local.generation = self.generation
        return local.session

    def request(self, url: str) -> Tuple[int, str, str]:
        """One GET, no retry. Returns ``(status, text, final_url)``."""
        resp = self._session().get(url, timeout=self.timeout)
        return resp.status_code, resp.text, str(resp.url)

    def rotate(self, seen_generation: int, reason: str = '') -> None:
        """Move every thread to the next strategy, unless another thread already did."""
        with self._lock:
            if seen_generation != self.generation:
                return
            old = self.strategy
            self._index = (self._index + 1) % len(self.ladder)
            self.generation += 1
            self.rotations += 1
            logging.warning(f'Request strategy {old!r} -> {self.strategy!r} ({reason or "refused"}).')


class Fetcher:
    """Validated GET with retry and strategy rotation on top of a :class:`Transport`."""

    def __init__(self, transport: Optional[Transport] = None, retries: int = 8, retry_wait: float = 0.4,
                 max_wait: float = 20.0, max_blocked_rounds: int = 3):
        self.transport = transport or Transport()
        self.retries = retries
        self.retry_wait = retry_wait
        self.max_wait = max_wait
        self.max_blocked_rounds = max_blocked_rounds
        self._lock = threading.Lock()
        self.requests = 0
        self.retried = 0
        self._blocked = 0

    def get(self, url: str, must: Sequence[str] = ()) -> Page:
        last = 'no attempt'
        for attempt in range(self.retries):
            generation = self.transport.generation
            with self._lock:
                self.requests += 1
            try:
                status, text, final = self.transport.request(url)
            except Exception as err:  # connection reset, timeout, TLS failure ...
                last = repr(err)[:160]
            else:
                if status in (404, 410):
                    raise SiteGone(f'{url} -> HTTP {status}')
                if status == 200 and is_valid(text, must):
                    with self._lock:
                        self._blocked = 0
                    return Page(text, final)
                last = f'HTTP {status}, {len(text)} bytes'
                if status == 403:
                    with self._lock:
                        self._blocked += 1
                        blocked = self._blocked
                    if blocked > self.max_blocked_rounds * len(self.transport.ladder):
                        raise SiteBlocked(f'{url} refused by every strategy: {last}')
                    self.transport.rotate(generation, last)
            with self._lock:
                self.retried += 1
            time.sleep(min(self.max_wait, self.retry_wait * 2 ** attempt) * (0.5 + random.random()))
        raise SiteError(f'{url} failed after {self.retries} attempts: {last}')


# ---------------------------------------------------------------------------------------------- parsing
def _id_and_slug(url: str) -> Tuple[int, str]:
    match = _ID_NAME.match(urlsplit(url).query)
    if not match:
        raise SiteError(f'No id in url {url!r}.')
    return int(match.group(1)), match.group(2)


def parse_show_list(page: Page) -> List[Show]:
    """Shows of one ``showList.php`` page."""
    shows = []
    for row in pq(page.text)('.single_post_content > .row').items():
        name = row('h4 a').text()
        if not name:
            continue
        url = urljoin(page.url, row('h4 a').attr('href'))
        show_id, slug = _id_and_slug(url)
        shows.append(Show(show_id, name, url, slug))
    return shows


def parse_episode_list(page: Page) -> List[Episode]:
    """Episode rows of one ``showimages.php`` page, in page order."""
    episodes = []
    for row in pq(page.text)('.post_title').parent()('.row').items():
        text = row('div:nth-child(1)').text()
        if not text:
            continue
        href = row('div:nth-child(2) a').attr('href')
        if not href:
            continue
        if not text.startswith(EPISODE_PREFIX):
            raise SiteError(f'Invalid episode row title {text!r} on {page.url!r}.')
        url = urljoin(page.url, href)
        episodes.append(Episode(_id_and_slug(url)[0], text[len(EPISODE_PREFIX):], url))
    return list(dict.fromkeys(episodes))


def parse_episode_images(page: Page) -> List[Tuple[int, str]]:
    """``(image id, thumbnail url)`` of one ``episodeimages.php`` page, the featured block excluded."""
    area = pq(page.text)('.single_post_area')
    area.remove('.topImages')
    images = []
    for item in area('div a > img').items():
        href = item.parent().attr('href')
        if href:
            images.append((int(href.split('/')[-1]), urljoin(page.url, item.attr('src'))))
    return images


def with_page(url: str, number: int) -> str:
    """The url of page ``number`` the way ``requests`` writes it: ``&page=N`` after an existing query."""
    return f'{url}{"&" if "?" in url else "?"}page={number}'


class FancapsClient:
    """The site as the rest of the package sees it. Every method is safe to call from many threads."""

    def __init__(self, fetcher: Optional[Fetcher] = None):
        self.fetcher = fetcher or Fetcher()

    def ensure_access(self, attempts: int = 12) -> str:
        """Probe one page; rotate strategies until one is let in. Returns the strategy that worked."""
        last = None
        for _ in range(attempts):
            try:
                self.fetcher.get(with_page(BASE + 'popular.php?paging=1', 1))
                logging.info(f'Site reachable with strategy {self.fetcher.transport.strategy!r}.')
                return self.fetcher.transport.strategy
            except SiteBlocked:
                raise
            except SiteError as err:
                last = err
                self.fetcher.transport.rotate(self.fetcher.transport.generation, 'warm-up failed')
        raise SiteBlocked(f'The site cannot be reached: {last!r}')

    def show_list(self) -> Dict[int, Show]:
        """Every show of the site: the 27 listing pages ``0`` and ``a`` to ``z``."""
        shows: Dict[int, Show] = {}
        for char in '0abcdefghijklmnopqrstuvwxyz':
            page = self.fetcher.get(f'{BASE}showList.php?{char}=1', must=('showimages.php',))
            for show in parse_show_list(page):
                shows.setdefault(show.id, show)
        return shows

    def list_page(self, show_id: int, slug: str, number: int) -> List[Episode]:
        """One page of the episode list of a show; an empty list past the last page."""
        page = self.fetcher.get(with_page(f'{BASE}showimages.php?{show_id}-{slug}', number), must=('post_title',))
        return parse_episode_list(page)

    def crawl_episode(self, episode_url: str) -> List[Tuple[int, str]]:
        """
        Every ``(image id, thumbnail url)`` of an episode, first appearance order.

        Pages are read until one brings nothing new. The caller must treat a raised error as "this
        episode is not done": a half-read episode is never returned.
        """
        seen: Dict[int, str] = {}
        for number in range(1, MAX_EPISODE_PAGES + 1):
            page = self.fetcher.get(with_page(episode_url, number), must=('single_post_area',))
            fresh = [(i, t) for i, t in parse_episode_images(page) if i not in seen]
            if not fresh:
                return list(seen.items())
            seen.update(fresh)
        raise SiteError(f'{episode_url} has more than {MAX_EPISODE_PAGES} pages.')


def page_count(episodes: int) -> int:
    """Pages an episode list of this length occupies."""
    return max(1, math.ceil(episodes / LIST_PAGE_SIZE))
