"""Finding the episodes the site lists, cheaply and without missing any.

The site lists the episodes of a show in *episode number* order, so a freshly uploaded episode can
land anywhere in the list, not at its end. The state kept between runs is therefore the full ordered
id list of every show, and discovery works in three layers:

1. A **tail probe** costs one request per show (two when the stored last page is exactly full): the
   last stored page is fetched again and compared with what was stored. An insertion anywhere before
   it shifts the tail and an appended episode changes it, so a pure addition always shows up here.
2. A show whose probe differs, and every show the table has never seen, is **walked**: all its pages
   are read and the stored list is replaced by the site's exact order. This is the only way the list
   is ever rewritten, so it cannot drift.
3. Once in a while (``full`` runs) *every* show is walked. A probe cannot see one episode added and
   another removed in the same show between two runs, because the tail then does not move; the full
   walk is what closes that gap, and it also notices removals.

A walk of the whole site is about nine thousand small requests, a few minutes; the daily probe of all
shows is about 2.7 thousand.
"""
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, NamedTuple, Optional, Protocol, Set

from ditk import logging
from tqdm import tqdm

from .client import Episode, LIST_PAGE_SIZE, MAX_LIST_PAGES, Show, SiteError


class Site(Protocol):
    def list_page(self, show_id: int, slug: str, number: int) -> List[Episode]:
        ...


def walk_show(site: Site, show_id: int, slug: str) -> List[Episode]:
    """All episodes of a show in site order: pages are read until the first empty one."""
    episodes: List[Episode] = []
    seen: Set[int] = set()
    for number in range(1, MAX_LIST_PAGES + 1):
        page = site.list_page(show_id, slug, number)
        fresh = [e for e in page if e.id not in seen]
        if not fresh:
            return episodes
        seen.update(e.id for e in fresh)
        episodes.extend(fresh)
    raise SiteError(f'Show {show_id} has more than {MAX_LIST_PAGES} list pages.')


def tail_changed(site: Site, show_id: int, slug: str, known: List[int]) -> bool:
    """Whether the site's episode list differs from ``known`` at its tail (see the module docstring)."""
    if not known:
        return True
    last_page = math.ceil(len(known) / LIST_PAGE_SIZE)
    ids = [e.id for e in site.list_page(show_id, slug, last_page)]
    if ids != known[LIST_PAGE_SIZE * (last_page - 1):]:
        return True
    return len(known) % LIST_PAGE_SIZE == 0 and bool(site.list_page(show_id, slug, last_page + 1))


class Refreshed(NamedTuple):
    #: ``{show id: {name, slug, url, episodes}}`` after this run; failed shows keep their previous entry.
    shows: Dict[str, dict]
    #: Episodes read by walks during this run.
    episodes: Dict[int, Episode]
    stats: Dict[str, int]


def _decide(site: Site, show: Show, stored: Optional[dict], full: bool):
    """``(how, episodes or None)`` for one show."""
    known = list(stored['episodes']) if stored else []
    if not known:
        return 'new', walk_show(site, show.id, show.slug)
    if full:
        return 'full', walk_show(site, show.id, show.slug)
    if tail_changed(site, show.id, show.slug, known):
        return 'changed', walk_show(site, show.id, show.slug)
    return 'same', None


def refresh_shows(site: Site, listing: Dict[int, Show], stored: Dict[str, dict], full: bool = False,
                  workers: int = 16, max_failed_ratio: float = 0.2, desc: str = 'Shows') -> Refreshed:
    """
    Bring the stored episode lists of every listed show up to date.

    Shows that vanished from the site listing keep their stored entry untouched. A show that fails
    keeps its previous entry and is retried next run; when more than ``max_failed_ratio`` of the shows
    fail the run is aborted, because the site is then not in a state worth recording.
    """
    shows = dict(stored)
    episodes: Dict[int, Episode] = {}
    stats = {'new': 0, 'full': 0, 'changed': 0, 'same': 0, 'failed': 0, 'delisted_episodes': 0}

    def _one(show: Show):
        try:
            return show, _decide(site, show, stored.get(str(show.id)), full), None
        except SiteError as err:
            return show, None, err

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for show, decision, err in tqdm(pool.map(_one, [listing[k] for k in sorted(listing)]), total=len(listing),
                                        desc=desc):
            if err is not None:
                stats['failed'] += 1
                logging.warning(f'Show {show.id} ({show.name!r}) skipped this run: {err!r}')
                continue
            how, found = decision
            stats[how] += 1
            entry = {'name': show.name, 'slug': show.slug, 'url': show.url,
                     'episodes': list(stored[str(show.id)]['episodes']) if str(show.id) in stored else []}
            if found is not None:
                ids = [e.id for e in found]
                stats['delisted_episodes'] += len(set(entry['episodes']) - set(ids))
                entry['episodes'] = ids
                episodes.update({e.id: e for e in found})
            shows[str(show.id)] = entry
    if listing and stats['failed'] / len(listing) > max_failed_ratio:
        raise SiteError(f'{stats["failed"]} of {len(listing)} shows failed, the site is not usable right now.')
    return Refreshed(shows, episodes, stats)
