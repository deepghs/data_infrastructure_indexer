"""A scripted stand-in for the site and a stand-in for the hub, shared by the fancaps tests."""
import json
import os
import shutil
from types import SimpleNamespace
from typing import Dict, List

from inf.fancaps.client import Episode, Show, SiteError, SiteGone


def episode_url(ep: int) -> str:
    return f'https://fancaps.net/anime/episodeimages.php?{ep}-S/E{ep}'


class FakeSite:
    """
    The site as :class:`inf.fancaps.client.FancapsClient` shows it.

    ``shows`` is ``{show id: {'name': ..., 'episodes': [episode id, ...] in site order}}``; every episode owns
    ``images[episode id]`` image ids. ``broken[ep]`` is an exception raised when the episode is read.
    """

    def __init__(self, shows: Dict[int, dict], images: Dict[int, List[int]] = None, thumb='https://ant.fancaps.net'):
        self.shows = shows
        self.images = images or {}
        self.broken: Dict[int, Exception] = {}
        self.thumb = thumb
        self.requests = {'list': 0, 'episode': 0}
        self.fetcher = SimpleNamespace(requests=0, retried=0,
                                       transport=SimpleNamespace(strategy='fake', rotations=0))

    def ensure_access(self):
        return 'fake'

    def show_list(self):
        return {sid: Show(sid, s['name'], f'https://fancaps.net/anime/showimages.php?{sid}-{s["name"]}', s['name'])
                for sid, s in self.shows.items()}

    def list_page(self, show_id, slug, number):
        self.requests['list'] += 1
        eps = self.shows[show_id]['episodes'][10 * (number - 1):10 * number]
        return [Episode(e, f'Episode {e}', episode_url(e)) for e in eps]

    def crawl_episode(self, url):
        self.requests['episode'] += 1
        ep = int(url.split('?')[1].split('-')[0])
        if ep in self.broken:
            raise self.broken[ep]
        return [(i, f'{self.thumb}/{i}.jpg') for i in self.images.get(ep, [])]


def gone(ep):
    return SiteGone(f'episode {ep} -> HTTP 404')


def fail(ep):
    return SiteError(f'episode {ep} failed after 8 attempts')


class FakeHub:
    """Files by repository; ``upload`` copies a directory in and remembers each commit."""

    def __init__(self, tmp_path):
        self.tmp, self.files, self.commits = tmp_path, {}, []
        self.on_commit = None

    def put(self, repo, name, path):
        self.files.setdefault(repo, {})[name] = str(path)

    def upload(self, local_directory, repo_id, repo_type, path_in_repo, message, **kw):
        dest = self.tmp / f'commit{len(self.commits)}'
        shutil.copytree(local_directory, dest)
        self.commits.append((message, dest))
        for dirpath, _, names in os.walk(dest):
            for name in names:
                full = os.path.join(dirpath, name)
                self.put(repo_id, os.path.relpath(full, dest), full)
        if self.on_commit:
            self.on_commit()

    def meta(self):
        with open(self.commits[-1][1] / 'meta.json') as f:
            return json.load(f)
