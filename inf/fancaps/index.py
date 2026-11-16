"""Index stage for the anime section of Fancaps, published as ``deepghs/fancaps_index``.

The published layout is the one the earlier job wrote, so everything that reads it keeps working::

    fancaps.parquet   one row per screenshot, nine columns, sorted by ``id`` descending
    meta.json         {exist_bangumi_ids, exist_episode_ids, fancaps_state}
    bangumi.json      bangumis with their episodes, ready made
    README.md         the dataset card, with a few distribution plots in figures/

``meta.json`` keeps its two original lists and gains one key, ``fancaps_state``, the resume state: the
ordered episode ids of every show (what discovery compares against, see :mod:`.discover`), the episodes
listed but not yet read, and a ledger of episodes that failed.

Every commit is consistent on its own and is a valid place to resume from:

* ``exist_episode_ids`` and ``exist_bangumi_ids`` list exactly the episodes and bangumis that have rows
  in the ``fancaps.parquet`` of the same commit;
* an episode enters the table only once *every* page of it was read; a failure, a challenge or a
  truncated page leaves it out and puts it in the ledger, it is never recorded as done;
* the resume state travels in the same commit as the rows it describes, nothing is kept locally.

The table is rewritten as a whole at each commit (a stream merge, see :mod:`.table`), the commit message
says how many images and episodes it adds, and a run that finds nothing new commits nothing, or only the
small ``meta.json`` when the listing state changed.
"""
import datetime
import json
import os
import resource
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List, NamedTuple, Optional

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs
from natsort import natsorted

from inf.utils.duration import duration_type
from inf.utils.safe import configure_hf_http_backend, safe_hf_hub_download, safe_upload_directory_as_directory
from .client import Episode, FancapsClient, Fetcher, SiteBlocked, SiteError, SiteGone, Transport
from .discover import refresh_shows, walk_show
from .readme import compute_stats, render_figures, render_readme
from .table import EpisodeInfo, check_table, empty_table, episode_rows, fresh_rows, merge_table, scan_table

STATE_VERSION = 1
#: Samples shown on the card.
SAMPLE_COUNT = 12


class Job(NamedTuple):
    bangumi_id: int
    bangumi_name: str
    bangumi_url: str
    episode: Episode


def _peak_rss_gb() -> float:
    """Peak resident memory of this process, for the log (Linux reports kilobytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _age_days(stamp: Optional[str]) -> Optional[float]:
    if not stamp:
        return None
    then = datetime.datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc)
    return (datetime.datetime.now(datetime.timezone.utc) - then).total_seconds() / 86400


def build_bangumi_json(episodes: Dict[int, EpisodeInfo]) -> List[dict]:
    """``bangumi.json``: bangumis by id, episodes in natural title order, as the earlier job wrote it."""
    result: Dict[int, dict] = {}
    for episode_id, info in episodes.items():
        item = result.setdefault(info.bangumi_id, {'id': info.bangumi_id, 'title': info.bangumi_name,
                                                   'url': info.bangumi_url, 'episodes': []})
        item['episodes'].append({'id': episode_id, 'title': info.name, 'url': info.url})
    for item in result.values():
        item['episodes'] = natsorted(item['episodes'], key=lambda x: x['title'])
    return sorted(result.values(), key=lambda x: x['id'])


def eligible(episode_id: int, ledger: Dict[str, dict], full_run: bool, max_attempts: int) -> bool:
    """Whether a listed, unread episode is tried this run (see the ledger rules in the module docstring)."""
    entry = ledger.get(str(episode_id))
    if entry is None or full_run:
        return True
    if entry['kind'] in ('gone', 'duplicate'):
        return False
    return entry['attempts'] < max_attempts


def _load_remote(hf_client, repository: str, work: str):
    """``(meta, local table path or None)``; the table is copied into ``work``, never edited in the cache."""
    meta = {'exist_bangumi_ids': [], 'exist_episode_ids': []}
    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='meta.json'):
        path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset', filename='meta.json')
        with open(path, 'r') as f:
            meta = json.load(f)
    base = None
    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='fancaps.parquet'):
        path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset', filename='fancaps.parquet')
        base = os.path.join(work, 'base.parquet')
        shutil.copyfile(path, base)
    return meta, base


def _samples(parquet_path: str) -> List[dict]:
    """One row for each of the bangumis holding the newest screenshots (the table starts with the newest)."""
    pf = pq.ParquetFile(parquet_path)
    columns = ['id', 'bangumi_name', 'bangumi_id', 'episode_name', 'episode_id', 'url']
    seen, out = set(), []
    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i, columns=columns)
        for row in table.to_pylist():
            if row['bangumi_id'] not in seen:
                seen.add(row['bangumi_id'])
                out.append(row)
                if len(out) >= SAMPLE_COUNT:
                    return out
    return out


def sync(repository: str, max_time_limit: Optional[float] = 5 * 3600, deploy_span: float = 3600.0, workers: int = 16,
         full_walk_days: float = 7.0, full_walk: bool = False, max_episodes: Optional[int] = None, retries: int = 8,
         max_attempts: int = 3, proxy: Optional[str] = None, dry_run: bool = False,
         output_dir: Optional[str] = None, client: Optional[FancapsClient] = None):
    """
    One run: discover the episodes the site lists and the table lacks, read them, commit.

    :param repository: Index dataset repository.
    :param max_time_limit: Stop scheduling episodes after this many seconds. ``None`` is unlimited.
    :param deploy_span: Minimum seconds between two commits while episodes keep arriving.
    :param workers: Threads reading episode pages.
    :param full_walk_days: Every show is walked completely when the last complete walk is older than this.
    :param full_walk: Force a complete walk now.
    :param max_episodes: Stop after scheduling this many episodes (smoke runs).
    :param retries: Attempts per page.
    :param max_attempts: Runs an episode may fail in a row before it waits for the next complete walk.
    :param proxy: Optional proxy url.
    :param dry_run: Build everything, upload nothing.
    :param output_dir: With ``dry_run``, where the files of the last commit are copied.
    :param client: A site client (tests).
    """
    start_time = time.time()
    deadline = start_time + max_time_limit if max_time_limit else None
    configure_hf_http_backend()
    delete_detached_cache()
    hf_client = get_hf_client()
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attrs = get_hf_fs().read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attrs += ['*.json filter=lfs diff=lfs merge=lfs -text', '*.csv filter=lfs diff=lfs merge=lfs -text']
        get_hf_fs().write_text(f'datasets/{repository}/.gitattributes', os.linesep.join(attrs))
        logging.info(f'Repository {repository!r} created.')

    client = client or FancapsClient(Fetcher(Transport(proxy=proxy), retries=retries))
    with TemporaryDirectory() as work:
        meta, base_path = _load_remote(hf_client, repository, work)
        scan = scan_table(base_path)
        known_ids = scan.ids
        episode_counts = dict(scan.episode_counts)
        infos: Dict[int, EpisodeInfo] = dict(scan.episodes)
        rows_total = scan.rows
        logging.info(f'Table loaded: {plural_word(rows_total, "image")}, {plural_word(len(infos), "episode")}.')

        state = meta.get('fancaps_state') or {}
        stored_shows: Dict[str, dict] = state.get('shows') or {}
        pending_meta: Dict[str, list] = dict(state.get('pending') or {})
        ledger: Dict[str, dict] = dict(state.get('failed') or {})
        last_full_walk: Optional[str] = state.get('last_full_walk')

        client.ensure_access()
        listing = client.show_list()
        logging.info(f'The site lists {plural_word(len(listing), "bangumi")}.')
        age = _age_days(last_full_walk)
        full_run = bool(full_walk or age is None or age >= full_walk_days)
        refreshed = refresh_shows(client, listing, stored_shows, full=full_run, workers=workers)
        logging.info(f'Listing refreshed ({"complete walk" if full_run else "tail probes"}): {refreshed.stats}')
        shows = refreshed.shows
        if full_run and refreshed.stats['failed'] == 0:
            last_full_walk = _now()

        ep_show: Dict[int, int] = {}
        for sid in listing:
            for e in shows[str(sid)]['episodes']:
                ep_show[e] = sid
        found = dict(refreshed.episodes)
        have = {int(k): Episode(int(k), v[1], v[2]) for k, v in pending_meta.items()}
        have.update(found)
        unread = set(ep_show) - set(infos)
        for e in sorted(unread - set(have)):                      # listed, unread, and no title / address kept
            sid = ep_show[e]
            for ep in walk_show(client, sid, shows[str(sid)]['slug']):
                have.setdefault(ep.id, ep)
        pending = sorted((e for e in unread if eligible(e, ledger, full_run, max_attempts)),
                         key=lambda e: (ep_show[e], e))
        logging.info(f'{plural_word(len(unread), "episode")} listed and not in the table, {len(pending)} to try now.')

        # --- state published with a commit -------------------------------------------------------------
        def _state() -> dict:
            unread_now = set(ep_show) - set(infos)
            return {
                'version': STATE_VERSION,
                'shows': shows,
                'pending': {str(e): [ep_show[e], have[e].name, have[e].url] for e in sorted(unread_now) if e in have},
                'failed': ledger,
                'last_full_walk': last_full_walk,
            }

        def _meta() -> dict:
            return {
                'exist_bangumi_ids': sorted({i.bangumi_id for i in infos.values()}),
                'exist_episode_ids': sorted(infos),
                'fancaps_state': _state(),
            }

        published_state = json.dumps(state.get('shows') or {}, sort_keys=True) + json.dumps(
            [state.get('failed') or {}, state.get('last_full_walk')], sort_keys=True)

        def _state_fingerprint() -> str:
            return json.dumps(shows, sort_keys=True) + json.dumps([ledger, last_full_walk], sort_keys=True)

        buffer_tables: List[pa.Table] = []
        buffer_infos: Dict[int, EpisodeInfo] = {}
        last_deploy = time.time()
        counters = {'images': 0, 'episodes': 0, 'commits': 0}
        base = {'path': base_path}

        def _deploy(reason: str):
            nonlocal known_ids, rows_total, published_state, buffer_tables, buffer_infos, last_deploy
            rows_added = bool(buffer_tables)
            if not rows_added and _state_fingerprint() == published_state:
                return
            added_images, added_episodes = 0, 0
            with TemporaryDirectory(dir=work) as ud:
                new = empty_table()
                if rows_added:
                    new = fresh_rows(pa.concat_tables(buffer_tables), known_ids)
                    written = set(np.unique(new['episode_id'].to_numpy()).tolist()) if new.num_rows else set()
                    for e, info in buffer_infos.items():
                        if e in written:
                            infos[e] = info
                        else:                                     # every image was already in the table elsewhere
                            logging.warning(f'Episode {e} adds no new image, set aside.')
                            ledger[str(e)] = {'kind': 'duplicate', 'attempts': max_attempts, 'at': _now(),
                                              'last': 'every image id is already in the table'}
                    added_images, added_episodes = new.num_rows, len(written)
                    if new.num_rows:
                        eps, counts = np.unique(new['episode_id'].to_numpy(), return_counts=True)
                        for e, c in zip(eps.tolist(), counts.tolist()):
                            episode_counts[e] = episode_counts.get(e, 0) + c
                        out = os.path.join(ud, 'fancaps.parquet')
                        merge_table(base['path'], new, out)
                        check_table(out, rows_total + new.num_rows)
                        new_ids = new['id'].to_numpy()
                        low = min(int(new_ids.min()), int(known_ids[0])) if len(known_ids) else int(new_ids.min())
                        high = max(int(new_ids.max()), int(known_ids[-1])) if len(known_ids) else int(new_ids.max())
                        stats = compute_stats(episode_counts, infos, low, high)
                        figures_dir = os.path.join(ud, 'figures')
                        render_figures(stats, figures_dir)
                        with open(os.path.join(ud, 'README.md'), 'w') as f:
                            f.write(render_readme(stats, figures_dir, _samples(out)))
                        with open(os.path.join(ud, 'bangumi.json'), 'w') as f:
                            json.dump(build_bangumi_json(infos), f)
                with open(os.path.join(ud, 'meta.json'), 'w') as f:
                    json.dump(_meta(), f)

                message = (f'Add {plural_word(added_images, "new image")}, with '
                           f'{plural_word(added_episodes, "new episode")}' if added_images
                           else 'Update the episode listing state')
                if dry_run:
                    logging.info(f'[dry-run] Would commit ({reason}): {message}')
                    if output_dir:
                        shutil.rmtree(output_dir, ignore_errors=True)
                        shutil.copytree(ud, output_dir)
                else:
                    logging.info(f'UPLOAD ({reason}): {message}')
                    started = time.time()
                    safe_upload_directory_as_directory(
                        local_directory=ud, repo_id=repository, repo_type='dataset', path_in_repo='.', message=message)
                    logging.info(f'UPLOAD done in {time.time() - started:.0f}s, peak memory {_peak_rss_gb():.1f} GB.')
                    counters['commits'] += 1
                if added_images:
                    # what was just published is the table the next merge starts from
                    current = os.path.join(work, 'base.parquet')
                    os.replace(os.path.join(ud, 'fancaps.parquet'), current)
                    base['path'] = current
                    known_ids = np.union1d(known_ids, new['id'].to_numpy())
                    rows_total += added_images
                    counters['images'] += added_images
                    counters['episodes'] += added_episodes
            buffer_tables, buffer_infos = [], {}
            published_state = _state_fingerprint()
            last_deploy = time.time()

        # state first: the listing alone is worth keeping (a restart then does not have to walk again)
        _deploy('listing state')

        # --- reading episodes ------------------------------------------------------------------------
        todo = iter(pending)
        scheduled = 0
        inflight: Dict[object, Job] = {}
        finished = 0
        last_log = time.time()

        def _read(job: Job):
            return client.crawl_episode(job.episode.url)

        def _job(e: int) -> Job:
            show = shows[str(ep_show[e])]
            return Job(ep_show[e], show['name'], show['url'], have[e])

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                while True:
                    while (len(inflight) < workers * 2 and (deadline is None or time.time() < deadline)
                           and (max_episodes is None or scheduled < max_episodes)):
                        e = next(todo, None)
                        if e is None:
                            break
                        job = _job(e)
                        inflight[pool.submit(_read, job)] = job
                        scheduled += 1
                    if not inflight:
                        break
                    done, _ = wait(list(inflight), timeout=5.0, return_when=FIRST_COMPLETED)
                    blocked: Optional[SiteBlocked] = None
                    for future in done:
                        job = inflight.pop(future)
                        key = str(job.episode.id)
                        finished += 1
                        try:
                            images = future.result()
                        except SiteBlocked as err:                # finish this batch first, then stop
                            blocked = err
                            continue
                        except SiteGone as err:
                            ledger[key] = {'kind': 'gone', 'attempts': ledger.get(key, {}).get('attempts', 0) + 1,
                                           'at': _now(), 'last': str(err)[:200]}
                            continue
                        except SiteError as err:
                            ledger[key] = {'kind': 'error', 'attempts': ledger.get(key, {}).get('attempts', 0) + 1,
                                           'at': _now(), 'last': str(err)[:200]}
                            logging.warning(f'Episode {job.episode.id} not read: {err}')
                            continue
                        if not images:
                            ledger[key] = {'kind': 'empty', 'attempts': ledger.get(key, {}).get('attempts', 0) + 1,
                                           'at': _now(), 'last': 'the episode page lists no image'}
                            continue
                        ledger.pop(key, None)
                        buffer_tables.append(episode_rows(job.bangumi_id, job.bangumi_name, job.bangumi_url,
                                                          job.episode, images))
                        buffer_infos[job.episode.id] = EpisodeInfo(job.bangumi_id, job.bangumi_name, job.bangumi_url,
                                                                   job.episode.name, job.episode.url)
                    if blocked is not None:
                        raise blocked
                    if time.time() - last_log >= 60:
                        logging.info(f'{finished}/{len(pending)} episodes read, {len(buffer_tables)} buffered, '
                                     f'{client.fetcher.requests} requests, {client.fetcher.retried} retried, '
                                     f'strategy {client.fetcher.transport.strategy!r}.')
                        last_log = time.time()
                    if buffer_tables and time.time() - last_deploy >= deploy_span:
                        _deploy('span')
        finally:
            _deploy('final')

        logging.info(f'Done: {plural_word(counters["images"], "image")} and '
                     f'{plural_word(counters["episodes"], "episode")} added in {counters["commits"]} commits, '
                     f'{client.fetcher.requests} requests '
                     f'({client.fetcher.retried} retried, {client.fetcher.transport.rotations} strategy changes), '
                     f'peak memory {_peak_rss_gb():.1f} GB.')
        remaining = len(set(ep_show) - set(infos))
        logging.info(f'{plural_word(remaining, "episode")} listed and not in the table; '
                     f'{plural_word(len(ledger), "episode")} in the ledger.')


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Index the anime section of Fancaps: find the episodes the site lists and the table lacks, read their '
         'screenshot ids, and commit them to the index repository.',
)
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_FC', default='deepghs/fancaps_index',
              show_default=True, help='Index dataset repository.')
@click.option('-t', '--max-time-limit', type=duration_type(), default='5h', show_default=True,
              help='Stop scheduling episodes after this long.')
@click.option('-s', '--deploy-span', type=duration_type(), default='1h', show_default=True,
              help='Minimum interval between two commits while episodes keep arriving.')
@click.option('-w', '--workers', type=int, default=16, show_default=True, help='Threads reading episode pages.')
@click.option('--full-walk-days', type=float, default=7.0, show_default=True,
              help='Walk every show completely when the last complete walk is older than this many days.')
@click.option('--full-walk', is_flag=True, default=False, help='Walk every show completely now.')
@click.option('-n', '--max-episodes', type=int, default=None, help='Stop after this many episodes (smoke runs).')
@click.option('-c', '--retries', type=int, default=8, show_default=True, help='Attempts per page.')
@click.option('--max-attempts', type=int, default=3, show_default=True,
              help='Failed runs in a row after which an episode waits for the next complete walk.')
@click.option('--proxy', type=str, envvar='FANCAPS_PROXY', default=None, help='Proxy url for every request.')
@click.option('--dry-run', is_flag=True, default=False, help='Build everything, upload nothing.')
@click.option('--output-dir', type=str, default=None, help='With --dry-run, copy the files of the last commit here.')
def cli(repository: str, max_time_limit: Optional[float], deploy_span: float, workers: int, full_walk_days: float,
        full_walk: bool, max_episodes: Optional[int], retries: int, max_attempts: int, proxy: Optional[str],
        dry_run: bool, output_dir: Optional[str]):
    logging.try_init_root(logging.INFO)
    sync(repository=repository, max_time_limit=max_time_limit, deploy_span=deploy_span, workers=workers,
         full_walk_days=full_walk_days, full_walk=full_walk, max_episodes=max_episodes, retries=retries,
         max_attempts=max_attempts, proxy=proxy, dry_run=dry_run, output_dir=output_dir)


if __name__ == '__main__':
    cli()
