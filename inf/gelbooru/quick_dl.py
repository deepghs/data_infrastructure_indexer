"""Staging download stage for Gelbooru.

Pull original image bytes for posts listed in ``deepghs/gelbooru_index`` and pack them into
hfutils-indexed tar volumes inside a private staging repository (default
``deepghs/gelbooru_newest_dl``). A later repack stage consumes those volumes instead of hitting
the image hosts again.

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
                       read-only baseline: every id ``gelbooru_full`` already holds, bad ids included
    README.md          statistics, coverage and usage

This repository therefore holds a *difference*, not a corpus: the posts the index knows about that
``gelbooru_full`` does not already store. Every commit carries one whole volume with its sidecar,
the table, the meta file and the README, so the repository is consistent at every commit.

What measuring on GitHub runners showed, and what the code does about it
=========================================================================

* ``img2.gelbooru.com`` (half of the index's links) serves a certificate for another name and, with
  verification off, only a 404 page; ``img3`` does not accept connections. ``img4`` serves the very
  same paths with the right bytes. Every download goes to ``img4``, and the table stores that url,
  the one actually used, not the one in the index.
* Without ``Referer: https://gelbooru.com/`` the host answers **HTTP 200 with an html page**. A 200
  is therefore never proof of success: the md5 recorded in the index is checked on every file.
* Plain ``requests`` with only a user agent is refused about half the time; a browser fingerprint
  (``curl_cffi``) is not.
* Up to eight threads were served in full. At sixteen and above the host started answering
  **404 for files that exist**. A 404 is therefore not proof that a post is gone either: it is
  retried, and a post is only recorded as bad after a slow second look, by one thread, at a time
  when a control download from the same volume proves the host is serving at all.
"""
import datetime
import json
import math
import os
import random
import re
import shutil
import tarfile
import time
from contextlib import contextmanager
from queue import Queue
from threading import Lock
from typing import Iterator, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from curl_cffi import requests as cffi_requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.index import tar_create_index_for_directory
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.utils import hf_normpath, number_to_tag
from natsort import natsorted
from pyrate_limiter import Rate, Limiter, Duration
from tqdm import tqdm

from inf.utils.download import AdaptiveRateLimiter, DownloadDigestMismatch, download_file, parallel_call, \
    get_free_disk_bytes, log_disk_usage
from inf.utils.duration import duration_type
from inf.utils.safe import configure_hf_http_backend, safe_hf_hub_download, \
    safe_upload_directory_as_directory

Image.MAX_IMAGE_PIXELS = 32768 ** 2

#: The only image host that serves usable files, see the module docstring.
IMAGE_HOST = 'img4.gelbooru.com'
#: Without it the host answers 200 with an html page.
REFERER = 'https://gelbooru.com/'
_IMAGE_HOST_URL = re.compile(r'^https?://img\d*\.gelbooru\.com/', re.IGNORECASE)

#: Columns read out of the index tables.
_SCAN_COLUMNS = ['id', 'file_url', 'md5', 'mimetype', 'width', 'height', 'rating']
#: Columns kept in the staging ``table.parquet``.
_TABLE_COLUMNS = ['id', 'filename', 'volume_file', 'file_size', 'mimetype', 'file_ext',
                  'width', 'height', 'rating', 'md5', 'file_url']

#: Browser fingerprints rotated through when a session is replaced.
_FINGERPRINTS = ['chrome131', 'chrome124', 'chrome136', 'chrome123', 'safari17_0', 'firefox135']

#: The index carries no file size. Bytes per pixel by type, measured on the first 32k stored images (the
#: first guesses of 0.40 / 1.0 planned 10 GB volumes that came out at 6.4-7.1 GB), used only to plan how many
#: posts one volume should try to hold; the run also rescales them from every finished volume, and the hard
#: ceiling decides what really fits.
_BYTES_PER_PIXEL = {'image/jpeg': 0.27, 'image/png': 0.66, 'image/gif': 7.5}
_DEFAULT_BYTES_PER_PIXEL = 0.45


def download_url(file_url: str) -> str:
    """The url a download actually uses: the image host that works, whichever the index names."""
    return _IMAGE_HOST_URL.sub(f'https://{IMAGE_HOST}/', file_url or '')


def estimate_bytes(row: dict) -> int:
    """A rough size for volume planning, from the pixel count and the file type."""
    pixels = float(row.get('width') or 0) * float(row.get('height') or 0)
    factor = _BYTES_PER_PIXEL.get((row.get('mimetype') or '').lower(), _DEFAULT_BYTES_PER_PIXEL)
    return int(max(pixels * factor, 50 * 1024))


def _volume_paths(volume_id: int) -> Tuple[str, str]:
    rel_tar = f'images/{volume_id // 1000}/{volume_id % 1000:03d}.tar'
    return rel_tar, f'{os.path.splitext(rel_tar)[0]}.json'


class GelbooruSessionPool:
    """
    A fixed set of browser-fingerprinted sessions leased to the download threads.

    A session that is refused is replaced by one with the next fingerprint; a healthy one is kept,
    since a warm connection is the cheapest throughput there is.
    """

    def __init__(self, size: int, fingerprints: Optional[List[str]] = None):
        self._fingerprints = list(fingerprints or _FINGERPRINTS)
        self._queue: Queue = Queue()
        self._lock = Lock()
        self._counter = 0
        self.replaced = 0
        for _ in range(max(size, 1)):
            self._queue.put(self._new())

    def _new(self):
        with self._lock:
            fingerprint = self._fingerprints[self._counter % len(self._fingerprints)]
            self._counter += 1
        return fingerprint, cffi_requests.Session(impersonate=fingerprint)

    @contextmanager
    def lease(self):
        slot = self._queue.get()
        holder = [slot]
        try:
            yield holder
        finally:
            self._queue.put(holder[0])

    def replace(self, holder: list):
        """Swap the leased session for one with another fingerprint."""
        try:
            holder[0][1].close()
        except Exception:  # pragma: no cover - closing is best effort
            pass
        holder[0] = self._new()
        self.replaced += 1

    def close(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()[1].close()
            except Exception:  # pragma: no cover
                pass


class UndecodableImage(Exception):
    """A download whose md5 matches the index but which the image decoder still refuses."""

    def __init__(self, post_id: int, url: str, reason: Exception):
        Exception.__init__(self, f'Post {post_id} downloaded intact from {url!r} but could not be decoded: '
                                 f'{reason!r}')
        self.post_id = post_id
        self.url = url
        self.reason = reason


def _status_of(err: Exception) -> Optional[int]:
    return getattr(getattr(err, 'response', None), 'status_code', None)


def classify_error(err: Exception) -> str:
    """
    ``'unusable'``, ``'missing'``, ``'rate_limit'``, ``'blocked'`` or ``'network'``.

    ``missing`` is a 404 or 410, which on this host is a *suspicion*, not a verdict: it also is what
    the host answers when it is being asked too fast. ``blocked`` covers a refused fingerprint and a
    200 whose body is not the file (the host's hotlink page), both cured by another session.
    """
    if isinstance(err, UndecodableImage):
        return 'unusable'
    status = _status_of(err)
    if status in (404, 410):
        return 'missing'
    if status == 429:
        return 'rate_limit'
    if status == 403 or isinstance(err, DownloadDigestMismatch):
        return 'blocked'
    return 'network'


def _load_baseline(hf_client, baseline_repository: str) -> List[int]:
    """Every id ``baseline_repository`` holds, bad ids included."""
    ids: set = set()
    for filename in ('exist_ids.json', 'bad_ids.json'):
        if hf_client.file_exists(repo_id=baseline_repository, repo_type='dataset', filename=filename):
            path = safe_hf_hub_download(hf_client, repo_id=baseline_repository, repo_type='dataset',
                                        filename=filename)
            with open(path, 'r') as f:
                ids.update(int(x) for x in json.load(f))
    return sorted(ids)


def _load_state(hf_client, repository: str, baseline_repository: str, glob_exist_ids_file: str):
    """
    The staging repository's state: ``(records, covered ids, bad ids, max volume id, baseline ids,
    baseline_is_new)``. The baseline is read from the repository when it holds one, and built from
    ``baseline_repository`` otherwise; in that case the first commit has to carry it.
    """
    records: List[dict] = []
    bad_image_ids: set = set()
    max_volume_id = 0

    baseline_is_new = False
    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename=glob_exist_ids_file):
        path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset',
                                    filename=glob_exist_ids_file)
        with open(path, 'r') as f:
            baseline = [int(x) for x in json.load(f)]
    else:
        baseline = _load_baseline(hf_client, baseline_repository)
        baseline_is_new = True
        logging.info(f'No {glob_exist_ids_file!r} in {repository!r}, built one from {baseline_repository!r}.')
    covered_ids = set(baseline)
    logging.info(f'Baseline loaded, {plural_word(len(baseline), "id")}.')

    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='table.parquet'):
        path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset', filename='table.parquet')
        df = pd.read_parquet(path).replace(np.NaN, None)
        records = df.to_dict('records')
        covered_ids.update(int(x) for x in df['id'])
        logging.info(f'Existing table loaded, {plural_word(len(records), "row")}.')

    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='meta.json'):
        path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset', filename='meta.json')
        with open(path, 'r') as f:
            meta = json.load(f)
        bad_image_ids.update(int(x) for x in (meta.get('bad_image_ids') or []))
        max_volume_id = meta.get('max_volume_id') or 0
        covered_ids.update(bad_image_ids)
        logging.info(f'Meta loaded, max_volume_id={max_volume_id}, {plural_word(len(bad_image_ids), "bad id")}.')

    return records, covered_ids, bad_image_ids, max_volume_id, baseline, baseline_is_new


def _iter_index_rows(src_repository: str, src_revision: str) -> Iterator[dict]:
    """The index's rows, streamed table by table through range requests with a column projection."""
    hf_fs = get_hf_fs()
    prefix = f'datasets/{src_repository}'
    if src_revision and src_revision != 'main':
        prefix = f'datasets/{src_repository}@{src_revision}'
    tables = natsorted(hf_fs.glob(f'{prefix}/tables/gelbooru-*.parquet'))
    logging.info(f'{plural_word(len(tables), "index table")} found in {src_repository!r}.')
    for table in tables:
        with hf_fs.open(table, 'rb') as f:
            pf = pq.ParquetFile(f)
            with tqdm(total=pf.metadata.num_rows, desc=f'Scanning {os.path.basename(table)}') as pg:
                for batch in pf.iter_batches(batch_size=200000, columns=_SCAN_COLUMNS):
                    rows = batch.to_pylist()
                    pg.update(len(rows))
                    yield from rows


def scan_candidates(rows: Iterator[dict], covered_ids: set) -> List[dict]:
    """
    The posts still to download, oldest id first, each with the url it is downloaded from.

    Only images with a url and an md5 qualify: without the md5 a download could not be told from the
    host's hotlink page.
    """
    candidates: List[dict] = []
    for row in rows:
        if row['id'] in covered_ids:
            continue
        if not (row['mimetype'] or '').startswith('image/'):
            continue
        if not row['file_url'] or not row['md5']:
            continue
        candidates.append({**row, 'id': int(row['id']), 'file_url': download_url(row['file_url'])})
    candidates.sort(key=lambda x: x['id'])
    return candidates


def take_volume(candidates: List[dict], max_volume_files: int, max_volume_bytes: int, scale: float = 1.0) -> int:
    """
    The exclusive end of the next volume-sized slice, by estimated bytes times ``scale``; a slice has one
    post at least.
    """
    end, total = 0, 0
    while end < len(candidates):
        size = estimate_bytes(candidates[end]) * scale
        if end > 0 and (end >= max_volume_files or total + size > max_volume_bytes):
            break
        total += size
        end += 1
    return end


def write_readme(md_file: str, df_table: pd.DataFrame, bad_image_ids: set, max_volume_id: int, src_repository: str,
                 repository: str, baseline_repository: str, baseline_size: int, glob_exist_ids_file: str):
    total_bytes = int(df_table['file_size'].sum()) if len(df_table) else 0
    current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    with open(md_file, 'w') as f:
        print('---', file=f)
        print('license: other', file=f)
        print('task_categories:', file=f)
        print('- image-classification', file=f)
        print('- zero-shot-image-classification', file=f)
        print('- text-to-image', file=f)
        print('language:', file=f)
        print('- en', file=f)
        print('tags:', file=f)
        print('- art', file=f)
        print('- anime', file=f)
        print('- not-for-all-audiences', file=f)
        print('size_categories:', file=f)
        print(f'- {number_to_tag(len(df_table))}', file=f)
        print('annotations_creators:', file=f)
        print('- no-annotation', file=f)
        print('source_datasets:', file=f)
        print('- gelbooru', file=f)
        print('---', file=f)
        print('', file=f)
        print('# Gelbooru Staging Download Dataset', file=f)
        print('', file=f)
        print(f'Raw original files of the Gelbooru posts listed in `{src_repository}` that '
              f'`{baseline_repository}` does not already hold. This is a staging area for the repack stage and '
              f'is deliberately incomplete on its own.', file=f)
        print('', file=f)
        print('# Coverage', file=f)
        print('', file=f)
        print(f'Before anything is downloaded, every id `{baseline_repository}` holds is removed from the candidate '
              f'list. That exclusion set ships here as `{glob_exist_ids_file}` ({baseline_size:,} ids, its bad ids '
              f'included); it is read-only, this job never adds to it.', file=f)
        print('', file=f)
        print('The complete set of Gelbooru images is therefore the union of the two repositories.', file=f)
        print('', file=f)
        print('# Information', file=f)
        print('', file=f)
        print(f'There are {plural_word(len(df_table), "image")} in {max_volume_id} volume(s), '
              f'{total_bytes / 1024 ** 3:.2f} GB. The maximum ID is '
              f'{int(df_table["id"].max()) if len(df_table) else 0}. {plural_word(len(bad_image_ids), "post")} '
              f'are recorded as permanently unavailable (`bad_image_ids` in `meta.json`). '
              f'Last updated at `{current_time}`.', file=f)
        print('', file=f)
        print('`file_url` is the url the file was actually downloaded from (`img4.gelbooru.com`), which can differ '
              'from the one in the index: `img2` and `img3` do not serve usable files.', file=f)
        print('', file=f)
        shown = df_table.sort_values(by=['id'], ascending=False)[:50][
            ['id', 'volume_file', 'filename', 'width', 'height', 'mimetype', 'file_size', 'file_url']]
        print(f'Recent {plural_word(len(shown), "image")}:', file=f)
        print('', file=f)
        print(shown.to_markdown(index=False), file=f)
        print('', file=f)


def sync(repository: str, src_repository: str, baseline_repository: str, src_revision: str = 'main',
         max_time_limit: Optional[float] = 5 * 60 * 60, max_volume_files: int = 10000,
         max_volume_bytes: int = 10 * 1024 ** 3, max_volume_hard_bytes: int = 12 * 1024 ** 3,
         download_workers: int = 8, min_free_disk: int = 24 * 1024 ** 3, upload_time_span: float = 30,
         glob_exist_ids_file: str = 'glob_exist_ids.json', max_volumes: Optional[int] = None,
         initial_rate: float = 4.0, max_rate: float = 12.0, retries: int = 6, retry_wait: float = 2.0,
         max_failed_ratio: float = 0.3, min_attempts_for_abort: int = 20, diagnose_attempts: int = 3,
         diagnose_wait: float = 3.0, cooldown: float = 60.0, dry_run: bool = False):
    """
    Download the missing Gelbooru originals into append-only tar volumes of the staging repository.

    :param repository: Staging dataset repository.
    :param src_repository: The index repository.
    :param baseline_repository: The repository whose ids are not downloaded again.
    :param download_workers: Download threads; eight is what the host serves in full, see the module docstring.
    :param retries: Attempts per post before it counts as failed, or as suspect when the answer is 404.
    :param diagnose_attempts: Slow looks, one thread at a time, taken at a suspect post before it is recorded bad.
    :param diagnose_wait: Seconds between two of them.
    :param cooldown: Seconds waited when the control download shows the host is not serving.
    :param max_failed_ratio: A volume whose failures reach this fraction is not published.
    :param min_attempts_for_abort: Fewer attempts than this are too few to say the host is blocking.
    :param dry_run: Download and build, but upload nothing.
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

    records, covered_ids, bad_image_ids, max_volume_id, baseline, baseline_is_new = _load_state(
        hf_client=hf_client, repository=repository, baseline_repository=baseline_repository,
        glob_exist_ids_file=glob_exist_ids_file)
    logging.info(f'{plural_word(len(covered_ids), "id")} already covered, current max volume id: {max_volume_id}.')

    candidates = scan_candidates(_iter_index_rows(src_repository, src_revision), covered_ids)
    if not candidates:
        logging.info('No candidate to download, quit.')
        return
    logging.info(f'{plural_word(len(candidates), "candidate")} to download, ids {candidates[0]["id"]} - '
                 f'{candidates[-1]["id"]}, about '
                 f'{sum(estimate_bytes(x) for x in candidates) / 1024 ** 4:.2f} TB estimated.')

    pool = GelbooruSessionPool(size=download_workers + 2)
    rate_limiter = AdaptiveRateLimiter(initial=initial_rate, maximum=max_rate)
    request_headers = {'Referer': REFERER, 'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
                       'Accept-Language': 'en-US,en;q=0.9'}

    volumes_done = 0
    size_scale = 1.0  # real bytes / estimated bytes, learned from the volumes this run has finished
    pending = candidates
    parked: List[dict] = []  # suspects the host would not settle this run; asked again next run
    aborted = False
    while pending:
        if max_time_limit is not None and start_time + max_time_limit < time.time():
            logging.info('Max time limit exceeded, stop scheduling new volumes.')
            break
        if max_volumes is not None and volumes_done >= max_volumes:
            logging.info(f'Reached --max-volumes {max_volumes}, stop.')
            break
        log_disk_usage(os.getcwd(), prefix='Disk before volume')
        free = get_free_disk_bytes(os.getcwd())
        if free < min_free_disk:
            logging.warning(f'Only {free / 1024 ** 3:.2f} GB free, below the {min_free_disk / 1024 ** 3:.2f} GB '
                            f'floor - stop cleanly.')
            break

        end = take_volume(pending, max_volume_files, max_volume_bytes, scale=size_scale)
        plan, rest = pending[:end], pending[end:]
        max_volume_id += 1
        rel_tar, rel_index = _volume_paths(max_volume_id)
        logging.info(f'Building volume #{max_volume_id} ({rel_tar}), {plural_word(len(plan), "post")}, '
                     f'about {sum(estimate_bytes(x) for x in plan) * size_scale / 1024 ** 3:.2f} GB planned '
                     f'(size scale {size_scale:.2f}) ...')

        with TemporaryDirectory() as td:
            upload_dir = os.path.join(td, 'upload')
            stage_dir = os.path.join(td, 'stage')
            os.makedirs(stage_dir, exist_ok=True)
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
            stats = {'ok': 0, 'gone': 0, 'unusable': 0, 'failed': 0, 'suspect': 0, 'retry': 0, 'deferred': 0}

            with tarfile.open(tar_file, 'w:') as tar:
                def _fetch(item, session, dst_file):
                    """Download one post and return ``(size, width, height)``; the md5 and the decoder both check it."""
                    size = download_file(item['file_url'], dst_file, session=session, expected_md5=item['md5'],
                                         headers=request_headers, timeout=60)
                    try:
                        with Image.open(dst_file) as image:
                            width, height = image.size
                    except Exception as decode_err:
                        raise UndecodableImage(item['id'], item['file_url'], decode_err) from decode_err
                    return size, width, height

                def _store(item, filename, dst_file, size, width, height):
                    """Put a downloaded file in the tar and its row in the records, under the lock."""
                    with lock:
                        tar.add(dst_file, filename)
                        volume_bytes[0] += size
                        stats['ok'] += 1
                        if control[0] is None:
                            control[0] = item
                        _, ext = os.path.splitext(filename)
                        new_records.append({
                            'id': item['id'], 'filename': filename, 'volume_file': hf_normpath(rel_tar),
                            'file_size': size, 'mimetype': item['mimetype'], 'file_ext': ext.lstrip('.'),
                            'width': width or item['width'], 'height': height or item['height'],
                            'rating': item['rating'], 'md5': item['md5'], 'file_url': item['file_url'],
                        })
                        if not sealed[0]:
                            if volume_bytes[0] >= max_volume_hard_bytes:
                                sealed[0] = (f'tar reached {volume_bytes[0] / 1024 ** 3:.2f} GB, over the '
                                             f'{max_volume_hard_bytes / 1024 ** 3:.2f} GB ceiling')
                            elif get_free_disk_bytes(stage_dir) < min_free_disk:
                                sealed[0] = f'free disk fell below {min_free_disk / 1024 ** 3:.2f} GB'
                            elif max_time_limit is not None and time.time() > start_time + max_time_limit:
                                sealed[0] = 'run deadline reached mid-volume'
                            if sealed[0]:
                                logging.warning(f'Sealing volume #{max_volume_id} early - {sealed[0]}; remaining '
                                                f'posts move to the next volume.')

                def _fn_download(item):
                    with lock:
                        if sealed[0]:
                            deferred.append(item)
                            stats['deferred'] += 1
                            return
                    filename = f'{item["id"]}{os.path.splitext(item["file_url"])[1]}'
                    dst_file = os.path.join(stage_dir, filename)
                    try:
                        for attempt in range(1, retries + 1):
                            failure = None
                            rate_limiter.acquire()
                            with pool.lease() as holder:
                                try:
                                    size, width, height = _fetch(item, holder[0][1], dst_file)
                                except Exception as err:
                                    failure = err
                                    if classify_error(err) in ('blocked', 'network'):
                                        pool.replace(holder)
                                else:
                                    rate_limiter.report_success()
                            if failure is None:
                                _store(item, filename, dst_file, size, width, height)
                                return
                            kind = classify_error(failure)
                            if kind in ('rate_limit', 'missing'):
                                # On this host a 404 under load is the meter speaking.
                                rate_limiter.report_throttled()
                            if kind == 'unusable':
                                with lock:
                                    volume_bad.append(item['id'])
                                    stats['unusable'] += 1
                                logging.warning(f'UNUSABLE post {item["id"]}: bytes match the index but will not '
                                                f'decode - {failure.reason!r}')
                                return
                            if attempt >= retries:
                                with lock:
                                    if kind == 'missing':
                                        suspects.append(item)
                                        stats['suspect'] += 1
                                    else:
                                        stats['failed'] += 1
                                logging.warning(f'{"SUSPECT" if kind == "missing" else "FAILED"} post {item["id"]} '
                                                f'after {attempt} attempts ({kind}, HTTP {_status_of(failure)}): '
                                                f'{failure!r}')
                                return
                            with lock:
                                stats['retry'] += 1
                            weight = 0.25 if kind == 'blocked' else 1.0
                            time.sleep(retry_wait * weight * attempt * (0.5 + random.random()))
                    finally:
                        if os.path.exists(dst_file):
                            os.remove(dst_file)

                parallel_call(plan, _fn_download, max_workers=download_workers, desc=f'Volume #{max_volume_id}',
                              postfix=lambda: dict(stats))

                # The second look at every post that answered 404 on every attempt: one thread, slowly, with
                # the control download first. Only a post that is still missing while the host demonstrably
                # serves is recorded as bad; if the control fails the host is not serving, and the suspects
                # wait for another run instead of being blacklisted for the host's mood.
                if suspects:
                    logging.info(f'Diagnosing {plural_word(len(suspects), "suspect")} one at a time ...')
                for index, item in enumerate(list(suspects)):
                    healthy = False
                    if control[0] is not None:
                        probe = os.path.join(stage_dir, f'control-{control[0]["id"]}')
                        try:
                            _fetch(control[0], cffi_requests.Session(impersonate=_FINGERPRINTS[0]), probe)
                            healthy = True
                        except Exception as err:
                            logging.warning(f'Control download failed ({err!r}); the host is not serving now.')
                        finally:
                            if os.path.exists(probe):
                                os.remove(probe)
                    if not healthy:
                        parked.extend(suspects[index:])
                        stats['suspect'] -= len(suspects[index:])
                        logging.warning(f'{len(suspects[index:])} suspect post(s) left for the next run, none '
                                        f'recorded as bad.')
                        time.sleep(cooldown)
                        break
                    resolved = None
                    filename = f'{item["id"]}{os.path.splitext(item["file_url"])[1]}'
                    dst_file = os.path.join(stage_dir, filename)
                    for attempt in range(1, diagnose_attempts + 1):
                        time.sleep(diagnose_wait)
                        session = cffi_requests.Session(impersonate=_FINGERPRINTS[attempt % len(_FINGERPRINTS)])
                        try:
                            size, width, height = _fetch(item, session, dst_file)
                        except Exception as err:
                            if classify_error(err) != 'missing':
                                resolved = 'undecided'
                                break
                        else:
                            _store(item, filename, dst_file, size, width, height)
                            stats['suspect'] -= 1
                            resolved = 'stored'
                            break
                        finally:
                            if os.path.exists(dst_file):
                                os.remove(dst_file)
                    if resolved is None:
                        volume_bad.append(item['id'])
                        stats['suspect'] -= 1
                        stats['gone'] += 1
                        logging.warning(f'GONE post {item["id"]}: 404 on every attempt, also on {diagnose_attempts} '
                                        f'slow looks while the control downloaded.')
                    elif resolved == 'undecided':
                        parked.append(item)
                        stats['suspect'] -= 1

            shutil.rmtree(stage_dir, ignore_errors=True)
            pending = sorted(deferred, key=lambda x: x['id']) + rest

            attempted = len(plan) - stats['deferred']
            logging.info(f'Rate limiter settled at {rate_limiter.rate:.1f} req/s after {rate_limiter.throttles} '
                         f'throttling responses; {pool.replaced} session(s) replaced.')
            logging.info(f'Volume #{max_volume_id} downloaded: {stats["ok"]} ok, {stats["gone"]} gone, '
                         f'{stats["unusable"]} unusable, {stats["failed"]} failed, {len(parked)} parked, '
                         f'{stats["deferred"]} deferred, {stats["retry"]} retries, '
                         f'{volume_bytes[0] / 1024 ** 3:.2f} GB.')
            estimated = sum(estimate_bytes(x) for x in new_records)
            if new_records and estimated > 0:
                size_scale = min(max(0.5 * size_scale + 0.5 * volume_bytes[0] / estimated, 0.3), 3.0)
            if stats['failed'] and attempted >= min_attempts_for_abort \
                    and stats['failed'] >= attempted * max_failed_ratio:
                logging.error(f'ABORT: {stats["failed"]} of {attempted} posts failed in volume #{max_volume_id}, '
                              f'at or above the {max_failed_ratio:.0%} threshold - not publishing this volume.')
                aborted = True
                break

            if not new_records:
                logging.warning(f'Volume #{max_volume_id} produced no file, discarded.')
                max_volume_id -= 1
                if volume_bad:
                    bad_image_ids.update(volume_bad)
                    covered_ids.update(volume_bad)
                pending = [x for x in pending if x['id'] not in set(volume_bad)]
                continue

            tar_create_index_for_directory(os.path.join(upload_dir, 'images'), silent=True)
            if not os.path.exists(os.path.join(upload_dir, rel_index)):
                raise RuntimeError(f'Sidecar {rel_index!r} was not produced for volume #{max_volume_id}, refusing to '
                                   f'publish an unindexed tar.')

            records.extend(new_records)
            bad_image_ids.update(volume_bad)
            covered_ids.update(x['id'] for x in new_records)
            covered_ids.update(volume_bad)

            df_table = pd.DataFrame(records)[_TABLE_COLUMNS].sort_values(by=['id'], ascending=[True])
            df_table.to_parquet(os.path.join(upload_dir, 'table.parquet'), index=False)
            with open(os.path.join(upload_dir, 'meta.json'), 'w') as f:
                json.dump({'max_volume_id': max_volume_id, 'bad_image_ids': sorted(bad_image_ids)}, f)
            if baseline_is_new:
                with open(os.path.join(upload_dir, glob_exist_ids_file), 'w') as f:
                    json.dump(baseline, f)
            write_readme(os.path.join(upload_dir, 'README.md'), df_table=df_table, bad_image_ids=bad_image_ids,
                         max_volume_id=max_volume_id, src_repository=src_repository, repository=repository,
                         baseline_repository=baseline_repository, baseline_size=len(baseline),
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
                elapsed = time.time() - upload_started
                logging.info(f'UPLOAD volume #{max_volume_id} done in {elapsed:.0f}s '
                             f'({actual_bytes / 1024 ** 2 / max(elapsed, 1e-6):.1f} MB/s).')
                baseline_is_new = False
            volumes_done += 1

        delete_detached_cache()
        log_disk_usage(os.getcwd(), prefix='Disk after volume')

    pool.close()
    logging.info(f'Done, {plural_word(volumes_done, "volume")} published in this run, '
                 f'{plural_word(len(records), "image")} stored in total, '
                 f'{plural_word(len(parked), "post")} parked for the next run'
                 f'{", run aborted by the failure threshold" if aborted else ""}.')


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Download the Gelbooru originals that the index lists and the baseline repository lacks into hfutils-indexed '
         'tar volumes of a private staging repository, one consistent commit per volume.',
)
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_GB_N_DL', default='deepghs/gelbooru_newest_dl',
              show_default=True, help='Staging dataset repository.')
@click.option('-i', '--src-repository', type=str, envvar='REMOTE_REPOSITORY_GB', default='deepghs/gelbooru_index',
              show_default=True, help='Index dataset repository to read posts from.')
@click.option('-b', '--baseline-repository', type=str, envvar='REMOTE_REPOSITORY_GB_FULL',
              default='deepghs/gelbooru_full', show_default=True,
              help='Repository whose ids (and bad ids) are not downloaded again.')
@click.option('--src-revision', type=str, default='main', show_default=True, help='Revision of the index to read.')
@click.option('-t', '--max-time-limit', type=duration_type(), default='5h', show_default=True,
              help='Stop scheduling work after this long.')
@click.option('-f', '--max-volume-files', type=int, default=10000, show_default=True, help='Entries per volume.')
@click.option('-B', '--max-volume-bytes', type=int, default=10 * 1024 ** 3, show_default=True,
              help='Approximate byte budget of a volume.')
@click.option('-H', '--max-volume-hard-bytes', type=int, default=12 * 1024 ** 3, show_default=True,
              help='Ceiling on the bytes written into one tar; crossing it seals the volume.')
@click.option('-w', '--download-workers', type=int, default=8, show_default=True,
              help='Download threads. Above eight the host starts answering 404 for files that exist.')
@click.option('-d', '--min-free-disk', type=int, default=24 * 1024 ** 3, show_default=True,
              help='Stop before a new volume when free disk falls below this many bytes.')
@click.option('-u', '--upload-time-span', type=duration_type(), default=30, show_default=True,
              help='Minimum interval between upload commits.')
@click.option('-V', '--max-volumes', type=int, default=None, help='Stop after this many volumes (smoke runs).')
@click.option('-I', '--initial-rate', type=float, default=4.0, show_default=True,
              help='Requests per second the adaptive limiter starts from.')
@click.option('-M', '--max-rate', type=float, default=12.0, show_default=True,
              help='Ceiling for the adaptive request rate.')
@click.option('-c', '--retries', type=int, default=6, show_default=True, help='Attempts per post.')
@click.option('-C', '--retry-wait', type=duration_type(), default=2.0, show_default=True,
              help='Base backoff between attempts.')
@click.option('--max-failed-ratio', type=float, default=0.3, show_default=True,
              help='A volume failing at this fraction is not published.')
@click.option('--diagnose-attempts', type=int, default=3, show_default=True,
              help='Slow looks at a post that answered 404 on every attempt, before it is recorded bad.')
@click.option('--diagnose-wait', type=duration_type(), default=3.0, show_default=True,
              help='Seconds between two slow looks.')
@click.option('--dry-run', is_flag=True, default=False, help='Download and build volumes, upload nothing.')
def cli(repository: str, src_repository: str, baseline_repository: str, src_revision: str,
        max_time_limit: Optional[float], max_volume_files: int, max_volume_bytes: int, max_volume_hard_bytes: int,
        download_workers: int, min_free_disk: int, upload_time_span: float, max_volumes: Optional[int],
        initial_rate: float, max_rate: float, retries: int, retry_wait: float, max_failed_ratio: float,
        diagnose_attempts: int, diagnose_wait: float, dry_run: bool):
    logging.try_init_root(logging.INFO)
    sync(repository=repository, src_repository=src_repository, baseline_repository=baseline_repository,
         src_revision=src_revision, max_time_limit=max_time_limit, max_volume_files=max_volume_files,
         max_volume_bytes=max_volume_bytes, max_volume_hard_bytes=max_volume_hard_bytes,
         download_workers=download_workers, min_free_disk=min_free_disk, upload_time_span=upload_time_span,
         max_volumes=max_volumes, initial_rate=initial_rate, max_rate=max_rate, retries=retries,
         retry_wait=retry_wait, max_failed_ratio=max_failed_ratio, diagnose_attempts=diagnose_attempts,
         diagnose_wait=diagnose_wait, dry_run=dry_run)


if __name__ == '__main__':
    cli()
