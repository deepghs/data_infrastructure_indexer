"""Staging download stage for Danbooru.

Pull original image bytes for posts listed in ``deepghs/danbooru_newest_index`` and pack them
into hfutils-indexed tar volumes inside a private staging repository (default
``deepghs/danbooru_newest_dl``). A later repack stage consumes those volumes instead of
hitting ``cdn.donmai.us`` again.

Layout published to the staging repository
==========================================

::

    images/
      0/000.tar      hfutils-indexed tar volume
      0/000.json     sibling sidecar -- per-entry {offset,size,sha256}
      0/001.tar
      ...
    table.parquet    one row per image actually written into a volume
    meta.json        {max_volume_id, bad_image_ids}
    glob_exist_ids.json
                     read-only baseline of ids already covered elsewhere
                     (seeded from deepghs/danbooru_newest-all); never written here
    README.md        statistics and preview

Why a staging repo at all
=========================

The destination dataset ``deepghs/danbooru_newest-all`` uses 1000 fixed ``id % 1000`` buckets
whose tars already average ~8 GB. Appending to it requires pulling a whole bucket back before
every write, which no free CI runner can sustain. Volumes here are append-only and never
rewritten, so a run only ever holds one volume on local disk.

Disk discipline
===============

This job is expected to run on a GitHub-hosted free runner, where free disk is the binding
constraint rather than CPU or memory:

* Volume boundaries are planned up-front from ``file_size`` in the index, so a volume never
  overshoots ``--max-volume-bytes``.
* Each worker deletes its downloaded file the moment it lands in the tar, keeping in-flight
  bytes at roughly ``download_workers x average file size``.
* The tar, its sidecar and the staging directory are removed immediately after the commit.
* Before starting another volume the job re-checks free space and stops cleanly below
  ``--min-free-disk`` rather than dying mid-upload.
* The 2.8 GB upstream ``records.parquet`` is streamed column-wise over HTTP range requests
  and never downloaded whole.

``table.parquet`` deliberately keeps only storage-locating and lightweight descriptive
columns. Tag strings and the rest of the upstream schema stay in the index repository; the
repack stage rejoins them by id.
"""
import datetime
import json
import logging as _logging
import math
import os
import shutil
import tarfile
import time
from hashlib import md5
from threading import Lock
from typing import List, Optional

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory, urlsplit
from hfutils.cache import delete_detached_cache
from hfutils.index import tar_create_index_for_directory
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.utils import hf_normpath, number_to_tag
from pyrate_limiter import Rate, Limiter, Duration
from tqdm import tqdm

from inf.utils.download import download_file, parallel_call, get_free_disk_bytes, log_disk_usage
from inf.utils.duration import duration_type
from inf.utils.safe import safe_hf_hub_download, safe_upload_directory_as_directory
from .base import get_danbooru_session, __site_url__  # noqa: F401 - re-exported for callers

# Danbooru serves genuinely huge originals; the default bomb guard would reject valid posts.
Image.MAX_IMAGE_PIXELS = 32768 ** 2

# One INFO line per download would bury the progress bar and the volume summaries.
_logging.getLogger('httpx').setLevel(_logging.WARNING)
_logging.getLogger('httpcore').setLevel(_logging.WARNING)

#: Columns streamed out of the upstream index. Kept minimal because every extra column
#: multiplies the bytes pulled over range requests on each run.
_SCAN_COLUMNS = ['id', 'file_url', 'mimetype', 'file_ext', 'file_size', 'image_width', 'image_height', 'rating', 'md5']

#: Columns persisted in the staging ``table.parquet``.
_TABLE_COLUMNS = ['id', 'filename', 'volume_file', 'file_size', 'mimetype', 'file_ext',
                  'width', 'height', 'rating', 'md5', 'file_url']


def _verify_md5(local_file: str, expected_md5: Optional[str], chunk_size: int = 1 << 20):
    """
    Check a downloaded file against the md5 recorded in the index.

    The index publishes an md5 for every post, so this turns "the byte count looked right" into
    a real integrity guarantee at the cost of one streaming pass. Reading in chunks keeps the
    multi-hundred-megabyte originals off the heap.

    :param local_file: Path of the file to check.
    :type local_file: str
    :param expected_md5: Expected digest. Verification is skipped when absent.
    :type expected_md5: Optional[str]
    :param chunk_size: Read size in bytes.
    :type chunk_size: int
    :raises ValueError: When the digest does not match.
    """
    if not expected_md5:
        return
    hash_obj = md5()
    with open(local_file, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            hash_obj.update(data)
    actual = hash_obj.hexdigest()
    if actual != expected_md5:
        raise ValueError(f'MD5 mismatch for {os.path.basename(local_file)!r} - '
                         f'{expected_md5!r} expected, {actual!r} found.')


class _SessionHolder:
    """
    Hold the shared Danbooru client and allow workers to swap it out after a challenge.

    Cloudflare rejects a fingerprint for the life of the connection, so recovering means
    building a whole new client rather than retrying on the old one. Workers pass the
    generation they were using; whoever reports first does the rebuild and the rest pick up
    the replacement, so a burst of concurrent 403s costs one warm-up instead of one per thread.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._lock = Lock()
        self._generation = 0
        self._session = get_danbooru_session(**kwargs)

    @property
    def session(self):
        return self._session

    @property
    def generation(self) -> int:
        return self._generation

    def refresh(self, seen_generation: int):
        """
        Replace the client unless another worker already replaced this generation.

        :param seen_generation: Generation the caller was using when it hit the challenge.
        :type seen_generation: int
        """
        with self._lock:
            if seen_generation != self._generation:
                return self._session
            logging.info(f'Cloudflare challenge hit, rebuilding session '
                         f'(generation {self._generation} -> {self._generation + 1}) ...')
            old = self._session
            self._session = get_danbooru_session(**self._kwargs)
            self._generation += 1
            try:
                old.close()
            except Exception:  # pragma: no cover - closing a dead client must not abort the run
                pass
            return self._session

    def close(self):
        try:
            self._session.close()
        except Exception:  # pragma: no cover
            pass


def _volume_paths(volume_id: int):
    """
    Map a volume id onto its in-repo tar and sidecar paths.

    Mirrors the Gelbooru staging layout: a two-level ``images/<thousands>/<nnn>.tar`` tree so a
    single directory never accumulates more than 1000 entries.

    :param volume_id: Monotonic volume counter, starting at 1.
    :type volume_id: int
    :returns: Tuple of (tar path, sidecar path), both relative to the repository root.
    :rtype: Tuple[str, str]
    """
    rel_tar = f'images/{volume_id // 1000}/{volume_id % 1000:03d}.tar'
    return rel_tar, f'{os.path.splitext(rel_tar)[0]}.json'


def _load_state(hf_client, repository: str, glob_exist_ids_file: str):
    """
    Read the staging repository's current state.

    :param hf_client: Hugging Face API client.
    :param repository: Staging dataset repository id.
    :type repository: str
    :param glob_exist_ids_file: Name of the read-only baseline id list in the repository.
    :type glob_exist_ids_file: str
    :returns: Tuple of (records, covered id set, bad id set, max volume id).
    """
    records: List[dict] = []
    covered_ids = set()
    bad_image_ids = set()
    max_volume_id = 0

    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename=glob_exist_ids_file):
        path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset',
                                    filename=glob_exist_ids_file)
        with open(path, 'r') as f:
            baseline = json.load(f)
        covered_ids.update(baseline)
        logging.info(f'Baseline {glob_exist_ids_file!r} loaded, {plural_word(len(baseline), "id")}.')
    else:
        logging.warning(f'No {glob_exist_ids_file!r} in {repository!r}, starting without a baseline.')

    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='table.parquet'):
        path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset',
                                    filename='table.parquet')
        df = pd.read_parquet(path).replace(np.NaN, None)
        records = df.to_dict('records')
        covered_ids.update(int(x) for x in df['id'])
        logging.info(f'Existing table loaded, {plural_word(len(records), "row")}.')

    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='meta.json'):
        path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset', filename='meta.json')
        with open(path, 'r') as f:
            meta = json.load(f)
        bad_image_ids.update(meta.get('bad_image_ids') or [])
        max_volume_id = meta.get('max_volume_id') or 0
        covered_ids.update(bad_image_ids)
        logging.info(f'Meta loaded, max_volume_id={max_volume_id}, '
                     f'{plural_word(len(bad_image_ids), "bad id")}.')

    return records, covered_ids, bad_image_ids, max_volume_id


def _scan_candidates(src_repository: str, src_revision: str, covered_ids: set,
                     include_non_image: bool) -> List[dict]:
    """
    Stream the upstream index and return the posts still missing from the staging repository.

    The index parquet is multiple gigabytes, so it is read through ``HfFileSystem`` range
    requests with an explicit column projection instead of being downloaded.

    :param src_repository: Upstream index dataset repository id.
    :type src_repository: str
    :param src_revision: Revision of the upstream index to read.
    :type src_revision: str
    :param covered_ids: Ids that must not be downloaded again.
    :type covered_ids: set
    :param include_non_image: Keep video/zip/flash posts as well when True.
    :type include_non_image: bool
    :returns: Candidate rows sorted by ascending id.
    :rtype: List[dict]
    """
    hf_fs = get_hf_fs()
    path = f'datasets/{src_repository}/records.parquet'
    if src_revision and src_revision != 'main':
        path = f'datasets/{src_repository}@{src_revision}/records.parquet'

    candidates: List[dict] = []
    with hf_fs.open(path, 'rb') as f:
        pf = pq.ParquetFile(f)
        total = pf.metadata.num_rows
        logging.info(f'Scanning {plural_word(total, "row")} of {src_repository!r} for candidates ...')
        with tqdm(total=total, desc='Scanning index') as pg:
            for batch in pf.iter_batches(batch_size=200000, columns=_SCAN_COLUMNS):
                rows = batch.to_pylist()
                pg.update(len(rows))
                for row in rows:
                    if row['id'] in covered_ids:
                        continue
                    mimetype = row['mimetype'] or ''
                    if not include_non_image and not mimetype.startswith('image/'):
                        continue
                    if not row['file_url']:
                        continue
                    candidates.append(row)

    candidates.sort(key=lambda x: x['id'])
    return candidates


def _plan_volumes(candidates: List[dict], max_volume_files: int, max_volume_bytes: int) -> List[List[dict]]:
    """
    Split candidates into volume-sized batches using the sizes announced by the index.

    Planning up-front keeps a volume from overshooting the disk budget, which a
    decide-as-you-download loop cannot guarantee.

    :param candidates: Candidate rows in ascending id order.
    :type candidates: List[dict]
    :param max_volume_files: Hard cap on entries per volume.
    :type max_volume_files: int
    :param max_volume_bytes: Soft cap on uncompressed bytes per volume; a single oversized post
        still forms a volume of its own rather than being skipped.
    :type max_volume_bytes: int
    :returns: List of per-volume candidate batches.
    :rtype: List[List[dict]]
    """
    plans: List[List[dict]] = []
    current: List[dict] = []
    current_bytes = 0
    for row in candidates:
        size = row['file_size'] or 0
        if current and (len(current) >= max_volume_files or current_bytes + size > max_volume_bytes):
            plans.append(current)
            current, current_bytes = [], 0
        current.append(row)
        current_bytes += size
    if current:
        plans.append(current)
    return plans


def _write_readme(md_file: str, df_table: pd.DataFrame, bad_image_ids: set, max_volume_id: int,
                  src_repository: str):
    """
    Render the staging repository README.

    :param md_file: Destination path.
    :type md_file: str
    :param df_table: Full staging table.
    :type df_table: pd.DataFrame
    :param bad_image_ids: Ids permanently skipped.
    :type bad_image_ids: set
    :param max_volume_id: Highest volume id written so far.
    :type max_volume_id: int
    :param src_repository: Upstream index dataset repository id.
    :type src_repository: str
    """
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
        print('- ja', file=f)
        print('tags:', file=f)
        print('- art', file=f)
        print('- anime', file=f)
        print('- not-for-all-audiences', file=f)
        print('size_categories:', file=f)
        print(f'- {number_to_tag(len(df_table))}', file=f)
        print('annotations_creators:', file=f)
        print('- no-annotation', file=f)
        print('source_datasets:', file=f)
        print('- danbooru', file=f)
        print('---', file=f)
        print('', file=f)

        print('# Danbooru Staging Download Dataset', file=f)
        print('', file=f)
        print(f'Raw original files pulled for posts listed in [{src_repository}]'
              f'(https://huggingface.co/datasets/{src_repository}). This repository is a staging '
              f'area for the repack stage, not a finished dataset.', file=f)
        print('', file=f)
        print('Images live in append-only hfutils-indexed tar volumes under `images/`, each with a '
              'sibling `.json` sidecar holding per-entry offset, size and sha256. `table.parquet` '
              'maps every stored post id to its `volume_file` and `filename`.', file=f)
        print('', file=f)

        print('# Information', file=f)
        print('', file=f)
        print(f'There are {plural_word(len(df_table), "image")} in total across '
              f'{plural_word(max_volume_id, "volume")}, {total_bytes / 1024 ** 4:.3f} TB of original '
              f'bytes. {plural_word(len(bad_image_ids), "post")} are recorded as permanently '
              f'unavailable. Last updated at `{current_time}`.', file=f)
        print('', file=f)

        if len(df_table):
            df_shown = df_table.sort_values(by=['id'], ascending=[False])[:30]
            df_shown = df_shown[['id', 'volume_file', 'filename', 'width', 'height',
                                 'mimetype', 'file_size']]
            print(f'These are the {plural_word(len(df_shown), "most recent image")}:', file=f)
            print('', file=f)
            print(df_shown.to_markdown(index=False), file=f)
            print('', file=f)


def sync(repository: str, src_repository: str, src_revision: str = 'main',
         max_time_limit: Optional[float] = (60 * 5) * 60, max_volume_files: int = 1000,
         max_volume_bytes: int = 2 * 1024 ** 3, download_workers: int = 16,
         min_free_disk: int = 8 * 1024 ** 3, upload_time_span: float = 30,
         include_non_image: bool = False, glob_exist_ids_file: str = 'glob_exist_ids.json',
         max_volumes: Optional[int] = None, cf_retries: int = 4, max_blocked_ratio: float = 0.3,
         username: Optional[str] = None, apitoken: Optional[str] = None,
         proxy_pool: Optional[str] = None):
    """
    Download missing Danbooru originals into append-only tar volumes in the staging repository.

    :param repository: Staging dataset repository to write to.
    :type repository: str
    :param src_repository: Upstream index dataset repository to read candidates from.
    :type src_repository: str
    :param src_revision: Revision of the upstream index to read.
    :type src_revision: str
    :param max_time_limit: Stop after this many seconds. None disables the limit.
    :type max_time_limit: Optional[float]
    :param max_volume_files: Maximum entries per tar volume.
    :type max_volume_files: int
    :param max_volume_bytes: Approximate byte budget per tar volume.
    :type max_volume_bytes: int
    :param download_workers: Concurrent download threads.
    :type download_workers: int
    :param min_free_disk: Stop before starting a volume when free disk drops below this.
    :type min_free_disk: int
    :param upload_time_span: Minimum seconds between commits.
    :type upload_time_span: float
    :param include_non_image: Also fetch video/zip/flash posts.
    :type include_non_image: bool
    :param glob_exist_ids_file: Name of the read-only baseline id list.
    :type glob_exist_ids_file: str
    :param max_volumes: Stop after this many volumes. Useful for smoke runs.
    :type max_volumes: Optional[int]
    :param cf_retries: Attempts per post before giving up, rebuilding the session between
        Cloudflare rejections.
    :type cf_retries: int
    :param max_blocked_ratio: Abort the run when this fraction of a volume fails for reasons
        other than the post being gone upstream.
    :type max_blocked_ratio: float
    :param username: Danbooru account name, used on the session warm-up call.
    :type username: Optional[str]
    :param apitoken: Danbooru API key, used on the session warm-up call.
    :type apitoken: Optional[str]
    :param proxy_pool: Proxy URL applied to every upstream request.
    :type proxy_pool: Optional[str]
    """
    start_time = time.time()
    delete_detached_cache()
    hf_client = get_hf_client()

    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        logging.info(f'Staging repository {repository!r} created.')

    records, covered_ids, bad_image_ids, max_volume_id = _load_state(
        hf_client=hf_client, repository=repository, glob_exist_ids_file=glob_exist_ids_file)
    logging.info(f'{plural_word(len(covered_ids), "id")} already covered, '
                 f'current max volume id: {max_volume_id}.')

    candidates = _scan_candidates(src_repository=src_repository, src_revision=src_revision,
                                  covered_ids=covered_ids, include_non_image=include_non_image)
    if not candidates:
        logging.info('No candidate to download, quit.')
        return
    total_bytes = sum(x['file_size'] or 0 for x in candidates)
    logging.info(f'{plural_word(len(candidates), "candidate")} to download, '
                 f'{total_bytes / 1024 ** 4:.3f} TB, id range '
                 f'{candidates[0]["id"]} - {candidates[-1]["id"]}.')

    plans = _plan_volumes(candidates, max_volume_files=max_volume_files,
                          max_volume_bytes=max_volume_bytes)
    logging.info(f'{plural_word(len(plans), "volume")} planned for the full backlog.')

    # Cloudflare rejects plain HTTP/1.1 clients on every donmai.us host, so the CDN is only
    # reachable through a warmed-up HTTP/2 session. See inf/danbooru/base.py.
    holder = _SessionHolder(proxy_pool=proxy_pool, username=username, apitoken=apitoken)

    volumes_done = 0
    for plan in plans:
        if max_time_limit is not None and start_time + max_time_limit < time.time():
            logging.info('Max time limit exceeded, stop scheduling new volumes.')
            break
        if max_volumes is not None and volumes_done >= max_volumes:
            logging.info(f'Reached --max-volumes {max_volumes}, stop.')
            break

        log_disk_usage(os.getcwd(), prefix='Disk before volume')
        free = get_free_disk_bytes(os.getcwd())
        if free < min_free_disk:
            logging.warning(f'Only {free / 1024 ** 3:.2f} GB free, below the '
                            f'{min_free_disk / 1024 ** 3:.2f} GB floor - stop cleanly.')
            break

        max_volume_id += 1
        rel_tar, rel_index = _volume_paths(max_volume_id)
        planned_bytes = sum(x['file_size'] or 0 for x in plan)
        logging.info(f'Building volume #{max_volume_id} ({rel_tar}), {plural_word(len(plan), "post")}, '
                     f'{planned_bytes / 1024 ** 3:.2f} GB planned ...')

        with TemporaryDirectory() as td:
            upload_dir = os.path.join(td, 'upload')
            stage_dir = os.path.join(td, 'stage')
            os.makedirs(stage_dir, exist_ok=True)
            tar_file = os.path.join(upload_dir, rel_tar)
            os.makedirs(os.path.dirname(tar_file), exist_ok=True)

            new_records: List[dict] = []
            volume_bad: List[int] = []
            blocked = [0]
            lock = Lock()

            with tarfile.open(tar_file, 'w:') as tar:
                def _fn_download(item):
                    _, ext = os.path.splitext(urlsplit(item['file_url']).filename)
                    filename = f'{item["id"]}{ext}'
                    dst_file = os.path.join(stage_dir, filename)
                    try:
                        size = width = height = None
                        for attempt in range(1, cf_retries + 1):
                            generation = holder.generation
                            try:
                                size = download_file(item['file_url'], dst_file,
                                                     session=holder.session,
                                                     expected_size=item['file_size'])
                                _verify_md5(dst_file, item['md5'])
                                # A 200 carrying an HTML error body would clear the size and md5
                                # checks only by miracle, but the header parse is nearly free.
                                with Image.open(dst_file) as image:
                                    width, height = image.size
                                break
                            except Exception as err:
                                status = getattr(getattr(err, 'response', None), 'status_code', None)
                                if status in (404, 410):
                                    # Genuinely gone upstream: record it so later runs skip it.
                                    with lock:
                                        volume_bad.append(item['id'])
                                    logging.warning(f'Post {item["id"]} is gone (HTTP {status}).')
                                    raise
                                if status == 403 and attempt < cf_retries:
                                    # Cloudflare re-challenged us. This is never a property of the
                                    # post, so it must not blacklist the id; swap in a session with
                                    # a fresh fingerprint and try again.
                                    with lock:
                                        blocked[0] += 1
                                    holder.refresh(generation)
                                    continue
                                logging.warning(f'Post {item["id"]} skipped - {err!r}')
                                raise

                        with lock:
                            tar.add(dst_file, filename)
                            new_records.append({
                                'id': item['id'],
                                'filename': filename,
                                'volume_file': hf_normpath(rel_tar),
                                'file_size': size,
                                'mimetype': item['mimetype'],
                                'file_ext': item['file_ext'],
                                'width': width or item['image_width'],
                                'height': height or item['image_height'],
                                'rating': item['rating'],
                                'md5': item['md5'],
                                'file_url': item['file_url'],
                            })
                    finally:
                        # Free the bytes immediately; in-flight footprint stays at roughly
                        # download_workers x average file size.
                        if os.path.exists(dst_file):
                            os.remove(dst_file)

                parallel_call(plan, _fn_download, max_workers=download_workers,
                              desc=f'Volume #{max_volume_id}')

            shutil.rmtree(stage_dir, ignore_errors=True)

            failed = len(plan) - len(new_records) - len(volume_bad)
            if failed and failed >= len(plan) * max_blocked_ratio:
                # Sustained blocking wastes a volume id and a commit per attempt, so stop the run
                # and let the next scheduled one start from a clean rate-limit window.
                logging.error(f'{plural_word(failed, "post")} of {len(plan)} failed in volume '
                              f'#{max_volume_id} ({blocked[0]} Cloudflare rejections) - '
                              f'aborting the run without publishing this volume.')
                break

            if not new_records:
                logging.warning(f'Volume #{max_volume_id} produced no file, discarded.')
                max_volume_id -= 1
                if volume_bad:
                    bad_image_ids.update(volume_bad)
                    covered_ids.update(volume_bad)
                continue

            tar_create_index_for_directory(os.path.join(upload_dir, 'images'), silent=True)
            if not os.path.exists(os.path.join(upload_dir, rel_index)):
                raise RuntimeError(f'Sidecar {rel_index!r} was not produced for volume '
                                   f'#{max_volume_id}, refusing to publish an unindexed tar.')

            records.extend(new_records)
            bad_image_ids.update(volume_bad)
            covered_ids.update(x['id'] for x in new_records)
            covered_ids.update(volume_bad)

            df_table = pd.DataFrame(records)[_TABLE_COLUMNS].sort_values(by=['id'], ascending=[True])
            df_table.to_parquet(os.path.join(upload_dir, 'table.parquet'), index=False)
            with open(os.path.join(upload_dir, 'meta.json'), 'w') as f:
                json.dump({
                    'max_volume_id': max_volume_id,
                    'bad_image_ids': sorted(bad_image_ids),
                }, f)
            _write_readme(os.path.join(upload_dir, 'README.md'), df_table=df_table,
                          bad_image_ids=bad_image_ids, max_volume_id=max_volume_id,
                          src_repository=src_repository)

            actual_bytes = os.path.getsize(tar_file)
            limiter.try_acquire('hf_upload')
            logging.info(f'Uploading volume #{max_volume_id} - {plural_word(len(new_records), "image")}, '
                         f'{actual_bytes / 1024 ** 3:.2f} GB, {plural_word(len(volume_bad), "bad id")} ...')
            safe_upload_directory_as_directory(
                local_directory=upload_dir,
                repo_id=repository,
                repo_type='dataset',
                path_in_repo='.',
                message=f'Sync volume #{max_volume_id}, with {plural_word(len(new_records), "image")}',
            )
            volumes_done += 1

        # TemporaryDirectory is gone here; drop the HF cache too so repeated volumes do not
        # accumulate blobs on a runner with a small disk.
        delete_detached_cache()
        log_disk_usage(os.getcwd(), prefix='Disk after volume')

    holder.close()
    logging.info(f'Done, {plural_word(volumes_done, "volume")} published in this run, '
                 f'{plural_word(len(records), "image")} stored in total.')


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Download Danbooru originals listed in the index repository into append-only, '
         'hfutils-indexed tar volumes inside a private staging repository. Designed to run '
         'repeatedly on a small-disk CI runner: each volume is planned, downloaded, indexed, '
         'uploaded and deleted before the next one starts.',
)
@click.option(
    '-r', '--repository',
    type=str,
    envvar='REMOTE_REPOSITORY_DB_N_DL',
    required=True,
    show_envvar=True,
    help='Staging Hugging Face dataset repository to write volumes into.',
)
@click.option(
    '-s', '--src-repository',
    type=str,
    envvar='REMOTE_REPOSITORY_DB_N',
    required=True,
    show_envvar=True,
    help='Upstream index dataset repository to read candidate posts from.',
)
@click.option(
    '-R', '--src-revision',
    type=str,
    default='main',
    show_default=True,
    help='Revision of the upstream index repository to read.',
)
@click.option(
    '-m', '--max-time-limit',
    type=duration_type(allow_none=True),
    default=5 * 60 * 60,
    show_default=True,
    help='Stop the run after this total runtime. Use none or unlimited to disable the limit.',
)
@click.option(
    '-f', '--max-volume-files',
    type=int,
    default=1000,
    show_default=True,
    help='Maximum number of entries packed into one tar volume.',
)
@click.option(
    '-b', '--max-volume-bytes',
    type=int,
    default=2 * 1024 ** 3,
    show_default=True,
    help='Approximate byte budget for one tar volume.',
)
@click.option(
    '-w', '--download-workers',
    type=int,
    default=16,
    show_default=True,
    help='Number of concurrent download threads.',
)
@click.option(
    '-d', '--min-free-disk',
    type=int,
    default=8 * 1024 ** 3,
    show_default=True,
    help='Stop before starting a new volume when free disk falls below this many bytes.',
)
@click.option(
    '-u', '--upload-time-span',
    type=duration_type(),
    default=30,
    show_default=True,
    help='Minimum interval between upload commits.',
)
@click.option(
    '-n', '--include-non-image/--no-include-non-image',
    default=False,
    show_default=True,
    help='Also download video, ugoira zip and flash posts instead of images only.',
)
@click.option(
    '-g', '--glob-exist-ids-file',
    type=str,
    default='glob_exist_ids.json',
    show_default=True,
    help='Name of the read-only baseline id list inside the staging repository.',
)
@click.option(
    '-V', '--max-volumes',
    type=int,
    default=None,
    help='Stop after publishing this many volumes. Intended for smoke runs.',
)
@click.option(
    '-c', '--cf-retries',
    type=int,
    default=4,
    show_default=True,
    help='Attempts per post, rebuilding the session between Cloudflare rejections.',
)
@click.option(
    '-B', '--max-blocked-ratio',
    type=float,
    default=0.3,
    show_default=True,
    help='Abort the run when this fraction of a volume fails for reasons other than the post '
         'being gone upstream.',
)
@click.option(
    '-U', '--username',
    type=str,
    envvar='DANBOORU_USERNAME',
    default=None,
    show_envvar=True,
    help='Danbooru account name used for authenticated upstream requests.',
)
@click.option(
    '-A', '--apitoken',
    type=str,
    envvar='DANBOORU_APITOKEN',
    default=None,
    show_envvar=True,
    help='Danbooru API key used for authenticated upstream requests.',
)
@click.option(
    '-p', '--proxy-pool',
    type=str,
    envvar='PP_DB',
    default=None,
    show_envvar=True,
    help='Proxy URL applied to every upstream request.',
)
def cli(repository: str, src_repository: str, src_revision: str, max_time_limit: Optional[float],
        max_volume_files: int, max_volume_bytes: int, download_workers: int, min_free_disk: int,
        upload_time_span: float, include_non_image: bool, glob_exist_ids_file: str,
        max_volumes: Optional[int], cf_retries: int, max_blocked_ratio: float,
        username: Optional[str], apitoken: Optional[str], proxy_pool: Optional[str]):
    logging.try_init_root(logging.INFO)
    return sync(
        repository=repository,
        src_repository=src_repository,
        src_revision=src_revision,
        max_time_limit=max_time_limit,
        max_volume_files=max_volume_files,
        max_volume_bytes=max_volume_bytes,
        download_workers=download_workers,
        min_free_disk=min_free_disk,
        upload_time_span=upload_time_span,
        include_non_image=include_non_image,
        glob_exist_ids_file=glob_exist_ids_file,
        max_volumes=max_volumes,
        cf_retries=cf_retries,
        max_blocked_ratio=max_blocked_ratio,
        username=username,
        apitoken=apitoken,
        proxy_pool=proxy_pool,
    )


if __name__ == '__main__':
    cli()
