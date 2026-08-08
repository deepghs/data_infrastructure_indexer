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
                     read-only baseline of ids already covered by the upstream collections
                     listed in ``_UPSTREAM_COLLECTIONS``; never written here
    README.md        statistics and preview

This repository therefore holds a *difference*, not a corpus: only the posts the index knows
about that no upstream collection already stores. Anyone wanting the complete set needs this
repository together with every entry in ``_UPSTREAM_COLLECTIONS``, and the generated README
says so explicitly so that a consumer who finds this repository on its own is not misled.

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

* Volume boundaries are planned up-front from ``file_size`` in the index, which keeps a volume
  near ``--max-volume-bytes``. Disk is not what bounds the size: a runner reports 145 GB total
  and peaked at 41 GB used through a run of 5 GB volumes, and the upload path streams from the
  tar rather than copying it, so peak usage is the tar itself.
* What does bound it is the job timeout. The run stops fetching at ``--max-time-limit`` and then
  has to upload whatever is in hand, so the last volume's upload must fit in the gap between
  that limit and the workflow's own timeout. Measured upload rates vary five-fold, 13 to
  161 MB/s, so the gap has to be costed at the low end: a 50 minute gap at 13 MB/s tops out
  near 39 GB. The 10 GB budget and 12 GB ceiling sit well inside that, trading some of the
  headroom for a smaller unit of loss when a run dies mid-volume.
* That plan is only an estimate, so a second, authoritative check runs against what actually
  lands on disk: once a tar passes ``--max-volume-hard-bytes``, or free space drops under
  ``--min-free-disk``, the volume is sealed on the spot, indexed, uploaded and deleted, and
  every post still queued for it moves to the next volume. A stale or wrong ``file_size`` in
  the index therefore cannot fill the runner.
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
import random
import shutil
import tarfile
import time
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
from hfutils.repository import hf_hub_repo_url
from hfutils.utils import hf_normpath, number_to_tag
from pyrate_limiter import Rate, Limiter, Duration
from tqdm import tqdm

from inf.utils.download import AdaptiveRateLimiter, download_file, parallel_call, \
    get_free_disk_bytes, log_disk_usage
from inf.utils.duration import duration_type
from inf.utils.brightdata import BrightDataError, ensure_proxy_access
from inf.utils.safe import configure_hf_http_backend, safe_hf_hub_download, \
    safe_upload_directory_as_directory
from .base import DanbooruSessionPool, __site_url__  # noqa: F401 - re-exported for callers

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

#: The public Hugging Face endpoint. Safe to write down; the self-hosted one is a secret and
#: must always come from ``HF_ENDPOINT`` instead of a literal.
_PUBLIC_ENDPOINT = 'https://huggingface.co'

#: Collections whose posts are excluded from this repository. ``glob_exist_ids.json`` is the
#: union of their ids, so a candidate is downloaded here only when none of these already hold
#: it. The entries are documentation rather than configuration: the baseline file is built once
#: and shipped read-only, and editing this list does not retroactively change what was skipped.
#:
#: Each entry carries enough detail for a reader - human or agent - to actually fetch a file
#: from that collection, because the three repositories do not share one access pattern. Two
#: live on a self-hosted endpoint and carry hfutils sidecars; the third is on the public hub
#: and carries none, so it needs a different retrieval path entirely.
_UPSTREAM_COLLECTIONS = [
    {
        'repo_id': 'deepghs/danbooru_newest-all',
        # None means "whatever HF_ENDPOINT points at". The self-hosted endpoint is a repository
        # secret, so it is never written as a literal here; URLs are built with
        # hf_hub_repo_url() at render time instead.
        'endpoint': None,
        'idx_repo_id': None,
        'tar_path': '`images/{id % 1000:04d}.tar`',
        'entry_name': '`{id}.{ext}`',
        'note': '1000 fixed buckets of ~8 GB, keyed on `id % 1000`. Sidecars live beside the '
                'tars as `images/{id % 1000:04d}.json`, so no separate index repository is '
                'needed.',
    },
    {
        'repo_id': 'nyanko7/danbooru2023',
        'endpoint': _PUBLIC_ENDPOINT,
        # The tars carry no sidecars of their own; deepghs publishes a mirror-shaped index
        # repository whose json paths match the tar paths one for one, which is exactly the
        # split-repository layout hfutils takes via idx_repo_id.
        'idx_repo_id': 'deepghs/danbooru2023_index',
        'tar_path': '`original/data-{id % 1000:04d}.tar` for the 2023 base, '
                    '`recent/data-1{id % 1000:03d}.tar` for later additions, and dated '
                    '`updates/<date>/dataset-*.tar` patches',
        'entry_name': '`./{id}.{ext}` (note the `./` prefix, unlike the other two)',
        'note': 'Posts up to id ~6,857,737 plus later patches. `exist_image_ids.json` in the '
                'index repository lists every id it holds.',
    },
]


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
    :returns: Tuple of (records, covered id set, bad id set, max volume id, baseline size).
    """
    records: List[dict] = []
    covered_ids = set()
    bad_image_ids = set()
    max_volume_id = 0
    baseline_size = 0

    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename=glob_exist_ids_file):
        path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset',
                                    filename=glob_exist_ids_file)
        with open(path, 'r') as f:
            baseline = json.load(f)
        covered_ids.update(baseline)
        baseline_size = len(baseline)
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

    return records, covered_ids, bad_image_ids, max_volume_id, baseline_size


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


#: Statuses meaning the post itself is gone. Only these may blacklist an id.
_PERMANENT_STATUS = (404, 410)


def _classify_error(err: Exception) -> str:
    """
    Decide how a failed download should be handled.

    The distinction that matters is between 403 and 429, which an earlier version lumped
    together. A 403 says this fingerprint is unwelcome, so the cure is a different session and
    an immediate retry. A 429 says the request rate is too high, so the cure is to slow the
    whole fleet down; swapping sessions there achieves nothing and retrying quickly makes it
    worse. Every rejection observed on CI has been a 429.

    :param err: Exception raised while fetching or validating a post.
    :type err: Exception
    :returns: ``'permanent'`` when the post is gone upstream, ``'blocked'`` when the fingerprint
        was refused, ``'rate_limit'`` when the site is metering us, ``'transient'`` otherwise.
    :rtype: str
    """
    status = getattr(getattr(err, 'response', None), 'status_code', None)
    if status in _PERMANENT_STATUS:
        return 'permanent'
    if status == 429:
        return 'rate_limit'
    if status == 403:
        return 'blocked'
    if status is not None and status // 100 == 5:
        return 'rate_limit'
    return 'transient'


def _take_volume(candidates: List[dict], start: int, max_volume_files: int, max_volume_bytes: int) -> int:
    """
    Return the exclusive end index of the next volume-sized slice of ``candidates``.

    Boundaries come from the ``file_size`` published by the index. A post larger than the whole
    budget still forms a volume of its own rather than being skipped.

    :param candidates: Candidate rows in ascending id order.
    :type candidates: List[dict]
    :param start: Index to start the slice at.
    :type start: int
    :param max_volume_files: Hard cap on entries per volume.
    :type max_volume_files: int
    :param max_volume_bytes: Soft cap on uncompressed bytes per volume.
    :type max_volume_bytes: int
    :returns: Exclusive end index of the slice.
    :rtype: int
    """
    end = start
    total = 0
    while end < len(candidates):
        size = candidates[end]['file_size'] or 0
        if end > start and (end - start >= max_volume_files or total + size > max_volume_bytes):
            break
        total += size
        end += 1
    return end


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
                  src_repository: str, repository: str, baseline_size: int,
                  glob_exist_ids_file: str):
    """
    Render the staging repository README.

    The coverage section is not decoration. This repository stores a set difference, so a
    consumer who takes it for a complete Danbooru mirror would silently lose every post the
    upstream collections already hold - millions of them. The README states the exclusion, names
    the collections, and spells out the union needed for a complete set.

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
    :param repository: This staging repository's own id, used in the usage examples.
    :type repository: str
    :param baseline_size: Number of ids in the read-only exclusion baseline.
    :type baseline_size: int
    :param glob_exist_ids_file: Name of the baseline file inside this repository.
    :type glob_exist_ids_file: str
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
        # The index repository lives only on the self-hosted endpoint; linking it to
        # huggingface.co would send every reader to a 404.
        print(f'Raw original files for Danbooru posts listed in [{src_repository}]'
              f'(https://hub.deepghs.org/datasets/{src_repository}) that **no existing upstream '
              f'collection already stores**. This repository is a staging area for the repack '
              f'stage, and it is deliberately incomplete on its own.', file=f)
        print('', file=f)

        print('# What is excluded, and what you need for a complete set', file=f)
        print('', file=f)
        print('Before anything is downloaded, every post id already covered by these collections '
              'is removed from the candidate list:', file=f)
        print('', file=f)
        for entry in _UPSTREAM_COLLECTIONS:
            url = hf_hub_repo_url(repo_id=entry['repo_id'], repo_type='dataset',
                                  endpoint=entry['endpoint'])
            print(f'- [`{entry["repo_id"]}`]({url})', file=f)
        print('', file=f)
        print(f'The union of their ids ships here as `{glob_exist_ids_file}` '
              f'({baseline_size:,} ids). That file is read-only: this job reads it to decide what '
              f'to skip and never adds to it. So what you find here is a set *difference* - only '
              f'the posts the index knows about that none of the collections above already hold.',
              file=f)
        print('', file=f)
        print('**This repository alone is not a complete Danbooru mirror.** Using it by itself '
              'silently omits every post covered upstream. For the complete, up-to-date set of '
              'originals you need all of the following together:', file=f)
        print('', file=f)
        print('```text', file=f)
        print(repository, file=f)
        for entry in _UPSTREAM_COLLECTIONS:
            print(f'  + {entry["repo_id"]}', file=f)
        print('```', file=f)
        print('', file=f)
        print('Post ids are globally unique and the three sets are disjoint by construction, so '
              'the union can be taken directly - no deduplication step is needed.', file=f)
        print('', file=f)

        print('# How to fetch one post', file=f)
        print('', file=f)
        print('Every one of the three collections is an hfutils-indexed tar store, so a single '
              'image is a range request rather than a whole-tar download. What differs is where '
              'the index lives and how an entry is named:', file=f)
        print('', file=f)
        print('| collection | tar holding post `id` | index | entry name |', file=f)
        print('|---|---|---|---|', file=f)
        print(f'| `{repository}` (this one) | `volume_file` from `table.parquet` | '
              f'beside the tars | `{"{id}.{ext}"}` |', file=f)
        for entry in _UPSTREAM_COLLECTIONS:
            idx = f'`{entry["idx_repo_id"]}`' if entry['idx_repo_id'] else 'beside the tars'
            print(f'| `{entry["repo_id"]}` | {entry["tar_path"]} | {idx} | '
                  f'{entry["entry_name"]} |', file=f)
        print('', file=f)
        for entry in _UPSTREAM_COLLECTIONS:
            print(f'- `{entry["repo_id"]}`: {entry["note"]}', file=f)
        print('', file=f)
        print('`nyanko7/danbooru2023` publishes no sidecars of its own, but '
              '`deepghs/danbooru2023_index` mirrors its tree exactly - `original/data-0000.json` '
              'indexes `original/data-0000.tar`, and so on. hfutils reads a split pair like that '
              'natively through `idx_repo_id`, so it needs no special handling.', file=f)
        print('', file=f)
        print('## Endpoints', file=f)
        print('', file=f)
        print(f'`nyanko7/danbooru2023` and `deepghs/danbooru2023_index` are public on '
              f'{_PUBLIC_ENDPOINT}. This repository and `deepghs/danbooru_newest-all` are '
              f'private and live on a self-hosted endpoint - set `HF_ENDPOINT` to it and '
              f'`HF_TOKEN` to a token with access. Because two endpoints are in play in one '
              f'script, pass `endpoint=` explicitly on the public calls rather than relying on '
              f'the environment for both.', file=f)
        print('', file=f)
        print('## Resolution order', file=f)
        print('', file=f)
        print('```python', file=f)
        print('import json, os', file=f)
        print('import pandas as pd', file=f)
        print('from huggingface_hub import hf_hub_download', file=f)
        print('from hfutils.index import hf_tar_file_download, hf_tar_file_exists', file=f)
        print('', file=f)
        print(f'PUBLIC = "{_PUBLIC_ENDPOINT}"', file=f)
        print('PRIVATE = os.environ["HF_ENDPOINT"]   # the self-hosted endpoint', file=f)
        print('', file=f)
        print('', file=f)
        print('def fetch(post_id: int, dst: str) -> str:', file=f)
        print('    """Download one original into dst; return which collection served it."""', file=f)
        print('    bucket = post_id % 1000', file=f)
        print('', file=f)
        print('    # 1. this repository - table.parquet names the exact volume', file=f)
        print('    table = pd.read_parquet(hf_hub_download(', file=f)
        print(f'        repo_id="{repository}", repo_type="dataset",', file=f)
        print('        filename="table.parquet", endpoint=PRIVATE))', file=f)
        print('    hit = table[table["id"] == post_id]', file=f)
        print('    if len(hit):', file=f)
        print('        row = hit.iloc[0]', file=f)
        print('        hf_tar_file_download(', file=f)
        print(f'            repo_id="{repository}", repo_type="dataset",', file=f)
        print('            archive_in_repo=row["volume_file"],', file=f)
        print('            file_in_archive=row["filename"], local_file=dst)', file=f)
        print('        return "staging"', file=f)
        print('', file=f)
        print('    # 2. danbooru_newest-all - fixed bucket, sidecar sits beside the tar', file=f)
        print('    sidecar = hf_hub_download(', file=f)
        print('        repo_id="deepghs/danbooru_newest-all", repo_type="dataset",', file=f)
        print('        filename=f"images/{bucket:04d}.json", endpoint=PRIVATE)', file=f)
        print('    for name in json.load(open(sidecar))["files"]:', file=f)
        print('        if name.rsplit(".", 1)[0] == str(post_id):', file=f)
        print('            hf_tar_file_download(', file=f)
        print('                repo_id="deepghs/danbooru_newest-all", repo_type="dataset",', file=f)
        print('                archive_in_repo=f"images/{bucket:04d}.tar",', file=f)
        print('                file_in_archive=name, local_file=dst)', file=f)
        print('            return "danbooru_newest-all"', file=f)
        print('', file=f)
        print('    # 3. danbooru2023 - tars in one repo, index in another', file=f)
        print('    #    HF_ENDPOINT must point at the public hub for this block; hfutils', file=f)
        print('    #    takes no per-call endpoint here, so set it around the call.', file=f)
        print('    was, os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT"), PUBLIC', file=f)
        print('    try:', file=f)
        print('        pair = dict(repo_id="nyanko7/danbooru2023", repo_type="dataset",', file=f)
        print('                    idx_repo_id="deepghs/danbooru2023_index",', file=f)
        print('                    idx_repo_type="dataset")', file=f)
        print('        for archive in (f"original/data-{bucket:04d}.tar",', file=f)
        print('                        f"recent/data-1{bucket:03d}.tar"):', file=f)
        print('            for ext in ("jpg", "png", "webp", "gif", "jpeg"):', file=f)
        print('                name = f"./{post_id}.{ext}"      # note the ./ prefix', file=f)
        print('                if hf_tar_file_exists(archive_in_repo=archive,', file=f)
        print('                                      file_in_archive=name, **pair):', file=f)
        print('                    hf_tar_file_download(archive_in_repo=archive,', file=f)
        print('                                         file_in_archive=name,', file=f)
        print('                                         local_file=dst, **pair)', file=f)
        print('                    return "danbooru2023"', file=f)
        print('    finally:', file=f)
        print('        if was is None:', file=f)
        print('            os.environ.pop("HF_ENDPOINT", None)', file=f)
        print('        else:', file=f)
        print('            os.environ["HF_ENDPOINT"] = was', file=f)
        print('', file=f)
        print('    raise KeyError(f"post {post_id} is in none of the three collections")', file=f)
        print('```', file=f)
        print('', file=f)
        print('The extension loop in step 3 is only needed when you do not already know it. '
              'Reading `exist_image_ids.json` once, or the index json for the bucket, gives you '
              'the exact entry name and skips the probing.', file=f)
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
        print('Volumes are numbered from 1 and laid out as `images/{n // 1000}/{n % 1000:03d}.tar`. '
              'They are append-only: an existing volume is never rewritten, so any tar and sidecar '
              'you have already downloaded stays valid forever and only new volumes need fetching '
              'on an update.', file=f)
        print('', file=f)
        print('`table.parquet` columns: ' +
              ', '.join(f'`{c}`' for c in _TABLE_COLUMNS) + '. `volume_file` and `filename` are '
              'the two you need to pull the bytes; `md5` and `file_size` come from the index and '
              'were verified at download time.', file=f)
        print('', file=f)
        print('`meta.json` carries `bad_image_ids`: posts the index lists but the CDN answers 404 '
              'or 410 for. They will never appear here and are not worth retrying.', file=f)
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
         max_time_limit: Optional[float] = (60 * 5) * 60, max_volume_files: int = 10000,
         max_volume_bytes: int = 10 * 1024 ** 3, max_volume_hard_bytes: int = 12 * 1024 ** 3,
         download_workers: int = 16, session_pool_size: int = 0,
         min_free_disk: int = 24 * 1024 ** 3, upload_time_span: float = 30,
         include_non_image: bool = False, glob_exist_ids_file: str = 'glob_exist_ids.json',
         max_volumes: Optional[int] = None, retire_after: int = 2,
         initial_rate: float = 4.0, max_rate: float = 64.0,
         cf_retries: int = 6, cf_retry_wait: float = 2.0,
         max_blocked_ratio: float = 0.3, proxy_pool: Optional[str] = None,
         brd_api_key: Optional[str] = None, brd_zone: Optional[str] = None):
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
    :param max_volume_bytes: Byte budget used when planning a volume from the index.
    :type max_volume_bytes: int
    :param max_volume_hard_bytes: Ceiling on the bytes actually written into a tar. Crossing it
        seals the volume immediately and moves the rest of the batch to the next one.
    :type max_volume_hard_bytes: int
    :param download_workers: Concurrent download threads.
    :type download_workers: int
    :param session_pool_size: Number of independent warmed-up clients to draw from.
    :type session_pool_size: int
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
    :param retire_after: Consecutive rejections a session slot survives before being replaced.
    :type retire_after: int
    :param initial_rate: Requests per second the adaptive limiter starts from.
    :type initial_rate: float
    :param max_rate: Ceiling for the adaptive limiter.
    :type max_rate: float
    :param cf_retries: Attempts per post before giving up, rebuilding the session between
        Cloudflare rejections.
    :type cf_retries: int
    :param cf_retry_wait: Base seconds to back off after a Cloudflare rejection; grows with the
        attempt number.
    :type cf_retry_wait: float
    :param max_blocked_ratio: Abort the run when this fraction of a volume fails for reasons
        other than the post being gone upstream.
    :type max_blocked_ratio: float
    :param proxy_pool: Proxy URL applied to every upstream request. Off by default: the direct
        route is rate limited but never blocked, while shared datacenter proxy addresses are
        blocked outright by the CDN.
    :type proxy_pool: Optional[str]
    :param brd_api_key: Bright Data API key, used only to allowlist this host's address.
    :type brd_api_key: Optional[str]
    :param brd_zone: Bright Data zone to allowlist into.
    :type brd_zone: Optional[str]
    """
    start_time = time.time()
    # Before any hub call: an untimed request against a stalled endpoint blocks for a quarter
    # of an hour, which is most of a scheduled run.
    configure_hf_http_backend()
    delete_detached_cache()
    hf_client = get_hf_client()

    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        logging.info(f'Staging repository {repository!r} created.')

    records, covered_ids, bad_image_ids, max_volume_id, baseline_size = _load_state(
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
    # Sized against the workers, not far above them: extra slots only dilute connection
    # reuse, and a hot connection is the single biggest lever on throughput here.
    if proxy_pool:
        # A Bright Data zone only serves allowlisted client addresses, and a runner's address is
        # different every job. The refusal reports the address, so the run can add its own.
        try:
            if not ensure_proxy_access(proxy_pool, api_key=brd_api_key, zone=brd_zone):
                logging.warning('Proxy is not usable; continuing over the direct route.')
                proxy_pool = None
        except BrightDataError as err:
            logging.warning(f'Proxy setup failed, continuing over the direct route - {err}')
            proxy_pool = None
    logging.info(f'Egress: {"proxy pool" if proxy_pool else "direct"}.')

    pool = DanbooruSessionPool(size=session_pool_size or download_workers + 4,
                               retire_after=retire_after, proxy_pool=proxy_pool)
    # The site meters requests and says so with 429. Discover the rate it will serve rather
    # than encoding a guess as a worker count.
    rate_limiter = AdaptiveRateLimiter(initial=initial_rate, maximum=max_rate)

    volumes_done = 0
    pending = candidates
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
            logging.warning(f'Only {free / 1024 ** 3:.2f} GB free, below the '
                            f'{min_free_disk / 1024 ** 3:.2f} GB floor - stop cleanly.')
            break

        end = _take_volume(pending, 0, max_volume_files, max_volume_bytes)
        plan, rest = pending[:end], pending[end:]

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
            deferred: List[dict] = []
            volume_bytes = [0]
            sealed = [None]
            lock = Lock()
            # One counter per outcome. Successes are counted and never logged: a per-item
            # success line adds nothing and drowns the lines that do need attention.
            stats = {'ok': 0, 'gone': 0, 'failed': 0, 'retry': 0, 'deferred': 0}

            with tarfile.open(tar_file, 'w:') as tar:
                def _handle(err, item, attempt) -> float:
                    """Record the outcome and return how long to wait before retrying."""
                    kind = _classify_error(err)
                    if kind == 'rate_limit':
                        rate_limiter.report_throttled()
                    status = getattr(getattr(err, 'response', None), 'status_code', None)
                    if kind == 'permanent':
                        # Genuinely gone upstream: record it so later runs skip it.
                        with lock:
                            volume_bad.append(item['id'])
                            stats['gone'] += 1
                        logging.warning(f'GONE post {item["id"]}: HTTP {status}.')
                        raise err
                    if attempt >= cf_retries:
                        with lock:
                            stats['failed'] += 1
                        logging.warning(f'FAILED post {item["id"]} after {attempt} attempts '
                                        f'({kind}, HTTP {status}): {err!r}')
                        raise err
                    with lock:
                        stats['retry'] += 1
                    logging.info(f'RETRY post {item["id"]} attempt {attempt}/{cf_retries} '
                                 f'({kind}, HTTP {status}).')
                    # A refused fingerprint costs only a new local object, so it earns a much
                    # lighter wait than a metered rate or a network fault. The jitter matters:
                    # without it every worker rejected in the same instant retries in the same
                    # instant, which is the burst that earned the rejection.
                    weight = 0.25 if kind == 'blocked' else 1.0
                    return cf_retry_wait * weight * attempt * (0.5 + random.random())

                def _fn_download(item):
                    with lock:
                        if sealed[0]:
                            # The valve tripped while this item was queued; hand it to the next
                            # volume instead of spending bandwidth we cannot store.
                            deferred.append(item)
                            stats['deferred'] += 1
                            return

                    _, ext = os.path.splitext(urlsplit(item['file_url']).filename)
                    filename = f'{item["id"]}{ext}'
                    dst_file = os.path.join(stage_dir, filename)
                    try:
                        size = width = height = None
                        for attempt in range(1, cf_retries + 1):
                            failure = None
                            rate_limiter.acquire()
                            with pool.lease() as (slot, generation, session):
                                try:
                                    size = download_file(item['file_url'], dst_file,
                                                         session=session,
                                                         expected_size=item['file_size'],
                                                         expected_md5=item['md5'])
                                    # Size and md5 already rule out a truncated or substituted
                                    # body; this only rejects formats PIL cannot open.
                                    with Image.open(dst_file) as image:
                                        width, height = image.size
                                except Exception as err:
                                    failure = err
                                    # Only a refused fingerprint is the session's fault. Swapping
                                    # sessions on a 429 would discard a healthy connection and
                                    # change nothing about the rate.
                                    if _classify_error(err) == 'blocked':
                                        pool.report_failure(slot, generation)
                                else:
                                    # Reported inside the lease: once released, another worker
                                    # may retire the slot and the credit would be lost.
                                    pool.report_success(slot)
                                    rate_limiter.report_success()
                            if failure is None:
                                break
                            # Outside the lease: a sleeping worker must not also hold a slot.
                            wait = _handle(failure, item, attempt)
                            if wait:
                                time.sleep(wait)
                        with lock:
                            tar.add(dst_file, filename)
                            volume_bytes[0] += size
                            stats['ok'] += 1
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
                            # Safety valves, checked against what actually landed on disk rather
                            # than what the index promised. Only the transition is interesting:
                            # workers already in flight when the valve tripped arrive here too,
                            # and the deadline test stays true forever once it is true, so
                            # re-evaluating would reprint the same line once per worker.
                            if not sealed[0]:
                                if volume_bytes[0] >= max_volume_hard_bytes:
                                    sealed[0] = (f'tar reached {volume_bytes[0] / 1024 ** 3:.2f} GB, over '
                                                 f'the {max_volume_hard_bytes / 1024 ** 3:.2f} GB ceiling')
                                elif get_free_disk_bytes(stage_dir) < min_free_disk:
                                    sealed[0] = f'free disk fell below {min_free_disk / 1024 ** 3:.2f} GB'
                                elif max_time_limit is not None and \
                                        time.time() > start_time + max_time_limit:
                                    # Without this the deadline is only tested between volumes, so
                                    # a volume that started just under it runs to completion and
                                    # pushes the job past its own timeout. Sealing here decouples
                                    # volume size from the schedule: what has been fetched ships.
                                    sealed[0] = 'run deadline reached mid-volume'
                                if sealed[0]:
                                    logging.warning(
                                        f'Sealing volume #{max_volume_id} early - {sealed[0]}; '
                                        f'remaining posts move to the next volume.')
                    finally:
                        # Free the bytes immediately; in-flight footprint stays at roughly
                        # download_workers x average file size.
                        if os.path.exists(dst_file):
                            os.remove(dst_file)

                parallel_call(plan, _fn_download, max_workers=download_workers,
                              desc=f'Volume #{max_volume_id}', postfix=lambda: dict(stats))

            shutil.rmtree(stage_dir, ignore_errors=True)
            pending = sorted(deferred, key=lambda x: x['id']) + rest

            attempted = len(plan) - stats['deferred']
            logging.info(f'Rate limiter settled at {rate_limiter.rate:.1f} req/s after '
                         f'{rate_limiter.throttles} throttling responses.')
            logging.info(f'Volume #{max_volume_id} downloaded: {stats["ok"]} ok, '
                         f'{stats["gone"]} gone, {stats["failed"]} failed, '
                         f'{stats["deferred"]} deferred, {stats["retry"]} retries, '
                         f'{volume_bytes[0] / 1024 ** 3:.2f} GB.')
            pool_stats = pool.stats()
            logging.info(f'Sessions: {pool_stats["reuse_rate"]:.0%} slot reuse, '
                         f'{pool_stats["total_ok"]} accepted / {pool_stats["total_bad"]} rejected; '
                         + ', '.join(f'{imp}={ok}/{bad}@{score}'
                                     for imp, ok, bad, score in pool_stats['fingerprints'][:5]))
            if stats['failed'] and stats['failed'] >= attempted * max_blocked_ratio:
                # Sustained blocking wastes a volume id and a commit per attempt, so stop the run
                # and let the next scheduled one start from a clean window.
                logging.error(f'ABORT: {stats["failed"]} of {attempted} posts failed in volume '
                              f'#{max_volume_id}, at or above the '
                              f'{max_blocked_ratio:.0%} threshold - not publishing this volume.')
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
                          src_repository=src_repository, repository=repository,
                          baseline_size=baseline_size,
                          glob_exist_ids_file=glob_exist_ids_file)

            actual_bytes = os.path.getsize(tar_file)
            limiter.try_acquire('hf_upload')
            logging.info(f'UPLOAD volume #{max_volume_id} starting - '
                         f'{plural_word(len(new_records), "image")}, '
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
            # Upload throughput, not download, is what bounds a run against this endpoint, so it
            # is worth a number in the log rather than a guess after the fact.
            logging.info(f'UPLOAD volume #{max_volume_id} done in {upload_elapsed:.0f}s '
                         f'({actual_bytes / 1024 ** 2 / max(upload_elapsed, 1e-6):.1f} MB/s).')
            volumes_done += 1

        # TemporaryDirectory is gone here; drop the HF cache too so repeated volumes do not
        # accumulate blobs on a runner with a small disk.
        delete_detached_cache()
        log_disk_usage(os.getcwd(), prefix='Disk after volume')

    pool.close()
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
    default=10000,
    show_default=True,
    help='Maximum number of entries packed into one tar volume. High enough that the byte budget '
         'is normally what closes a volume.',
)
@click.option(
    '-b', '--max-volume-bytes',
    type=int,
    default=10 * 1024 ** 3,
    show_default=True,
    help='Approximate byte budget for one tar volume.',
)
@click.option(
    '-H', '--max-volume-hard-bytes',
    type=int,
    default=12 * 1024 ** 3,
    show_default=True,
    help='Ceiling on the bytes actually written into a tar. Crossing it seals the volume '
         'immediately, uploads it, and moves the rest of the batch to the next volume.',
)
@click.option(
    '-w', '--download-workers',
    type=int,
    default=16,
    show_default=True,
    help='Number of concurrent download threads. Throughput is capped by the site meter, not by '
         'this, so extra workers only help drain the initial burst allowance faster.',
)
@click.option(
    '-P', '--session-pool-size',
    type=int,
    default=0,
    show_default=True,
    help='Session pool size. 0 sizes it just above the worker count, which is what keeps '
         'connections hot; a much larger pool only dilutes reuse.',
)
@click.option(
    '-d', '--min-free-disk',
    type=int,
    default=24 * 1024 ** 3,
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
    '-I', '--initial-rate',
    type=float,
    default=4.0,
    show_default=True,
    help='Requests per second the adaptive limiter starts from before probing upwards.',
)
@click.option(
    '-M', '--max-rate',
    type=float,
    default=64.0,
    show_default=True,
    help='Ceiling for the adaptive request rate.',
)
@click.option(
    '-T', '--retire-after',
    type=int,
    default=2,
    show_default=True,
    help='Consecutive rejections a session slot survives before it is replaced. Dropping a '
         'working connection costs a handshake and a fresh Cloudflare verdict, so one isolated '
         'rejection is not worth acting on.',
)
@click.option(
    '-c', '--cf-retries',
    type=int,
    default=6,
    show_default=True,
    help='Attempts per post, rebuilding the session between Cloudflare rejections.',
)
@click.option(
    '-C', '--cf-retry-wait',
    type=duration_type(),
    default=2.0,
    show_default=True,
    help='Base backoff after a rejection; grows with the attempt number. A refused fingerprint '
         'waits a quarter as long, since only the session needs changing.',
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
    '-p', '--proxy-pool',
    type=str,
    envvar='PP_DB',
    default=None,
    show_envvar=True,
    help='Proxy URL applied to every upstream request. Leave unset to go direct, which is the '
         'faster route: it is metered but never blocked, whereas shared datacenter proxy exits '
         'are refused outright by the CDN.',
)
@click.option(
    '--brd-api-key',
    type=str,
    envvar='BRD_API_KEY',
    default=None,
    show_envvar=True,
    help='Bright Data API key, used only to add this host to the zone allowlist.',
)
@click.option(
    '--brd-zone',
    type=str,
    envvar='BRD_ZONE',
    default=None,
    show_envvar=True,
    help='Bright Data zone to allowlist this host into.',
)
def cli(repository: str, src_repository: str, src_revision: str, max_time_limit: Optional[float],
        max_volume_files: int, max_volume_bytes: int, max_volume_hard_bytes: int,
        download_workers: int, session_pool_size: int, min_free_disk: int,
        upload_time_span: float, include_non_image: bool, glob_exist_ids_file: str,
        max_volumes: Optional[int], initial_rate: float, max_rate: float,
        retire_after: int, cf_retries: int, cf_retry_wait: float,
        max_blocked_ratio: float, proxy_pool: Optional[str],
        brd_api_key: Optional[str], brd_zone: Optional[str]):
    logging.try_init_root(logging.INFO)
    return sync(
        repository=repository,
        src_repository=src_repository,
        src_revision=src_revision,
        max_time_limit=max_time_limit,
        max_volume_files=max_volume_files,
        max_volume_bytes=max_volume_bytes,
        max_volume_hard_bytes=max_volume_hard_bytes,
        download_workers=download_workers,
        session_pool_size=session_pool_size,
        min_free_disk=min_free_disk,
        upload_time_span=upload_time_span,
        include_non_image=include_non_image,
        glob_exist_ids_file=glob_exist_ids_file,
        max_volumes=max_volumes,
        retire_after=retire_after,
        initial_rate=initial_rate,
        max_rate=max_rate,
        cf_retries=cf_retries,
        cf_retry_wait=cf_retry_wait,
        max_blocked_ratio=max_blocked_ratio,
        proxy_pool=proxy_pool,
        brd_api_key=brd_api_key,
        brd_zone=brd_zone,
    )


if __name__ == '__main__':
    cli()
