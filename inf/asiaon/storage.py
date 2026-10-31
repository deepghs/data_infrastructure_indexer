"""Package storage for asiaon posts.

Continues the published raw storage dataset exactly as the pyskeb jobs left it: one tar per post under
``images/<md5(post id)[:2]>/<post id>.tar`` with a same-named ``.json`` index, plus ``posts.parquet``,
``files.parquet``, ``tags.parquet``, ``categories.parquet`` and ``README.md``. Rows already published
are never rewritten and file ids continue from the highest one in use.

Which posts, and from where
===========================

A post is a candidate when the index lists it, storage does not hold it, and it has a link on a host
this job can fetch (see :mod:`inf.asiaon.hosts`). The newest are taken first, so a run cut short by its
time budget has still covered what is most recent. For each candidate the hosts are tried in their fixed
order, and a post is stored from the first host whose package downloads *and* unpacks; anything less
leaves it out of storage altogether and the next run simply tries it again. Nothing is recorded about
failures: a post is either in ``posts.parquet`` or it is not.

The password for each archive is worked out and checked against the archive itself, see
:mod:`inf.asiaon.passwords`; the index's own ``unzip_password`` is not used as it stands.

Batches
=======

The tables hold two million file rows, so rewriting them for every post is not an option. Packages
collect in a batch directory and go up together once ``deploy_span`` has passed or the batch holds
``max_batch_bytes``; the first package goes up at once, as a check that the whole path works. Each
post's rows join the live tables only after the batch carrying them has been committed, and the batch
directory is emptied after each upload, so what is on disk is the batch being assembled and the post
in hand, never the dataset.

Dead links and faults
=====================

A link the host says is gone is an expected end for an old post, not a fault. A run stops with an
error after ``max_consecutive_failures`` posts in a row that failed for any other reason, which means
something systematic, and stops quietly after ``max_dead_streak`` posts in a row with nothing alive, since
past that point the rest of the list is almost certainly dead too. A host that reports it is throttling
or refusing the run is not asked again until the next one.

History
=======

This job only ever adds commits and never rewrites history: squashing the repository is a manual
decision, made by hand when wanted.
"""
import io
import json
import math
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
from collections import deque
from datetime import timezone
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List, Optional, Set, Tuple

import click
import numpy as np
import pandas as pd
import requests
from ditk import logging
from hbutils.encoding import md5
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.archive import archive_pack
from hfutils.cache import delete_detached_cache
from hfutils.index import tar_get_index_info
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.utils import number_to_tag
from pyrate_limiter import Duration, Limiter, Rate

from inf.utils.archive_safety import ArchiveUnsafe, check_archive_members, validate_tree
from inf.utils.duration import duration_type
from inf.utils.safe import configure_hf_http_backend, safe_hf_hub_download, safe_upload_directory_as_directory
from inf.utils.transfer import DownloadError
from .hosts import HOSTS
from .passwords import WrongPassword, find_password

mimetypes.add_type('image/webp', '.webp')

#: Attempts per host for a failure that is likely to be momentary; each retry starts from a fresh link.
_HOST_ATTEMPTS = 2
#: Abandoned slow transfers after which a host is set aside for the rest of the run.
_SLOW_STRIKES = 2
#: Refusals (HTTP 403 or 429) in a row from one host before it is set aside; one can be a single file.
_REFUSED_STRIKES = 3
_MAX_CONSECUTIVE_FAILURES = 10
_MAX_DEAD_STREAK = 200
_MAX_BATCH_BYTES = 6 * 1024 ** 3
#: Seconds to wait for a finished post before looking again at whether a commit is due.
_WAIT_SLICE = 15.0
#: When every host has been set aside, the job waits this long and then asks them all again: the limits
#: seen so far (Cloudflare challenges, pixeldrain's throttle, gofile and mega rate limits) ease with time.
_COOLDOWN = 20 * 60
#: How many times in one run the job waits like that before giving up on the hosts for this run.
_MAX_COOLDOWNS = 8
#: Posts in a row left for later because of a set-aside host, with at least one host still set aside, after
#: which the job waits like that too: hosts that are not set aside cannot help with the posts it is walking.
_DEFER_STREAK = 30


class PackageUnavailable(Exception):
    """Every host that was asked says the package is gone, or the post has no link anyone can fetch."""


class PackageFailed(Exception):
    """The package could not be fetched or unpacked for a reason that may not last."""


class PackageDeferred(Exception):
    """
    Nothing could be said about the package this run: the hosts that were asked say it is gone, or were
    not asked at all because they are refusing the run, so a host that might still have it was skipped.
    It is neither a fault nor a dead link, and it is simply tried again next run.
    """


class UnpackError(Exception):
    """The downloaded archive could not be turned into a usable directory of files."""


def utf8_encode_safe(text: str) -> str:
    """``text`` with every character that cannot be written as UTF-8 replaced by an underscore."""
    with io.StringIO() as buf:
        for ch in text:
            try:
                ch.encode('utf-8')
            except UnicodeEncodeError:
                buf.write('_')
            else:
                buf.write(ch)
        return buf.getvalue()


def _to_none(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def _is_transient(err: Exception) -> bool:
    return isinstance(err, requests.RequestException) or (isinstance(err, DownloadError) and err.retryable)


def _fix_unsafe_names(directory: str) -> None:
    """Rename files whose names cannot be written as UTF-8, as the original job did before packing."""
    for root, _, files in os.walk(directory):
        for name in files:
            src = os.path.join(root, name)
            rel = os.path.relpath(src, directory)
            fixed = utf8_encode_safe(rel)
            if fixed != rel:
                dst = os.path.join(directory, fixed)
                logging.info(f'Moving file {rel!r} --> {fixed!r} due to filename safe issue.')
                os.makedirs(os.path.dirname(dst) or directory, exist_ok=True)
                shutil.move(src, dst)


def unpack_package(archive_file: str, directory: str, stored_password: Optional[str]) -> int:
    """
    Unpack ``archive_file`` into ``directory`` and return the number of files.

    The member list is checked before extracting and the tree after, see
    :mod:`inf.utils.archive_safety`, and the password is the one the archive itself accepted.
    """
    try:
        check_archive_members(archive_file, stored_password)
        password = find_password(archive_file, stored_password)
        cmd = ['7z', 'x', '-y', f'-o{directory}', f'-p{password}' if password else '-p', archive_file]
        # UTF-8 names have to survive extraction whatever locale the runner has.
        proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=7200,
                              env={**os.environ, 'LC_ALL': 'C.UTF-8'})
        if proc.returncode != 0:
            raise UnpackError(f'7z exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-200:]!r}')
        _fix_unsafe_names(directory)
        return validate_tree(directory)
    except ArchiveUnsafe as err:
        raise UnpackError(f'Refused unsafe archive - {err}') from err
    except WrongPassword as err:
        raise UnpackError(f'Password not found - {err}') from err
    except UnpackError:
        raise
    except Exception as err:
        raise UnpackError(f'Unpack failed - {err!r}') from err


#: Where posts whose links are confirmed dead are recorded, inside the storage dataset itself.
_GONE_FILE = 'meta/gone.json'


def read_gone(hf_client, hf_fs, repository: str) -> Dict[str, Dict]:
    """
    The posts recorded as gone: ``{post id: {'dbid': ..., 'hosts': {host: {'url', 'kind', 'reason', 'at'}}}}``.

    A file that exists but cannot be read is an error, never an empty record: carrying on would write a
    near-empty file over the real one at the next commit.
    """
    if not hf_fs.exists(f'datasets/{repository}/{_GONE_FILE}'):
        return {}
    path = safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset', filename=_GONE_FILE)
    with open(path, encoding='utf-8') as f:
        return json.load(f)['posts']


def write_gone(directory: str, posts: Dict[str, Dict]) -> None:
    os.makedirs(os.path.join(directory, os.path.dirname(_GONE_FILE)), exist_ok=True)
    with open(os.path.join(directory, _GONE_FILE), 'w', encoding='utf-8') as f:
        json.dump({'version': 1, 'posts': posts}, f, ensure_ascii=False, sort_keys=True, indent=1)


def merge_gone(live: Dict[str, Dict], pending: Dict[str, Dict]) -> Dict[str, Dict]:
    """``live`` with ``pending`` added; nothing is ever removed from a record."""
    merged = {pid: {'dbid': entry.get('dbid'), 'hosts': dict(entry['hosts'])} for pid, entry in live.items()}
    for pid, entry in pending.items():
        target = merged.setdefault(pid, {'dbid': entry.get('dbid'), 'hosts': {}})
        target['hosts'].update(entry['hosts'])
    return merged


def is_known_gone(record: Optional[Dict], host_name: str, url: str) -> bool:
    """Whether this host's link for this post is recorded as gone. A different link is a new link."""
    entry = ((record or {}).get('hosts') or {}).get(host_name)
    return bool(entry) and entry.get('url') == url


def usable_url(host, post: Dict) -> Optional[str]:
    url = _to_none(post.get(host.column))
    # Some old records hold the site's own data-API address where a download link should be.
    if not url or '_next/data' in url or not host.accepts(url):
        return None
    return url


def fetch_package(post: Dict, work_dir: str, disabled: Set[str], strikes: Optional[Dict[str, int]] = None,
                  known_gone: Optional[Dict] = None, gone_hosts: Optional[Dict[str, Dict]] = None) -> Tuple[str, str]:
    """
    Download and unpack a post's package from the first host that delivers.

    :returns: The host used and the directory the files were unpacked into.
    :raises PackageUnavailable: When no host has a live link for this post.
    :raises PackageFailed: When a host had one and the package still could not be had.

    ``known_gone`` is this post's entry in the gone record: hosts recorded there are not asked again.
    Each host that says the link is gone this time is added to ``gone_hosts`` with its kind and reason,
    whatever the post's final outcome, so the caller can record it.
    """
    outcomes: List[Tuple[str, str, str]] = []  # (host, 'gone' | 'fault' | 'skipped', text)
    for host in HOSTS:
        url = usable_url(host, post)
        if not url:
            continue
        if is_known_gone(known_gone, host.name, url):
            outcomes.append((host.name, 'gone', f'{host.name}: recorded as gone'))
            continue
        if host.name in disabled:
            outcomes.append((host.name, 'skipped', f'{host.name}: skipped, refusing this run'))
            continue

        download_dir = os.path.join(work_dir, 'download', host.name)
        unpack_dir = os.path.join(work_dir, 'unpack', host.name)
        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(unpack_dir, exist_ok=True)
        verdict = None
        for attempt in range(1, _HOST_ATTEMPTS + 1):
            archive_file = None
            try:
                archive_file = host.download(url, download_dir)
                logging.info(f'Unpacking {os.path.basename(archive_file)!r} ...')
                unpack_package(archive_file, unpack_dir, _to_none(post.get('unzip_password')))
                if strikes is not None:
                    strikes.pop(f'{host.name}:refused', None)
                return host.name, unpack_dir
            except Exception as err:
                logging.warning(f'Post {post["id"]!r} failed on {host.name} (attempt {attempt}/{_HOST_ATTEMPTS}): '
                                f'{err!r}')
                if isinstance(err, DownloadError) and err.blocked:
                    disabled.add(host.name)
                if isinstance(err, DownloadError) and err.refused and strikes is not None:
                    key = f'{host.name}:refused'
                    strikes[key] = strikes.get(key, 0) + 1
                    if strikes[key] >= _REFUSED_STRIKES:
                        logging.warning(f'{host.name} refused {strikes[key]} times in a row, setting it aside.')
                        disabled.add(host.name)
                if isinstance(err, DownloadError) and err.slow and strikes is not None:
                    strikes[host.name] = strikes.get(host.name, 0) + 1
                    if strikes[host.name] >= _SLOW_STRIKES:
                        logging.warning(f'{host.name} was too slow {strikes[host.name]} times, setting it aside.')
                        disabled.add(host.name)
                verdict = err
                # A failed attempt may have unpacked part of the archive; the next starts from nothing.
                shutil.rmtree(unpack_dir, ignore_errors=True)
                os.makedirs(unpack_dir, exist_ok=True)
                if not (_is_transient(err) and attempt < _HOST_ATTEMPTS):
                    break
                time.sleep(5)
            finally:
                # The archive is no longer needed either way; free the disk. The unpacked files are
                # not touched here: on success they are what goes into the package.
                if archive_file and os.path.exists(archive_file):
                    os.remove(archive_file)
        if host.name in disabled and not (isinstance(verdict, DownloadError) and verdict.gone):
            # The host is refusing this run as a whole, which says nothing about the post: it is not a fault.
            kind = 'skipped'
        else:
            kind = 'gone' if isinstance(verdict, DownloadError) and verdict.gone else 'fault'
        if kind == 'gone' and gone_hosts is not None:
            gone_hosts[host.name] = {'kind': verdict.kind, 'reason': str(verdict)[:300]}
        outcomes.append((host.name, kind, f'{host.name}: {verdict!r}'))

    text = '; '.join(item[2] for item in outcomes)
    asked = [item for item in outcomes if item[1] != 'skipped']
    skipped = len(outcomes) - len(asked)
    if not outcomes:
        raise PackageUnavailable('no link on a host this job can fetch')
    if all(item[1] == 'gone' for item in asked):
        # Every host that was asked says it is gone. If another was not asked, that one may still have it.
        raise (PackageDeferred(text) if skipped else PackageUnavailable(text))
    raise PackageFailed(text)


def build_package(post_id: str, unpack_dir: str, batch_dir: str) -> Tuple[str, int, int, List[str]]:
    """Repack ``unpack_dir`` into the post's tar inside the batch and write its index."""
    package_file = f'images/{md5(post_id.encode())[:2]}/{post_id}.tar'
    tar_file = os.path.join(batch_dir, *package_file.split('/'))
    os.makedirs(os.path.dirname(tar_file), exist_ok=True)
    archive_pack('tar', directory=unpack_dir, archive_file=tar_file)

    meta_data = tar_get_index_info(tar_file, 1 << 20, with_hash=True)
    with open(os.path.splitext(tar_file)[0] + '.json', 'w') as f:
        json.dump(meta_data, f)
    return package_file, os.path.getsize(tar_file), len(meta_data['files']), list(meta_data['files'].keys())


def process_post(post: Dict, staging_root: str, disabled: Set[str], strikes: Dict[str, int],
                 known_gone: Optional[Dict] = None) -> Dict:
    """
    Download, unpack and repack one post, leaving the tar in a staging directory of its own.

    Runs in a worker thread, so nothing here touches the batch or the tables: the tar and its index
    are built under ``staging_root`` and handed to the main thread, which moves them into the batch
    once it is this post's turn. The hosts that said the post's link is gone come back too, as
    ``gone_hosts`` in the result, or as an attribute of the exception when the post did not make it.
    """
    gone_hosts: Dict[str, Dict] = {}
    try:
        with TemporaryDirectory() as td:
            host_name, unpack_dir = fetch_package(post, os.path.join(td, 'work'), disabled, strikes,
                                                  known_gone, gone_hosts)
            stage = tempfile.mkdtemp(dir=staging_root)
            try:
                package_file, package_size, file_count, members = build_package(post['id'], unpack_dir, stage)
                sizes = [os.path.getsize(os.path.join(unpack_dir, member)) for member in members]
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                raise
    except Exception as err:
        err.gone_hosts = gone_hosts
        raise
    return {'host': host_name, 'package_file': package_file, 'package_size': package_size,
            'file_count': file_count, 'members': members, 'sizes': sizes, 'stage': stage,
            'gone_hosts': gone_hosts}


def adopt_staged(stage: str, batch_dir: str) -> None:
    """Move a staged package into the batch and drop the staging directory."""
    for root, _, files in os.walk(stage):
        for name in files:
            src = os.path.join(root, name)
            dst = os.path.join(batch_dir, os.path.relpath(src, stage))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
    shutil.rmtree(stage, ignore_errors=True)


def _read_table(hf_client, hf_fs, repository: str, filename: str) -> Optional[pd.DataFrame]:
    if not hf_fs.exists(f'datasets/{repository}/{filename}'):
        return None
    df = pd.read_parquet(safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset', filename=filename))
    # Parquet nulls come back as NaN; the rest of the pipeline expects None.
    return df.astype(object).where(df.notna(), None)


def _read_table_typed(hf_client, hf_fs, repository: str, filename: str) -> Optional[pd.DataFrame]:
    """A published table with its column types intact; the big file table must not be turned into objects."""
    if not hf_fs.exists(f'datasets/{repository}/{filename}'):
        return None
    return pd.read_parquet(safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset', filename=filename))


def _with_sorted_download_urls(df: pd.DataFrame) -> pd.DataFrame:
    """Every ``download_urls`` struct with the same keys in alphabetical order, as published."""
    if 'download_urls' not in df.columns:
        return df
    urls = [d if isinstance(d, dict) else {} for d in df['download_urls']]
    keys = sorted({key for d in urls for key in d})
    df = df.copy()
    df['download_urls'] = [{key: d.get(key) for key in keys} for d in urls]
    return df


def _write_readme(file: str, df_posts: pd.DataFrame, df_files: pd.DataFrame, df_tags: pd.DataFrame,
                  df_categories: pd.DataFrame) -> None:
    with open(file, 'w') as f:
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
        print('- photo', file=f)
        print('- anime', file=f)
        print('- not-for-all-audiences', file=f)
        print('size_categories:', file=f)
        print(f'- {number_to_tag(len(df_files))}', file=f)
        print('source_datasets:', file=f)
        print('- asiaon', file=f)
        print('---', file=f)
        print('', file=f)

        print('This is a middleware repository for website [asiaon.top](https://asiaon.top/). '
              'These data will be published after some more cleaning, annotation and repacking. '
              'It will be quite useful for training image generation or '
              'making image processing tools on real nsfw images.', file=f)
        print('', file=f)
        print('## Posts', file=f)
        print('', file=f)
        df_posts_shown = df_posts[:50][[
            'dbid', 'id', 'title', 'page_url', 'tags', 'categories',
            'resolution', 'model', 'file_size_str', 'unzip_password',
            'file_count', 'package_file', 'package_size', 'created_at',
        ]].copy()
        df_posts_shown['dbid'] = df_posts_shown['dbid'].map(int)
        df_posts_shown = df_posts_shown.replace(np.nan, '')
        print(f'{plural_word(len(df_posts), "post")} in total. '
              f'Only {plural_word(len(df_posts_shown), "post")} shown.', file=f)
        print('', file=f)
        print(df_posts_shown.to_markdown(index=False), file=f)
        print('', file=f)

        print('## Files', file=f)
        print('', file=f)
        df_files_shown = df_files[:50].replace(np.nan, '')
        print(f'{plural_word(len(df_files), "file")} in total, '
              f'{plural_word(len(df_files_shown), "file")} shown.', file=f)
        print('', file=f)
        print('', file=f)
        print(df_files_shown.to_markdown(index=False), file=f)
        print('', file=f)

        for title, noun, df in [('Tags', 'tag', df_tags), ('Categories', 'category', df_categories)]:
            print(f'## {title}', file=f)
            print('', file=f)
            df_shown = df[:50].replace(np.nan, '')
            print(f'{plural_word(len(df), noun)} in total, {plural_word(len(df_shown), noun)} shown.', file=f)
            print('', file=f)
            print('', file=f)
            print(df_shown.to_markdown(index=False), file=f)
            print('', file=f)


def write_tables(directory: str, df_posts: pd.DataFrame, df_files: pd.DataFrame,
                 tags: Dict[str, Dict], categories: Dict[str, Dict]) -> None:
    df_posts = _with_sorted_download_urls(df_posts).sort_values(by=['dbid'], ascending=False)
    df_posts.to_parquet(os.path.join(directory, 'posts.parquet'), engine='pyarrow', index=False)

    df_files = df_files.sort_values(by=['id'], ascending=[False])
    df_files.to_parquet(os.path.join(directory, 'files.parquet'), engine='pyarrow', index=False)

    df_tags = pd.DataFrame(list(tags.values()))
    if len(df_tags) > 0:
        df_tags = df_tags.sort_values(['count', 'id'], ascending=[False, True])
        df_tags.to_parquet(os.path.join(directory, 'tags.parquet'), engine='pyarrow', index=False)

    df_categories = pd.DataFrame(list(categories.values()))
    if len(df_categories) > 0:
        df_categories = df_categories.sort_values(['count', 'id'], ascending=[False, True])
        df_categories.to_parquet(os.path.join(directory, 'categories.parquet'), engine='pyarrow', index=False)

    _write_readme(os.path.join(directory, 'README.md'), df_posts, df_files, df_tags, df_categories)


def _count_terms(live: Dict[str, Dict], pending: Dict[str, Dict], origin: Dict[str, Dict], ids) -> None:
    """Count a post's terms into ``pending`` as new rows, leaving ``live`` as published until the commit lands."""
    for term_id in ids:
        row = pending.get(term_id) or live.get(term_id)
        row = dict(row) if row is not None else {**origin[term_id], 'count': 0}
        row['count'] += 1
        pending[term_id] = row


def sync(index_repository: str, repository: str, max_time_limit: Optional[float] = 5 * 60 * 60,
         deploy_span: float = 5 * 60, max_batch_bytes: int = _MAX_BATCH_BYTES, max_posts: Optional[int] = None,
         dry_run: bool = False, max_consecutive_failures: int = _MAX_CONSECUTIVE_FAILURES,
         max_dead_streak: int = _MAX_DEAD_STREAK, workers: int = 1, created_before: Optional[float] = None,
         created_after: Optional[float] = None):
    start_time = time.time()
    delete_detached_cache()
    # Without a timeout a stalled hub request blocks for about sixteen minutes, which is how a run was lost once.
    configure_hf_http_backend(timeout=300)
    upload_limiter = Limiter(Rate(120, int(math.ceil(Duration.HOUR * 1))), max_delay=1 << 32)

    hf_client = get_hf_client(hf_token=os.environ['HF_TOKEN'])
    hf_fs = get_hf_fs(hf_token=os.environ['HF_TOKEN'])
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        raise RuntimeError(f'Repository {repository!r} does not exist, refusing to create a fresh storage.')

    df_index = _read_table(hf_client, hf_fs, index_repository, 'posts.parquet')
    index_tags = _read_table(hf_client, hf_fs, index_repository, 'tags.parquet')
    index_categories = _read_table(hf_client, hf_fs, index_repository, 'categories.parquet')
    origin_tags = {r['id']: r for r in index_tags.to_dict('records')}
    origin_categories = {r['id']: r for r in index_categories.to_dict('records')}

    df_posts = _read_table_typed(hf_client, hf_fs, repository, 'posts.parquet')
    df_files = _read_table_typed(hf_client, hf_fs, repository, 'files.parquet')
    # An empty table has no columns to read ids from, and nothing to be kept in front of the new rows.
    df_posts = None if df_posts is not None and df_posts.empty else df_posts
    df_files = None if df_files is not None and df_files.empty else df_files
    tags_table = _read_table(hf_client, hf_fs, repository, 'tags.parquet')
    categories_table = _read_table(hf_client, hf_fs, repository, 'categories.parquet')
    live_tags = {r['id']: r for r in tags_table.to_dict('records')} if tags_table is not None else {}
    live_categories = {r['id']: r for r in categories_table.to_dict('records')} if categories_table is not None else {}
    exist_post_ids = set(df_posts['id']) if df_posts is not None else set()
    live_gone = read_gone(hf_client, hf_fs, repository)
    max_file_id = int(df_files['id'].max()) if df_files is not None and len(df_files) else 0
    logging.info(f'Storage {repository!r} holds {plural_word(len(exist_post_ids), "post")} and '
                 f'{plural_word(0 if df_files is None else len(df_files), "file")}, last file id {max_file_id}.')

    candidates = df_index.sort_values(by=['created_at', 'dbid'], ascending=[False, False]).to_dict('records')

    def _worth_asking(post: Dict) -> bool:
        # some host with a link this job can use that is not already recorded as gone for that link
        return any(usable_url(h, post) and not is_known_gone(live_gone.get(post['id']), h.name, usable_url(h, post))
                   for h in HOSTS)

    candidates = [r for r in candidates
                  if r['id'] not in exist_post_ids and any(usable_url(h, r) for h in HOSTS)
                  and (created_before is None or r['created_at'] < created_before)
                  and (created_after is None or r['created_at'] >= created_after)]
    recorded_dead = sum(1 for r in candidates if not _worth_asking(r))
    candidates = [r for r in candidates if _worth_asking(r)]
    logging.info(f'{plural_word(len(live_gone), "post")} recorded as gone, {recorded_dead} of them among the '
                 f'posts that would otherwise be tried, skipped for good.')
    other = len(df_index) - len(exist_post_ids & set(df_index['id'])) - len(candidates)
    logging.info(f'{plural_word(len(candidates), "post")} to try from {[h.name for h in HOSTS]!r}; '
                 f'{plural_word(other, "other post")} not stored and not fetchable here.')

    state = {'last_deploy': None, 'batch_bytes': 0, 'packages': 0, 'files': 0}
    pending_posts: List[Dict] = []
    pending_files: List[Dict] = []
    pending_tags: Dict[str, Dict] = {}
    pending_categories: Dict[str, Dict] = {}
    pending_gone: Dict[str, Dict] = {}
    disabled: Set[str] = set()
    counts = {'stored': 0, 'failed': 0, 'unavailable': 0, 'deferred': 0}
    failed_ids: List[str] = []

    strikes: Dict[str, int] = {}
    with TemporaryDirectory() as batch_dir, TemporaryDirectory() as staging_root, \
            ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        def _clean_batch():
            for name in os.listdir(batch_dir):
                path = os.path.join(batch_dir, name)
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

        def _note_gone(post: Dict, gone_hosts: Optional[Dict[str, Dict]]) -> None:
            for host_name, verdict in (gone_hosts or {}).items():
                host = next(h for h in HOSTS if h.name == host_name)
                dbid = _to_none(post.get('dbid'))
                entry = pending_gone.setdefault(post['id'], {'dbid': None if dbid is None else int(dbid), 'hosts': {}})
                entry['hosts'][host_name] = {'url': usable_url(host, post), 'kind': verdict['kind'],
                                             'reason': verdict['reason'], 'at': int(time.time())}

        def _deploy(force: bool = False):
            nonlocal df_posts, df_files
            if not pending_posts and not pending_gone:
                return
            due = state['last_deploy'] is None or time.time() - state['last_deploy'] >= deploy_span \
                or state['batch_bytes'] >= max_batch_bytes
            if not (force or due):
                return

            # Everything consumed so far, packages, rows and dead-link verdicts alike, goes up in this one
            # commit, and the live state advances to exactly that: a post's package and what was learned
            # about its links can never be split across two commits. The two-million-row tables are only
            # rewritten when there is a package to add; a commit that only records dead links carries the
            # small gone file alone.
            if pending_posts:
                new_posts = pd.DataFrame(pending_posts)
                new_files = pd.DataFrame(pending_files)
                all_posts = new_posts if df_posts is None else pd.concat([df_posts, new_posts], ignore_index=True)
                all_files = new_files if df_files is None else pd.concat([df_files, new_files], ignore_index=True)
                write_tables(batch_dir, all_posts, all_files, {**live_tags, **pending_tags},
                             {**live_categories, **pending_categories})
            if pending_gone:
                write_gone(batch_dir, merge_gone(live_gone, pending_gone))
            gone_note = f'record {plural_word(len(pending_gone), "post")} as gone'
            if pending_posts:
                message = f'Add {plural_word(state["packages"], "package")}, with {plural_word(state["files"], "file")}'
                message += f', {gone_note}' if pending_gone else ''
            else:
                message = gone_note.capitalize()
            if dry_run:
                logging.info(f'[dry-run] Would upload: {message}, {state["batch_bytes"]} bytes.')
            else:
                upload_limiter.try_acquire('hf upload limit')
                safe_upload_directory_as_directory(
                    repo_id=repository,
                    repo_type='dataset',
                    local_directory=batch_dir,
                    path_in_repo='.',
                    message=message,
                    hf_token=os.environ['HF_TOKEN'],
                )
            # The batch is committed: its rows become part of the live tables and its files leave the disk.
            if pending_posts:
                df_posts, df_files = all_posts, all_files
                live_tags.update(pending_tags)
                live_categories.update(pending_categories)
            live_gone.update(merge_gone(live_gone, pending_gone))
            for rows in (pending_posts, pending_files):
                rows.clear()
            pending_tags.clear()
            pending_categories.clear()
            pending_gone.clear()
            state.update(last_deploy=time.time(), batch_bytes=0, packages=0, files=0)
            _clean_batch()

        dead_streak, fail_streak, cooldowns, deferred_streak = 0, 0, 0, 0
        todo, inflight, submitted, no_more = iter(candidates), deque(), 0, False
        requeue, waiting = deque(), []  # waiting: posts deferred since the last pause, asked again after it

        def _submit_more():
            nonlocal submitted, no_more
            while not no_more and len(inflight) < max(workers, 1):
                if max_time_limit is not None and start_time + max_time_limit < time.time():
                    logging.info('Time budget used up, no new posts will be started.')
                    no_more = True
                elif max_posts is not None and submitted >= max_posts:
                    no_more = True
                else:
                    post = requeue.popleft() if requeue else next(todo, None)
                    if post is None:
                        no_more = True
                        continue
                    submitted += 1
                    logging.info(f'Post {post["id"]!r} (attempt {submitted}, {counts["stored"]} stored, '
                                 f'{counts["unavailable"]} gone, {counts["deferred"]} deferred, '
                                 f'{counts["failed"]} failed) ...')
                    inflight.append((post, pool.submit(process_post, post, staging_root, disabled, strikes,
                                                       live_gone.get(post['id']))))

        def _abandon():
            for _, future in inflight:
                future.cancel()
            inflight.clear()

        while True:
            # Checked on every turn, not only after a post is stored: a long stretch of dead links stores
            # nothing, and packages that are already built must not wait for the next stored post to be committed.
            _deploy()
            all_set_aside = bool(HOSTS) and all(host.name in disabled for host in HOSTS)
            stalled = bool(disabled) and bool(waiting) and (max_posts is None or submitted < max_posts) and \
                (deferred_streak >= _DEFER_STREAK or no_more)
            if not inflight and ((all_set_aside and not no_more) or stalled):
                # Nothing can be fetched right now. Walking on would only defer the whole list in a few
                # seconds, so commit what is built, wait for the limits to ease and ask every host again.
                out_of_time = max_time_limit is not None and start_time + max_time_limit < time.time() + _COOLDOWN
                if cooldowns >= _MAX_COOLDOWNS or out_of_time:
                    logging.warning('The hosts are still refusing, nothing more can be stored in this run.')
                    break
                cooldowns += 1
                _deploy(force=True)
                logging.warning(f'Hosts are set aside, waiting {_COOLDOWN // 60} minutes before asking them '
                                f'again ({cooldowns}/{_MAX_COOLDOWNS}).')
                time.sleep(_COOLDOWN)
                disabled.clear()
                strikes.clear()
                requeue.extend(waiting)
                waiting.clear()
                no_more = False  # _submit_more puts it back at once if the budget or the list is used up
                deferred_streak = 0
                fail_streak = 0  # a new round: the refusals of the last one are not held against the next
            _submit_more()
            if not inflight:
                break
            # Whichever post finishes first is taken first: waiting on the oldest one would let a single slow
            # multi-gigabyte download hold back every verdict and package that is already done behind it.
            done, _ = wait([f for _, f in inflight], timeout=_WAIT_SLICE, return_when=FIRST_COMPLETED)
            if not done:
                continue
            entry = next(item for item in inflight if item[1] in done)
            inflight.remove(entry)
            post, future = entry
            try:
                result = future.result()
            except PackageDeferred as err:
                _note_gone(post, getattr(err, 'gone_hosts', None))
                logging.warning(f'Post {post["id"]!r} left for the next run - {err}')
                counts['deferred'] += 1
                waiting.append(post)
                deferred_streak += 1
                continue
            except PackageUnavailable as err:
                _note_gone(post, getattr(err, 'gone_hosts', None))
                logging.warning(f'Post {post["id"]!r} is gone - {err}')
                counts['unavailable'] += 1
                deferred_streak = 0
                dead_streak += 1
                fail_streak = 0
                if max_dead_streak and dead_streak >= max_dead_streak:
                    logging.info(f'{dead_streak} posts in a row with nothing alive, the rest of the list is '
                                 f'almost certainly dead too, stopping.')
                    _abandon()
                    break
                continue
            except Exception as err:
                _note_gone(post, getattr(err, 'gone_hosts', None))
                logging.error(f'Post {post["id"]!r} not stored - {err!r}')
                counts['failed'] += 1
                deferred_streak = 0
                failed_ids.append(post['id'])
                fail_streak += 1
                if fail_streak >= max_consecutive_failures:
                    _abandon()
                    _deploy(force=True)
                    raise RuntimeError(f'{fail_streak} posts in a row failed, giving up: '
                                       f'{failed_ids[-fail_streak:]!r}. Last error: {err!r}') from err
                continue
            dead_streak = fail_streak = deferred_streak = 0

            _note_gone(post, result['gone_hosts'])
            adopt_staged(result['stage'], batch_dir)
            file_ids = []
            for member, size in zip(result['members'], result['sizes']):
                max_file_id += 1
                _, ext = os.path.splitext(member)
                mimetype, _ = mimetypes.guess_type(member.lower())
                pending_files.append({
                    'id': max_file_id,
                    'filename': f'{max_file_id}{ext}',
                    'mimetype': mimetype,
                    'post_id': post['id'],
                    'package_file': result['package_file'],
                    'file_in_package': member,
                    'file_size': size,
                })
                file_ids.append(max_file_id)
            pending_posts.append({
                **post,
                'file_count': result['file_count'],
                'package_size': result['package_size'],
                'package_file': result['package_file'],
                'file_ids': json.dumps(file_ids),
            })
            _count_terms(live_tags, pending_tags, origin_tags, post['tags'])
            _count_terms(live_categories, pending_categories, origin_categories, post['categories'])
            exist_post_ids.add(post['id'])
            state['batch_bytes'] += result['package_size']
            state['packages'] += 1
            state['files'] += result['file_count']
            counts['stored'] += 1
            logging.info(f'Post {post["id"]!r} stored via {result["host"]}, '
                         f'{plural_word(result["file_count"], "file")}, {result["package_size"]} bytes.')
            _deploy()
        _deploy(force=True)

    logging.info(f'Done: {plural_word(counts["stored"], "post")} stored, {plural_word(counts["unavailable"], "post")} '
                 f'gone, {plural_word(counts["deferred"], "post")} deferred, '
                 f'{plural_word(counts["failed"], "post")} failed {failed_ids[:20]!r}, '
                 f'{plural_word(len(exist_post_ids), "post")} in storage.')
    if failed_ids:
        logging.warning('Failed posts stay out of storage and are tried again on the next run.')


def _epoch(day) -> Optional[float]:
    return day.replace(tzinfo=timezone.utc).timestamp() if day is not None else None


@click.command()
@click.option('-i', '--index-repository', type=str, envvar='REMOTE_REPOSITORY_AO',
              default='deepghs/asiaon_index', show_default=True, help='Index dataset repository to read posts from.')
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_AO_S1',
              default='deepghs/asiaon_raw_storage', show_default=True,
              help='Storage dataset repository. It must already exist.')
@click.option('-t', '--max-time-limit', type=duration_type(allow_none=True), default='5h', show_default=True,
              help='Stop starting new posts after this duration, leaving room for the batch in flight. '
                   'Use "none" to disable.')
@click.option('-d', '--deploy-span', type=duration_type(), default=5 * 60, show_default=True,
              help='Minimum interval between commits.')
@click.option('-B', '--max-batch-gib', type=float, default=6.0, show_default=True,
              help='Commit as soon as the batch holds this many GiB of packages.')
@click.option('-n', '--max-posts', type=int, default=None,
              help='Attempt at most this many posts, stored, gone or failed, for trial runs.')
@click.option('-w', '--workers', type=int, default=3, show_default=True,
              help='Posts downloaded, unpacked and repacked at the same time.')
@click.option('--created-before', type=click.DateTime(formats=['%Y-%m-%d']), default=None,
              help='Only posts created before this date (UTC). A sweep of the old posts, whose links are the ones '
                   'still dying, can be run ahead of the new ones this way.')
@click.option('--created-after', type=click.DateTime(formats=['%Y-%m-%d']), default=None,
              help='Only posts created on or after this date (UTC).')
@click.option('--max-dead-streak', type=int, default=_MAX_DEAD_STREAK, show_default=True,
              help='Stop after this many posts in a row with nothing alive. 0 walks every candidate, which is '
                   'what a one-off sweep of the old posts wants.')
@click.option('--dry-run', is_flag=True, default=False, help='Download, unpack and repack, but upload nothing.')
def cli(index_repository: str, repository: str, max_time_limit: Optional[float], deploy_span: float,
        max_batch_gib: float, max_posts: Optional[int], workers: int, created_before, created_after,
        max_dead_streak: int, dry_run: bool):
    """Store asiaon post packages."""
    logging.try_init_root(logging.INFO)
    sync(
        index_repository=index_repository,
        repository=repository,
        max_time_limit=max_time_limit,
        deploy_span=deploy_span,
        max_batch_bytes=int(max_batch_gib * 1024 ** 3),
        max_posts=max_posts,
        max_dead_streak=max_dead_streak,
        workers=workers,
        created_before=_epoch(created_before),
        created_after=_epoch(created_after),
        dry_run=dry_run,
    )


if __name__ == '__main__':
    cli()
