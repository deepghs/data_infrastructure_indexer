"""Package storage for cosplaytele posts.

Continues the published raw storage dataset exactly as the pyskeb jobs left it: one tar per post
under ``images/<md5(post id)[:2]>/<post id>.tar`` with a same-named ``.json`` index, plus
``posts.parquet``, ``files.parquet``, ``tags.parquet`` and ``README.md``. Rows already published are
never rewritten, file ids continue from the highest one in use, and one commit is made per post.

Which posts, and from where
===========================

The index lists every post with a column per download host. A post is a candidate when it has a
link on a host this job can fetch (see :mod:`inf.cosplaytele.hosts`) and is not stored yet; the
newest are taken first, so a run cut short by its time budget has still covered what is most recent.

For each candidate the hosts are tried in their fixed order. A post is stored from the first host
whose package downloads *and* unpacks; anything less leaves it out of storage altogether, so the
next run simply tries it again. Nothing is recorded about failures - a post is either in
``posts.parquet`` or it is not.

History
=======

This job only ever adds commits and never rewrites history: squashing the repository is a manual
decision, made by hand when wanted.

Disk
====

Each post is handled in its own temporary directory: download, unpack, repack, upload, delete. Only
the package being uploaded is ever on disk, so a runner needs room for roughly three copies of the
largest package (archive, unpacked files, tar), not for the dataset.
"""
import copy
import json
import math
import mimetypes
import os
import time
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import requests
from ditk import logging
from hbutils.encoding import md5
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.archive import archive_pack, archive_unpack
from hfutils.cache import delete_detached_cache
from hfutils.index import tar_get_index_info
from hfutils.operate import get_hf_client, get_hf_fs
from pyrate_limiter import Duration, Limiter, Rate

from inf.utils.duration import duration_type
from inf.utils.safe import safe_upload_directory_as_directory
from inf.utils.archive_safety import ArchiveUnsafe, check_archive_members, validate_tree
from .hosts import HOSTS, DownloadError, get_download_session
from .index import _read_table

mimetypes.add_type('image/webp', '.webp')

#: Posts in a row that may fail before the run gives up. A single post can fail for its own reasons,
#: such as a removed share, but a long unbroken run of failures means something systematic - a
#: rotated link key, a changed page, a check rejecting everything - and carrying on would download
#: gigabytes per post for nothing until the time budget ran out.
_MAX_CONSECUTIVE_FAILURES = 10


#: Attempts per host for a failure that is likely to be momentary. Each retry fetches a fresh signed
#: link, since the old one may have expired or been cut off.
_HOST_ATTEMPTS = 2


def _is_transient(err: Exception) -> bool:
    return isinstance(err, requests.RequestException) or (isinstance(err, DownloadError) and err.retryable)


class UnpackError(Exception):
    """The downloaded archive could not be turned into a usable directory of files."""


def _to_none(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def unpack_package(archive_file: str, directory: str, password: Optional[str]) -> int:
    """
    Unpack ``archive_file`` into ``directory`` and return the number of files.

    The member list is checked before extracting and the tree after, see
    :mod:`inf.utils.archive_safety`; a package only counts as unpacked when both pass and it
    produced at least one file.
    """
    try:
        check_archive_members(archive_file, password)
        archive_unpack(archive_file=archive_file, directory=directory, password=password)
        return validate_tree(directory)
    except ArchiveUnsafe as err:
        raise UnpackError(f'Refused unsafe archive - {err}') from err
    except Exception as err:
        raise UnpackError(f'Unpack failed - {err!r}') from err


def fetch_package(post: Dict, work_dir: str, session) -> Tuple[str, str]:
    """
    Download and unpack a post's package from the first host that delivers.

    :returns: The host used and the directory the files were unpacked into.
    :raises RuntimeError: When no host could deliver, with every failure in the message.
    """
    failures = []
    for host in HOSTS:
        url = _to_none(post.get(host.column))
        if not url:
            continue

        download_dir = os.path.join(work_dir, 'download', host.name)
        unpack_dir = os.path.join(work_dir, 'unpack', host.name)
        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(unpack_dir, exist_ok=True)
        for attempt in range(1, _HOST_ATTEMPTS + 1):
            archive_file = None
            try:
                archive_file = host.download(url, download_dir, session)
                logging.info(f'Unpacking {os.path.basename(archive_file)!r} ...')
                unpack_package(archive_file, unpack_dir, _to_none(post.get('unzip_password')))
                return host.name, unpack_dir
            except Exception as err:
                transient = _is_transient(err)
                logging.warning(f'Post {post["id"]!r} failed on {host.name} (attempt {attempt}/{_HOST_ATTEMPTS}): '
                                f'{err!r}')
                if not (transient and attempt < _HOST_ATTEMPTS):
                    failures.append(f'{host.name}: {err!r}')
                    break
                time.sleep(5)
            finally:
                # The archive is no longer needed either way; free the disk before the next attempt.
                if archive_file and os.path.exists(archive_file):
                    os.remove(archive_file)

    raise RuntimeError('; '.join(failures) if failures else 'no supported host link')


def build_package(post_id: str, unpack_dir: str, upload_dir: str) -> Tuple[str, int, int, List[str]]:
    """
    Repack ``unpack_dir`` into the post's tar and write its index.

    :returns: The tar's path in the repository, its size, its file count and the member names.
    """
    package_file = f'images/{md5(post_id.encode())[:2]}/{post_id}.tar'
    tar_file = os.path.join(upload_dir, *package_file.split('/'))
    os.makedirs(os.path.dirname(tar_file), exist_ok=True)
    archive_pack('tar', directory=unpack_dir, archive_file=tar_file)

    meta_data = tar_get_index_info(tar_file, 1 << 20, with_hash=True)
    with open(os.path.splitext(tar_file)[0] + '.json', 'w') as f:
        json.dump(meta_data, f)
    return package_file, os.path.getsize(tar_file), len(meta_data['files']), list(meta_data['files'].keys())


def _write_readme(file: str, df_posts: pd.DataFrame, df_files: pd.DataFrame, df_tags: pd.DataFrame):
    with open(file, 'w') as f:
        print('---', file=f)
        print('license: other', file=f)
        print('---', file=f)
        print('', file=f)
        print('## Records', file=f)
        print('', file=f)
        df_posts_shown = df_posts[:50][['id', 'title', 'page_url', 'tags', 'unzip_password',
                                        'file_count', 'package_file', 'package_size', 'created_at']]
        df_posts_shown = df_posts_shown.replace(np.nan, '')
        print(f'{plural_word(len(df_posts), "record")} in total. '
              f'Only {plural_word(len(df_posts_shown), "record")} shown.', file=f)
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

        print('## Tags', file=f)
        print('', file=f)
        df_tags_shown = df_tags[:50].replace(np.nan, '')
        print(f'{plural_word(len(df_tags), "tag")} in total, '
              f'{plural_word(len(df_tags_shown), "tag")} shown.', file=f)
        print('', file=f)
        print('', file=f)
        print(df_tags_shown.to_markdown(index=False), file=f)
        print('', file=f)


def write_tables(directory: str, posts: List[Dict], files: List[Dict], d_tags: Dict) -> None:
    df_posts = pd.DataFrame(posts).sort_values(by=['created_at', 'id'], ascending=[False, False])
    df_posts.to_parquet(os.path.join(directory, 'posts.parquet'), engine='pyarrow', index=False)

    df_files = pd.DataFrame(files).sort_values(by=['id'], ascending=[False])
    df_files.to_parquet(os.path.join(directory, 'files.parquet'), engine='pyarrow', index=False)

    df_tags = pd.DataFrame(list(d_tags.values()))
    if len(df_tags) > 0:
        df_tags = df_tags.sort_values(['count', 'tag'], ascending=[False, True])
        df_tags.to_parquet(os.path.join(directory, 'tags.parquet'), engine='pyarrow', index=False)

    _write_readme(os.path.join(directory, 'README.md'), df_posts, df_files, df_tags)


def _tag_updates(post: Dict, d_tags: Dict, d_origin_tags: Dict) -> Dict:
    """The tag rows this post changes, as new dicts so the live table is untouched until commit."""
    updates = {}
    for token in (tuple(ts.split('/', maxsplit=1)) for ts in json.loads(post['tags'])):
        origin = d_origin_tags.get(token)
        current = updates.get(token) or d_tags.get(token)
        if current is None:
            current = copy.deepcopy(origin) if origin else {
                'tag': token[1], 'name': token[1], 'type': token[0], 'category': None,
            }
            current['count'] = 0
        current = dict(current)
        count = current['count']
        if origin:
            current.update(origin)
        current['count'] = count + 1
        updates[token] = current
    return updates


def sync(index_repository: str, repository: str, max_time_limit: Optional[float] = 5 * 60 * 60,
         hf_upload_interval: float = 30.0, max_posts: Optional[int] = None, dry_run: bool = False,
         max_consecutive_failures: int = _MAX_CONSECUTIVE_FAILURES):
    start_time = time.time()
    delete_detached_cache()
    upload_limiter = Limiter(Rate(1, int(math.ceil(Duration.SECOND * hf_upload_interval))), max_delay=1 << 32)

    hf_client = get_hf_client(hf_token=os.environ['HF_TOKEN'])
    hf_fs = get_hf_fs(hf_token=os.environ['HF_TOKEN'])
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        raise RuntimeError(f'Repository {repository!r} does not exist, refusing to create a fresh storage.')

    df_index = _read_table(hf_client, hf_fs, index_repository, 'posts.parquet')
    df_index_tags = _read_table(hf_client, hf_fs, index_repository, 'tags.parquet')
    d_origin_tags = {(r['type'], r['tag']): r for r in df_index_tags.to_dict('records')}

    df_posts = _read_table(hf_client, hf_fs, repository, 'posts.parquet')
    df_files = _read_table(hf_client, hf_fs, repository, 'files.parquet')
    df_tags = _read_table(hf_client, hf_fs, repository, 'tags.parquet')
    posts = df_posts.to_dict('records') if df_posts is not None else []
    files = df_files.to_dict('records') if df_files is not None else []
    d_tags = {(r['type'], r['tag']): r for r in df_tags.to_dict('records')} if df_tags is not None else {}
    exist_post_ids = {r['id'] for r in posts}
    max_file_id = max((int(r['id']) for r in files), default=0)
    logging.info(f'Storage {repository!r} holds {plural_word(len(posts), "post")} and '
                 f'{plural_word(len(files), "file")}, last file id {max_file_id}.')

    candidates = df_index.sort_values(by=['created_at', 'id'], ascending=[False, False]).to_dict('records')
    candidates = [r for r in candidates
                  if r['id'] not in exist_post_ids and any(_to_none(r.get(h.column)) for h in HOSTS)]
    unsupported = len(df_index) - len(exist_post_ids & set(df_index['id'])) - len(candidates)
    logging.info(f'{plural_word(len(candidates), "post")} to store from {[h.name for h in HOSTS]!r}; '
                 f'{plural_word(max(unsupported, 0), "other post")} not stored and not fetchable here.')

    session = get_download_session()
    stored, failed, streak = 0, [], 0
    for post in candidates:
        if max_time_limit is not None and start_time + max_time_limit < time.time():
            logging.info('Time budget used up, stopping.')
            break
        if max_posts is not None and stored + len(failed) >= max_posts:
            break

        logging.info(f'Post {post["id"]!r} ({stored + len(failed) + 1}/{len(candidates)}) ...')
        with TemporaryDirectory() as td:
            upload_dir = os.path.join(td, 'upload')
            os.makedirs(upload_dir)
            try:
                host_name, unpack_dir = fetch_package(post, os.path.join(td, 'work'), session)
                package_file, package_size, file_count, members = build_package(post['id'], unpack_dir, upload_dir)
            except Exception as err:
                logging.error(f'Post {post["id"]!r} not stored - {err!r}')
                failed.append(post['id'])
                streak += 1
                if streak >= max_consecutive_failures:
                    raise RuntimeError(f'{streak} posts in a row failed, giving up: {failed[-streak:]!r}. '
                                       f'Last error: {err!r}') from err
                continue
            streak = 0

            # Everything this post adds is worked out on copies and only merged once the commit lands.
            new_files, file_ids, next_id = [], [], max_file_id
            for member in members:
                next_id += 1
                _, ext = os.path.splitext(member)
                mimetype, _ = mimetypes.guess_type(member.lower())
                new_files.append({
                    'id': next_id,
                    'filename': f'{next_id}{ext}',
                    'mimetype': mimetype,
                    'post_id': post['id'],
                    'package_file': package_file,
                    'file_in_package': member,
                    'file_size': os.path.getsize(os.path.join(unpack_dir, member)),
                })
                file_ids.append(next_id)
            new_post = {
                **post,
                'file_count': file_count,
                'package_size': package_size,
                'package_file': package_file,
                'file_ids': json.dumps(file_ids),
            }
            tag_updates = _tag_updates(post, d_tags, d_origin_tags)

            write_tables(upload_dir, [*posts, new_post], [*files, *new_files], {**d_tags, **tag_updates})
            if dry_run:
                logging.info(f'[dry-run] Would upload {package_file!r} from {host_name}, '
                             f'{plural_word(file_count, "file")}, {package_size} bytes.')
            else:
                upload_limiter.try_acquire('hf upload limit')
                safe_upload_directory_as_directory(
                    repo_id=repository,
                    repo_type='dataset',
                    local_directory=upload_dir,
                    path_in_repo='.',
                    message=f'Add package {post["id"]!r}, with {plural_word(file_count, "file")}',
                    hf_token=os.environ['HF_TOKEN'],
                )
            posts.append(new_post)
            files.extend(new_files)
            d_tags.update(tag_updates)
            exist_post_ids.add(post['id'])
            max_file_id = next_id
            stored += 1
            logging.info(f'Post {post["id"]!r} stored via {host_name}, '
                         f'{plural_word(file_count, "file")}, {package_size} bytes.')

    logging.info(f'Done: {plural_word(stored, "post")} stored, {plural_word(len(failed), "post")} failed '
                 f'{failed[:20]!r}, {plural_word(len(posts), "post")} in storage.')
    if failed:
        logging.warning('Failed posts stay out of storage and are tried again on the next run.')


@click.command()
@click.option('-i', '--index-repository', type=str, envvar='REMOTE_REPOSITORY_CTEL',
              default='deepghs/cosplaytele_index', show_default=True,
              help='Index dataset repository to read posts from.')
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_CTEL_S1',
              default='deepghs/cosplaytele_raw_storage', show_default=True,
              help='Storage dataset repository. It must already exist.')
@click.option('-t', '--max-time-limit', type=duration_type(allow_none=True), default='5h',
              show_default=True, help='Stop starting new posts after this duration, leaving room for '
                                      'the package in flight. Use "none" to disable.')
@click.option('-u', '--upload-interval', type=float, default=30.0, show_default=True,
              help='Minimum seconds between commits.')
@click.option('-n', '--max-posts', type=int, default=None,
              help='Attempt at most this many posts, stored or failed, for trial runs.')
@click.option('--dry-run', is_flag=True, default=False,
              help='Download, unpack and repack, but upload nothing.')
def cli(index_repository: str, repository: str, max_time_limit: Optional[float], upload_interval: float,
        max_posts: Optional[int], dry_run: bool):
    """Store cosplaytele post packages."""
    logging.try_init_root(logging.INFO)
    sync(
        index_repository=index_repository,
        repository=repository,
        max_time_limit=max_time_limit,
        hf_upload_interval=upload_interval,
        max_posts=max_posts,
        dry_run=dry_run,
    )


if __name__ == '__main__':
    cli()
