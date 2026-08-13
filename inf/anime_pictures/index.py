"""Index sync for anime-pictures.net.

Ported from the pyskeb prototype, with three things changed after checking the live API rather
than trusting the old code.

``cloudscraper`` is gone. A ``curl_cffi`` session with a browser fingerprint is admitted on the
handshake, so the challenge solver and its proxy-pool fallback are both unnecessary.

Two fields the prototype read no longer exist on the post object. ``position`` is simply absent
now. ``redirect_id`` did not disappear, it moved: a merged post answers ``410`` with
``{"redirect": {"post_id": N}}`` in the body, so the value is still recoverable and this keeps
recording it - see :func:`_fetch_post`.

The stored table is held as an Arrow table rather than being converted to a list of dicts. At
643k rows that conversion is survivable where it was not for a larger dataset, but it buys
nothing and costs several times the memory.

Layout published to the target repository
=========================================

::

    anime_pictures.parquet   one row per post
    tags.parquet             one row per tag, with a usage count
    README.md                statistics and preview

Pacing
======

The site publishes roughly 420 posts a week, so a run has very little to do in the steady
state. ``--no-recent`` holds back posts younger than its threshold: tags and scores keep moving
for a while after publication, and re-fetching a post is not something this job does, so it is
better to let a post settle before recording it.

Publication rate
================

About 430 posts a week, measured 2026-08-13 by comparing the newest page's ``pubtime`` against
pages far behind it, at 80 posts a page: 1,600 posts back spans 24.97 days (448/week) and 8,000
posts back spans 132.59 days (422/week). Two depths an order of magnitude apart agreeing is what
makes this trustworthy - a shallow sample would only show the last few hours.
"""
import datetime
import json
import math
import mimetypes
import os
import time
from typing import List, Optional
from urllib.parse import quote_plus

import click
import dateparser
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ditk import logging
from hbutils.color import Color
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.utils import number_to_tag

from inf.utils.duration import duration_type
from inf.utils.safe import safe_hf_hub_download, safe_upload_directory_as_directory
from inf.utils.session import REQUEST_ERRORS
from .base import __api_url__, get_anime_pictures_session

mimetypes.add_type('image/webp', '.webp')

#: Tag categories, as the site numbers them.
_TAG_TYPES = {
    0: 'unknown',
    1: 'character',
    2: 'reference',
    3: 'copyright (product)',
    4: 'author',
    5: 'game copyright',
    6: 'other copyright',
    7: 'object',
}

#: Columns of ``anime_pictures.parquet``, in the order the prototype wrote them. Kept identical
#: so the published table stays a continuation of the existing one rather than a new shape.
_TABLE_COLUMNS = [
    'id', 'width', 'height', 'file_size', 'mimetype', 'filename',
    'md5', 'md5_pixels', 'erotics', 'ext', 'status', 'status_type', 'redirect_id',
    'spoiler', 'have_alpha', 'color', 'artifacts_degree', 'smooth_degree',
    'tags_count', 'tags',
    'small_preview_url', 'medium_preview_url', 'big_preview_url', 'file_url',
    'score', 'score_number', 'downloads', 'favorites', 'position',
    'user_id', 'user_name', 'moderator_id', 'moderator_name',
    'published_at', 'created_at',
]

#: Statuses meaning the post will never be retrievable under this id.
_GONE_STATUS = (403, 404, 410)


def _as_int(value) -> Optional[int]:
    """
    Coerce to int where that is lossless, otherwise None.

    :returns: Integer, or None.
    :rtype: Optional[int]
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if not math.isnan(value) and float(value).is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_float(value) -> Optional[float]:
    """
    Coerce to float, otherwise None.

    Separate from :func:`_as_int` because several of the site's quality metrics are genuinely
    fractional - ``artefacts_degree`` comes back as 8.122037921611959 - and putting them through
    an integer coercion silently nulls every value that is not a whole number.

    :returns: Float, or None.
    :rtype: Optional[float]
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_timestamp(value) -> Optional[float]:
    """
    Parse one of the site's datetime strings into a POSIX timestamp.

    :returns: Timestamp, or None when the value cannot be read.
    :rtype: Optional[float]
    """
    if not value:
        return None
    parsed = dateparser.parse(value)
    return parsed.timestamp() if parsed else None


def _hex_color(rgb) -> Optional[str]:
    """
    Render the site's ``[r, g, b]`` triple as a hex string.

    :returns: Hex colour, or None when the triple is unusable.
    :rtype: Optional[str]
    """
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        return None
    try:
        r, g, b = (float(v) / 255.0 for v in rgb)
    except (TypeError, ValueError):
        return None
    return str(Color.from_rgb(r, g, b))


class PostGone(Exception):
    """Raised when a post cannot be retrieved and never will be under this id."""

    def __init__(self, post_id: int, status: int, redirect_id: Optional[int] = None):
        Exception.__init__(self, f'Post {post_id} is gone (HTTP {status})'
                                 + (f', merged into {redirect_id}' if redirect_id else ''))
        self.post_id = post_id
        self.status = status
        self.redirect_id = redirect_id


def _fetch_post(session, post_id: int) -> dict:
    """
    Fetch one post.

    A merged post answers 410 with ``{"redirect": {"post_id": N}}``. That body is the only place
    the old ``redirect_id`` field survives, so it is read out and carried on the exception rather
    than discarded with the response.

    :param session: Session to fetch with.
    :param post_id: Post to fetch.
    :type post_id: int
    :returns: The decoded post payload.
    :rtype: dict
    :raises PostGone: When the post is forbidden, missing or merged away.
    """
    resp = session.get(f'{__api_url__}/api/v3/posts/{post_id}')
    if resp.status_code in _GONE_STATUS:
        redirect_id = None
        try:
            body = resp.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            redirect_id = _as_int((body.get('redirect') or {}).get('post_id'))
        raise PostGone(post_id, resp.status_code, redirect_id)
    resp.raise_for_status()
    return resp.json()


def build_row(item: dict, redirect_id: Optional[int] = None) -> dict:
    """
    Turn an API payload into a table row.

    Field types are coerced rather than trusted. ``status_type`` is optional on some posts and
    ``position`` no longer exists at all, so both are allowed to be None instead of raising.

    :param item: Payload from ``/api/v3/posts/<id>``.
    :type item: dict
    :param redirect_id: Post this one was merged into, when known.
    :type redirect_id: Optional[int]
    :returns: A row keyed by :data:`_TABLE_COLUMNS`.
    :rtype: dict
    """
    post = item.get('post') or {}
    filename = item.get('file_url')
    url = f'{__api_url__}/pictures/download_image/{quote_plus(filename)}' if filename else None
    mimetype, _ = mimetypes.guess_type(url) if url else (None, None)
    user = item.get('user') or {}
    moderator = item.get('moderator') or {}
    return {
        'id': _as_int(post.get('id')),
        'width': _as_int(post.get('width')),
        'height': _as_int(post.get('height')),
        'file_size': _as_int(post.get('size')),
        'mimetype': mimetype,
        'filename': filename,

        'md5': post.get('md5'),
        'md5_pixels': post.get('md5_pixels'),
        'erotics': _as_int(post.get('erotics')),
        'ext': post.get('ext'),
        'status': _as_int(post.get('status')),
        # Optional on some posts, absent entirely on others; null in 41% of the
        # stored rows already. Stored as double.
        'status_type': _as_float(post.get('status_type')),
        # No longer a field on the post object. Recoverable only from a 410 body, and
        # already null in every stored row bar four, so nothing of substance is lost.
        'redirect_id': _as_float(redirect_id),
        'spoiler': post.get('spoiler'),
        'have_alpha': post.get('have_alpha'),
        'color': _hex_color(post.get('color')),
        # Fractional in the API and stored as double; see _as_float.
        'artifacts_degree': _as_float(post.get('artefacts_degree')),
        'smooth_degree': _as_float(post.get('smooth_degree')),

        'tags_count': _as_int(post.get('tags_count')),
        'tags': json.dumps([
            tag_item['tag']['tag']
            for tag_item in (item.get('tags') or [])
            if isinstance(tag_item.get('tag'), dict) and tag_item['tag'].get('tag')
        ]),

        'small_preview_url': post.get('small_preview'),
        'medium_preview_url': post.get('medium_preview'),
        'big_preview_url': post.get('big_preview'),
        'file_url': url,

        'score': _as_int(post.get('score')),
        'score_number': _as_int(post.get('score_number')),
        'downloads': _as_int(post.get('download_count')),
        'favorites': len(item.get('favorites_users') or []),
        # Dropped by the API, and null in all 642,989 stored rows, so it never carried
        # anything. Kept so the table shape does not change.
        'position': item.get('position'),

        'user_id': _as_int(user.get('id')),
        'user_name': user.get('name'),
        'moderator_id': _as_int(moderator.get('id')) if moderator else None,
        'moderator_name': moderator.get('name') if moderator else None,

        'published_at': _as_timestamp(post.get('pubtime')),
        'created_at': _as_timestamp(post.get('datetime')),
    }


def sync(repository: str, max_time_limit: Optional[float] = 5 * 60 * 60,
         upload_time_span: float = 30, deploy_span: float = 15 * 60,
         no_recent: float = 60 * 60 * 24 * 15, max_empty_pages: int = 10,
         proxy_pool: Optional[str] = None):
    """
    Sync anime-pictures post metadata into the target Hugging Face dataset repository.

    :param repository: Target dataset repository.
    :type repository: str
    :param max_time_limit: Stop fetching after this many seconds, leaving room for the final
        upload. None disables the limit.
    :type max_time_limit: Optional[float]
    :param upload_time_span: Minimum seconds between uploads.
    :type upload_time_span: float
    :param deploy_span: Minimum seconds between commits.
    :type deploy_span: float
    :param no_recent: Skip posts published more recently than this. Tags and scores keep moving
        for a while after publication and this job never revisits a post.
    :type no_recent: float
    :param max_empty_pages: Stop after this many consecutive listing pages with nothing new.
    :type max_empty_pages: int
    :param proxy_pool: Optional proxy URL; not needed in practice.
    :type proxy_pool: Optional[str]
    """
    from pyrate_limiter import Duration, Limiter, Rate

    start_time = time.time()
    delete_detached_cache()
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    session = get_anime_pictures_session(proxy_pool=proxy_pool)

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(f'datasets/{repository}/.gitattributes', os.linesep.join(attr_lines))

    # Held as Arrow, never as a list of dicts: a dict per row turns every column into millions
    # of boxed Python objects for no benefit.
    if hf_client.file_exists(repo_id=repository, repo_type='dataset',
                             filename='anime_pictures.parquet'):
        base_table = pq.read_table(safe_hf_hub_download(
            hf_client, repo_id=repository, repo_type='dataset',
            filename='anime_pictures.parquet'))
        exist_ids = set(base_table.column('id').to_pylist())
        logging.info(f'Existing table loaded, {plural_word(base_table.num_rows, "row")}.')
    else:
        base_table = None
        exist_ids = set()

    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='tags.parquet'):
        df_tags = pd.read_parquet(safe_hf_hub_download(
            hf_client, repo_id=repository, repo_type='dataset',
            filename='tags.parquet')).replace(np.NaN, None)
        for column in ('parent', 'alias'):
            if column in df_tags.columns:
                df_tags[column] = df_tags[column].map(lambda x: int(float(x)) if x else None)
        d_tags = {item['id']: item for item in df_tags.to_dict('records')}
        logging.info(f'Existing tags loaded, {plural_word(len(d_tags), "tag")}.')
    else:
        d_tags = {}

    records: List[dict] = []
    stats = {'ok': 0, 'gone': 0, 'failed': 0, 'recent': 0}
    _total_count = base_table.num_rows if base_table is not None else 0
    _last_update, has_update = None, False

    def _merged_table():
        """Combine the stored table with this run's rows, newest id first."""
        if not records:
            return base_table
        fresh = pa.Table.from_pylist(
            [{column: row.get(column) for column in _TABLE_COLUMNS} for row in records],
            schema=base_table.schema if base_table is not None else None)
        if base_table is None:
            return fresh.sort_by([('id', 'descending')])
        # from_pylist infers types from a few rows, so a column that is all-None in this batch
        # comes back null-typed and refuses to concatenate; the stored schema stays authoritative.
        return pa.concat_tables([base_table, fresh.cast(base_table.schema)]) \
            .sort_by([('id', 'descending')])

    def _deploy(force: bool = False):
        nonlocal _last_update, has_update, _total_count
        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            table = _merged_table()
            pq.write_table(table, os.path.join(td, 'anime_pictures.parquet'))
            total_rows = table.num_rows
            preview = table.slice(0, 50).to_pandas()
            del table

            df_tags = pd.DataFrame(list(d_tags.values()))
            df_tags = df_tags.sort_values(['count', 'type'], ascending=[False, True])
            df_tags.to_parquet(os.path.join(td, 'tags.parquet'), index=False)

            _write_readme(os.path.join(td, 'README.md'), total_rows=total_rows,
                          preview=preview, df_tags=df_tags)

            limiter.try_acquire('hf upload limit')
            added = total_rows - _total_count
            logging.info(f'UPLOAD starting - {plural_word(added, "new post")}, '
                         f'{total_rows:,} rows in total.')
            upload_started = time.time()
            safe_upload_directory_as_directory(
                repo_id=repository, repo_type='dataset', local_directory=td, path_in_repo='.',
                message=f'Add {plural_word(added, "new post")} into index',
            )
            logging.info(f'UPLOAD done in {time.time() - upload_started:.0f}s.')
            has_update = False
            _last_update = time.time()
            _total_count = total_rows

    def _ping_tags(item: dict):
        """Fold a post's tags into the tag table, keeping a usage count."""
        for tag_item in item.get('tags') or []:
            tag = tag_item.get('tag')
            if not isinstance(tag, dict) or tag.get('id') is None:
                continue
            entry = d_tags.setdefault(tag['id'], {**tag, 'count': 0})
            count = entry.get('count', 0)
            entry.update(tag)
            entry['count'] = count + 1

    def _listing(page_no: int) -> dict:
        resp = session.get(f'{__api_url__}/api/v3/posts', params={
            'page': str(page_no), 'order_by': 'date', 'ldate': '0', 'lang': 'en',
        })
        resp.raise_for_status()
        return resp.json()

    first = _listing(0)
    max_pages = first['max_pages']
    posts_per_page = first['posts_per_page']
    logging.info(f'Site reports {first.get("posts_count"):,} posts across '
                 f'{max_pages} pages of {posts_per_page}; '
                 f'{plural_word(len(exist_ids), "id")} already indexed.')

    empty_pages = 0
    try:
        for page_no in range(0, max_pages):
            if max_time_limit is not None and start_time + max_time_limit < time.time():
                logging.info('Run deadline reached, stopping the walk.')
                break

            try:
                page = _listing(page_no)
            except REQUEST_ERRORS as err:
                logging.warning(f'Listing page {page_no} failed - {err!r}, moving on.')
                continue

            logging.info(f'Listing page {page_no}/{max_pages}: '
                         f'{len(page.get("posts") or [])} posts.')
            fresh_on_page = 0
            for post_item in page.get('posts') or []:
                if max_time_limit is not None and start_time + max_time_limit < time.time():
                    break
                post_id = _as_int(post_item.get('id'))
                if post_id is None or post_id in exist_ids:
                    continue

                published_at = _as_timestamp(post_item.get('pubtime'))
                if published_at and published_at + no_recent > time.time():
                    # Counted as fresh so the walk does not mistake the unsettled head of the
                    # site for having caught up.
                    fresh_on_page += 1
                    stats['recent'] += 1
                    logging.info(f'Post {post_id} too recent, held back.')
                    continue

                redirect_id = None
                try:
                    item = _fetch_post(session, post_id)
                except PostGone as err:
                    logging.info(f'Post {post_id} gone (HTTP {err.status})'
                                 + (f', merged into {err.redirect_id}' if err.redirect_id else '')
                                 + '.')
                    exist_ids.add(post_id)
                    stats['gone'] += 1
                    fresh_on_page += 1
                    continue
                except REQUEST_ERRORS as err:
                    logging.warning(f'Post {post_id} skipped - {err!r}.')
                    stats['failed'] += 1
                    continue

                records.append(build_row(item, redirect_id=redirect_id))
                _ping_tags(item)
                exist_ids.add(post_id)
                stats['ok'] += 1
                fresh_on_page += 1
                has_update = True
                # One line per record. Without it the fetch phase is silent for minutes at a
                # time and an healthy run is indistinguishable from a hung one.
                logging.info(f'Post {post_id} confirmed '
                             f'(page {page_no}/{max_pages}, {stats["ok"]} added this run).')
                _deploy()

            if fresh_on_page:
                empty_pages = 0
            else:
                empty_pages += 1
                if empty_pages >= max_empty_pages:
                    logging.info(f'Stopping: {empty_pages} consecutive pages with nothing new.')
                    break
    finally:
        _deploy(force=True)

    logging.info(f'Done. {stats["ok"]} added, {stats["gone"]} gone, '
                 f'{stats["failed"]} failed, {stats["recent"]} held back as too recent.')


def _write_readme(md_file: str, total_rows: int, preview: pd.DataFrame, df_tags: pd.DataFrame):
    """
    Render the dataset README.

    :param md_file: Destination path.
    :type md_file: str
    :param total_rows: Row count of the published table.
    :type total_rows: int
    :param preview: The newest rows, for the sample table.
    :type preview: pd.DataFrame
    :param df_tags: Tag table, already sorted.
    :type df_tags: pd.DataFrame
    """
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
        print('- ru', file=f)
        print('tags:', file=f)
        print('- art', file=f)
        print('- anime', file=f)
        print('- not-for-all-audiences', file=f)
        print('size_categories:', file=f)
        print(f'- {number_to_tag(total_rows)}', file=f)
        print('annotations_creators:', file=f)
        print('- no-annotation', file=f)
        print('source_datasets:', file=f)
        print('- anime-pictures', file=f)
        print('---', file=f)
        print('', file=f)

        print('## Records', file=f)
        print('', file=f)
        print(f'{plural_word(total_rows, "record")} in total, '
              f'last updated at `{current_time}`. '
              f'Only {plural_word(len(preview), "record")} shown.', file=f)
        print('', file=f)
        shown = preview[['id', 'width', 'height', 'file_size', 'mimetype', 'file_url']]
        print(shown.to_markdown(index=False), file=f)
        print('', file=f)

        print('## Tags', file=f)
        print('', file=f)
        print(f'{plural_word(len(df_tags), "tag")} in total.', file=f)
        print('', file=f)
        for type_id in sorted(set(df_tags['type'])):
            df_type = df_tags[df_tags['type'] == type_id][
                ['id', 'tag', 'tag_jp', 'tag_ru', 'type', 'count']]
            df_shown = df_type[:30].replace(np.NaN, '')
            print(f'These are the top {plural_word(len(df_shown), "tag")} '
                  f'({plural_word(len(df_type), "tag")} in total) '
                  f'of type `{_TAG_TYPES.get(type_id, type_id)} ({type_id})`:', file=f)
            print('', file=f)
            print(df_shown.to_markdown(index=False), file=f)
            print('', file=f)


@click.command(context_settings={'help_option_names': ['-h', '--help']},
               help='Sync anime-pictures.net post metadata into a Hugging Face dataset '
                    'repository. Walks the listing newest-first and stops once it has seen '
                    'enough consecutive pages with nothing new.')
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_AP', required=True,
              show_envvar=True,
              help='Target Hugging Face dataset repository to read from and write to.')
@click.option('-m', '--max-time-limit', type=duration_type(allow_none=True), default=5 * 60 * 60,
              show_default=True,
              help='Stop fetching after this duration, leaving room for the final upload. '
                   'Use none or unlimited to disable.')
@click.option('-u', '--upload-time-span', type=duration_type(), default=30, show_default=True,
              help='Minimum interval between upload batches.')
@click.option('-d', '--deploy-span', type=duration_type(), default=15 * 60, show_default=True,
              help='Minimum interval between commits.')
@click.option('-n', '--no-recent', type=duration_type(), default=60 * 60 * 24 * 15,
              show_default=True,
              help='Hold back posts published more recently than this. Tags and scores keep '
                   'moving after publication and this job never revisits a post.')
@click.option('-E', '--max-empty-pages', type=int, default=10, show_default=True,
              help='Stop after this many consecutive listing pages with nothing new. Raise it '
                   'to walk past a stretch that is already indexed.')
@click.option('-p', '--proxy-pool', type=str, envvar='PP_AP', default=None, show_envvar=True,
              help='Optional proxy URL. Not needed in practice; the direct route is accepted.')
def cli(repository: str, max_time_limit: Optional[float], upload_time_span: float,
        deploy_span: float, no_recent: float, max_empty_pages: int, proxy_pool: Optional[str]):
    logging.try_init_root(logging.INFO)
    sync(
        repository=repository,
        max_time_limit=max_time_limit,
        upload_time_span=upload_time_span,
        deploy_span=deploy_span,
        no_recent=no_recent,
        max_empty_pages=max_empty_pages,
        proxy_pool=proxy_pool,
    )


if __name__ == '__main__':
    cli()
