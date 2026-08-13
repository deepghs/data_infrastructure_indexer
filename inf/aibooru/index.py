"""Index sync for aibooru.online.

Ported from the pyskeb prototype (``test/prepare/aib/index.py``). Three things changed.

Paging is by cursor. The prototype walked ``page=1,2,3...``, and the site refuses ``page`` past
1000 with ``PaginationExtension::PaginationError`` - at 200 posts a page that ceiling sits at
200,000 posts, and the site already holds 172,895. It would have stopped working within the year.
``page=b{id}`` means "before this id" and has no depth limit: take the lowest id from each page and
ask for the next one below it. Verified against the live site, including chaining a second hop.

Known posts are re-examined instead of skipped, so an edited row is corrected rather than frozen
at whatever it looked like when first seen. See :mod:`inf.utils.upsert` for the rules; the one
that matters is that a field arriving as ``None`` never overwrites a stored value.

The stored table is Arrow rather than a list of dicts, as elsewhere in this repository.

Layout published to the target repository
=========================================

::

    aibooru.parquet             one row per post, 28 columns
    index_tags.parquet          tag metadata, written by a separate job, untouched here
    index_tag_aliases.parquet   alias metadata, likewise

What is recorded, and what the count means
==========================================

Deleted posts are skipped, and that is a schema decision rather than a preference: the published
28 columns carry no ``is_deleted``, so a deleted post could not be marked as such if it were
stored. Recording them would mean adding a column, which is a change to the published structure
and should be decided deliberately rather than as a side effect of a port.

So the row count is measured against the site excluding deleted posts, not against the plain
total. As of 2026-08-13 ``counts/posts.json`` reports 172,895 either plainly or with
``status:any`` - it includes deleted posts - of which ``status:deleted`` is 26,349. Sampling the
600 newest posts found 58 deleted, close to a tenth. Audit against the deleted-excluding figure or
the gap will look ten times worse than it is.

Posts with no ``md5`` are skipped too. Those are uploads still being processed - one in the 600
sampled, and it was also deleted - and they are temporary: once the file lands, the post is simply
an id we have not seen and gets picked up normally. Neither kind counts towards the "nothing new"
counter, or a page carrying a couple of them would look like progress forever.

The prototype also held back anything newer than 15 days, to let tags and scores settle before
recording them. That is not carried over: the walk now revisits known posts every run and corrects
them in place, so there is no longer a cost to recording something early.

Expect a lot of updates, and know why
=====================================

The stored rows stop at 2025-08-24, and tags kept being edited after that. Re-fetching 25 rows at
random found 19 whose tags had changed - id 1100, for instance, gained ``aqua_eyes``, ``aqua_hair``
and ``alternate_breast_size_(larger)``. So roughly three quarters of the older rows carry stale
tags.

This does not mean a run rewrites three quarters of the table. The walk stops 20 pages after it
crosses into already-indexed territory, so only those pages get corrected on an ordinary run.
Refreshing the tags of everything already stored is a deliberate act: run with a very large
``--max-empty-pages`` so the walk continues to the bottom of the site.

Publication rate
================

About 950 posts a week, measured 2026-08-13 from ``counts/posts.json`` with an ``age`` filter,
which counts server-side and needs no sampling: 927 over the last week, 5,059 over the last month
(1,180/week) and 11,974 over three months (921/week). The three agree, so the figure is steady
rather than a burst.

Note the count includes deleted posts unless excluded - plain, ``status:any`` and the default all
report the same total, of which ``status:deleted`` was 26,349 of 172,895. This job records only
live posts, so compare against ``-status:deleted`` (146,543 at the same date) when auditing.
"""
import datetime
import math
import os
import time
from typing import List, Optional

import click
import dateparser
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client
from hfutils.utils import number_to_tag
from pyrate_limiter import Duration, Limiter, Rate

from inf.utils.duration import duration_type
from inf.utils.safe import safe_hf_hub_download, safe_upload_directory_as_directory
from inf.utils.session import REQUEST_ERRORS
from inf.utils.upsert import apply_updates
from inf.utils.upsert import row_signature as _row_signature
from inf.utils.upsert import table_signatures as _table_signatures
from .base import __site_url__, get_aibooru_session

#: Posts per request.
#:
#: The site caps this at 200 and does so *silently*: asking for 320, 321 or 1000 all return 200
#: rows with HTTP 200 and no warning. Requesting more than 200 therefore costs the same and gains
#: nothing, while looking like it worked.
_POSTS_PER_PAGE = 200

#: Rows buffered as dicts before folding into an Arrow chunk.
_PENDING_FLUSH = 20000

#: Attempts per page before giving up on the run.
#:
#: Cursor paging cannot skip a failed page the way an offset walk can: leaving ``before_id`` alone
#: would re-request the same page forever, and advancing it past the failure would silently drop
#: those posts. So a page is retried, and if it will not come, the run stops and reports where -
#: ``--start-below-id`` resumes from there without re-walking what is already indexed.
_PAGE_ATTEMPTS = 5

#: Fields whose change makes a stored row worth rewriting.
#:
#: Not every column. ``score``, ``up_score``, ``down_score``, ``fav_count`` and ``views`` drift on
#: their own, and ``updated_at`` moves whenever any of them do, so including any of them would mark
#: much of the table changed on every pass and rewrite it for nothing.
_UPDATE_TRIGGER_FIELDS = (
    'md5', 'file_url', 'large_file_url', 'preview_file_url',
    'file_ext', 'file_size', 'width', 'height',
    'tags', 'rating', 'source', 'parent_id', 'pixiv_id',
    'has_children', 'has_active_children', 'has_visible_children', 'has_large',
    'bit_flags',
)


def to_timestamp(value: Optional[str]) -> Optional[float]:
    """
    Parse an API timestamp the way the published table stores it: a float epoch.

    ``datetime.fromisoformat`` handles what this API sends (``2026-08-13T04:12:11.123-04:00``) and
    is far cheaper than a general parser, which matters across tens of thousands of rows.
    ``dateparser`` remains the fallback, and is what the prototype used, so anything it could read
    is still read.

    :param value: Timestamp as sent by the API.
    :type value: Optional[str]
    :returns: Seconds since the epoch, or None.
    :rtype: Optional[float]
    """
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        parsed = dateparser.parse(value)
        return parsed.timestamp() if parsed else None


def is_recordable(item: dict) -> bool:
    """
    Whether a post belongs in the table.

    Deleted posts are out because the published schema has no column to mark them with, and posts
    without an ``md5`` are still being processed and will be picked up on a later run.

    :param item: One entry from ``/posts.json``.
    :type item: dict
    :returns: True when the post should be stored.
    :rtype: bool
    """
    return not item.get('is_deleted') and bool(item.get('md5'))


def build_row(item: dict) -> dict:
    """
    Turn an API item into a table row.

    The column names are the prototype's and are kept exactly, including the three that differ
    from what the API calls them: ``tags`` is ``tag_string``, ``width`` is ``image_width`` and
    ``height`` is ``image_height``.

    :param item: One entry from ``/posts.json``.
    :type item: dict
    :returns: The row to store, in the published column set.
    :rtype: dict
    """
    return {
        'id': item.get('id'),
        'uploader_id': item.get('uploader_id'),
        'approver_id': item.get('approver_id'),
        'up_score': item.get('up_score'),
        'down_score': item.get('down_score'),
        'score': item.get('score'),
        'fav_count': item.get('fav_count'),
        'source': item.get('source'),
        'md5': item.get('md5'),
        'rating': item.get('rating'),
        'tags': item.get('tag_string'),
        'file_ext': item.get('file_ext'),
        'file_size': item.get('file_size'),
        'width': item.get('image_width'),
        'height': item.get('image_height'),
        'parent_id': item.get('parent_id'),
        'has_children': item.get('has_children'),
        'has_active_children': item.get('has_active_children'),
        'has_visible_children': item.get('has_visible_children'),
        'pixiv_id': item.get('pixiv_id'),
        'bit_flags': item.get('bit_flags'),
        'views': item.get('views'),
        'has_large': item.get('has_large'),
        'file_url': item.get('file_url'),
        'large_file_url': item.get('large_file_url'),
        'preview_file_url': item.get('preview_file_url'),
        'created_at': to_timestamp(item.get('created_at')),
        'updated_at': to_timestamp(item.get('updated_at')),
    }


def row_signature(row: dict) -> int:
    """
    Fingerprint this site's trigger fields for one row.

    :param row: Row built from an API item, or read back from the stored table.
    :type row: dict
    :returns: Hash over :data:`_UPDATE_TRIGGER_FIELDS`.
    :rtype: int
    """
    return _row_signature(row, _UPDATE_TRIGGER_FIELDS)


def table_signatures(table: pa.Table) -> dict:
    """
    Fingerprint every stored row, keyed by id.

    :param table: Table as read from the hub.
    :type table: pa.Table
    :returns: Mapping of post id to fingerprint.
    :rtype: dict
    """
    return _table_signatures(table, _UPDATE_TRIGGER_FIELDS)


def sync(repository: str, max_time_limit: Optional[float] = 5 * 60 * 60,
         upload_time_span: float = 30, deploy_span: float = 15 * 60,
         max_empty_pages: int = 20, start_below_id: Optional[int] = None,
         username: Optional[str] = None, api_key: Optional[str] = None,
         proxy_pool: Optional[str] = None, brd_api_key: Optional[str] = None,
         brd_zone: Optional[str] = None):
    """
    Sync AIBooru post metadata into the target Hugging Face dataset repository.

    :param repository: Target dataset repository.
    :type repository: str
    :param max_time_limit: Stop fetching after this many seconds, leaving room for the final
        upload. None disables the limit.
    :type max_time_limit: Optional[float]
    :param upload_time_span: Minimum seconds between upload attempts.
    :type upload_time_span: float
    :param deploy_span: Minimum seconds between commits.
    :type deploy_span: float
    :param max_empty_pages: Stop after this many consecutive pages carrying no unseen id.
    :type max_empty_pages: int
    :param start_below_id: Begin the walk just below this id rather than at the newest post.
    :type start_below_id: Optional[int]
    :param username: Site login, optional.
    :type username: Optional[str]
    :param api_key: Matching API key, optional.
    :type api_key: Optional[str]
    :param proxy_pool: Bright Data proxy URL, used only if direct access is refused.
    :type proxy_pool: Optional[str]
    :param brd_api_key: Bright Data API key, needed to allowlist the runner on the zone.
    :type brd_api_key: Optional[str]
    :param brd_zone: Zone to allowlist into.
    :type brd_zone: Optional[str]
    """
    start_time = time.time()
    delete_detached_cache()
    hf_client = get_hf_client()
    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    # A fixed session id keeps the proxy's exit address stable across the run, if it comes to
    # that; direct access is tried first.
    session = get_aibooru_session(username=username, api_key=api_key, proxy_pool=proxy_pool,
                              proxy_session='aibindex', brd_api_key=brd_api_key, brd_zone=brd_zone)

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)

    if hf_client.file_exists(repo_id=repository, repo_type='dataset',
                             filename='aibooru.parquet'):
        base_table = pq.read_table(safe_hf_hub_download(
            hf_client, repo_id=repository, repo_type='dataset', filename='aibooru.parquet'))
        table_schema = base_table.schema
        exist_sigs = table_signatures(base_table)
        logging.info(f'Existing table loaded, {plural_word(base_table.num_rows, "row")}, '
                     f'{plural_word(len(table_schema.names), "column")}.')
    else:
        base_table = None
        table_schema = None
        exist_sigs = {}

    chunks: List[pa.Table] = [base_table] if base_table is not None else []
    pending: List[dict] = []
    updates: dict = {}
    stats = {'ok': 0, 'updated': 0, 'skipped': 0, 'deleted': 0, 'pending_file': 0, 'failed': 0}
    _total_count = base_table.num_rows if base_table is not None else 0
    _last_update, has_update = None, False

    def _flush_pending():
        nonlocal table_schema
        if not pending:
            return
        rows = [{column: row.get(column) for column in table_schema.names} for row in pending] \
            if table_schema is not None else list(pending)
        fresh = pa.Table.from_pylist(rows, schema=table_schema)
        if table_schema is None:
            table_schema = fresh.schema
        chunks.append(fresh)
        pending.clear()

    def _merged_table() -> pa.Table:
        nonlocal chunks
        _flush_pending()
        if not chunks:
            return None
        table = chunks[0] if len(chunks) == 1 else pa.concat_tables(chunks)
        table = apply_updates(table, updates)
        table = table.sort_by([('id', 'descending')])
        chunks = [table]
        updates.clear()
        return table

    def _deploy(force: bool = False):
        nonlocal _last_update, has_update, _total_count
        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            table = _merged_table()
            pq.write_table(table, os.path.join(td, 'aibooru.parquet'))
            total_rows = table.num_rows
            preview = table.slice(0, 100).to_pandas()
            del table

            _write_readme(os.path.join(td, 'README.md'), total_rows=total_rows, preview=preview)

            limiter.try_acquire('hf upload limit')
            added = total_rows - _total_count
            logging.info(f'UPLOAD starting - {plural_word(added, "new post")}, '
                         f'{total_rows:,} rows in total.')
            started = time.time()
            safe_upload_directory_as_directory(
                repo_id=repository, repo_type='dataset', local_directory=td, path_in_repo='.',
                message=f'Add {plural_word(added, "new record")} into index',
            )
            logging.info(f'UPLOAD done in {time.time() - started:.0f}s.')
            has_update = False
            _last_update = time.time()
            _total_count = total_rows

    def _get_posts(below_id: Optional[int]) -> list:
        params = {'limit': str(_POSTS_PER_PAGE)}
        if below_id is not None:
            params['page'] = f'b{below_id}'
        resp = session.get(f'{__site_url__}/posts.json', params=params)
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    def _iter_items():
        """
        Walk the site newest-first by cursor, each page bounded by the lowest id of the last.
        """
        below_id = start_below_id
        empty_pages = 0
        while True:
            if max_time_limit is not None and start_time + max_time_limit < time.time():
                logging.info('Run deadline reached, stopping the walk.')
                return

            items = None
            for attempt in range(_PAGE_ATTEMPTS):
                try:
                    items = _get_posts(below_id)
                    break
                except REQUEST_ERRORS as err:
                    stats['failed'] += 1
                    wait = 2 ** attempt
                    logging.warning(f'Page below {below_id} failed ({attempt + 1}/'
                                    f'{_PAGE_ATTEMPTS}) - {err!r}, retrying in {wait}s.')
                    time.sleep(wait)
            if items is None:
                logging.error(f'Giving up below id {below_id}. Resume with '
                              f'--start-below-id {below_id} once the site recovers.')
                return
            if not items:
                logging.info(f'No posts below {below_id}; the walk has reached the bottom.')
                return

            fresh = 0
            lowest = None
            for item in items:
                post_id = item.get('id')
                if post_id is None:
                    continue
                lowest = post_id if lowest is None else min(lowest, post_id)
                if not is_recordable(item):
                    # Not counted as fresh: these never become rows, so treating them as new work
                    # would hold the counter at zero and walk the whole site every run.
                    if item.get('is_deleted'):
                        stats['deleted'] += 1
                    else:
                        stats['pending_file'] += 1
                        logging.warning(f'Post {post_id} has no md5 yet (still processing); '
                                        f'it will be picked up once its file lands.')
                    continue
                if post_id not in exist_sigs:
                    fresh += 1
                yield item

            if lowest is None:
                logging.info(f'Page below {below_id}: {len(items)} posts, none carrying an id.')
                return
            logging.info(f'Page below {below_id}: {len(items)} posts, {fresh} new, '
                         f'lowest id {lowest:,}.')
            below_id = lowest

            if fresh:
                empty_pages = 0
            else:
                empty_pages += 1
                if empty_pages >= max_empty_pages:
                    logging.info(f'Stopping: {empty_pages} consecutive pages with nothing new.')
                    return

    try:
        for item in _iter_items():
            if max_time_limit is not None and start_time + max_time_limit < time.time():
                break
            post_id = item['id']
            row = build_row(item)
            signature = row_signature(row)
            known = post_id in exist_sigs

            if known and exist_sigs[post_id] == signature:
                stats['skipped'] += 1
                continue

            if known:
                updates[post_id] = row
                stats['updated'] += 1
                logging.info(f'Post {post_id} changed ({stats["updated"]} updated this run).')
            else:
                pending.append(row)
                if len(pending) >= _PENDING_FLUSH:
                    _flush_pending()
                stats['ok'] += 1
                logging.info(f'Post {post_id} confirmed ({stats["ok"]} added this run).')

            exist_sigs[post_id] = signature
            has_update = True
            _deploy()
    finally:
        _deploy(force=True)

    logging.info(f'Done. {stats["ok"]} added, {stats["updated"]} updated, '
                 f'{stats["skipped"]} unchanged, {stats["deleted"]} deleted skipped, '
                 f'{stats["pending_file"]} awaiting a file, {stats["failed"]} request failures.')


def _write_readme(md_file: str, total_rows: int, preview: pd.DataFrame):
    """
    Render the dataset README, keeping the prototype's shape.

    :param md_file: Destination path.
    :type md_file: str
    :param total_rows: Row count of the published table.
    :type total_rows: int
    :param preview: Newest rows, for the sample table.
    :type preview: pd.DataFrame
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
        print('tags:', file=f)
        print('- art', file=f)
        print('- anime', file=f)
        print('- not-for-all-audiences', file=f)
        print('size_categories:', file=f)
        print(f'- {number_to_tag(total_rows)}', file=f)
        print('annotations_creators:', file=f)
        print('- no-annotation', file=f)
        print('source_datasets:', file=f)
        print('- aibooru', file=f)
        print('---', file=f)
        print('', file=f)
        print('## Records', file=f)
        print('', file=f)
        print(f'{plural_word(total_rows, "record")} in total, last updated on '
              f'`{current_time}`. Only {plural_word(len(preview), "record")} shown.', file=f)
        print('', file=f)
        columns = [c for c in ('id', 'width', 'height', 'rating', 'tags', 'file_size',
                               'file_url', 'created_at') if c in preview.columns]
        print(preview[columns].to_markdown(index=False), file=f)
        print('', file=f)


@click.command()
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_AIB',
              default='deepghs/aibooru_index', show_default=True,
              help='Target dataset repository.')
@click.option('-t', '--max-time-limit', type=duration_type(allow_none=True), default='5h',
              show_default=True, help='Stop fetching after this duration, leaving room for the '
                                      'final upload. Use "none" to disable.')
@click.option('-u', '--upload-time-span', type=duration_type(), default=30, show_default=True,
              help='Minimum interval between upload attempts.')
@click.option('-d', '--deploy-span', type=duration_type(), default=15 * 60, show_default=True,
              help='Minimum interval between commits.')
@click.option('-E', '--max-empty-pages', type=int, default=20, show_default=True,
              help='Stop after this many consecutive pages carrying no id we have not seen '
                   'before.')
@click.option('-B', '--start-below-id', type=int, envvar='START_BELOW_ID', default=None,
              help='Begin the walk just below this id instead of at the newest post. Use it to '
                   'resume a backfill that ran out of time, or to work a known gap: without it '
                   'the walk starts at the newest page and stops after --max-empty-pages pages '
                   'of already-indexed posts, long before reaching one.')
@click.option('-U', '--username', type=str, envvar='AIBOORU_USERNAME', default=None,
              help='Site login. Optional; the API answers anonymously.')
@click.option('-K', '--api-key', type=str, envvar='AIBOORU_APIKEY', default=None,
              help='Matching API key.')
@click.option('--proxy-pool', type=str, envvar='BRD_PROXY_URL', default=None,
              help='Bright Data proxy URL. Only used if direct access is refused, since the pool '
                   'is billed per request.')
@click.option('--brd-api-key', type=str, envvar='BRD_API_KEY', default=None,
              help='Bright Data API key, needed to allowlist this host on the zone.')
@click.option('--brd-zone', type=str, envvar='BRD_ZONE', default=None,
              help='Bright Data zone to allowlist into.')
def cli(repository: str, max_time_limit: Optional[float], upload_time_span: float,
        deploy_span: float, max_empty_pages: int, start_below_id: Optional[int],
        username: Optional[str], api_key: Optional[str], proxy_pool: Optional[str],
        brd_api_key: Optional[str], brd_zone: Optional[str]):
    """Sync the AIBooru index."""
    logging.try_init_root(logging.INFO)
    sync(
        repository=repository,
        max_time_limit=max_time_limit,
        upload_time_span=upload_time_span,
        deploy_span=deploy_span,
        max_empty_pages=max_empty_pages,
        start_below_id=start_below_id,
        username=username,
        api_key=api_key,
        proxy_pool=proxy_pool,
        brd_api_key=brd_api_key,
        brd_zone=brd_zone,
    )


if __name__ == '__main__':
    cli()
