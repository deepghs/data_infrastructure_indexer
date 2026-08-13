"""Index sync for e6ai.net.

Ported from the pyskeb prototype (``test/prepare/e6ai/index.py``). The walk and the storage
changed; the published row did not.

The prototype asked for ``limit=1000``. The site refuses anything over 320 with
``410 Limit must be between 0 and 320``, so that request never worked - the same ceiling this
repository already hit on e621. Paging is by ``page=b{id}`` cursor, which the prototype also used
and which has no depth limit; a numeric ``page`` is refused past 750 here.

Known posts are now re-examined rather than skipped, so an edited row is corrected instead of
frozen at whatever it looked like when first seen. See :mod:`inf.utils.upsert`; the rule that
matters is that a field arriving as ``None`` never overwrites a stored value.

Layout published to the target repository
=========================================

::

    e6ai.parquet                one row per post, 48 columns
    tags.parquet                per-tag usage counts
    index_tags.parquet          tag metadata, written by a separate job, untouched here
    index_tag_aliases.parquet   alias metadata, likewise

The 48 columns are not a hand-picked set
========================================

The prototype built each row by popping the API's seven nested objects - ``file``, ``flags``,
``preview``, ``sample``, ``score``, ``tags``, ``relationships`` - flattening them under prefixes,
and then spreading whatever remained of the post on top. So the column set is a snapshot of what
the API returned when it last ran, which is why there are columns like ``preview_alt`` and
``uploader_name`` that the current API may or may not still send.

That makes the stored schema the contract, not the API: rows are written through the published
column list, so a field the API has since dropped becomes null and one it has added is discarded
rather than silently widening the table. Changing those 48 columns should be a deliberate act.

Two shapes in there need care. ``tags`` is a ``list<string>``, the concatenation of every tag
category, not a space-joined string as on the danbooru-derived sites. And ``sample_alternates`` is
a deeply nested struct whose Arrow type includes a ``__dummy`` field: Arrow cannot infer a type for
an empty ``{}``, which is what the API sends for most posts, so empty objects are rewritten as
``{'__dummy': None}`` at every depth. :func:`parquet_safe` does that, and dropping it would make
the write fail on the first post without alternates.

Posts with no file url are recorded anyway
==========================================

The prototype skipped them, reasoning that a higher-privileged account might be needed. They are
stored here instead: the published schema has ``file_url`` and ``md5`` nullable and carries
``is_deleted``, so it can represent such a post honestly, and a row written now is what lets the
url be filled in later. This is the same choice made for atfbooru, and it differs from aibooru only
because aibooru's schema has no column to mark a deleted post with.

In practice this is rare - all 320 of the newest posts sampled on 2026-08-13 had a url, and none
were deleted - but it is warned about when it happens, and such posts do not count towards the
"nothing new" counter.

Publication rate
================

About 750 posts a week, measured 2026-08-13: the 320 newest posts span three days, and the site's
newest id was 179,784 against 126,075 in the table left by the prototype.

Like e621, this API has no counts endpoint, so this is a sample rather than a server-side count -
though a three-day span makes it considerably steadier than a same-hour one would be.
"""
import datetime
import math
import mimetypes
import os
import time
from itertools import chain
from typing import List, Optional

import click
import numpy as np
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
from inf.utils.upsert import adds_anything, apply_updates
from inf.utils.upsert import row_signature as _row_signature
from inf.utils.upsert import table_signatures as _table_signatures
from .base import __site_url__, get_e6ai_session

mimetypes.add_type('image/webp', '.webp')

#: Posts per request. 320 is the site's ceiling; 321 is refused outright.
_POSTS_PER_PAGE = 320

#: Rows buffered as dicts before folding into an Arrow chunk.
_PENDING_FLUSH = 20000

#: Attempts per page before giving up on the run. Cursor paging cannot skip a failed page: leaving
#: the cursor alone re-requests it forever, and advancing past it drops those posts silently.
_PAGE_ATTEMPTS = 5

#: Tag categories as the API groups them, and the numbering the tag table uses.
_TAG_CATEGORIES = {
    0: 'general',
    1: 'artist',
    3: 'copyright',
    4: 'character',
    5: 'species',
    6: 'invalid',
    7: 'meta',
    8: 'lore',
}
_TAG_INV_CATEGORIES = {name: index for index, name in _TAG_CATEGORIES.items()}

#: Fields whose change makes a stored row worth rewriting.
#:
#: Not every column. ``score``, ``up_score``, ``down_score``, ``fav_count``, ``comment_count`` and
#: ``change_seq`` drift on their own, and ``updated_at`` moves whenever any of them do, so
#: including any of them would mark much of the table changed on every pass.
_UPDATE_TRIGGER_FIELDS = (
    'md5', 'file_url', 'file_ext', 'file_size', 'width', 'height', 'mimetype',
    'tags', 'locked_tags', 'rating', 'sources', 'pools', 'description',
    'parent_id', 'has_children', 'has_active_children', 'children',
    'preview_url', 'sample_url', 'sample_has',
    'is_deleted', 'is_pending', 'is_flagged', 'is_note_locked', 'is_status_locked',
    'is_rating_locked', 'duration',
)

#: Of the trigger fields, the lists whose order the API does not hold stable.
#:
#: ``tags`` is built by concatenating every tag category, and the site does not guarantee an order
#: within or between them: re-fetching 12 stored posts found one whose tags held exactly the same
#: values in a different order. Comparing it as ordered would mark that row changed on every run
#: forever, since rewriting it does not make the next fetch agree. ``children`` and ``pools`` are
#: id lists with no inherent order either.
#:
#: What gets stored still keeps the order the API sent; only the comparison is order-blind.
_UNORDERED_TRIGGER_FIELDS = frozenset({'tags', 'locked_tags', 'sources', 'pools', 'children'})


def parquet_safe(value):
    """
    Rewrite empty objects so Arrow can type them.

    Arrow cannot infer a struct type for ``{}``, and the API sends exactly that for
    ``sample.alternates`` on most posts. The prototype's fix, kept here because the published
    column type depends on it, is a ``__dummy`` field at every depth where an object would
    otherwise be empty.

    :param value: Any decoded JSON value.
    :returns: The same value with empty objects replaced.
    """
    if isinstance(value, dict):
        if not value:
            return {'__dummy': None}
        return {key: parquet_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(parquet_safe(item) for item in value)
    return value


def build_row(post: dict) -> dict:
    """
    Turn an API post into a table row, flattening the way the published table expects.

    The API's nested objects are lifted out and re-attached under prefixes, then whatever remains
    of the post is spread on top - which is how the 48 published columns came to be. The input is
    not mutated, unlike the prototype which popped from it directly.

    :param post: One entry from ``/posts.json``.
    :type post: dict
    :returns: The row to store, before it is narrowed to the published columns.
    :rtype: dict
    """
    rest = dict(post)
    file_info = rest.pop('file', None) or {}
    flags_info = rest.pop('flags', None) or {}
    preview_info = rest.pop('preview', None) or {}
    sample_info = rest.pop('sample', None) or {}
    score_info = rest.pop('score', None) or {}
    tags_info = rest.pop('tags', None) or {}
    relationships_info = rest.pop('relationships', None) or {}

    file_url = file_info.get('url')
    mimetype = mimetypes.guess_type(file_url)[0] if file_url else None

    row = {
        'id': rest.get('id'),

        'mimetype': mimetype,
        'file_ext': file_info.get('ext'),
        'width': file_info.get('width'),
        'height': file_info.get('height'),
        'md5': file_info.get('md5'),
        'file_url': file_url,
        'file_size': file_info.get('size'),
        'rating': rest.get('rating'),

        'tags': list(chain(*tags_info.values())) if tags_info else [],

        'uploader_id': rest.get('uploader_id'),
        'approver_id': rest.get('approver_id'),

        'score': score_info.get('total'),
        'up_score': score_info.get('up'),
        'down_score': score_info.get('down'),
        'fav_count': rest.get('fav_count'),

        **{f'preview_{key}': value for key, value in preview_info.items()},
        **{f'sample_{key}': value for key, value in sample_info.items()},
        **{f'is_{key}': value for key, value in flags_info.items()},

        **relationships_info,
        **rest,
    }
    if 'sample_alternates' in row:
        row['sample_alternates'] = parquet_safe(row['sample_alternates'])
    return row


def tags_by_category(post: dict) -> dict:
    """
    The post's tags grouped by category id, for the usage counts in ``tags.parquet``.

    :param post: One entry from ``/posts.json``.
    :type post: dict
    :returns: Mapping of category id to tag names.
    :rtype: dict
    """
    grouped = {}
    for name, values in (post.get('tags') or {}).items():
        category = _TAG_INV_CATEGORIES.get(name)
        if category is None:
            continue
        grouped[category] = list(values or [])
    return grouped


def has_file_url(post: dict) -> bool:
    """
    Whether the API gave us a file for this post.

    :param post: One entry from ``/posts.json``.
    :type post: dict
    :returns: True when a file url is present.
    :rtype: bool
    """
    return bool(((post.get('file') or {}).get('url') or '').strip())


def row_signature(row: dict) -> int:
    """
    Fingerprint this site's trigger fields for one row.

    :param row: Row built from an API post, or read back from the stored table.
    :type row: dict
    :returns: Hash over :data:`_UPDATE_TRIGGER_FIELDS`.
    :rtype: int
    """
    return _row_signature(row, _UPDATE_TRIGGER_FIELDS, _UNORDERED_TRIGGER_FIELDS)


def table_signatures(table: pa.Table) -> dict:
    """
    Fingerprint every stored row, keyed by id.

    :param table: Table as read from the hub.
    :type table: pa.Table
    :returns: Mapping of post id to fingerprint.
    :rtype: dict
    """
    return _table_signatures(table, _UPDATE_TRIGGER_FIELDS, _UNORDERED_TRIGGER_FIELDS)


def sync(repository: str, max_time_limit: Optional[float] = 5 * 60 * 60,
         upload_time_span: float = 30, deploy_span: float = 15 * 60,
         max_empty_pages: int = 20, start_below_id: Optional[int] = None,
         username: Optional[str] = None, api_key: Optional[str] = None,
         proxy_pool: Optional[str] = None, brd_api_key: Optional[str] = None,
         brd_zone: Optional[str] = None):
    """
    Sync E6AI post metadata into the target Hugging Face dataset repository.

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
    session = get_e6ai_session(username=username, api_key=api_key, proxy_pool=proxy_pool,
                              proxy_session='e6aiindex', brd_api_key=brd_api_key, brd_zone=brd_zone)

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)

    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='e6ai.parquet'):
        base_table = pq.read_table(safe_hf_hub_download(
            hf_client, repo_id=repository, repo_type='dataset', filename='e6ai.parquet'))
        table_schema = base_table.schema
        exist_sigs = table_signatures(base_table)
        # id -> row offset, so a stored row can be read back when its fingerprint moves. Needed
        # because "did this change?" has to be asked of the merge result, not the fetched row:
        # see inf.utils.upsert.adds_anything. Costs one int pair per row rather than the fields
        # themselves, and the lookup is a one-row slice.
        base_index = {post_id: offset for offset, post_id
                      in enumerate(base_table.column('id').to_pylist())}
        logging.info(f'Existing table loaded, {plural_word(base_table.num_rows, "row")}, '
                     f'{plural_word(len(table_schema.names), "column")}.')
    else:
        base_table = None
        table_schema = None
        exist_sigs = {}
        base_index = {}

    d_index_tags = {}
    if hf_client.file_exists(repo_id=repository, repo_type='dataset',
                            filename='index_tags.parquet'):
        df_index_tags = pd.read_parquet(safe_hf_hub_download(
            hf_client, repo_id=repository, repo_type='dataset',
            filename='index_tags.parquet')).replace(np.NaN, None)
        d_index_tags = {(item['category'], item['name']): item
                        for item in df_index_tags.to_dict('records')}
        del df_index_tags
        logging.info(f'Tag metadata loaded, {plural_word(len(d_index_tags), "tag")}.')

    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='tags.parquet'):
        df_tags = pd.read_parquet(safe_hf_hub_download(
            hf_client, repo_id=repository, repo_type='dataset',
            filename='tags.parquet')).replace(np.NaN, None)
        d_tags = {(item['category'], item['name']): item for item in df_tags.to_dict('records')}
        del df_tags
    else:
        d_tags = {}

    chunks: List[pa.Table] = [base_table] if base_table is not None else []
    pending: List[dict] = []
    updates: dict = {}
    urlless_ids = set()
    stats = {'ok': 0, 'updated': 0, 'skipped': 0, 'no_gain': 0, 'urlless': 0, 'failed': 0}
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
            pq.write_table(table, os.path.join(td, 'e6ai.parquet'))
            total_rows = table.num_rows
            preview = table.slice(0, 50).to_pandas()
            del table

            df_out = pd.DataFrame(list(d_tags.values()))
            if len(df_out):
                df_out = df_out.sort_values(['count', 'category'], ascending=[False, True])
                df_out.to_parquet(os.path.join(td, 'tags.parquet'), index=False)

            _write_readme(os.path.join(td, 'README.md'), total_rows=total_rows, preview=preview,
                          df_tags=df_out)

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

    def _ping_tags(post: dict):
        """Fold a post's tags into the tag table, keeping a usage count."""
        for category, names in tags_by_category(post).items():
            for name in names:
                token = (category, name)
                known = d_index_tags.get(token)
                if token not in d_tags:
                    d_tags[token] = {
                        'id': known['id'] if known else -1,
                        'name': name,
                        'category': category,
                        'total_count': known['post_count'] if known else 0,
                        'count': 0,
                    }
                d_tags[token]['count'] = d_tags[token].get('count', 0) + 1

    def _stored_row(post_id: int) -> Optional[dict]:
        offset = base_index.get(post_id)
        if offset is None or base_table is None:
            return None
        return base_table.slice(offset, 1).to_pylist()[0]

    def _get_posts(below_id: Optional[int]) -> list:
        params = {'limit': str(_POSTS_PER_PAGE)}
        if below_id is not None:
            params['page'] = f'b{below_id}'
        resp = session.get(f'{__site_url__}/posts.json', params=params)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            payload = payload.get('posts') or []
        return payload if isinstance(payload, list) else []

    def _iter_items():
        """Walk the site newest-first by cursor, each page bounded by the lowest id of the last."""
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
            for post in items:
                post_id = post.get('id')
                if post_id is None:
                    continue
                lowest = post_id if lowest is None else min(lowest, post_id)
                if not has_file_url(post) and post_id not in urlless_ids:
                    urlless_ids.add(post_id)
                    stats['urlless'] += 1
                    logging.warning(f'Post {post_id} carries no file url '
                                    f'(deleted={(post.get("flags") or {}).get("deleted")}) - '
                                    f'recorded anyway; a later run fills the url in without '
                                    f'overwriting a known value.')
                if post_id not in exist_sigs:
                    fresh += 1
                yield post

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
        for post in _iter_items():
            if max_time_limit is not None and start_time + max_time_limit < time.time():
                break
            post_id = post['id']
            row = build_row(post)
            signature = row_signature(row)
            known = post_id in exist_sigs

            if known and exist_sigs[post_id] == signature:
                stats['skipped'] += 1
                continue

            if known and not adds_anything(_stored_row(post_id), row, exist_sigs[post_id],
                                           _UPDATE_TRIGGER_FIELDS, _UNORDERED_TRIGGER_FIELDS):
                # The fingerprint moved but the merge would not: a field the API has stopped
                # sending. Rewriting would change nothing and the next run would ask again.
                stats['no_gain'] += 1
                continue

            if known:
                updates[post_id] = row
                stats['updated'] += 1
                logging.info(f'Post {post_id} changed ({stats["updated"]} updated this run).')
            else:
                pending.append(row)
                if len(pending) >= _PENDING_FLUSH:
                    _flush_pending()
                # Tag counts accumulate on first sight only. Re-counting on every update would
                # inflate them, and undoing a previous contribution would mean keeping each post's
                # old tag list around.
                _ping_tags(post)
                stats['ok'] += 1
                logging.info(f'Post {post_id} confirmed ({stats["ok"]} added this run).')

            exist_sigs[post_id] = signature
            has_update = True
            _deploy()
    finally:
        _deploy(force=True)

    logging.info(f'Done. {stats["ok"]} added, {stats["updated"]} updated, '
                 f'{stats["skipped"]} unchanged, {stats["no_gain"]} offering nothing new, '
                 f'{stats["urlless"]} without a file url, {stats["failed"]} request failures.')


def _write_readme(md_file: str, total_rows: int, preview: pd.DataFrame, df_tags: pd.DataFrame):
    """
    Render the dataset README.

    :param md_file: Destination path.
    :type md_file: str
    :param total_rows: Row count of the published table.
    :type total_rows: int
    :param preview: Newest rows, for the sample table.
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
        print('tags:', file=f)
        print('- art', file=f)
        print('- anime', file=f)
        print('- not-for-all-audiences', file=f)
        print('size_categories:', file=f)
        print(f'- {number_to_tag(total_rows)}', file=f)
        print('annotations_creators:', file=f)
        print('- no-annotation', file=f)
        print('source_datasets:', file=f)
        print('- e6ai', file=f)
        print('---', file=f)
        print('', file=f)
        print('## Records', file=f)
        print('', file=f)
        print(f'{plural_word(total_rows, "record")} in total, last updated on '
              f'`{current_time}`. Only {plural_word(len(preview), "record")} shown.', file=f)
        print('', file=f)
        columns = [c for c in ('id', 'width', 'height', 'rating', 'mimetype', 'file_size',
                               'file_url') if c in preview.columns]
        print(preview[columns].to_markdown(index=False), file=f)
        print('', file=f)
        if df_tags is not None and len(df_tags):
            print('## Tags', file=f)
            print('', file=f)
            print(f'{plural_word(len(df_tags), "tag")} in total.', file=f)
            print('', file=f)
            shown = df_tags[:30][[c for c in ('id', 'name', 'category', 'count', 'total_count')
                                  if c in df_tags.columns]]
            print(shown.to_markdown(index=False), file=f)
            print('', file=f)


@click.command()
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_E6AI',
              default='deepghs/e6ai_index', show_default=True,
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
                   'resume a backfill that ran out of time, or to work a known gap.')
@click.option('-U', '--username', type=str, envvar='E6AI_USERNAME', default=None,
              help='Site login. Optional; the API answers anonymously.')
@click.option('-K', '--api-key', type=str, envvar='E6AI_APIKEY', default=None,
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
    """Sync the E6AI index."""
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
