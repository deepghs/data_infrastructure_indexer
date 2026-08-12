"""Index sync for booru.allthefallen.moe.

Ported from the pyskeb prototype. Two things had to change.

The site now gates every request behind a proof-of-work challenge of its own - not Cloudflare -
so the prototype's ``ATFBooruSource`` receives an HTML page where it expects JSON. That is
handled in :mod:`inf.atfbooru.base`; nothing about it leaks into this module beyond asking for
the session.

The stored table is held as an Arrow table rather than a list of dicts. The prototype called
``to_dict('records')`` on it, which at 46 columns costs several kilobytes a row; the backfill
this needs is around 865k posts, so that representation would not fit on a CI runner.

Layout published to the target repository
=========================================

::

    records.parquet             one row per post, 46 columns
    tags.parquet                per-tag usage counts
    index_tags.parquet          tag metadata, read only
    index_tag_aliases.parquet   alias metadata, read only

Walking the site
================

Results come back newest-first and the API caps how deep ``page`` may go, so a plain page walk
cannot reach far. The way through is the one the prototype used: page forward until the cap,
then restart at page 1 with ``tags=id:<lowest id seen so far``, which moves the window instead
of the offset.

What an anonymous run can and cannot see
========================================

Large stretches of the id space are invisible without an account, and the shape of that is worth
writing down because it looks exactly like a crawler bug.

``counts/posts.json`` reports the whole database while ``posts.json`` returns only what the
caller may see. Measured anonymously: counts puts 896,697 posts in ``id:712322..1613216`` while a
full walk of that range yields 67,695. Individual fetches agree - ``/posts/1613216.json`` is 200
but ``/posts/1000123.json`` is 403 - and a narrow window such as ``id:1000000..1000500`` returns
nothing even though counts says 498. The boundary is not "old is hidden": id 712,321 is
readable, and the refusals sit in the middle of the range.

So a gap audit on an anonymously-built table will always show large holes, and they are not
misses. 97 gaps of 1000+ ids, about 204k ids in total, fall in that category. Closing them needs
``ATFBOORU_USERNAME`` and ``ATFBOORU_APIKEY``; the walk itself already handles them, since it
simply never sees those posts.

New posts are visible anonymously, so a scheduled run keeps up with the site regardless.
"""
import datetime
import html
import math
import mimetypes
import os
import re
import time
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

from inf.utils.duration import duration_type
from inf.utils.safe import safe_hf_hub_download, safe_upload_directory_as_directory
from inf.utils.session import REQUEST_ERRORS
from .base import __site_url__, get_atfbooru_session

mimetypes.add_type('image/webp', '.webp')

#: Tag categories, as danbooru-derived sites number them.
_TAG_TYPES = {
    -1: 'unknown',
    0: 'general',
    1: 'artist',
    3: 'copyright',
    4: 'character',
    5: 'meta',
}

#: Nested object the API returns that the table does not carry.
_DROPPED_FIELDS = ('media_asset',)

#: Rows buffered as dicts before folding into an Arrow chunk. Small on purpose: 46 columns of
#: dict per row is what makes the naive approach unaffordable.
_PENDING_FLUSH = 20000

#: Posts per API request. The site accepts 200, which is also what the prototype used.
_POSTS_PER_PAGE = 200


def build_row(item: dict) -> dict:
    """
    Turn an API item into a table row.

    :param item: One entry from ``/posts.json``.
    :type item: dict
    :returns: The row to store.
    :rtype: dict
    """
    row = {key: value for key, value in item.items() if key not in _DROPPED_FIELDS}
    file_url = row.get('file_url')
    row['mimetype'] = mimetypes.guess_type(file_url)[0] if file_url else None
    return row


def sync(repository: str, max_time_limit: Optional[float] = 5 * 60 * 60,
         upload_time_span: float = 30, deploy_span: float = 15 * 60,
         max_page: int = 1000, max_empty_pages: int = 10,
         username: Optional[str] = None, api_key: Optional[str] = None):
    """
    Sync ATFBooru post metadata into the target Hugging Face dataset repository.

    :param repository: Target dataset repository.
    :type repository: str
    :param max_time_limit: Stop fetching after this many seconds, leaving room for the final
        upload. None disables the limit.
    :type max_time_limit: Optional[float]
    :param upload_time_span: Minimum seconds between uploads.
    :type upload_time_span: float
    :param deploy_span: Minimum seconds between commits.
    :type deploy_span: float
    :param max_page: Page number at which to restart the walk with a lower ``id:<`` bound. The
        API refuses to page indefinitely, so this is how the window advances.
    :type max_page: int
    :param max_empty_pages: Stop after this many consecutive pages containing nothing new.
    :type max_empty_pages: int
    :param username: Site login for authenticated requests.
    :type username: Optional[str]
    :param api_key: Matching API key.
    :type api_key: Optional[str]
    """
    from pyrate_limiter import Duration, Limiter, Rate

    start_time = time.time()
    delete_detached_cache()
    hf_client = get_hf_client()
    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    session = get_atfbooru_session(username=username, api_key=api_key)

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)

    # Held as Arrow throughout. The published file and its 46 columns are unchanged; only the
    # in-process representation differs from the prototype.
    if hf_client.file_exists(repo_id=repository, repo_type='dataset',
                             filename='records.parquet'):
        base_table = pq.read_table(safe_hf_hub_download(
            hf_client, repo_id=repository, repo_type='dataset', filename='records.parquet'))
        table_schema = base_table.schema
        exist_ids = set(base_table.column('id').to_pylist())
        logging.info(f'Existing table loaded, {plural_word(base_table.num_rows, "row")}, '
                     f'{plural_word(len(table_schema.names), "column")}.')
    else:
        base_table = None
        table_schema = None
        exist_ids = set()

    df_index_tags = pd.read_parquet(safe_hf_hub_download(
        hf_client, repo_id=repository, repo_type='dataset',
        filename='index_tags.parquet')).replace(np.NaN, None)
    d_index_tags = {item['name']: item for item in df_index_tags.to_dict('records')}
    del df_index_tags
    logging.info(f'Tag metadata loaded, {plural_word(len(d_index_tags), "tag")}.')

    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='tags.parquet'):
        df_tags = pd.read_parquet(safe_hf_hub_download(
            hf_client, repo_id=repository, repo_type='dataset',
            filename='tags.parquet')).replace(np.NaN, None)
        d_tags = {item['name']: item for item in df_tags.to_dict('records')}
        del df_tags
    else:
        d_tags = {}

    chunks: List[pa.Table] = [base_table] if base_table is not None else []
    pending: List[dict] = []
    stats = {'ok': 0, 'skipped': 0, 'failed': 0}
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
        _flush_pending()
        if not chunks:
            return None
        table = chunks[0] if len(chunks) == 1 else pa.concat_tables(chunks)
        return table.sort_by([('id', 'descending')])

    def _deploy(force: bool = False):
        nonlocal _last_update, has_update, _total_count
        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            table = _merged_table()
            pq.write_table(table, os.path.join(td, 'records.parquet'))
            total_rows = table.num_rows
            preview = table.slice(0, 50).to_pandas()
            del table

            df_out = pd.DataFrame(list(d_tags.values()))
            df_out = df_out.sort_values(['count', 'category'], ascending=[False, True])
            df_out.to_parquet(os.path.join(td, 'tags.parquet'), index=False)

            _write_readme(os.path.join(td, 'README.md'), total_rows=total_rows,
                          preview=preview, df_tags=df_out)

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

    def _ping_tags(item: dict):
        """Fold a post's tags into the tag table, keeping a usage count."""
        for tag in filter(bool, re.split(r'\s+', item.get('tag_string') or '')):
            tag = html.unescape(tag)
            known = d_index_tags.get(tag)
            info = {
                'id': known['id'] if known else -1,
                'name': tag,
                'total': known['post_count'] if known else 0,
                'category': known['category'] if known else -1,
                'is_deprecated': known['is_deprecated'] if known else False,
            }
            if tag not in d_tags:
                d_tags[tag] = dict(info)
                current = 0
            else:
                current = d_tags[tag].get('count', 0)
                if info['category'] != -1:
                    d_tags[tag].update(info)
            d_tags[tag]['count'] = current + 1

    def _get_posts(page: int, below_id: Optional[int]) -> list:
        params = {'limit': str(_POSTS_PER_PAGE), 'page': str(page)}
        if below_id is not None:
            params['tags'] = f'id:<{below_id}'
        resp = session.get(f'{__site_url__}/posts.json', params=params)
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    def _iter_items():
        """
        Walk the site newest-first, moving the window rather than the offset.

        The API stops honouring ``page`` past a certain depth, so on reaching ``max_page`` the
        walk restarts at page 1 bounded by the lowest id it has seen. Without that it could only
        ever read the first few hundred pages.
        """
        below_id = None
        lowest_seen = None
        page = 1
        empty_pages = 0
        while True:
            if max_time_limit is not None and start_time + max_time_limit < time.time():
                logging.info('Run deadline reached, stopping the walk.')
                return
            try:
                items = _get_posts(page, below_id)
            except REQUEST_ERRORS as err:
                logging.warning(f'Page {page} (below {below_id}) failed - {err!r}, moving on.')
                page += 1
                continue

            if not items:
                if below_id is None and page == 1:
                    logging.info('Site returned nothing at all; stopping.')
                    return
                # Exhausted this window; drop to the next one.
                if lowest_seen is None:
                    return
                logging.info(f'Window below {below_id} exhausted at page {page}; '
                             f'continuing below {lowest_seen}.')
                below_id, page = lowest_seen, 1
                continue

            fresh = 0
            for item in items:
                post_id = item.get('id')
                if post_id is None:
                    continue
                if lowest_seen is None or post_id < lowest_seen:
                    lowest_seen = post_id
                if post_id not in exist_ids:
                    fresh += 1
                    yield item

            logging.info(f'Page {page} (below {below_id}): {len(items)} posts, {fresh} new, '
                         f'lowest id seen {lowest_seen:,}.')
            if fresh:
                empty_pages = 0
            else:
                empty_pages += 1
                if empty_pages >= max_empty_pages:
                    logging.info(f'Stopping: {empty_pages} consecutive pages with nothing new.')
                    return

            page += 1
            if page > max_page:
                logging.info(f'Reached page cap {max_page}; continuing below {lowest_seen}.')
                below_id, page = lowest_seen, 1

    try:
        for item in _iter_items():
            if max_time_limit is not None and start_time + max_time_limit < time.time():
                break
            post_id = item['id']
            if post_id in exist_ids:
                stats['skipped'] += 1
                continue
            pending.append(build_row(item))
            if len(pending) >= _PENDING_FLUSH:
                _flush_pending()
            _ping_tags(item)
            exist_ids.add(post_id)
            stats['ok'] += 1
            has_update = True
            logging.info(f'Post {post_id} confirmed ({stats["ok"]} added this run).')
            _deploy()
    finally:
        _deploy(force=True)

    logging.info(f'Done. {stats["ok"]} added, {stats["skipped"]} already known, '
                 f'{stats["failed"]} failed. Challenges answered: {session.challenges}.')


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
        print('- atfbooru', file=f)
        print('---', file=f)
        print('', file=f)

        print('## Records', file=f)
        print('', file=f)
        print(f'{plural_word(total_rows, "record")} in total, last updated at '
              f'`{current_time}`. Only {plural_word(len(preview), "record")} shown.', file=f)
        print('', file=f)
        columns = [c for c in ('id', 'image_width', 'image_height', 'rating', 'mimetype',
                               'file_url') if c in preview.columns]
        print(preview[columns].to_markdown(index=False), file=f)
        print('', file=f)

        print('## Tags', file=f)
        print('', file=f)
        print(f'{plural_word(len(df_tags), "tag")} in total.', file=f)
        print('', file=f)
        for category in sorted(set(df_tags['category'])):
            df_cat = df_tags[df_tags['category'] == category][
                ['id', 'name', 'category', 'total', 'count']]
            df_shown = df_cat[:30].replace(np.NaN, '')
            print(f'These are the top {plural_word(len(df_shown), "tag")} '
                  f'({plural_word(len(df_cat), "tag")} in total) '
                  f'of category `{_TAG_TYPES.get(category, category)} ({category})`:', file=f)
            print('', file=f)
            print(df_shown.to_markdown(index=False), file=f)
            print('', file=f)


@click.command(context_settings={'help_option_names': ['-h', '--help']},
               help='Sync ATFBooru post metadata into a Hugging Face dataset repository. The '
                    'site guards itself with a proof-of-work challenge, which the session layer '
                    'answers without a browser.')
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_ATF', required=True,
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
@click.option('-P', '--max-page', type=int, default=1000, show_default=True,
              help='Page number at which to restart the walk with a lower id bound. The API '
                   'will not page indefinitely, so this is how the window advances.')
@click.option('-E', '--max-empty-pages', type=int, default=10, show_default=True,
              help='Stop after this many consecutive pages with nothing new. Raise it to walk '
                   'past a stretch that is already indexed.')
@click.option('-U', '--username', type=str, envvar='ATFBOORU_USERNAME', default=None,
              show_envvar=True, help='Site username for authenticated requests.')
@click.option('-K', '--api-key', type=str, envvar='ATFBOORU_APIKEY', default=None,
              show_envvar=True, help='Site API key for authenticated requests.')
def cli(repository: str, max_time_limit: Optional[float], upload_time_span: float,
        deploy_span: float, max_page: int, max_empty_pages: int,
        username: Optional[str], api_key: Optional[str]):
    logging.try_init_root(logging.INFO)
    sync(
        repository=repository,
        max_time_limit=max_time_limit,
        upload_time_span=upload_time_span,
        deploy_span=deploy_span,
        max_page=max_page,
        max_empty_pages=max_empty_pages,
        username=username,
        api_key=api_key,
    )


if __name__ == '__main__':
    cli()
