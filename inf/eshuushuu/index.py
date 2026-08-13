"""Index sync for e-shuushuu.net, rebuilt on the site's v1 API.

This is a rebuild, not a port. The pyskeb prototype scraped HTML fifteen images at a time; that
markup is gone. What replaced it, ``/api/v1/images``, carries far more than the old table stored,
so the published repository was renamed to ``eshuushuu_index_deprecate`` and this writes a new one
from scratch.

What the old table was missing
==============================

Sampling 120 images across six eras, every field the old 16 columns held maps cleanly onto the API
- but the API also carries, for every image, an ``md5_hash`` and a per-tag ``type`` that the old
table had nowhere to put. The old ``tags`` column is a flat list of names with Artist, Character,
Source and Theme collapsed together and the distinction discarded; the lone ``old_characters``
column was the only surviving trace of it, and it was almost always empty. Also present now:
``source_url`` (a pixiv or twitter link, on about one image in six), ``user_id``, ``num_ratings``,
and thumbnail/medium/large urls.

Two of the old columns were lossy. ``file_size_text`` stored ``"534.7 kB"`` where the API gives
547,580 bytes, and ``created_at`` was truncated to the minute where the API gives seconds. Both are
stored exactly here.

``old_characters`` has no source in the API and is not carried over. Everything else the old table
had is present, under the same name where the name still made sense.

Tags live in their own table
============================

``tags.parquet`` keys on the site's own ``tag_id``, so ``tag_ids`` on a row joins straight to it.
Each tag carries its category, this index's own usage count, and the site-wide ``usage_count`` the
API reports - the second is the site's number, the first is how many of our rows actually use it,
and they answer different questions.

Walking the site
================

Paging is by offset and the page size is fixed: ``limit`` is accepted and ignored, every response
holds exactly 20 images, and there is no cursor - ``before_id``, ``since_id``, ``min_id``, ``sort``
and ``order`` are all silently ignored, each returning the newest page. Depth is unlimited
(page 10,863 answers normally).

54,315 pages at 0.6s each is nine hours, past what a run gets, and splitting it across two runs
would invite the offset to shift underneath: new uploads push everything down a page, and a
resumed sweep would skip whatever crossed the boundary. That is the likeliest explanation for the
25,000 images the old table was missing from inside its own id range - not deletions, since
sampling 40 of those absent ids found all 40 still live.

So pages are fetched concurrently instead, which brings a full sweep to around three hours and
keeps it in one run. Measured: 1 worker 0.598s/page, 4 workers 0.245s, 8 workers 0.199s, no
throttling at any of them. Six is used - the returns past that are small and this is a small site.
Fetching is concurrent; parsing, counting and writing stay on one thread and in page order.

Offsets shift, so ids repeat
============================

None of that removes the fundamental problem with offset paging: every upload during the sweep
pushes each following page down by one image. Over three hours that is a few dozen images, so some
will be served twice on adjacent pages and some will slip through the boundary unseen. Concurrency
does not cause this and does not worsen it - a serial sweep of the same length shifts exactly as
much.

Duplicates are therefore expected, not exceptional, and are handled in two places: an id seen
earlier in the run is dropped immediately, and the table is checked for duplicate ids before every
write, keeping the last occurrence if any survive. The second is redundant if the first works,
which is the point - a duplicated row is far harder to notice once it is published.

Images missed at a boundary cannot be recovered by deduplication; they are what the audit after
the sweep is for, and single images can be fetched by id from ``/api/v1/images/{id}``.
"""
import datetime
import math
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
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
from .base import __site_url__, get_eshuushuu_session

mimetypes.add_type('image/webp', '.webp')

#: Images per response. Fixed by the site - ``limit`` is accepted and ignored.
_PER_PAGE = 20

#: Concurrent fetchers. Six sits just past the knee of the curve; more buys little.
_WORKERS = 6

#: Pages requested per round. Fetched concurrently, then handled in page order.
_PREFETCH = _WORKERS * 4

#: Rows buffered as dicts before folding into an Arrow chunk.
_PENDING_FLUSH = 20000

#: Attempts per page before the run gives up and reports where to resume.
_PAGE_ATTEMPTS = 5

#: Tag categories as the API names them, lowercased for the column suffix.
_TAG_CATEGORIES = ('artist', 'character', 'source', 'theme')

#: Fields whose change makes a stored row worth rewriting.
#:
#: Not every column. ``favorites``, ``num_ratings``, ``rating``, ``score`` and ``posts`` drift on
#: their own, so including them would mark much of the table changed on every pass.
_UPDATE_TRIGGER_FIELDS = (
    'md5_hash', 'file_url', 'cdn_url', 'thumbnail_url', 'medium_url', 'large_url',
    'filename', 'ext', 'src_filename', 'original_filename', 'mimetype',
    'file_size', 'width', 'height', 'status', 'caption', 'source_url', 'misc_metadata',
    'replacement_id', 'username', 'user_id', 'created_at',
    'tags', 'tag_ids', 'tags_artist', 'tags_character', 'tags_source', 'tags_theme',
)

#: Of those, the lists whose order the API does not hold stable.
_UNORDERED_TRIGGER_FIELDS = frozenset({
    'tags', 'tag_ids', 'tags_artist', 'tags_character', 'tags_source', 'tags_theme',
})


def to_timestamp(value: Optional[str]) -> Optional[float]:
    """
    Parse the API's ``date_added`` into an epoch, keeping the seconds.

    The old table truncated these to the minute; there is no reason to.

    :param value: Timestamp such as ``2026-08-13T04:39:44Z``.
    :type value: Optional[str]
    :returns: Seconds since the epoch, or None.
    :rtype: Optional[float]
    """
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (TypeError, ValueError):
        return None


def split_tags(item: dict) -> dict:
    """
    Group an image's tags by the API's category, and keep the flat forms too.

    :param item: One entry from ``/api/v1/images``.
    :type item: dict
    :returns: Keys ``tags``, ``tag_ids`` and one ``tags_<category>`` per known category.
    :rtype: dict
    """
    grouped = {f'tags_{name}': [] for name in _TAG_CATEGORIES}
    names, ids = [], []
    for tag in item.get('tags') or []:
        title = tag.get('title')
        if title is None:
            continue
        names.append(title)
        ids.append(tag.get('tag_id'))
        column = f"tags_{(tag.get('type_name') or '').lower()}"
        if column in grouped:
            grouped[column].append(title)
    return {'tags': names, 'tag_ids': ids, **grouped}


def build_row(item: dict) -> dict:
    """
    Turn an API image into a table row.

    :param item: One entry from ``/api/v1/images``.
    :type item: dict
    :returns: The row to store.
    :rtype: dict
    """
    filename, ext = item.get('filename'), item.get('ext')
    src_filename = f'{filename}.{ext}' if filename and ext else None
    user = item.get('user') or {}
    return {
        'id': item.get('image_id'),
        'username': user.get('username'),
        'user_id': item.get('user_id'),
        'original_filename': item.get('original_filename'),
        'filename': filename,
        'ext': ext,
        'src_filename': src_filename,
        # The pre-CDN url still resolves and is what the previous table recorded, so it stays the
        # canonical one; the CDN url is kept beside it rather than replacing it.
        'file_url': f'{__site_url__}/images/{src_filename}' if src_filename else None,
        'cdn_url': item.get('url'),
        'thumbnail_url': item.get('thumbnail_url'),
        'medium_url': item.get('medium_url'),
        'large_url': item.get('large_url'),
        'md5_hash': item.get('md5_hash'),
        'file_size': item.get('filesize'),
        'width': item.get('width'),
        'height': item.get('height'),
        'mimetype': mimetypes.guess_type(src_filename)[0] if src_filename else None,
        'rating': item.get('rating'),
        'score': item.get('bayesian_rating'),
        'num_ratings': item.get('num_ratings'),
        'favorites': item.get('favorites'),
        'posts': item.get('posts'),
        'status': item.get('status'),
        'caption': item.get('caption') or None,
        'source_url': item.get('source_url') or None,
        'misc_metadata': item.get('miscmeta') or None,
        'replacement_id': item.get('replacement_id'),
        'created_at': to_timestamp(item.get('date_added')),
        **split_tags(item),
    }


def _drop_duplicate_ids(table: pa.Table) -> pa.Table:
    """
    Keep one row per id, the last written.

    A backstop rather than the mechanism: ids repeated within a run are already dropped as they
    arrive. This exists because a duplicated row is cheap to catch here and expensive to notice
    once it has been published, and because the count is vectorised - it costs nothing on a table
    that is already clean.

    :param table: Table about to be written.
    :type table: pa.Table
    :returns: The table, with any duplicate ids reduced to their last occurrence.
    :rtype: pa.Table
    """
    if table.num_rows == 0:
        return table
    distinct = pc.count_distinct(table.column('id')).as_py()
    if distinct == table.num_rows:
        return table
    logging.warning(f'{table.num_rows - distinct} duplicate id(s) in the table; keeping the last '
                    f'occurrence of each.')
    ids = table.column('id').to_numpy(zero_copy_only=False)
    # np.unique on the reversed array gives the first index of each id from the end, which maps
    # back to the last occurrence in the original order.
    _, first_from_end = np.unique(ids[::-1], return_index=True)
    keep = np.sort(len(ids) - 1 - first_from_end)
    return table.take(pa.array(keep))


def row_signature(row: dict) -> int:
    """
    Fingerprint this site's trigger fields for one row.

    :param row: Row built from an API image, or read back from the table.
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
    :returns: Mapping of image id to fingerprint.
    :rtype: dict
    """
    return _table_signatures(table, _UPDATE_TRIGGER_FIELDS, _UNORDERED_TRIGGER_FIELDS)


def sync(repository: str, max_time_limit: Optional[float] = 5 * 60 * 60,
         upload_time_span: float = 30, deploy_span: float = 20 * 60,
         max_empty_pages: int = 20, start_page: int = 1, workers: int = _WORKERS,
         proxy_pool: Optional[str] = None, brd_api_key: Optional[str] = None,
         brd_zone: Optional[str] = None):
    """
    Sync e-shuushuu image metadata into the target Hugging Face dataset repository.

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
    :param start_page: First page to request, for resuming a sweep.
    :type start_page: int
    :param workers: Concurrent fetchers.
    :type workers: int
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

    # One session per worker: curl_cffi sessions are not safe to share across threads, and a
    # distinct proxy session id per worker keeps their exit addresses stable if the pool is used.
    sessions = [get_eshuushuu_session(proxy_pool=proxy_pool, proxy_session=f'ess{i}',
                                      brd_api_key=brd_api_key, brd_zone=brd_zone)
                for i in range(workers)]

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)

    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='table.parquet'):
        base_table = pq.read_table(safe_hf_hub_download(
            hf_client, repo_id=repository, repo_type='dataset', filename='table.parquet'))
        table_schema = base_table.schema
        exist_sigs = table_signatures(base_table)
        logging.info(f'Existing table loaded, {plural_word(base_table.num_rows, "row")}, '
                     f'{plural_word(len(table_schema.names), "column")}.')
    else:
        base_table = None
        table_schema = None
        exist_sigs = {}
        logging.info('No table yet; building this index from scratch.')

    d_tags = {}
    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='tags.parquet'):
        df_tags = pd.read_parquet(safe_hf_hub_download(
            hf_client, repo_id=repository, repo_type='dataset', filename='tags.parquet'))
        d_tags = {int(item['id']): dict(item) for item in df_tags.to_dict('records')}
        del df_tags
        logging.info(f'Tag table loaded, {plural_word(len(d_tags), "tag")}.')

    chunks: List[pa.Table] = [base_table] if base_table is not None else []
    pending: List[dict] = []
    updates: dict = {}
    #: Ids already handled in this run. An offset walk re-serves images whenever an upload shifts
    #: the page boundaries, so this is a normal occurrence rather than a fault.
    seen_this_run = set()
    stats = {'ok': 0, 'updated': 0, 'skipped': 0, 'repeated': 0, 'failed': 0}
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

    def _merged_table() -> Optional[pa.Table]:
        nonlocal chunks
        _flush_pending()
        if not chunks:
            return None
        table = chunks[0] if len(chunks) == 1 else pa.concat_tables(chunks)
        table = apply_updates(table, updates)
        table = _drop_duplicate_ids(table)
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
            if table is None:
                return
            pq.write_table(table, os.path.join(td, 'table.parquet'))
            total_rows = table.num_rows
            preview = table.slice(0, 50).to_pandas()
            del table

            df_out = pd.DataFrame(list(d_tags.values()))
            if len(df_out):
                df_out = df_out.sort_values(['count', 'id'], ascending=[False, True])
                df_out.to_parquet(os.path.join(td, 'tags.parquet'), index=False)

            _write_readme(os.path.join(td, 'README.md'), total_rows=total_rows,
                          preview=preview, df_tags=df_out)

            limiter.try_acquire('hf upload limit')
            added = total_rows - _total_count
            logging.info(f'UPLOAD starting - {plural_word(added, "new image")}, '
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
        """Fold an image's tags into the tag table, keeping this index's own usage count."""
        for tag in item.get('tags') or []:
            tag_id = tag.get('tag_id')
            if tag_id is None:
                continue
            tag_id = int(tag_id)
            if tag_id not in d_tags:
                d_tags[tag_id] = {'id': tag_id, 'name': tag.get('title'),
                                  'type': tag.get('type'), 'type_name': tag.get('type_name'),
                                  'count': 0, 'usage_count': tag.get('usage_count')}
            entry = d_tags[tag_id]
            # The site's own numbers can move; ours is a count of the rows in this table.
            entry['name'] = tag.get('title') or entry.get('name')
            entry['type'] = tag.get('type') if tag.get('type') is not None else entry.get('type')
            entry['type_name'] = tag.get('type_name') or entry.get('type_name')
            if tag.get('usage_count') is not None:
                entry['usage_count'] = tag.get('usage_count')
            entry['count'] = (entry.get('count') or 0) + 1

    def _get_page(page: int, index: int = 0) -> Optional[list]:
        """Fetch one page, or None when it could not be had."""
        for attempt in range(_PAGE_ATTEMPTS):
            try:
                resp = sessions[index % len(sessions)].get(
                    f'{__site_url__}/api/v1/images', params={'page': str(page)})
                resp.raise_for_status()
                payload = resp.json()
                return payload.get('images') or []
            except REQUEST_ERRORS as err:
                wait = 2 ** attempt
                logging.warning(f'Page {page} failed ({attempt + 1}/{_PAGE_ATTEMPTS}) - '
                                f'{err!r}, retrying in {wait}s.')
                time.sleep(wait)
            except Exception as err:
                logging.warning(f'Page {page} gave an unusable response - {err!r}.')
                return None
        return None

    def _iter_items():
        """
        Walk the site newest-first by page, fetching a batch at a time in parallel.

        Fetching is concurrent; everything after it happens in page order on one thread, so the
        tag counts and the stop condition stay deterministic.
        """
        page = start_page
        empty_pages = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                if max_time_limit is not None and start_time + max_time_limit < time.time():
                    logging.info(f'Run deadline reached at page {page}. Resume with '
                                 f'--start-page {page}.')
                    return

                batch = list(range(page, page + _PREFETCH))
                fetched = list(pool.map(lambda pair: _get_page(*pair),
                                        [(p, i) for i, p in enumerate(batch)]))

                exhausted = False
                for number, items in zip(batch, fetched):
                    if items is None:
                        stats['failed'] += 1
                        logging.error(f'Giving up on page {number}. Resume with '
                                      f'--start-page {number}.')
                        return
                    if not items:
                        logging.info(f'Page {number} is empty; the sweep has reached the end.')
                        exhausted = True
                        break

                    fresh = 0
                    for item in items:
                        if item.get('image_id') is not None:
                            if item['image_id'] not in exist_sigs:
                                fresh += 1
                            yield item
                    if fresh:
                        empty_pages = 0
                    else:
                        empty_pages += 1
                        if empty_pages >= max_empty_pages:
                            logging.info(f'Stopping at page {number}: {empty_pages} consecutive '
                                         f'pages with nothing new.')
                            return
                if exhausted:
                    return
                page += _PREFETCH
                if (page // _PREFETCH) % 20 == 0:
                    logging.info(f'... reached page {page:,}, {stats["ok"]:,} recorded so far.')

    try:
        for item in _iter_items():
            if max_time_limit is not None and start_time + max_time_limit < time.time():
                break
            image_id = item['image_id']
            if image_id in seen_this_run:
                # Served twice because the offset moved under us. Nothing to do: the first sight
                # already recorded it, and counting its tags again would inflate them.
                stats['repeated'] += 1
                continue
            seen_this_run.add(image_id)
            row = build_row(item)
            signature = row_signature(row)
            known = image_id in exist_sigs

            if known and exist_sigs[image_id] == signature:
                stats['skipped'] += 1
                continue

            if known:
                updates[image_id] = row
                stats['updated'] += 1
            else:
                pending.append(row)
                if len(pending) >= _PENDING_FLUSH:
                    _flush_pending()
                # Counted on first sight only; re-counting on an update would inflate the totals.
                _ping_tags(item)
                stats['ok'] += 1

            exist_sigs[image_id] = signature
            has_update = True
            _deploy()
    finally:
        _deploy(force=True)

    logging.info(f'Done. {stats["ok"]} added, {stats["updated"]} updated, '
                 f'{stats["skipped"]} unchanged, {stats["repeated"]} re-served by a shifted '
                 f'offset, {stats["failed"]} pages failed. {len(d_tags):,} tags tracked, '
                 f'{len(seen_this_run):,} distinct ids seen this run.')


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
        print('size_categories:', file=f)
        print(f'- {number_to_tag(total_rows)}', file=f)
        print('annotations_creators:', file=f)
        print('- no-annotation', file=f)
        print('source_datasets:', file=f)
        print('- e-shuushuu', file=f)
        print('---', file=f)
        print('', file=f)
        print('## Records', file=f)
        print('', file=f)
        print(f'{plural_word(total_rows, "record")} in total, last updated on '
              f'`{current_time}`. Only {plural_word(len(preview), "record")} shown.', file=f)
        print('', file=f)
        columns = [c for c in ('id', 'width', 'height', 'mimetype', 'file_size', 'md5_hash',
                               'file_url') if c in preview.columns]
        print(preview[columns].to_markdown(index=False), file=f)
        print('', file=f)
        if df_tags is not None and len(df_tags):
            print('## Tags', file=f)
            print('', file=f)
            print(f'{plural_word(len(df_tags), "tag")} in total. `tag_ids` on a record joins to '
                  f'`id` here. `count` is how many records in this index carry the tag; '
                  f'`usage_count` is what the site reports site-wide.', file=f)
            print('', file=f)
            shown = df_tags[:30][[c for c in ('id', 'name', 'type', 'type_name', 'count',
                                              'usage_count') if c in df_tags.columns]]
            print(shown.to_markdown(index=False), file=f)
            print('', file=f)


@click.command()
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_ESS',
              default='deepghs/eshuushuu_index', show_default=True,
              help='Target dataset repository.')
@click.option('-t', '--max-time-limit', type=duration_type(allow_none=True), default='5h',
              show_default=True, help='Stop fetching after this duration, leaving room for the '
                                      'final upload. Use "none" to disable.')
@click.option('-u', '--upload-time-span', type=duration_type(), default=30, show_default=True,
              help='Minimum interval between upload attempts.')
@click.option('-d', '--deploy-span', type=duration_type(), default=20 * 60, show_default=True,
              help='Minimum interval between commits.')
@click.option('-E', '--max-empty-pages', type=int, default=20, show_default=True,
              help='Stop after this many consecutive pages carrying no id we have not seen '
                   'before. Raise it hugely for a full sweep.')
@click.option('-P', '--start-page', type=int, envvar='START_PAGE', default=1, show_default=True,
              help='First page to request. Use it to resume a sweep that ran out of time.')
@click.option('-w', '--workers', type=int, default=_WORKERS, show_default=True,
              help='Concurrent fetchers. The site tolerated eight without throttling; six is '
                   'used by default since the returns past that are small.')
@click.option('--proxy-pool', type=str, envvar='BRD_PROXY_URL', default=None,
              help='Bright Data proxy URL. Only used if direct access is refused, since the pool '
                   'is billed per request.')
@click.option('--brd-api-key', type=str, envvar='BRD_API_KEY', default=None,
              help='Bright Data API key, needed to allowlist this host on the zone.')
@click.option('--brd-zone', type=str, envvar='BRD_ZONE', default=None,
              help='Bright Data zone to allowlist into.')
def cli(repository: str, max_time_limit: Optional[float], upload_time_span: float,
        deploy_span: float, max_empty_pages: int, start_page: int, workers: int,
        proxy_pool: Optional[str], brd_api_key: Optional[str], brd_zone: Optional[str]):
    """Sync the e-shuushuu index."""
    logging.try_init_root(logging.INFO)
    sync(
        repository=repository,
        max_time_limit=max_time_limit,
        upload_time_span=upload_time_span,
        deploy_span=deploy_span,
        max_empty_pages=max_empty_pages,
        start_page=start_page,
        workers=workers,
        proxy_pool=proxy_pool,
        brd_api_key=brd_api_key,
        brd_zone=brd_zone,
    )


if __name__ == '__main__':
    cli()
