"""Index sync for safebooru.org.

Ported from the pyskeb prototype (``test/prepare/sb/index.py``). The published rows are unchanged;
what changed is how much of the table a run has to hold, how deep it can walk, and how much it
rewrites.

Why the prototype stopped
=========================

It kept the whole table as a list of dicts. At 5.76M rows and 25 columns that is 14-23 GB, which
does not start on a 16 GB runner - and the published data stops at 2025-08-03, which is about when
the table would have crossed that line.

Nothing here loads that table. Whether an id is already recorded is answered from ``meta.json``,
which the prototype already maintained: a sorted list of every id it has seen, 50.8 MB of JSON
against 2.4 GB of parquet. Only the shard currently being written to is held in memory.

Shards
======

The layout is ``tables/safebooru-N.parquet`` and the intent was a 2.5M-row cap per shard, though
the only shard that exists holds 5.76M - the rotation never took effect. Existing shards are left
exactly as they are; new rows go to a new shard, and a shard's row count is read from the parquet
footer rather than by downloading it.

This bounds a run at about 2 GB: the id set is around 350 MB, the shard being written at most
2.5M rows of Arrow. It also bounds what gets uploaded, since a deploy rewrites one shard rather
than the whole 2.4 GB table - at ~19,700 new posts a week that is the difference between tens of
megabytes and gigabytes per run.

The cost is that rows in sealed shards are never revisited. The 5.76M rows written before
2025-08-03 keep the tags they had then, however much the site has edited them since. Refreshing
those means deliberately reading a 2.4 GB shard back, which is a separate decision and a separate
run.

Walking the site
================

The gap is one contiguous stretch: the table reaches id 5,974,383 and the site's newest is
7,050,375, so about 1.05M posts sit above everything already held.

``pid`` is refused past 1000, which at ``limit=200`` is only 200k posts deep - not enough to cross
that gap. ``limit`` turns out to accept 1000 rather than the 200 the prototype used (1001 is
silently truncated to 1000, so asking for more looks like it worked), and ``tags=id:<N`` is a
working cursor with no depth limit. So the walk is a pure cursor at 1000 posts a request, which
puts the backlog at about 1,050 requests.

One trap worth knowing: of the id filters, only ``id:<N`` is honoured. ``id:>=N`` and ``id:A..B``
are **silently ignored** - both return the newest page instead of erroring - so neither can be used
to check a range or to seek forwards.

Two details that must match the existing rows
=============================================

``tags`` is not stored as the API sends it. The prototype wrote ``' '.join(['', *tags, ''])``, so
each value is surrounded by spaces (``" 1girl solo "``), which is what makes ``LIKE '% tag %'``
match a whole tag rather than a prefix. Tag names are also normalised through ``index_tags.parquet``
and any rename recorded in ``meta.json``'s ``tag_mapping``.

``scraped_at`` is the time the row was written, so it moves on every fetch. It must never be part
of the staleness comparison or every row would read as changed.

One prototype behaviour is deliberately not carried over: for a tag missing from
``index_tags.parquet`` it fetched that tag from the tag API, one request per unknown tag. Across
1.05M posts of roughly twenty tags each that is thousands of extra requests to fill in a table that
a separate job already maintains. Unknown tags are recorded with ``id: -1`` instead and gain their
metadata when that job next runs.
"""
import datetime
import html
import json
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
from huggingface_hub import HfFileSystem
from pyrate_limiter import Duration, Limiter, Rate

from inf.utils.duration import duration_type
from inf.utils.safe import safe_hf_hub_download, safe_upload_directory_as_directory
from inf.utils.session import REQUEST_ERRORS
from inf.utils.upsert import adds_anything, apply_updates
from inf.utils.upsert import row_signature as _row_signature
from .base import __site_url__, get_safebooru_session

mimetypes.add_type('image/webp', '.webp')

#: Posts per request. The API accepts 1000 and silently truncates anything larger, so asking for
#: more than this gains nothing while looking like it worked. The prototype used 200.
_POSTS_PER_PAGE = 1000

#: Rows buffered as dicts before folding into an Arrow chunk.
_PENDING_FLUSH = 20000

#: Attempts per page before giving up. A cursor walk cannot skip a failed page: leaving the cursor
#: alone re-requests it forever, and advancing past it drops those posts silently.
_PAGE_ATTEMPTS = 5

#: Tag categories, as this API numbers them.
_TAG_TYPES = {
    -1: 'unknown',
    0: 'general',
    1: 'artist',
    3: 'copyright',
    4: 'character',
    5: 'meta',
}

#: Fields whose change makes a stored row worth rewriting.
#:
#: Not every column. ``score`` and ``change`` drift on their own, and ``scraped_at`` is set to the
#: current time on every fetch - including it would mark every row changed on every pass.
_UPDATE_TRIGGER_FIELDS = (
    'hash', 'file_url', 'preview_url', 'sample_url', 'directory', 'image', 'filename',
    'mimetype', 'width', 'height', 'rating', 'tags', 'source', 'parent_id',
    'sample', 'sample_width', 'sample_height', 'status', 'has_notes', 'owner',
)

#: Of those, the ones holding space-delimited lists whose order the site does not hold stable.
#: ``tags`` is compared as a set, so a reordered tag string does not read as an edit.
_UNORDERED_TRIGGER_FIELDS = frozenset({'tags'})


def format_tags(tags: List[str]) -> str:
    """
    Render a tag list the way the published table stores it.

    Surrounded by spaces on both sides, which is what lets a consumer match a whole tag with
    ``LIKE '% tag %'`` rather than catching prefixes. Matches the 5.76M rows already published.

    :param tags: Tag names, already normalised and deduplicated.
    :type tags: List[str]
    :returns: The stored representation.
    :rtype: str
    """
    return ' '.join(['', *tags, ''])


def parse_tags(value: Optional[str]) -> List[str]:
    """
    Split a tag string, whether it came from the API or back out of the table.

    :param value: Space-delimited tags, possibly with the surrounding spaces this table adds.
    :type value: Optional[str]
    :returns: Tag names, deduplicated, order preserved.
    :rtype: List[str]
    """
    raw = (html.unescape(part) for part in re.split(r'\s+', value or '') if part)
    return list(dict.fromkeys(raw))


def build_row(item: dict, tags: List[str]) -> dict:
    """
    Turn an API item into a table row.

    Three of the 25 published columns are not sent by the API: ``filename`` is the API's ``image``,
    ``mimetype`` is guessed from the file url, and ``scraped_at`` is now. Everything else the API
    sends is carried through, which is how the column set came to be.

    :param item: One entry from the post index API.
    :type item: dict
    :param tags: Normalised tag names for this post.
    :type tags: List[str]
    :returns: The row to store.
    :rtype: dict
    """
    file_url = item.get('file_url')
    return {
        'id': item.get('id'),
        'width': item.get('width'),
        'height': item.get('height'),
        'filename': item.get('image'),
        'mimetype': mimetypes.guess_type(file_url)[0] if file_url else None,
        'rating': item.get('rating'),
        'file_url': file_url,
        **item,
        'tags': format_tags(tags),
        'scraped_at': time.time(),
    }


def row_signature(row: dict) -> int:
    """
    Fingerprint this site's trigger fields for one row.

    :param row: Row built from an API item, or read back from the stored table.
    :type row: dict
    :returns: Hash over :data:`_UPDATE_TRIGGER_FIELDS`.
    :rtype: int
    """
    return _row_signature(_set_valued(row), _UPDATE_TRIGGER_FIELDS, _UNORDERED_TRIGGER_FIELDS)


def table_signatures(table: pa.Table) -> dict:
    """
    Fingerprint every row of a shard, keyed by id.

    :param table: Shard as read from the hub.
    :type table: pa.Table
    :returns: Mapping of post id to fingerprint.
    :rtype: dict
    """
    signatures = {}
    for batch in table.to_batches(max_chunksize=65536):
        for row in batch.to_pylist():
            signatures[row['id']] = row_signature(row)
    return signatures


def _set_valued(row: dict) -> dict:
    """
    Present the space-delimited fields as lists, so ordering can be normalised away.

    ``tags`` is stored as text, so the shared comparison cannot see it as a list and would treat a
    reordered tag string as an edit. Splitting it first lets the unordered-field handling apply.

    :param row: Row to prepare.
    :type row: dict
    :returns: A shallow copy with those fields split.
    :rtype: dict
    """
    prepared = dict(row)
    for field in _UNORDERED_TRIGGER_FIELDS:
        if isinstance(prepared.get(field), str):
            prepared[field] = parse_tags(prepared[field])
    return prepared


def shard_number(path: str) -> int:
    """
    The N in ``tables/safebooru-N.parquet``.

    :param path: Shard path.
    :type path: str
    :returns: Its number, or -1 when the name does not match.
    :rtype: int
    """
    match = re.search(r'safebooru-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1


def sync(repository: str, max_time_limit: Optional[float] = 5 * 60 * 60,
         upload_time_span: float = 30, deploy_span: float = 15 * 60,
         max_part_rows: int = 2500000, max_empty_pages: int = 20,
         start_below_id: Optional[int] = None, proxy_pool: Optional[str] = None,
         brd_api_key: Optional[str] = None, brd_zone: Optional[str] = None):
    """
    Sync Safebooru post metadata into the target Hugging Face dataset repository.

    :param repository: Target dataset repository.
    :type repository: str
    :param max_time_limit: Stop fetching after this many seconds, leaving room for the final
        upload. None disables the limit.
    :type max_time_limit: Optional[float]
    :param upload_time_span: Minimum seconds between upload attempts.
    :type upload_time_span: float
    :param deploy_span: Minimum seconds between commits.
    :type deploy_span: float
    :param max_part_rows: Rows a shard may hold before the next one is started.
    :type max_part_rows: int
    :param max_empty_pages: Stop after this many consecutive pages carrying no unseen id.
    :type max_empty_pages: int
    :param start_below_id: Begin the walk just below this id rather than at the newest post.
    :type start_below_id: Optional[int]
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
    hf_fs = HfFileSystem(token=os.environ.get('HF_TOKEN'),
                         endpoint=os.environ.get('HF_ENDPOINT'))
    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    session = get_safebooru_session(proxy_pool=proxy_pool, proxy_session='sbindex',
                                    brd_api_key=brd_api_key, brd_zone=brd_zone)

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)

    # Which ids are already held, and where the last walk stopped. This is the whole reason the
    # 2.4 GB table never has to be read.
    exist_ids = set()
    tag_mapping = {}
    last_min_id = None
    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='meta.json'):
        with open(safe_hf_hub_download(hf_client, repo_id=repository, repo_type='dataset',
                                       filename='meta.json'), 'r') as f:
            meta_info = json.load(f)
        exist_ids = set(meta_info.get('exist_ids') or ())
        tag_mapping = dict(meta_info.get('tag_mapping') or {})
        last_min_id = meta_info.get('last_min_id')
        logging.info(f'Known ids loaded from meta.json: {plural_word(len(exist_ids), "id")}, '
                     f'{plural_word(len(tag_mapping), "tag rename")}, '
                     f'last walk reached {last_min_id}.')

    d_origin_tags = {}
    if hf_client.file_exists(repo_id=repository, repo_type='dataset',
                             filename='index_tags.parquet'):
        df_index_tags = pd.read_parquet(safe_hf_hub_download(
            hf_client, repo_id=repository, repo_type='dataset',
            filename='index_tags.parquet')).replace(np.NaN, None)
        d_origin_tags = {item['name']: item for item in df_index_tags.to_dict('records')}
        del df_index_tags
        logging.info(f'Tag metadata loaded, {plural_word(len(d_origin_tags), "tag")}.')

    d_tags = {}
    if hf_client.file_exists(repo_id=repository, repo_type='dataset', filename='tags.parquet'):
        df_tags = pd.read_parquet(safe_hf_hub_download(
            hf_client, repo_id=repository, repo_type='dataset',
            filename='tags.parquet')).replace(np.NaN, None)
        d_tags = {item['name']: item for item in df_tags.to_dict('records')}
        del df_tags

    # Pick the shard to write into. A sealed shard is never downloaded; its row count comes from
    # the parquet footer, which is a few kilobytes.
    shard_paths = sorted(hf_fs.glob(f'datasets/{repository}/tables/safebooru-*.parquet'),
                         key=shard_number)
    base_table = None
    table_schema = None
    if shard_paths:
        newest = shard_paths[-1]
        with hf_fs.open(newest, 'rb') as fp:
            parquet_file = pq.ParquetFile(fp)
            newest_rows = parquet_file.metadata.num_rows
            table_schema = parquet_file.schema_arrow
        if newest_rows >= max_part_rows:
            current_ptr = shard_number(newest) + 1
            logging.info(f'Shard {newest.split("/")[-1]} holds '
                         f'{plural_word(newest_rows, "row")}, at or past the {max_part_rows:,} '
                         f'cap; starting shard {current_ptr} and leaving it untouched.')
        else:
            current_ptr = shard_number(newest)
            logging.info(f'Shard {newest.split("/")[-1]} holds '
                         f'{plural_word(newest_rows, "row")} and has room; loading it.')
            base_table = pq.read_table(safe_hf_hub_download(
                hf_client, repo_id=repository, repo_type='dataset',
                filename=f'tables/safebooru-{current_ptr}.parquet'))
    else:
        current_ptr = 1
        logging.info('No shard exists yet; starting at shard 1.')

    exist_sigs = table_signatures(base_table) if base_table is not None else {}
    base_index = ({post_id: offset for offset, post_id
                   in enumerate(base_table.column('id').to_pylist())}
                  if base_table is not None else {})

    chunks: List[pa.Table] = [base_table] if base_table is not None else []
    pending: List[dict] = []
    updates: dict = {}
    stats = {'ok': 0, 'updated': 0, 'skipped': 0, 'sealed': 0, 'no_gain': 0, 'failed': 0}
    _total_count = len(exist_ids)
    _shard_rows = base_table.num_rows if base_table is not None else 0
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
        table = table.sort_by([('id', 'descending')])
        chunks = [table]
        updates.clear()
        return table

    def _rotate_if_full():
        """Seal the shard being written once it reaches the cap and start the next one."""
        nonlocal current_ptr, chunks, exist_sigs, base_index, base_table, _shard_rows
        if _shard_rows < max_part_rows:
            return
        logging.info(f'Shard {current_ptr} reached {_shard_rows:,} rows; sealing it and '
                     f'starting shard {current_ptr + 1}.')
        current_ptr += 1
        # Drop every reference to the sealed shard: it is on the hub, it will not be rewritten,
        # and holding it would keep gigabytes alive for the rest of the run.
        chunks = []
        base_table = None
        exist_sigs = {}
        base_index = {}
        _shard_rows = 0

    def _deploy(force: bool = False):
        nonlocal _last_update, has_update, _total_count, _shard_rows
        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            table = _merged_table()
            if table is None:
                return
            os.makedirs(os.path.join(td, 'tables'), exist_ok=True)
            pq.write_table(table, os.path.join(td, 'tables',
                                               f'safebooru-{current_ptr}.parquet'))
            _shard_rows = table.num_rows
            preview = table.slice(0, 50).to_pandas()
            del table

            with open(os.path.join(td, 'meta.json'), 'w') as f:
                json.dump({
                    'exist_ids': sorted(exist_ids),
                    'tag_mapping': tag_mapping,
                    'last_min_id': last_min_id,
                }, f)

            df_out = pd.DataFrame(list(d_tags.values()))
            if len(df_out):
                df_out = df_out.sort_values(['count', 'type'], ascending=[False, True])
                df_out.to_parquet(os.path.join(td, 'tags.parquet'), index=False)

            _write_readme(os.path.join(td, 'README.md'), total_rows=len(exist_ids),
                          shard=current_ptr, shard_rows=_shard_rows, preview=preview,
                          df_tags=df_out)

            limiter.try_acquire('hf upload limit')
            added = len(exist_ids) - _total_count
            logging.info(f'UPLOAD starting - {plural_word(added, "new post")}, '
                         f'{len(exist_ids):,} ids in total, shard {current_ptr} at '
                         f'{_shard_rows:,} rows.')
            started = time.time()
            # Only the shard being written, meta.json, tags.parquet and the README go up. Sealed
            # shards are not in the directory, so they are not touched.
            safe_upload_directory_as_directory(
                repo_id=repository, repo_type='dataset', local_directory=td, path_in_repo='.',
                message=f'Add {plural_word(added, "new record")} into index',
            )
            logging.info(f'UPLOAD done in {time.time() - started:.0f}s.')
            has_update = False
            _last_update = time.time()
            _total_count = len(exist_ids)
        _rotate_if_full()

    def _normalise_tags(item: dict) -> List[str]:
        """
        Resolve a post's tags through the tag metadata, keeping usage counts.

        Unlike the prototype, a tag missing from ``index_tags.parquet`` is not looked up over the
        network - that would be thousands of extra requests to populate a table another job owns.
        """
        current = []
        for tag in parse_tags(item.get('tags')):
            if tag not in d_tags:
                if tag in d_origin_tags:
                    d_tags[tag] = {**d_origin_tags[tag], 'count': 0}
                else:
                    d_tags[tag] = {'id': -1, 'type': -1, 'name': tag, 'count': 0,
                                   'ambiguous': False}
            count = d_tags[tag].get('count', 0)
            if tag in d_origin_tags:
                d_tags[tag].update(d_origin_tags[tag])

            origin_tag = tag
            tag = d_tags[origin_tag].get('name') or origin_tag
            if origin_tag != tag:
                tag_mapping[origin_tag] = tag
                if tag in d_tags and tag != origin_tag:
                    total = d_tags[tag].get('count', 0) + d_tags[origin_tag].get('count', 0)
                    d_tags[tag] = d_tags.pop(origin_tag)
                    d_tags[tag]['count'] = total
                else:
                    d_tags[tag] = d_tags.pop(origin_tag)
            else:
                tag_mapping.pop(origin_tag, None)

            if tag in current:
                d_tags[tag]['count'] = count
            else:
                d_tags[tag]['count'] = count + 1
                current.append(tag)
        return current

    def _get_posts(below_id: Optional[int]) -> list:
        params = {'page': 'dapi', 's': 'post', 'q': 'index', 'json': '1',
                  'limit': str(_POSTS_PER_PAGE)}
        if below_id is not None:
            params['tags'] = f'id:<{below_id}'
        resp = session.get(f'{__site_url__}/index.php', params=params)
        resp.raise_for_status()
        # Past the end this API answers with an XML error document rather than JSON or an empty
        # list, so a parse failure here means "no more", not a transport problem.
        try:
            payload = resp.json()
        except Exception:
            return []
        if isinstance(payload, dict):
            payload = payload.get('post') or []
        return payload if isinstance(payload, list) else []

    def _stored_row(post_id: int) -> Optional[dict]:
        offset = base_index.get(post_id)
        if offset is None or base_table is None:
            return None
        return base_table.slice(offset, 1).to_pylist()[0]

    def _iter_items():
        """Walk the site newest-first by cursor, each request bounded by the lowest id seen."""
        nonlocal last_min_id
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
                if post_id not in exist_ids:
                    fresh += 1
                yield item

            if lowest is None:
                logging.info(f'Page below {below_id}: {len(items)} posts, none carrying an id.')
                return
            logging.info(f'Page below {below_id}: {len(items)} posts, {fresh} new, '
                         f'lowest id {lowest:,}.')
            if last_min_id is None or lowest < last_min_id:
                last_min_id = lowest
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
            known_here = post_id in exist_sigs

            if post_id in exist_ids and not known_here:
                # Held in a sealed shard, which this run does not open. Nothing to compare
                # against and nothing to write.
                stats['sealed'] += 1
                continue

            row = build_row(item, _normalise_tags(item))
            signature = row_signature(row)

            if known_here and exist_sigs[post_id] == signature:
                stats['skipped'] += 1
                continue

            stored = _stored_row(post_id)
            if known_here and stored is not None and not adds_anything(
                    _set_valued(stored), _set_valued(row), exist_sigs[post_id],
                    _UPDATE_TRIGGER_FIELDS, _UNORDERED_TRIGGER_FIELDS):
                # The fingerprint moved but the merge would not: a field the API stopped sending.
                stats['no_gain'] += 1
                continue

            if known_here:
                updates[post_id] = row
                stats['updated'] += 1
                logging.info(f'Post {post_id} changed ({stats["updated"]} updated this run).')
            else:
                pending.append(row)
                if len(pending) >= _PENDING_FLUSH:
                    _flush_pending()
                stats['ok'] += 1
                logging.info(f'Post {post_id} confirmed ({stats["ok"]} added this run).')

            exist_ids.add(post_id)
            exist_sigs[post_id] = signature
            has_update = True
            _deploy()
    finally:
        _deploy(force=True)

    logging.info(f'Done. {stats["ok"]} added, {stats["updated"]} updated, '
                 f'{stats["skipped"]} unchanged, {stats["no_gain"]} offering nothing new, '
                 f'{stats["sealed"]} in sealed shards, {stats["failed"]} request failures. '
                 f'{len(exist_ids):,} ids known in total.')


def _write_readme(md_file: str, total_rows: int, shard: int, shard_rows: int,
                  preview: pd.DataFrame, df_tags: pd.DataFrame):
    """
    Render the dataset README.

    :param md_file: Destination path.
    :type md_file: str
    :param total_rows: Ids known across every shard.
    :type total_rows: int
    :param shard: Shard currently being written.
    :type shard: int
    :param shard_rows: Rows in that shard.
    :type shard_rows: int
    :param preview: Newest rows of that shard, for the sample table.
    :type preview: pd.DataFrame
    :param df_tags: Tag table, already sorted.
    :type df_tags: pd.DataFrame
    """
    current_time = datetime_now()
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
        print('size_categories:', file=f)
        print(f'- {number_to_tag(total_rows)}', file=f)
        print('annotations_creators:', file=f)
        print('- no-annotation', file=f)
        print('source_datasets:', file=f)
        print('- safebooru', file=f)
        print('---', file=f)
        print('', file=f)
        print('## Records', file=f)
        print('', file=f)
        print(f'{plural_word(total_rows, "record")} in total, last updated on '
              f'`{current_time}`. Shard {shard} holds {plural_word(shard_rows, "row")}; '
              f'only {plural_word(len(preview), "record")} shown.', file=f)
        print('', file=f)
        columns = [c for c in ('id', 'width', 'height', 'rating', 'mimetype', 'tags', 'file_url')
                   if c in preview.columns]
        print(preview[columns].to_markdown(index=False), file=f)
        print('', file=f)
        if df_tags is not None and len(df_tags):
            print('## Tags', file=f)
            print('', file=f)
            print(f'{plural_word(len(df_tags), "tag")} in total.', file=f)
            print('', file=f)
            shown = df_tags[:30][[c for c in ('id', 'name', 'type', 'count')
                                  if c in df_tags.columns]]
            print(shown.to_markdown(index=False), file=f)
            print('', file=f)


def datetime_now() -> str:
    """
    Local time, formatted for the README.

    :returns: Timestamp with timezone.
    :rtype: str
    """
    return datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')


@click.command()
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_SB',
              default='deepghs/safebooru_index', show_default=True,
              help='Target dataset repository.')
@click.option('-t', '--max-time-limit', type=duration_type(allow_none=True), default='5h',
              show_default=True, help='Stop fetching after this duration, leaving room for the '
                                      'final upload. Use "none" to disable.')
@click.option('-u', '--upload-time-span', type=duration_type(), default=30, show_default=True,
              help='Minimum interval between upload attempts.')
@click.option('-d', '--deploy-span', type=duration_type(), default=15 * 60, show_default=True,
              help='Minimum interval between commits.')
@click.option('-p', '--max-part-rows', type=int, default=2500000, show_default=True,
              help='Rows a shard may hold before the next one is started. Existing shards are '
                   'never rewritten, whatever they hold.')
@click.option('-E', '--max-empty-pages', type=int, default=20, show_default=True,
              help='Stop after this many consecutive pages carrying no id we have not seen '
                   'before.')
@click.option('-B', '--start-below-id', type=int, envvar='START_BELOW_ID', default=None,
              help='Begin the walk just below this id instead of at the newest post. Use it to '
                   'resume a backfill that ran out of time, or to work a known gap.')
@click.option('--proxy-pool', type=str, envvar='BRD_PROXY_URL', default=None,
              help='Bright Data proxy URL. Only used if direct access is refused, since the pool '
                   'is billed per request.')
@click.option('--brd-api-key', type=str, envvar='BRD_API_KEY', default=None,
              help='Bright Data API key, needed to allowlist this host on the zone.')
@click.option('--brd-zone', type=str, envvar='BRD_ZONE', default=None,
              help='Bright Data zone to allowlist into.')
def cli(repository: str, max_time_limit: Optional[float], upload_time_span: float,
        deploy_span: float, max_part_rows: int, max_empty_pages: int,
        start_below_id: Optional[int], proxy_pool: Optional[str], brd_api_key: Optional[str],
        brd_zone: Optional[str]):
    """Sync the Safebooru index."""
    logging.try_init_root(logging.INFO)
    sync(
        repository=repository,
        max_time_limit=max_time_limit,
        upload_time_span=upload_time_span,
        deploy_span=deploy_span,
        max_part_rows=max_part_rows,
        max_empty_pages=max_empty_pages,
        start_below_id=start_below_id,
        proxy_pool=proxy_pool,
        brd_api_key=brd_api_key,
        brd_zone=brd_zone,
    )


if __name__ == '__main__':
    cli()
