import gc
import json
import math
import mimetypes
import os
import re
import time
from functools import partial
from typing import Optional, List

import click
import httpx
import json_repair
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.utils import number_to_tag
from pyrate_limiter import Duration, Limiter, Rate

from inf.utils.duration import duration_type
from inf.utils.safe import safe_hf_hub_download, safe_upload_directory_as_directory
from inf.utils.session import REQUEST_ERRORS, srequest
from .base import get_session
from .tag import _get_tag_info

mimetypes.add_type('image/webp', '.webp')


def loads_zerochan_json(text: str) -> dict:
    """
    Parse a zerochan JSON body, repairing the site's unescaped quotes.

    Zerochan builds its JSON by string concatenation and never escapes quotes inside values, so
    a tag whose own name contains one produces a body no strict parser will accept::

        "Kokonose "Konoha" Haruka"        should have been  "Kokonose \\"Konoha\\" Haruka"
        "Don't Say "Lazy""                should have been  "Don't Say \\"Lazy\\""

    This is not rare and it is not random: 17 of 17 sampled ``failed_ids`` failed for exactly
    this reason, which is most of why that list had grown to 10,673 entries. Those posts exist
    and serve fine - only the encoding is broken - so they were being discarded over a site-side
    formatting bug.

    ``json_repair`` recovers them without loss. Verified on those 17: every key came back, tag
    counts matched a line-by-line reading of the raw body exactly, and quote-bearing tags such
    as ``Kokonose "Konoha" Haruka`` survived verbatim. It is also safe on well-formed input -
    on 8 valid responses its output was identical to ``json.loads``.

    :param text: Raw response body.
    :type text: str
    :returns: Parsed object.
    :rtype: dict
    :raises json.JSONDecodeError: When even repair cannot produce an object.
    """
    # Every exit from here is either a dict or a JSONDecodeError. Callers already treat that
    # exception as "record this id as failed and carry on", so anything else escaping - a bare
    # ValueError, a TypeError on a None body - kills a run that may have collected hundreds of
    # records. One such escape did exactly that.
    if not isinstance(text, str) or not text.strip():
        raise json.JSONDecodeError('empty body', text if isinstance(text, str) else '', 0)

    try:
        strict = json.loads(text)
    except json.JSONDecodeError:
        pass
    except RecursionError as err:
        # The standard library's scanner recurses per nesting level, so a deeply nested body
        # blows the stack here too - before any repair is attempted. RecursionError descends
        # from RuntimeError, not ValueError, so it is not caught by the clause above either.
        raise json.JSONDecodeError(f'body nests too deeply to parse: {err}', text, 0) from err
    else:
        # A valid JSON array is still not a record; `[1, 2, 3]` must not reach the caller.
        if isinstance(strict, dict):
            return strict
        raise json.JSONDecodeError(f'body is a {type(strict).__name__}, not an object', text, 0)

    # Not every non-JSON body is worth repairing. Some ids answer 200 with an HTML page for a
    # different post - a merged duplicate - and handing that to a repair pass is pointless and
    # hazardous: it recurses until the parser blows its stack.
    if text.lstrip().startswith('<'):
        raise json.JSONDecodeError('body is HTML, not JSON', text, 0)

    try:
        repaired = json_repair.loads(text)
    except json.JSONDecodeError:
        raise
    except (ValueError, RecursionError) as err:
        # json_repair reports a blown recursion limit as a bare ValueError, which is *not* a
        # JSONDecodeError - JSONDecodeError subclasses ValueError, not the other way round, so a
        # caller catching JSONDecodeError sees it sail straight through.
        raise json.JSONDecodeError(f'repair failed: {err}', text, 0) from err

    if not isinstance(repaired, dict) or not repaired:
        raise json.JSONDecodeError(
            f'repaired body is a {type(repaired).__name__}, not a record', text, 0)
    return repaired


def _as_text(value) -> Optional[str]:
    """
    Return ``value`` if it is a usable string, otherwise None.

    :returns: Non-empty string, or None.
    :rtype: Optional[str]
    """
    return value.strip() or None if isinstance(value, str) else None


def _as_int(value) -> Optional[int]:
    """
    Coerce ``value`` to an int when that is lossless and meaningful, otherwise None.

    :returns: Integer, or None.
    :rtype: Optional[int]
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def normalise_record(item: dict) -> dict:
    """
    Coerce a parsed record's fields to the types the rest of the pipeline assumes.

    A repaired body is structurally a dict but says nothing about what is *inside* it. Repairing
    ``"tags": ["Female", "Don't Say "Lazy"", ...]`` can leave a fragment parsed as a number, and
    a number is truthy, so `filter(bool, tags)` passed it straight through to ``quote_plus``,
    which raised ``TypeError: quote_from_bytes() expected bytes`` and killed a run that had
    already collected thousands of records.

    Rather than patch that one call site, everything the record contributes downstream is
    normalised here: strings must be non-empty strings, ids and dimensions must be integers, and
    ``tags`` must be a list of strings. Anything else becomes None or is dropped, which the
    callers already handle - an id that cannot be read is a failed record, not a crash.

    :param item: Parsed record, possibly repaired.
    :type item: dict
    :returns: Record with predictable field types.
    :rtype: dict
    :raises ValueError: When the record has no usable id, since nothing can be keyed without it.
    """
    post_id = _as_int(item.get('id'))
    if post_id is None:
        raise ValueError(f'record has no usable id: {item.get("id")!r}')
    tags = [t for t in (item.get('tags') if isinstance(item.get('tags'), list) else []) or []
            if _as_text(t)]
    return {
        'id': post_id,
        'width': _as_int(item.get('width')),
        'height': _as_int(item.get('height')),
        'size': _as_int(item.get('size')),
        'full': _as_text(item.get('full')),
        'small': _as_text(item.get('small')),
        'medium': _as_text(item.get('medium')),
        'large': _as_text(item.get('large')),
        'hash': _as_text(item.get('hash')),
        'source': _as_text(item.get('source')),
        'primary': _as_text(item.get('primary')),
        'tags': [_as_text(t) for t in tags],
    }


def parse_id_ranges(text: Optional[str]) -> List[int]:
    """
    Expand a ``lo-hi,lo-hi`` string into a list of ids.

    Needed because some posts never appear in the listing at all. Verified on the gap
    4,353,199..4,353,341: all three ids sampled fetch fine one at a time, yet paging down from
    4,353,342 jumps straight to 4,353,198 and skips every one of the 143 ids between. Historic
    crawls walked the listing, which is why those stretches were never filled and why no
    ``--max-empty-pages`` value can reach them - they have to be requested by id.

    A single id may be given on its own, so ``100-105,200`` is six ids.

    :param text: Ranges as ``lo-hi`` or bare ids, comma separated. None or blank yields nothing.
    :type text: Optional[str]
    :returns: Sorted, de-duplicated ids.
    :rtype: List[int]
    :raises ValueError: On a malformed range, rather than silently fetching the wrong span.
    """
    if not text or not text.strip():
        return []
    ids = set()
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk.lstrip('-'):
            low, _, high = chunk.partition('-')
            start, end = int(low), int(high)
            if end < start:
                raise ValueError(f'range {chunk!r} ends before it starts')
            if end - start > 1_000_000:
                raise ValueError(f'range {chunk!r} spans more than a million ids')
            ids.update(range(start, end + 1))
        else:
            ids.add(int(chunk))
    return sorted(ids)


def _prefix_ids(extra_ids: Optional[List[int]], failed_ids, try_failed_first: bool) -> List[int]:
    """
    Build the id list to visit before walking the listing, newest first.

    :returns: Sorted descending, de-duplicated.
    :rtype: List[int]
    """
    prefix = set(extra_ids or ())
    if try_failed_first:
        prefix.update(failed_ids)
    return sorted(prefix, reverse=True)


def get_record(zerochan_id: int, session: Optional[requests.Session] = None):
    session = session or get_session()
    resp = srequest(
        session, 'GET', f'https://www.zerochan.net/{zerochan_id}',
        params={'json': '1'}
    )
    return normalise_record(loads_zerochan_json(resp.text))


def sync(repository: str, max_time_limit: Optional[float] = 50 * 60, upload_time_span: float = 30,
         tag_refresh_time: float = 365 * 24 * 60 * 60, deploy_span: float = 45 * 60, sync_mode: bool = False,
         try_failed_ids_first: bool = False, start_from_id: Optional[int] = None,
         max_tag_refresh: int = 300, max_empty_pages: int = 10,
         extra_ids: Optional[List[int]] = None):
    """Sync Zerochan post metadata and tag state into the target Hugging Face dataset repository."""
    start_time = time.time()
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    logging.info(f'Try failed ids first: {try_failed_ids_first!r}')
    session = get_session()
    delete_detached_cache()

    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    # The table is held as an Arrow table and never materialised as Python dicts. Measured on
    # the live 4.1M-row file: read_parquet + to_dict('records') + DataFrame(records) peaked at
    # 11.41 GB, which is what killed three runs on a 16 GB runner with "the hosted runner lost
    # communication with the server". The same work through Arrow peaks at 6.67 GB, because a
    # dict-per-row turns every column into millions of boxed Python objects.
    if hf_fs.exists(f'datasets/{repository}/zerochan.parquet'):
        base_table = pq.read_table(safe_hf_hub_download(
            hf_client,
            repo_id=repository,
            repo_type='dataset',
            filename='zerochan.parquet',
        ))
        exist_ids = set(base_table.column('id').to_pylist())
        pre_ids = set(exist_ids)
    else:
        base_table = None
        exist_ids = set()
        pre_ids = set()
    # Only rows fetched during this run live here, so it stays small.
    records = []

    if hf_fs.exists(f'datasets/{repository}/meta.json'):
        meta_info = json.loads(hf_fs.read_text(f'datasets/{repository}/meta.json'))
        failed_ids = set(meta_info['failed_ids'])
    else:
        failed_ids = set()

    if hf_fs.exists(f'datasets/{repository}/tags.json'):
        tags_raw = json.loads(hf_fs.read_text(f'datasets/{repository}/tags.json'))
        d_tags = {item['name']: item for item in tags_raw}
    else:
        d_tags = {}

    ids_in_table = [*pre_ids, *failed_ids]
    if ids_in_table:
        min_id = min(ids_in_table)
    else:
        min_id = None

    # Refreshing a stale tag costs a serial request against a rate-limited site, and the cache
    # is old enough that every tag looks stale: of 564,908 cached tags the median age is 830
    # days. Under the previous 15-day window that meant re-fetching essentially all of them,
    # which is why one run managed 35 posts in 22 minutes while spending 306 requests on 305
    # distinct tags. Tag metadata - category, aliases, description - changes on a scale of
    # years, so a stale entry is still a good answer. Refresh a bounded number per run and
    # serve the rest from cache; over successive runs the whole cache still turns over, but no
    # single run is held hostage to it. A tag with no cached entry at all is always fetched:
    # there is nothing to fall back on.
    tag_refreshes = [0]

    def ping_tag(tag, primary: bool = False):
        cached = d_tags.get(tag)
        fresh = cached is not None and cached['created_at'] + tag_refresh_time > time.time()
        budget_spent = tag_refreshes[0] >= max_tag_refresh
        if cached is not None and (fresh or budget_spent):
            if primary:
                d_tags[tag]['strict'] += 1
            else:
                d_tags[tag]['count'] += 1
            return tag
        else:
            if cached is not None:
                tag_refreshes[0] += 1
            logging.info(f'Query for tag {tag!r}.')
            tag_info = _get_tag_info(tag)
            if not tag_info:
                logging.warning(f'Empty tag, dropped - {tag!r}.')
                return None

            tag = tag_info['name']
            if tag in d_tags:
                strict = d_tags[tag]['strict']
                count = d_tags[tag]['count']
            else:
                strict = 0
                count = 0

            d_tags[tag] = tag_info
            d_tags[tag]['created_at'] = time.time()
            if primary:
                d_tags[tag]['strict'] = strict + 1
                d_tags[tag]['count'] = count
            else:
                d_tags[tag]['strict'] = strict
                d_tags[tag]['count'] = count + 1
            return tag

    def _iter_image_ids(offset: Optional[int] = None, prefix_ids: Optional[List[int]] = None):
        nonlocal exist_ids
        prefix_ids = list(prefix_ids or [])
        for id_ in prefix_ids:
            if id_ not in exist_ids:
                yield id_
                exist_ids.add(id_)

        ptc = 0
        while True:
            if max_time_limit is not None and start_time + max_time_limit < time.time():
                return

            params = {'json': '1'}
            if offset is not None and offset > 0:
                params['o'] = str(offset)
            resp = srequest(
                session, 'GET', 'https://www.zerochan.net/',
                params=params
            )
            ids = list(map(int, re.findall(r'"id":\s*(\d+)\s*,', resp.text)))
            has_new = False
            new_count = 0
            for id_ in ids:
                if id_ not in exist_ids:
                    yield id_
                    has_new = True
                    new_count += 1

            if not has_new or new_count <= 1:
                ptc += 1
            else:
                ptc = 0
            logging.info(f'Current continuous empty pages: {ptc!r}, has new: {has_new!r}, new count: {new_count!r}')
            if sync_mode and ptc >= max_empty_pages:
                logging.info(f'Stopping: {ptc} consecutive pages with nothing new, at or above '
                             f'the --max-empty-pages limit of {max_empty_pages}.')
                break
            if not ids:
                break
            offset = min(ids)

    def _merged_table():
        """
        Combine the stored table with this run's new rows, newest id first.

        Built through Arrow rather than pandas: the stored table is over four million rows, and
        turning it into dicts to rebuild a DataFrame is what pushed three runs past the runner's
        memory and cost them with "the hosted runner lost communication with the server".
        """
        if not records:
            return base_table
        fresh = pa.Table.from_pylist(
            records, schema=base_table.schema if base_table is not None else None)
        if base_table is None:
            return fresh.sort_by([('id', 'descending')])
        # from_pylist infers types from a handful of rows, so a column that is all-None in this
        # batch would come back null-typed and refuse to concatenate. Casting first keeps the
        # stored schema authoritative.
        fresh = fresh.cast(base_table.schema)
        return pa.concat_tables([base_table, fresh]).sort_by([('id', 'descending')])

    _last_update, has_update = None, False
    _total_count = base_table.num_rows if base_table is not None else 0

    def _deploy(force=False):
        nonlocal _last_update, has_update, _total_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            parquet_file = os.path.join(td, 'zerochan.parquet')
            table = _merged_table()
            pq.write_table(table, parquet_file)
            total_rows = table.num_rows
            preview_rows = table.slice(0, 50)
            del table
            gc.collect()

            df_tags = pd.DataFrame([
                {
                    'name': d_item['name'],
                    'category': d_item['category'],
                    'raw_category': d_item.get('raw_category'),
                    **{
                        f'lang_{k}': (v or '')
                        for k, v in d_item['langs'].items()
                    },
                    'count': d_item['count'],
                    'strict': d_item['strict'],
                }
                for d_item in d_tags.values()
            ])
            df_tags = df_tags.replace(np.NaN, '')
            df_tags = df_tags.sort_values(['count', 'category'], ascending=[False, True])
            pcolumns = ['name', 'category', 'raw_category', 'count', 'strict']
            columns = [*pcolumns, *filter(lambda x: x.startswith('lang_'), df_tags.columns)]
            df_tags = df_tags[columns]
            s_columns = [name for name in columns if name != 'raw_category']

            with open(os.path.join(td, 'tags.json'), 'w') as f:
                json.dump(list(d_tags.values()), f)

            with open(os.path.join(td, 'meta.json'), 'w') as f:
                json.dump({
                    'failed_ids': sorted(failed_ids),
                    'exist_ids': sorted(exist_ids),
                }, f)

            with open(os.path.join(td, 'README.md'), 'w') as f:
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
                print('- zerochan', file=f)
                print('---', file=f)
                print('', file=f)

                print('## Records', file=f)
                print(f'', file=f)
                # Only the preview rows are pulled into pandas; the full table stays in Arrow.
                df_records_shown = preview_rows.to_pandas()[
                    ['id', 'width', 'height', 'file_size', 'mimetype', 'primary_tag', 'file_url', ]]
                print(f'{plural_word(total_rows, "record")} in total. '
                      f'Only {plural_word(len(df_records_shown), "record")} shown.', file=f)
                print(f'', file=f)
                print(df_records_shown.to_markdown(index=False), file=f)
                print(f'', file=f)
                print('## Tags', file=f)
                print(f'', file=f)
                print(f'{plural_word(len(df_tags), "tag")} in total.', file=f)
                print(f'', file=f)
                for type_id in sorted(set(df_tags['category'])):
                    df_tags_type = df_tags[df_tags['category'] == type_id]
                    if type_id != 'unknown':
                        df_tags_type = df_tags_type[s_columns]
                    df_tags_shown = df_tags_type[:30]
                    print(f'These are the top {plural_word(len(df_tags_shown), "tag")} '
                          f'({plural_word(len(df_tags_type), "tag")} in total) '
                          f'of type `{type_id}`:', file=f)
                    print('', file=f)
                    print(df_tags_shown.to_markdown(index=False), file=f)
                    print('', file=f)

            limiter.try_acquire('hf upload limit')
            safe_upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Add {plural_word(total_rows - _total_count, "new record")} into index',
            )
            has_update = False
            _last_update = time.time()
            _total_count = total_rows

    is_data_safe = True
    try:
        for post_id in _iter_image_ids(
                offset=min_id if not sync_mode else start_from_id,
                prefix_ids=_prefix_ids(extra_ids, failed_ids, try_failed_ids_first),
        ):
            if max_time_limit is not None and start_time + max_time_limit < time.time():
                break
            # if post_id in pre_ids and sync_mode:
            #     break
            if post_id in exist_ids:
                continue

            logging.info(f'Post {post_id!r} confirmed.')
            try:
                item = get_record(post_id, session=session)
            except (*REQUEST_ERRORS, httpx.HTTPError, json.JSONDecodeError, ValueError) as err:
                logging.info(f'Post {post_id!r} skipped due to error - {err!r}.')
                failed_ids.add(post_id)
                has_update = True
                continue

            if not item['full']:
                logging.warning(f'Post {post_id!r} has no file url, skipped.')
                failed_ids.add(post_id)
                has_update = True
                continue
            mimetype, _ = mimetypes.guess_type(item['full'])
            tags = item['tags']
            try:
                row = {
                    'id': item['id'],
                    'width': item['width'],
                    'height': item['height'],
                    'file_size': item['size'],
                    'mimetype': mimetype,
                    'file_url': item['full'],
                    'small_url': item['small'],
                    'medium_url': item['medium'],
                    'large_url': item['large'],
                    'hash': item['hash'],
                    'source': item.get('source'),
                    'primary_tag': ping_tag(item['primary'], primary=True) if item['primary'] else None,
                    'tags': json.dumps(list(filter(bool, map(partial(ping_tag, primary=False), tags)))),
                }
                records.append(row)
                exist_ids.add(item['id'])
                if item['id'] in failed_ids:
                    failed_ids.remove(item['id'])
            except:
                is_data_safe = False
                raise

            has_update = True
            _deploy()

    finally:
        if is_data_safe:
            _deploy(force=True)


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Sync Zerochan post metadata and tag state into the target Hugging Face dataset repository. '
         'The command iterates upstream post IDs, refreshes tag metadata on demand, '
         'and periodically writes parquet, tag and meta snapshots back to the repository.',
)
@click.option(
    '-r', '--repository',
    type=str,
    envvar='REMOTE_REPOSITORY_ZC',
    required=True,
    show_envvar=True,
    help='Target Hugging Face dataset repository to read from and write to.',
)
@click.option(
    '-m', '--max-time-limit',
    type=duration_type(allow_none=True),
    # The final deploy happens after this deadline and takes as long as a full parquet upload
    # (16 minutes when measured). 5.7h left only two minutes before the job's own 6h ceiling.
    default=5 * 60 * 60,
    show_default=True,
    help='Stop the sync after this total runtime. Use none or unlimited to disable the limit.',
)
@click.option(
    '-u', '--upload-time-span',
    type=duration_type(),
    default=30,
    show_default=True,
    help='Minimum interval between upload batches.',
)
@click.option(
    '-t', '--tag-refresh-time',
    type=duration_type(),
    default=365 * 24 * 60 * 60,
    show_default=True,
    help='Refresh cached tag metadata when older than this threshold. Tag metadata changes on '
         'a scale of years, and every refresh is a serial request against a rate-limited site.',
)
@click.option(
    '-R', '--extra-id-ranges',
    type=str,
    default=None,
    help='Fetch these ids directly before walking the listing, as `lo-hi,lo-hi` or bare ids. '
         'Some posts never appear in the listing even though they fetch fine one at a time, so '
         'no amount of paging reaches them; this is the only way to fill those stretches.',
)
@click.option(
    '-E', '--max-empty-pages',
    type=int,
    default=10,
    show_default=True,
    help='In sync mode, stop after this many consecutive listing pages containing nothing new. '
         'Ten suits day-to-day catch-up, where anything new sits at the head. It is far too '
         'small for filling a gap in the middle of the range: the walk starts at the head, and '
         'ten pages is under 500 posts, so an already-covered stretch longer than that ends the '
         'run before the gap is ever reached. Raise it when backfilling.',
)
@click.option(
    '-T', '--max-tag-refresh',
    type=int,
    default=300,
    show_default=True,
    help='Refresh at most this many already-cached tags per run. Tags with no cached entry are '
         'always fetched; this only bounds re-fetching stale ones, so a cold cache cannot '
         'starve the post backlog.',
)
@click.option(
    '-d', '--deploy-span',
    type=duration_type(),
    default=45 * 60,
    show_default=True,
    help='Minimum interval between deploy or upload commits. Every deploy re-uploads the whole '
         'parquet, which measured 16 minutes at 723 MB, so a short interval spends most of the '
         'run re-sending the same file rather than fetching posts.',
)
@click.option(
    '-s', '--sync-mode/--no-sync-mode',
    default=True,
    show_default=True,
    help='Continue incremental sync behavior instead of a fresh rebuild.',
)
@click.option(
    '-f', '--try-failed-ids-first/--no-try-failed-ids-first',
    default=False,
    show_default=True,
    help='Retry previously failed record IDs before scanning new ones.',
)
@click.option(
    '-i', '--start-from-id',
    type=int,
    default=None,
    help='Start scanning from this explicit record ID instead of the stored pointer.',
)
def cli(repository: str, max_time_limit: Optional[float], upload_time_span: float, tag_refresh_time: float,
        max_tag_refresh: int, max_empty_pages: int, extra_id_ranges: Optional[str],
        deploy_span: float, sync_mode: bool, try_failed_ids_first: bool, start_from_id: Optional[int]):
    logging.try_init_root(logging.INFO)
    return sync(
        repository=repository,
        max_time_limit=max_time_limit,
        upload_time_span=upload_time_span,
        tag_refresh_time=tag_refresh_time,
        max_tag_refresh=max_tag_refresh,
        max_empty_pages=max_empty_pages,
        extra_ids=parse_id_ranges(extra_id_ranges),
        deploy_span=deploy_span,
        sync_mode=sync_mode,
        try_failed_ids_first=try_failed_ids_first,
        start_from_id=start_from_id,
    )


if __name__ == '__main__':
    cli()
