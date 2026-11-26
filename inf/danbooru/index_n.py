"""Index sync for danbooru.donmai.us.

Publication rate
================

About 49,600 posts a week, measured 2026-08-13 from ``counts/posts.json?tags=age:..1w``, which
returned 49,641. That endpoint counts server-side over the whole database, so it needs no sampling
and no assumption about how evenly uploads land.

Sampling the 200 newest posts instead would have suggested 56,499/week, but those 200 span only
0.02 days - half an hour - so that number describes a moment, not a week. Prefer the counts
endpoint wherever a site offers one.
"""
import gc
import math
import mimetypes
import os
import shutil
import tempfile
import time
from typing import Dict, Iterator, List, Optional, Tuple

import click
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.utils import number_to_tag
from pyrate_limiter import Rate, Duration, Limiter
from waifuc.source import DanbooruSource

from inf.utils.duration import duration_type
from inf.utils.safe import safe_hf_hub_download, safe_upload_directory_as_directory
from inf.utils.session import srequest

mimetypes.add_type('image/webp', '.webp')


#: Rows per row group of the written table, and per batch while merging. Small enough that a batch of 46 columns
#: of text stays around a hundred MB; the stored table had one million rows (1.4 GB) per row group before.
_ROW_GROUP_ROWS = 100000

#: Columns shown in the README preview.
_PREVIEW_COLUMNS = ['id', 'image_width', 'image_height', 'rating', 'mimetype', 'file_size', 'file_url']


def _iter_stored_batches(path: str, batch_size: int, columns: Optional[List[str]] = None) -> Iterator[pa.RecordBatch]:
    """
    The batches of a parquet file, a row group at a time, each row group through a reader of its own.

    One reader walked over the whole file keeps what it decoded: measured on the 5.5 million row table, its
    live arrow memory grew by about 600 MB per row group, 3.2 GB by the end. A reader per row group frees
    everything between groups, 73 MB live, and halves the peak.
    """
    for group in range(pq.ParquetFile(path).metadata.num_row_groups):
        reader = pq.ParquetFile(path)
        yield from reader.iter_batches(batch_size=batch_size, row_groups=[group], columns=columns)
        del reader


class _StoredIndex:
    """
    The ids and file urls of the stored table, which is all the crawl needs to tell a new post from a known one.

    The table is millions of rows by 46 columns, about 7.5 GB once decoded, so it is never loaded. Only these two
    columns are, as arrow data and a sorted id array, and lookups are binary searches rather than a python dict.
    """

    def __init__(self, path: Optional[str]):
        batches = [] if path is None else list(_iter_stored_batches(path, _ROW_GROUP_ROWS, ['id', 'file_url']))
        if not batches:
            self._ids = np.zeros(0, dtype=np.int64)
            self._order = np.zeros(0, dtype=np.int64)
            self._urls = None
        else:
            table = pa.Table.from_batches(batches)
            ids = table.column('id').to_numpy().astype(np.int64)
            self._order = np.argsort(ids, kind='stable')
            self._ids = ids[self._order]
            self._urls = table.column('file_url')
        self._changed: Dict[int, Optional[str]] = {}

    def __len__(self):
        return len(self._ids)

    def get(self, post_id: int) -> Tuple[bool, Optional[str]]:
        """``(known, file_url)`` of a post, counting the posts recorded since this index was loaded."""
        if post_id in self._changed:
            return True, self._changed[post_id]
        pos = int(np.searchsorted(self._ids, post_id))
        if pos < len(self._ids) and self._ids[pos] == post_id:
            return True, self._urls[int(self._order[pos])].as_py()
        return False, None

    def record(self, post_id: int, file_url: Optional[str]):
        self._changed[post_id] = file_url


def _merged_schema(old_schema: Optional[pa.Schema], records: List[dict]) -> pa.Schema:
    """The old schema, plus a column for every key the new records carry that the old table did not have."""
    inferred = pa.Table.from_pylist(records).schema
    if old_schema is None:
        fields = list(inferred)
    else:
        fields = list(old_schema)
        known = set(old_schema.names)
        fields.extend(f for f in inferred if f.name not in known)
    # a column that is all null in the new records has no type to infer
    return pa.schema([pa.field(f.name, pa.large_string()) if pa.types.is_null(f.type) else f for f in fields])


def _conform(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """The table with exactly the columns of ``schema``, in order, missing columns filled with nulls."""
    columns = []
    for field in schema:
        if field.name in table.column_names:
            column = table.column(field.name)
            columns.append(column if column.type == field.type else column.cast(field.type))
        else:
            columns.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(columns, schema=schema)


def _sort_by_id_desc(table: pa.Table) -> pa.Table:
    return table.take(pc.sort_indices(table, sort_keys=[('id', 'descending')]))


def _merge_records(old_path: Optional[str], new_records: List[dict], out_path: str) -> Tuple[int, pa.Table, int]:
    """
    Write ``out_path``: the stored table with ``new_records`` merged in, newest id first, a new record replacing
    the stored row of the same id.

    The stored table is streamed batch by batch and the output written batch by batch, so what is in memory is a
    few batches and the new records, however large the table is. The stored table is expected to be sorted by id,
    descending, as this job has always written it; each stored batch is merged with the new records that fall
    into its id range, and a table that is not sorted loses no row, it only comes out less ordered.

    :returns: ``(rows written, first rows for the README, rows in the stored table)``
    """
    new_ids = np.array([int(x['id']) for x in new_records], dtype=np.int64)
    if old_path is None:
        schema = _merged_schema(None, new_records)
        new_table = _sort_by_id_desc(_conform(pa.Table.from_pylist(new_records), schema))
        pq.write_table(new_table, out_path, row_group_size=_ROW_GROUP_ROWS)
        return new_table.num_rows, new_table.slice(0, 50), 0

    pf = pq.ParquetFile(old_path)
    old_rows = pf.metadata.num_rows
    schema = _merged_schema(pf.schema_arrow.remove_metadata(), new_records)
    del pf
    pending = _sort_by_id_desc(_conform(pa.Table.from_pylist(new_records), schema))
    pending_ids = pending.column('id').to_numpy()  # descending

    written, head, replaced = 0, None, 0
    with pq.ParquetWriter(out_path, schema, compression='snappy') as writer:

        def _emit(table: pa.Table):
            nonlocal written, head
            if table.num_rows == 0:
                return
            if head is None or head.num_rows < 50:
                head = table.slice(0, 50) if head is None else pa.concat_tables([head, table]).slice(0, 50)
            writer.write_table(table, row_group_size=_ROW_GROUP_ROWS)
            written += table.num_rows

        for batch in _iter_stored_batches(old_path, _ROW_GROUP_ROWS):
            table = pa.Table.from_batches([batch])
            keep = ~np.isin(table.column('id').to_numpy(), new_ids)
            replaced += int(len(keep) - keep.sum())
            table = _conform(table.filter(pa.array(keep)), schema)
            if table.num_rows == 0:
                continue
            # the new records at or above this batch's lowest id that are not written yet join the batch
            lowest = int(pc.min(table.column('id')).as_py())
            take = int(np.searchsorted(-pending_ids, -lowest, side='right'))
            if take:
                table = _sort_by_id_desc(pa.concat_tables([pending.slice(0, take), table]))
                pending, pending_ids = pending.slice(take), pending_ids[take:]
            _emit(table)
        _emit(pending)
    logging.info(f'Merged {plural_word(len(new_records), "new record")}, {plural_word(replaced, "row")} replaced.')
    return written, head, old_rows


def sync(repository: str, upload_time_span: float = 30, deploy_span: float = 5 * 60,
         max_time_limit: Optional[float] = 50 * 60, sync_mode: bool = False,
         site_username: Optional[str] = None, site_apikey: Optional[str] = None,
         site_golden: bool = False, start_from_id: Optional[int] = None):
    """Sync Danbooru post metadata into the target Hugging Face dataset repository."""
    start_time = time.time()
    delete_detached_cache()
    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    hf_client = get_hf_client(hf_token=os.environ['HF_TOKEN'])
    hf_fs = get_hf_fs(hf_token=os.environ['HF_TOKEN'])

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        hf_client.update_repo_visibility(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.db filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    # The stored table is never loaded whole: its path is kept, its ids and urls are indexed, and every deploy
    # streams it into the next version of the file (see _merge_records).
    stored_path: Optional[str] = None
    if hf_fs.exists(f'datasets/{repository}/records.parquet'):
        stored_path = safe_hf_hub_download(
            hf_client,
            repo_id=repository,
            repo_type='dataset',
            filename='records.parquet',
        )
    known = _StoredIndex(stored_path)
    gc.collect()
    logging.info(f'Stored table indexed, {plural_word(len(known), "row")}.')

    new_records: Dict[int, dict] = {}
    _last_update, has_update = None, False
    _total_count = len(known)
    # the files this run wrote and may delete; the downloaded file lives in the hub cache and is never touched
    work_dir = tempfile.mkdtemp(prefix='danbooru_index_n_')

    def _deploy(force=False):
        nonlocal stored_path, new_records, _last_update, has_update, _total_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            table_parquet_file = os.path.join(td, 'records.parquet')
            total, df_head, previous_count = _merge_records(stored_path, list(new_records.values()),
                                                            table_parquet_file)
            if total < previous_count:
                raise RuntimeError(
                    f'Refusing to deploy: combined record count {total} '
                    f'is smaller than previous {previous_count}. Aborting to protect remote data.'
                )

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
                print(f'- {number_to_tag(total)}', file=f)
                print('annotations_creators:', file=f)
                print('- no-annotation', file=f)
                print('source_datasets:', file=f)
                print('- danbooru', file=f)
                print('---', file=f)
                print('', file=f)

                print('## Records', file=f)
                print('', file=f)
                df_records_shown = df_head.select(_PREVIEW_COLUMNS).to_pandas()
                print(f'{plural_word(total, "record")} in total. '
                      f'Only {plural_word(len(df_records_shown), "record")} shown.', file=f)
                print('', file=f)
                print(df_records_shown.to_markdown(index=False), file=f)
                print('', file=f)

            limiter.try_acquire('hf upload limit')
            safe_upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Adding {plural_word(total - _total_count, "new record")} into index',
                hf_token=os.environ['HF_TOKEN'],
            )

            # the file just uploaded is the stored table of the next deploy
            kept_path = os.path.join(work_dir, 'records.parquet.next')
            shutil.move(table_parquet_file, kept_path)
        if stored_path is not None and os.path.dirname(stored_path) == work_dir:
            os.remove(stored_path)
        stored_path = os.path.join(work_dir, 'records.parquet')
        os.replace(kept_path, stored_path)

        new_records = {}
        has_update = False
        _last_update = time.time()
        _total_count = total
        del df_head, df_records_shown
        gc.collect()

    if site_apikey and site_username:
        logging.info(f'Initializing session with username {site_username!r} and api_key {site_apikey!r} ...')
        source = DanbooruSource(['1girl'], api_key=site_apikey, username=site_username)
    else:
        logging.info('Initializing session without any API key ...')
        source = DanbooruSource(['1girl'])
    source._prune_session()
    session = source.session

    max_query_pages = 5000 if site_golden else 1000
    min_image_id: Optional[int] = None
    q_tags = []
    image_id_lower_bound: int = 7306660
    if start_from_id is not None:
        min_image_id = start_from_id
        q_tags = [f'id:<{min_image_id}']
        logging.info(f'Start from explicit image id boundary {start_from_id!r}.')

    def _iter_items():
        nonlocal min_image_id, q_tags
        no_item_cnt = 0
        page = 1
        while True:
            if page > max_query_pages:
                page = 1
                assert min_image_id is not None
                q_tags = [f'id:<{min_image_id}']

            logging.info(f'Requesting for newest page {page!r} ...')
            resp = srequest(session, 'GET', 'https://danbooru.donmai.us/posts.json', params={
                'limit': '200',
                'page': str(page),
                "tags": ' '.join(q_tags),
            }, auth=source.auth)
            has_new_items = False
            for pitem in resp.json():
                is_known, known_url = known.get(pitem['id'])
                if not is_known or known_url != pitem.get('file_url'):
                    yield pitem
                    if pitem.get('file_url'):
                        has_new_items = True
                if not min_image_id or pitem['id'] < min_image_id:
                    min_image_id = pitem['id']

            if has_new_items:
                no_item_cnt = 0
            else:
                no_item_cnt = no_item_cnt + 1
            if sync_mode and no_item_cnt >= 10:
                return
            if min_image_id and min_image_id < image_id_lower_bound:
                return

            page += 1

    try:
        for item in _iter_items():
            if max_time_limit is not None and start_time + max_time_limit < time.time():
                break
            # if item['id'] in exist_ids:
            #     logging.info(f'Post {item["id"]!r} already crawled, skipped.')
            #     continue
            if not item.get('file_url'):
                logging.info(f'Empty url post {item["id"]!r}, maybe you need a golden account to scrape that, skipped.')
                continue

            logging.info(f'Post {item["id"]!r} confirmed!')
            if known.get(item['id'])[0]:
                logging.info(f'Post {item["id"]!r} already exist, but has file_url changed.')
            del item['media_asset']
            if item.get('file_url'):
                item['mimetype'], _ = mimetypes.guess_type(item['file_url'])
            else:
                item['mimetype'] = None
            new_records[item['id']] = item
            known.record(item['id'], item.get('file_url'))
            has_update = True
            _deploy()
        _deploy(force=True)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Sync Danbooru posts into the target Hugging Face dataset repository. '
         'The command pulls upstream post metadata in batches, refreshes local parquet snapshots, '
         'and periodically deploys repository artifacts during the run.',
)
@click.option(
    '-r', '--repository',
    type=str,
    envvar='REMOTE_REPOSITORY_DB_N',
    required=True,
    show_envvar=True,
    help='Target Hugging Face dataset repository to read from and write to.',
)
@click.option(
    '-u', '--upload-time-span',
    type=duration_type(),
    default=30,
    show_default=True,
    help='Minimum interval between upload batches.',
)
@click.option(
    '-d', '--deploy-span',
    type=duration_type(),
    default=5 * 60,
    show_default=True,
    help='Minimum interval between deploy or upload commits.',
)
@click.option(
    '-m', '--max-time-limit',
    type=duration_type(allow_none=True),
    default=5.7 * 60 * 60,
    show_default=True,
    help='Stop the sync after this total runtime. Use none or unlimited to disable the limit.',
)
@click.option(
    '-s', '--sync-mode/--no-sync-mode',
    default=True,
    show_default=True,
    help='Continue incremental sync behavior instead of a fresh rebuild.',
)
@click.option(
    '-U', '--site-username',
    type=str,
    envvar='DANBOORU_USERNAME',
    default=None,
    show_envvar=True,
    help='Site username used for authenticated upstream requests.',
)
@click.option(
    '-A', '--site-apikey',
    type=str,
    envvar='DANBOORU_APITOKEN',
    default=None,
    show_envvar=True,
    help='Site API key used for authenticated upstream requests.',
)
@click.option(
    '-g', '--site-golden/--no-site-golden',
    default=True,
    show_default=True,
    help='Enable the Danbooru golden-account request mode.',
)
@click.option(
    '-i', '--start-from-id',
    type=int,
    default=None,
    help='Start scanning below this explicit Danbooru post ID boundary.',
)
def cli(repository: str, upload_time_span: float, deploy_span: float, max_time_limit: Optional[float],
        sync_mode: bool, site_username: Optional[str], site_apikey: Optional[str], site_golden: bool,
        start_from_id: Optional[int]):
    logging.try_init_root(logging.INFO)
    return sync(
        repository=repository,
        upload_time_span=upload_time_span,
        deploy_span=deploy_span,
        max_time_limit=max_time_limit,
        sync_mode=sync_mode,
        site_username=site_username,
        site_apikey=site_apikey,
        site_golden=site_golden,
        start_from_id=start_from_id,
    )


if __name__ == '__main__':
    cli()
