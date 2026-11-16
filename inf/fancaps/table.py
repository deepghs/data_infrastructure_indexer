"""The ``fancaps.parquet`` table: its fixed layout, scanning it, and merging new rows into it.

The table is **one file** with nine columns and is kept sorted by ``id`` descending, which is how the
earlier job wrote it (a whole-frame sort, ``to_parquet(index=False)``). At 29 million rows that frame
does not fit a CI runner twice over, so both the scan and the merge here stream row groups and never
hold more than a few of them.

Layout, copied from the file already published and checked against it:

* ``id``, ``bangumi_id``, ``episode_id`` are ``int64``; the six other columns are ``string``;
* Snappy, dictionary encoding on every column, row groups of 1,048,576 rows, the pandas metadata block;
* ``url`` is always ``https://cdni.fancaps.net/file/fancaps-animeimages/{id}.jpg``; ``thumbnail_url`` is the
  address the site's own page gives (the host changed over the years, so it is not derivable);
* every episode id maps to exactly one ``(bangumi, episode)`` description.
"""
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .client import CDN_URL, Episode

COLUMNS = ['id', 'url', 'thumbnail_url', 'bangumi_id', 'bangumi_name', 'bangumi_url', 'episode_id', 'episode_name',
           'episode_url']
ROW_GROUP_SIZE = 1 << 20
_PANDAS_METADATA = (
    '{"index_columns": [], "column_indexes": [], "columns": ['
    '{"name": "id", "field_name": "id", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, '
    '{"name": "url", "field_name": "url", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, '
    '{"name": "thumbnail_url", "field_name": "thumbnail_url", "pandas_type": "unicode", "numpy_type": "object", '
    '"metadata": null}, '
    '{"name": "bangumi_id", "field_name": "bangumi_id", "pandas_type": "int64", "numpy_type": "int64", '
    '"metadata": null}, '
    '{"name": "bangumi_name", "field_name": "bangumi_name", "pandas_type": "unicode", "numpy_type": "object", '
    '"metadata": null}, '
    '{"name": "bangumi_url", "field_name": "bangumi_url", "pandas_type": "unicode", "numpy_type": "object", '
    '"metadata": null}, '
    '{"name": "episode_id", "field_name": "episode_id", "pandas_type": "int64", "numpy_type": "int64", '
    '"metadata": null}, '
    '{"name": "episode_name", "field_name": "episode_name", "pandas_type": "unicode", "numpy_type": "object", '
    '"metadata": null}, '
    '{"name": "episode_url", "field_name": "episode_url", "pandas_type": "unicode", "numpy_type": "object", '
    '"metadata": null}], "creator": {"library": "pyarrow", "version": "15.0.2"}, "pandas_version": "2.3.1"}'
)
SCHEMA = pa.schema([
    ('id', pa.int64()), ('url', pa.string()), ('thumbnail_url', pa.string()), ('bangumi_id', pa.int64()),
    ('bangumi_name', pa.string()), ('bangumi_url', pa.string()), ('episode_id', pa.int64()),
    ('episode_name', pa.string()), ('episode_url', pa.string()),
], metadata={'pandas': _PANDAS_METADATA})


class EpisodeInfo(NamedTuple):
    bangumi_id: int
    bangumi_name: str
    bangumi_url: str
    name: str
    url: str


class TableScan(NamedTuple):
    rows: int
    #: Every image id, ascending and unique.
    ids: np.ndarray
    #: ``{episode id: image count}``.
    episode_counts: Dict[int, int]
    #: ``{episode id: description}``.
    episodes: Dict[int, EpisodeInfo]


def empty_table() -> pa.Table:
    return SCHEMA.empty_table()


def episode_rows(bangumi_id: int, bangumi_name: str, bangumi_url: str, episode: Episode,
                 images: List[Tuple[int, str]]) -> pa.Table:
    """The rows of one finished episode, in the published layout."""
    n = len(images)
    return pa.table({
        'id': pa.array([i for i, _ in images], pa.int64()),
        'url': pa.array([CDN_URL.format(id=i) for i, _ in images], pa.string()),
        'thumbnail_url': pa.array([t for _, t in images], pa.string()),
        'bangumi_id': pa.array([bangumi_id] * n, pa.int64()),
        'bangumi_name': pa.array([bangumi_name] * n, pa.string()),
        'bangumi_url': pa.array([bangumi_url] * n, pa.string()),
        'episode_id': pa.array([episode.id] * n, pa.int64()),
        'episode_name': pa.array([episode.name] * n, pa.string()),
        'episode_url': pa.array([episode.url] * n, pa.string()),
    }, schema=SCHEMA)


def scan_table(path: Optional[str]) -> TableScan:
    """Ids, per-episode counts and episode descriptions of a published table, one row group at a time."""
    if not path:
        return TableScan(0, np.zeros(0, dtype=np.int64), {}, {})
    pf = pq.ParquetFile(path)
    chunks, counts, infos, rows = [], {}, {}, 0
    keys = ['bangumi_id', 'bangumi_name', 'bangumi_url', 'episode_id', 'episode_name', 'episode_url']
    for batch in pf.iter_batches(batch_size=ROW_GROUP_SIZE, columns=['id'] + keys):
        table = pa.Table.from_batches([batch])
        rows += table.num_rows
        chunks.append(table['id'].to_numpy())
        eps, n = np.unique(table['episode_id'].to_numpy(), return_counts=True)
        for e, c in zip(eps.tolist(), n.tolist()):
            counts[e] = counts.get(e, 0) + c
        distinct = table.select(keys).group_by(keys).aggregate([])
        for row in zip(*[distinct[k].to_pylist() for k in keys]):
            infos[row[3]] = EpisodeInfo(row[0], row[1], row[2], row[4], row[5])
    ids = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    ids = np.unique(ids)
    if len(ids) != rows:
        raise ValueError(f'{path!r} holds {rows} rows but only {len(ids)} distinct image ids.')
    return TableScan(rows, ids, counts, infos)


def fresh_rows(new: pa.Table, known_ids: np.ndarray) -> pa.Table:
    """``new`` without ids the table already holds and without repeats, sorted by id descending."""
    if new.num_rows == 0:
        return new
    ids = new['id'].to_numpy()
    keep = ~np.isin(ids, known_ids, assume_unique=False)
    new = new.filter(pa.array(keep))
    if new.num_rows:
        _, first = np.unique(new['id'].to_numpy(), return_index=True)
        new = new.take(pa.array(np.sort(first)))
    return new.sort_by([('id', 'descending')])


def _merged(base: Optional[pq.ParquetFile], new: pa.Table, batch_rows: int) -> Iterator[pa.Table]:
    """Both inputs are sorted by id descending; yield the merge in the same order, a batch at a time."""
    new_ids = new['id'].to_numpy()
    cursor = 0
    if base is not None:
        for batch in base.iter_batches(batch_size=batch_rows):
            table = pa.Table.from_batches([batch]).cast(SCHEMA)
            ids = table['id'].to_numpy()
            high, low = ids[0], ids[-1]
            # new rows above this batch come first; ``new_ids`` is descending so negate for searchsorted
            above = cursor + int(np.searchsorted(-new_ids[cursor:], -high, side='right'))
            if above > cursor:
                yield new.slice(cursor, above - cursor)
                cursor = above
            inside = cursor + int(np.searchsorted(-new_ids[cursor:], -low, side='right'))
            if inside > cursor:
                combined = pa.concat_tables([table, new.slice(cursor, inside - cursor)])
                yield combined.sort_by([('id', 'descending')])
                cursor = inside
            else:
                yield table
    if cursor < new.num_rows:
        yield new.slice(cursor)


def merge_table(base_path: Optional[str], new: pa.Table, out_path: str, batch_rows: int = 1 << 18) -> int:
    """
    Write ``base`` plus ``new`` to ``out_path``, sorted by id descending, in the published layout.

    ``new`` must already be free of ids ``base`` holds (see :func:`fresh_rows`). Row groups of exactly
    :data:`ROW_GROUP_SIZE` rows are produced, so the result has the same physical shape as a whole-frame
    ``to_parquet``. Returns the number of rows written.
    """
    new = new.cast(SCHEMA).sort_by([('id', 'descending')])
    base = pq.ParquetFile(base_path) if base_path else None
    written, pending = 0, []
    pending_rows = 0
    with pq.ParquetWriter(out_path, SCHEMA, compression='snappy', use_dictionary=True, write_statistics=True) as writer:
        def _flush(final: bool):
            nonlocal pending, pending_rows, written
            if not pending:
                return
            table = pa.concat_tables(pending)
            offset = 0
            while table.num_rows - offset >= ROW_GROUP_SIZE or (final and offset < table.num_rows):
                chunk = table.slice(offset, min(ROW_GROUP_SIZE, table.num_rows - offset))
                writer.write_table(chunk, row_group_size=ROW_GROUP_SIZE)
                written += chunk.num_rows
                offset += chunk.num_rows
            rest = table.slice(offset)
            pending, pending_rows = ([rest] if rest.num_rows else []), rest.num_rows

        for table in _merged(base, new, batch_rows):
            pending.append(table)
            pending_rows += table.num_rows
            if pending_rows >= ROW_GROUP_SIZE:
                _flush(final=False)
        _flush(final=True)
    return written


def check_table(path: str, rows: int) -> None:
    """Cheap structural check of a written table: layout, row count and that ids descend at the seams."""
    pf = pq.ParquetFile(path)
    if not pf.schema_arrow.equals(SCHEMA, check_metadata=False):
        raise ValueError(f'{path!r} has an unexpected schema: {pf.schema_arrow}')
    if pf.metadata.num_rows != rows:
        raise ValueError(f'{path!r} holds {pf.metadata.num_rows} rows, expected {rows}.')
    previous_low = None
    for i in range(pf.metadata.num_row_groups):
        ids = pf.read_row_group(i, columns=['id'])['id'].to_numpy()
        if len(ids) > 1 and (ids[1:] >= ids[:-1]).any():
            raise ValueError(f'{path!r} row group {i} is not strictly descending.')
        if previous_low is not None and ids[0] >= previous_low:
            raise ValueError(f'{path!r} row groups {i - 1} and {i} overlap.')
        previous_low = ids[-1]
