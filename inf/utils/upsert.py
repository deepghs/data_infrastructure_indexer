"""Accumulating updates for index tables held as Arrow.

A site index is not write-once: a post already recorded can change - tags get edited, a file
finishes processing, a post is deleted or restored. Re-fetching it should improve the stored row,
never degrade it.

Two rules make that work, and both are here rather than in each site package:

A field arriving as ``None`` never overwrites a stored value. Sites withhold data selectively -
atfbooru serves banned posts with ``file_url`` and ``md5`` blanked - so a post banned after being
indexed must keep the url captured while it was still readable. Empty string, ``False`` and ``0``
are real values and do replace; a post whose tags were all removed genuinely has no tags.

Staleness is decided from a chosen subset of fields, not all of them, and from a fingerprint
rather than the values. Scores, favourite counts and "last activity" timestamps drift on their
own, so including them would mark most of a table changed on every pass and rewrite it for
nothing. Keeping the fields themselves for millions of rows does not fit a CI runner; an integer
per row does.

Each site supplies its own trigger fields, because the column names differ: atfbooru has
``tag_string``, aibooru calls the same thing ``tags``, e6ai stores it as a list.
"""
from typing import Dict, Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc


def merge_row(old: dict, new: dict) -> dict:
    """
    Fold a freshly fetched row onto the stored one, never losing a known value.

    :param old: Row as currently stored.
    :type old: dict
    :param new: Row built from the API.
    :type new: dict
    :returns: The merged row.
    :rtype: dict
    """
    merged = dict(old)
    for key, value in new.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def row_signature(row: dict, fields: Sequence[str]) -> int:
    """
    Fingerprint the fields that decide whether a stored row is stale.

    :param row: Row built from an API item, or read back from the stored table.
    :type row: dict
    :param fields: Trigger fields, in a fixed order.
    :type fields: Sequence[str]
    :returns: Hash over those fields.
    :rtype: int
    """
    return hash(tuple(_hashable(row.get(field)) for field in fields))


def table_signatures(table: pa.Table, fields: Sequence[str]) -> Dict[int, int]:
    """
    Fingerprint every row of a stored table, keyed by id.

    Must agree with :func:`row_signature` exactly - the two are compared against each other, so
    any disagreement marks the whole table stale and rewrites it on every run. Columns are read in
    ``fields`` order and absent ones filled with ``None`` for that reason.

    Done one record batch at a time: materialising the trigger fields for millions of rows at once
    would peak at gigabytes.

    :param table: Table as read from the hub.
    :type table: pa.Table
    :param fields: Trigger fields, in the same order used for :func:`row_signature`.
    :type fields: Sequence[str]
    :returns: Mapping of post id to fingerprint.
    :rtype: Dict[int, int]
    """
    names = set(table.schema.names)
    signatures = {}
    for batch in table.to_batches(max_chunksize=65536):
        ids = batch.column('id').to_pylist()
        columns = [batch.column(field).to_pylist() if field in names
                   else [None] * batch.num_rows for field in fields]
        for post_id, values in zip(ids, zip(*columns)):
            signatures[post_id] = hash(tuple(_hashable(v) for v in values))
        del ids, columns
    return signatures


def apply_updates(table: pa.Table, updates: Dict[int, dict]) -> pa.Table:
    """
    Rewrite the rows whose trigger fields moved, merging rather than replacing.

    One vectorised pass: the affected rows are lifted out together, merged as dicts, and put back.
    Row by row against an Arrow table would mean a full scan per update.

    An id in ``updates`` the table does not hold is inserted instead of dropped. That should not
    happen - a row reaches ``updates`` only once it is known - but silently losing a post would be
    much worse than carrying an extra one.

    :param table: Table to update.
    :type table: pa.Table
    :param updates: New rows keyed by post id.
    :type updates: Dict[int, dict]
    :returns: Table with the merged rows in place, id order not preserved.
    :rtype: pa.Table
    """
    if not updates:
        return table
    keys = pa.array(list(updates.keys()), type=table.schema.field('id').type)
    mask = pc.fill_null(pc.is_in(table.column('id'), value_set=keys), False)
    stale = table.filter(mask).to_pylist()
    kept = table.filter(pc.invert(mask))

    merged = [merge_row(old, updates[old['id']]) for old in stale]
    present = {old['id'] for old in stale}
    merged.extend(row for post_id, row in updates.items() if post_id not in present)

    rows = [{column: row.get(column) for column in table.schema.names} for row in merged]
    return pa.concat_tables([kept, pa.Table.from_pylist(rows, schema=table.schema)])


def _hashable(value):
    """
    Make an API value hashable without changing what counts as equal.

    Lists and dicts appear in these tables - e6ai stores tags as a list and ``sample_alternates``
    as a nested object - and both are unhashable. Converting to tuples keeps order significant,
    which is what we want: a reordered tag list is a change worth recording, and Arrow will hand
    the list back in the order it was written, so comparing a stored row against a fetched one
    stays stable.

    :param value: Field value.
    :returns: Something :func:`hash` accepts.
    """
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    return value


def unchanged(signatures: Dict[int, int], post_id: int, signature: int) -> bool:
    """
    Whether a post is known and its trigger fields have not moved.

    :param signatures: Fingerprints of the stored rows.
    :type signatures: Dict[int, int]
    :param post_id: Post id.
    :type post_id: int
    :param signature: Fingerprint of the freshly built row.
    :type signature: int
    :returns: True when the stored row is already current.
    :rtype: bool
    """
    return post_id in signatures and signatures[post_id] == signature


def iter_missing(signatures: Dict[int, int], ids: Iterable[int]) -> Iterable[int]:
    """
    Ids from ``ids`` that are not in the table yet.

    :param signatures: Fingerprints of the stored rows.
    :type signatures: Dict[int, int]
    :param ids: Candidate ids.
    :type ids: Iterable[int]
    :returns: The ids not present.
    :rtype: Iterable[int]
    """
    return (post_id for post_id in ids if post_id not in signatures)
