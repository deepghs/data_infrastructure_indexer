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
from typing import Collection, Dict, Iterable, Optional, Sequence

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


def row_signature(row: dict, fields: Sequence[str],
                  unordered_fields: Collection[str] = ()) -> int:
    """
    Fingerprint the fields that decide whether a stored row is stale.

    :param row: Row built from an API item, or read back from the stored table.
    :type row: dict
    :param fields: Trigger fields, in a fixed order.
    :type fields: Sequence[str]
    :param unordered_fields: Of those, the list-valued ones whose order carries no meaning. See
        :func:`_normalise` for why this exists.
    :type unordered_fields: Collection[str]
    :returns: Hash over those fields.
    :rtype: int
    """
    return hash(tuple(_normalise(field, row.get(field), unordered_fields) for field in fields))


def table_signatures(table: pa.Table, fields: Sequence[str],
                     unordered_fields: Collection[str] = ()) -> Dict[int, int]:
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
    :param unordered_fields: Must match what :func:`row_signature` is given, for the same reason
        the field order must.
    :type unordered_fields: Collection[str]
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
            signatures[post_id] = hash(tuple(
                _normalise(field, value, unordered_fields)
                for field, value in zip(fields, values)))
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


def adds_anything(stored_row: Optional[dict], fetched_row: dict, stored_signature: int,
                  fields: Sequence[str], unordered_fields: Collection[str] = ()) -> bool:
    """
    Whether a fetched row would actually change the stored one.

    Comparing the fetched row's fingerprint against the stored one is not enough, because the merge
    protects stored values from ``None``. A field the API has stopped sending therefore reads as a
    difference that rewriting cannot resolve: the stored row keeps its value, the next fetch still
    sends ``None``, and the row is marked changed again. Forever.

    Measured on e6ai over 60 re-fetched posts: ``sample_url`` went value-to-``None`` on 18% of them,
    and ``file_url``, ``preview_url`` and ``mimetype`` on 7% - the posts that had since been
    deleted. Left alone, every run would rewrite a fifth of everything it looked at and report it as
    changed, drowning the real edits.

    So the question is asked of the merge result rather than the fetched row.

    :param stored_row: The row as currently stored, or None when it cannot be read back.
    :type stored_row: Optional[dict]
    :param fetched_row: Row built from the API.
    :type fetched_row: dict
    :param stored_signature: Fingerprint of the stored row.
    :type stored_signature: int
    :param fields: Trigger fields.
    :type fields: Sequence[str]
    :param unordered_fields: Trigger fields whose list order carries no meaning.
    :type unordered_fields: Collection[str]
    :returns: True when the merge would differ from what is stored.
    :rtype: bool
    """
    if stored_row is None:
        return True
    merged = merge_row(stored_row, fetched_row)
    return row_signature(merged, fields, unordered_fields) != stored_signature


def _normalise(field: str, value, unordered_fields: Collection[str]):
    """
    Prepare one field value for hashing, sorting the ones whose order means nothing.

    Some APIs return a list in an unstable order. Measured on e6ai: re-fetching 12 stored posts
    found one whose ``tags`` held exactly the same values in a different order. Left alone that row
    reads as changed on every single run - and rewriting it changes nothing, so it reads as changed
    again next time. It never converges.

    Sorting only happens for the fields a site names, because order is not always noise: a list of
    child post ids or of sample variants may well be meaningful. What is stored keeps the API's own
    order either way; this affects the comparison only.

    :param field: Field name.
    :type field: str
    :param value: Field value.
    :param unordered_fields: Fields whose list order carries no meaning.
    :type unordered_fields: Collection[str]
    :returns: Something :func:`hash` accepts.
    """
    if field in unordered_fields and isinstance(value, list):
        # key=repr keeps mixed-type lists sortable rather than raising.
        value = sorted(value, key=repr)
    return _hashable(value)


def _hashable(value):
    """
    Make an API value hashable without changing what counts as equal.

    Lists and dicts appear in these tables - e6ai stores tags as a list and ``sample_alternates``
    as a nested object - and both are unhashable. Converting to tuples is order-preserving, which
    is the right default: Arrow hands a list back in the order it was written, so a stored row and
    a freshly fetched one compare stably. Where a site's API does not hold that order stable,
    :func:`_normalise` sorts the field first.

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
