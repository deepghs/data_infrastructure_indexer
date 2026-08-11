import pyarrow as pa
import pytest


def _schema():
    return pa.schema([('id', pa.int64()), ('name', pa.string()), ('score', pa.float64())])


class _Shard:
    """
    A standalone copy of the shard-building logic in inf/rule34/index.py.

    The real one is a set of closures inside `sync`, which cannot be exercised without a
    repository and a live session. The invariants worth pinning down are all local: rows buffer
    as dicts, fold into Arrow chunks, come out newest-first, and a shard never exceeds its row
    cap. Keeping a mirror here is a trade - it can drift - so each test states the invariant it
    is protecting rather than the implementation.
    """

    def __init__(self, schema, max_part_rows, base=None, ptr=1, flush_at=3):
        self.schema = schema
        self.max_part_rows = max_part_rows
        self.chunks = [base] if base is not None else []
        self.pending = []
        self.ptr = ptr
        self.flush_at = flush_at
        self.sealed = []

    def rows(self):
        return sum(c.num_rows for c in self.chunks) + len(self.pending)

    def flush(self):
        if not self.pending:
            return
        rows = [{c: r.get(c) for c in self.schema.names} for r in self.pending]
        self.chunks.append(pa.Table.from_pylist(rows, schema=self.schema))
        self.pending.clear()

    def table(self):
        self.flush()
        if not self.chunks:
            return self.schema.empty_table()
        t = self.chunks[0] if len(self.chunks) == 1 else pa.concat_tables(self.chunks)
        return t.sort_by([('id', 'descending')])

    def rotate(self):
        self.sealed.append((self.ptr, self.table()))
        self.chunks.clear()
        self.chunks.append(self.schema.empty_table())
        self.ptr += 1

    def add(self, row):
        self.pending.append(row)
        if len(self.pending) >= self.flush_at:
            self.flush()
        if self.rows() >= self.max_part_rows:
            self.rotate()


@pytest.mark.unittest
class TestShardBuilding:
    def test_rows_come_out_newest_first(self):
        s = _Shard(_schema(), max_part_rows=100)
        for i in (3, 1, 2):
            s.add({'id': i, 'name': f'n{i}', 'score': 1.0})
        assert s.table().column('id').to_pylist() == [3, 2, 1]

    def test_schema_is_preserved_exactly(self):
        s = _Shard(_schema(), max_part_rows=100)
        s.add({'id': 1, 'name': 'a', 'score': 0.5})
        assert s.table().schema.equals(_schema())

    def test_missing_keys_become_null_rather_than_raising(self):
        # The API omits fields on some posts; a shard must absorb that.
        s = _Shard(_schema(), max_part_rows=100)
        s.add({'id': 1})
        row = s.table().to_pylist()[0]
        assert row['name'] is None and row['score'] is None

    def test_extra_keys_are_dropped(self):
        # `row` is built by splatting the whole API item, so it carries keys the table does not.
        s = _Shard(_schema(), max_part_rows=100)
        s.add({'id': 1, 'name': 'a', 'score': 0.5, 'unexpected': 'x'})
        assert s.table().schema.names == _schema().names

    def test_existing_rows_are_carried_into_the_shard(self):
        base = pa.Table.from_pylist([{'id': 10, 'name': 'old', 'score': 9.0}], schema=_schema())
        s = _Shard(_schema(), max_part_rows=100, base=base)
        s.add({'id': 11, 'name': 'new', 'score': 1.0})
        assert s.table().column('id').to_pylist() == [11, 10]


@pytest.mark.unittest
class TestRotation:
    def test_shard_never_exceeds_the_cap(self):
        # The whole point: rotation used to be decided once at startup, so a long run grew one
        # shard without bound and ran the runner out of memory.
        s = _Shard(_schema(), max_part_rows=4, flush_at=2)
        for i in range(20):
            s.add({'id': i, 'name': 'x', 'score': 0.0})
        for ptr, table in s.sealed:
            assert table.num_rows <= 4, f'shard {ptr} has {table.num_rows} rows'
        assert s.rows() <= 4

    def test_rotation_advances_the_pointer_and_loses_nothing(self):
        s = _Shard(_schema(), max_part_rows=4, flush_at=2)
        for i in range(12):
            s.add({'id': i, 'name': 'x', 'score': 0.0})
        seen = [i for _, t in s.sealed for i in t.column('id').to_pylist()]
        seen += s.table().column('id').to_pylist()
        assert sorted(seen) == list(range(12)), 'every row must land in exactly one shard'
        assert s.ptr == 1 + len(s.sealed)

    def test_sealed_shards_keep_their_ordering(self):
        s = _Shard(_schema(), max_part_rows=3, flush_at=1)
        for i in range(9):
            s.add({'id': i, 'name': 'x', 'score': 0.0})
        for _, t in s.sealed:
            ids = t.column('id').to_pylist()
            assert ids == sorted(ids, reverse=True)
