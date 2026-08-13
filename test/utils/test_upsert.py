import pyarrow as pa
import pytest

from inf.utils.upsert import (_hashable, apply_updates, iter_missing, merge_row, row_signature,
                              table_signatures, unchanged)

FIELDS = ('md5', 'tags', 'rating', 'nested')


@pytest.mark.unittest
class TestHashable:
    def test_scalars_pass_through(self):
        for value in ('a', 1, 1.5, True, None):
            assert _hashable(value) == value

    def test_lists_become_tuples(self):
        assert _hashable(['a', 'b']) == ('a', 'b')

    def test_nested_lists(self):
        assert _hashable([['a'], ['b', 'c']]) == (('a',), ('b', 'c'))

    def test_dicts_become_sorted_pairs(self):
        assert _hashable({'b': 2, 'a': 1}) == (('a', 1), ('b', 2))

    def test_dict_key_order_does_not_matter(self):
        assert _hashable({'a': 1, 'b': 2}) == _hashable({'b': 2, 'a': 1})

    def test_list_order_does_matter(self):
        # A reordered tag list is a real change; Arrow returns lists in write order, so comparing
        # a stored row against a fetched one stays stable.
        assert _hashable(['a', 'b']) != _hashable(['b', 'a'])

    def test_dict_holding_a_list(self):
        assert _hashable({'urls': ['x', 'y']}) == (('urls', ('x', 'y')),)

    def test_everything_it_returns_is_hashable(self):
        hash(_hashable({'a': [1, {'b': [2]}]}))


@pytest.mark.unittest
class TestRowSignatureWithContainers:
    def test_list_valued_field(self):
        a = {'md5': 'x', 'tags': ['a', 'b'], 'rating': 'e', 'nested': None}
        b = dict(a, tags=['a', 'b', 'c'])
        assert row_signature(a, FIELDS) == row_signature(dict(a), FIELDS)
        assert row_signature(a, FIELDS) != row_signature(b, FIELDS)

    def test_nested_struct_field(self):
        a = {'md5': 'x', 'tags': [], 'rating': 'e', 'nested': {'has': False, 'urls': []}}
        b = dict(a, nested={'has': True, 'urls': []})
        assert row_signature(a, FIELDS) != row_signature(b, FIELDS)

    def test_empty_dict_and_none_are_distinguishable(self):
        a = {'md5': 'x', 'tags': [], 'rating': 'e', 'nested': {}}
        b = dict(a, nested=None)
        assert row_signature(a, FIELDS) != row_signature(b, FIELDS)


@pytest.mark.unittest
class TestTableSignaturesWithContainers:
    def _table(self, rows):
        return pa.table({
            'id': [r['id'] for r in rows],
            'md5': [r['md5'] for r in rows],
            'tags': [r['tags'] for r in rows],
            'rating': [r['rating'] for r in rows],
            'nested': [r['nested'] for r in rows],
        }, schema=pa.schema([
            ('id', pa.int64()), ('md5', pa.string()),
            ('tags', pa.list_(pa.string())), ('rating', pa.string()),
            ('nested', pa.struct([('has', pa.bool_()), ('n', pa.int64())])),
        ]))

    def test_agrees_with_row_signature_through_arrow(self):
        # The contract: a list column read back out of Arrow must fingerprint identically to the
        # Python list it was written from, or every row reads as stale on the next run.
        rows = [
            {'id': 1, 'md5': 'a', 'tags': ['x', 'y'], 'rating': 'e',
             'nested': {'has': True, 'n': 1}},
            {'id': 2, 'md5': 'b', 'tags': [], 'rating': 'g', 'nested': {'has': False, 'n': 0}},
            {'id': 3, 'md5': None, 'tags': ['z'], 'rating': 's', 'nested': None},
        ]
        table = self._table(rows)
        sigs = table_signatures(table, FIELDS)
        for stored, original in zip(table.to_pylist(), rows):
            assert sigs[original['id']] == row_signature(stored, FIELDS)

    def test_list_order_survives_the_round_trip(self):
        rows = [{'id': 1, 'md5': 'a', 'tags': ['b', 'a'], 'rating': 'e', 'nested': None}]
        table = self._table(rows)
        assert table_signatures(table, FIELDS)[1] == row_signature(table.to_pylist()[0], FIELDS)


@pytest.mark.unittest
class TestApplyUpdatesWithContainers:
    def _table(self, rows):
        return pa.table({'id': [r['id'] for r in rows], 'tags': [r['tags'] for r in rows]},
                        schema=pa.schema([('id', pa.int64()), ('tags', pa.list_(pa.string()))]))

    def test_list_column_updates_in_place(self):
        table = self._table([{'id': 1, 'tags': ['a']}, {'id': 2, 'tags': ['b']}])
        out = apply_updates(table, {1: {'id': 1, 'tags': ['a', 'c']}})
        by_id = {r['id']: r for r in out.to_pylist()}
        assert by_id[1]['tags'] == ['a', 'c']
        assert by_id[2]['tags'] == ['b']
        assert out.num_rows == 2

    def test_none_list_does_not_wipe_a_stored_list(self):
        table = self._table([{'id': 1, 'tags': ['a', 'b']}])
        out = apply_updates(table, {1: {'id': 1, 'tags': None}})
        assert out.to_pylist()[0]['tags'] == ['a', 'b']

    def test_empty_list_does_replace(self):
        # Losing every tag is a real change, unlike a missing field.
        table = self._table([{'id': 1, 'tags': ['a', 'b']}])
        out = apply_updates(table, {1: {'id': 1, 'tags': []}})
        assert out.to_pylist()[0]['tags'] == []


@pytest.mark.unittest
class TestHelpers:
    def test_unchanged(self):
        sigs = {1: 111, 2: 222}
        assert unchanged(sigs, 1, 111)
        assert not unchanged(sigs, 1, 999)
        assert not unchanged(sigs, 3, 111)

    def test_iter_missing(self):
        assert list(iter_missing({1: 0, 2: 0}, [1, 2, 3, 4])) == [3, 4]

    def test_merge_row_is_shared(self):
        assert merge_row({'a': 1}, {'a': None})['a'] == 1
