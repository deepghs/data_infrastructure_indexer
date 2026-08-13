import pyarrow as pa
import pytest

from inf.atfbooru.index import (_UPDATE_TRIGGER_FIELDS, build_row, row_signature,
                                table_signatures)
from inf.utils.upsert import apply_updates, merge_row


def _post(**overrides):
    row = {
        'id': 100, 'md5': 'abc', 'file_url': 'https://x/a.png',
        'large_file_url': 'https://x/a_l.png', 'preview_file_url': 'https://x/a_p.png',
        'mimetype': 'image/png', 'file_ext': 'png', 'file_size': 1234,
        'image_width': 800, 'image_height': 600,
        'tag_string': 'a b c', 'tag_string_general': 'a b', 'tag_string_character': 'c',
        'tag_string_copyright': '', 'tag_string_artist': '', 'tag_string_meta': '',
        'rating': 'e', 'source': 'https://src', 'parent_id': None, 'pixiv_id': None,
        'is_deleted': False, 'is_banned': False, 'is_pending': False, 'is_flagged': False,
        'score': 5, 'fav_count': 2,
    }
    row.update(overrides)
    return row


@pytest.mark.unittest
class TestMergeRow:
    def test_none_never_overwrites_a_known_value(self):
        old = _post(file_url='https://x/a.png', md5='abc')
        new = _post(file_url=None, md5=None)
        merged = merge_row(old, new)
        assert merged['file_url'] == 'https://x/a.png'
        assert merged['md5'] == 'abc'

    def test_a_post_banned_later_keeps_its_url(self):
        # The exact case the rule exists for: it was readable when recorded, now it is not.
        old = _post(file_url='https://x/a.png', md5='abc', is_banned=False)
        new = _post(file_url=None, md5=None, is_banned=True)
        merged = merge_row(old, new)
        assert merged['file_url'] == 'https://x/a.png'
        assert merged['md5'] == 'abc'
        assert merged['is_banned'] is True

    def test_url_less_row_gains_a_url(self):
        old = _post(file_url=None, md5=None, is_banned=True)
        new = _post(file_url='https://x/a.png', md5='abc', is_banned=False)
        merged = merge_row(old, new)
        assert merged['file_url'] == 'https://x/a.png'
        assert merged['md5'] == 'abc'

    def test_real_values_do_replace(self):
        merged = merge_row(_post(tag_string='a b'), _post(tag_string='a b c'))
        assert merged['tag_string'] == 'a b c'

    def test_empty_string_is_a_real_value(self):
        # All tags removed is a genuine change, not missing data.
        merged = merge_row(_post(tag_string='a b'), _post(tag_string=''))
        assert merged['tag_string'] == ''

    def test_false_is_a_real_value(self):
        merged = merge_row(_post(is_deleted=True), _post(is_deleted=False))
        assert merged['is_deleted'] is False

    def test_zero_is_a_real_value(self):
        merged = merge_row(_post(file_size=1234), _post(file_size=0))
        assert merged['file_size'] == 0

    def test_does_not_mutate_its_inputs(self):
        old, new = _post(tag_string='a'), _post(tag_string='b')
        merge_row(old, new)
        assert old['tag_string'] == 'a'
        assert new['tag_string'] == 'b'

    def test_keeps_columns_absent_from_the_new_row(self):
        merged = merge_row(_post(), {'id': 100, 'tag_string': 'z'})
        assert merged['file_url'] == 'https://x/a.png'
        assert merged['tag_string'] == 'z'


@pytest.mark.unittest
class TestRowSignature:
    def test_same_row_same_signature(self):
        assert row_signature(_post()) == row_signature(_post())

    @pytest.mark.parametrize('field,value', [
        ('tag_string', 'a b c d'),
        ('file_url', 'https://x/other.png'),
        ('md5', 'def'),
        ('rating', 'g'),
        ('is_deleted', True),
        ('is_banned', True),
        ('parent_id', 7.0),
        ('file_size', 999),
    ])
    def test_trigger_fields_change_it(self, field, value):
        assert row_signature(_post()) != row_signature(_post(**{field: value}))

    @pytest.mark.parametrize('field,value', [
        ('score', 999),
        ('fav_count', 42),
        ('up_score', 7),
    ])
    def test_drifting_fields_do_not(self, field, value):
        # Otherwise a large share of the table would look changed on every pass.
        assert row_signature(_post()) == row_signature(_post(**{field: value}))

    def test_missing_field_reads_as_none(self):
        row = _post()
        del row['source']
        assert row_signature(row) == row_signature(_post(source=None))


@pytest.mark.unittest
class TestTableSignatures:
    def _table(self, rows):
        columns = ['id'] + [f for f in _UPDATE_TRIGGER_FIELDS]
        return pa.table({c: [r.get(c) for r in rows] for c in columns})

    def test_agrees_with_row_signature(self):
        # The contract that matters: disagreement would mark every row stale on every run.
        rows = [_post(id=1), _post(id=2, tag_string='x y'), _post(id=3, file_url=None, md5=None)]
        sigs = table_signatures(self._table(rows))
        for row in rows:
            assert sigs[row['id']] == row_signature(row), f'mismatch on id {row["id"]}'

    def test_agrees_across_batch_boundaries(self):
        rows = [_post(id=i, tag_string=f'tag{i}') for i in range(500)]
        table = pa.concat_tables([self._table(rows[:200]), self._table(rows[200:])])
        sigs = table_signatures(table)
        assert len(sigs) == 500
        for row in rows:
            assert sigs[row['id']] == row_signature(row)

    def test_absent_trigger_column_reads_as_none(self):
        rows = [_post(id=1, source=None)]
        table = self._table(rows).drop_columns(['source'])
        assert table_signatures(table)[1] == row_signature(_post(id=1, source=None))

    def test_empty_table(self):
        assert table_signatures(self._table([])) == {}


@pytest.mark.unittest
class TestApplyUpdates:
    def _table(self, rows):
        columns = ['id'] + list(_UPDATE_TRIGGER_FIELDS)
        return pa.table({c: [r.get(c) for r in rows] for c in columns})

    def test_no_updates_returns_the_table(self):
        table = self._table([_post(id=1), _post(id=2)])
        assert apply_updates(table, {}).num_rows == 2

    def test_replaces_in_place_without_duplicating(self):
        table = self._table([_post(id=1), _post(id=2), _post(id=3)])
        out = apply_updates(table, {2: _post(id=2, tag_string='new tags')})
        assert out.num_rows == 3
        assert sorted(out.column('id').to_pylist()) == [1, 2, 3]
        by_id = {r['id']: r for r in out.to_pylist()}
        assert by_id[2]['tag_string'] == 'new tags'
        assert by_id[1]['tag_string'] == 'a b c'

    def test_merge_protection_holds_through_the_table(self):
        table = self._table([_post(id=1, file_url='https://x/a.png', md5='abc')])
        out = apply_updates(table, {1: _post(id=1, file_url=None, md5=None, is_banned=True)})
        row = out.to_pylist()[0]
        assert row['file_url'] == 'https://x/a.png'
        assert row['md5'] == 'abc'
        assert row['is_banned'] is True

    def test_unknown_id_is_inserted_not_dropped(self):
        table = self._table([_post(id=1)])
        out = apply_updates(table, {9: _post(id=9)})
        assert sorted(out.column('id').to_pylist()) == [1, 9]

    def test_schema_is_preserved(self):
        table = self._table([_post(id=1), _post(id=2)])
        out = apply_updates(table, {1: _post(id=1, rating='g')})
        assert out.schema == table.schema

    def test_updating_every_row(self):
        rows = [_post(id=i) for i in range(50)]
        table = self._table(rows)
        out = apply_updates(table, {i: _post(id=i, rating='g') for i in range(50)})
        assert out.num_rows == 50
        assert set(out.column('rating').to_pylist()) == {'g'}

    def test_signature_after_update_matches_the_new_row(self):
        # What the next run compares against: the rewritten row must not read as stale again.
        table = self._table([_post(id=1)])
        new = _post(id=1, tag_string='x y z')
        out = apply_updates(table, {1: new})
        assert table_signatures(out)[1] == row_signature(new)


@pytest.mark.unittest
class TestBuildRow:
    def test_drops_media_asset(self):
        row = build_row({'id': 1, 'file_url': 'https://x/a.png', 'media_asset': {'id': 9}})
        assert 'media_asset' not in row
        assert row['id'] == 1

    def test_guesses_mimetype(self):
        assert build_row({'id': 1, 'file_url': 'https://x/a.png'})['mimetype'] == 'image/png'
        assert build_row({'id': 2, 'file_url': 'https://x/a.webp'})['mimetype'] == 'image/webp'

    def test_mimetype_none_without_url(self):
        assert build_row({'id': 1, 'file_url': None})['mimetype'] is None
