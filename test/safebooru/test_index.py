import pyarrow as pa
import pytest

from inf.safebooru.index import (_UPDATE_TRIGGER_FIELDS, build_row, format_tags, parse_tags,
                                 row_signature, shard_number, table_signatures)

#: The published column set. A contract: new shards sit beside 5.76M rows written to this shape.
PUBLISHED_COLUMNS = (
    'preview_url', 'sample_url', 'file_url', 'directory', 'hash', 'width', 'height', 'id',
    'image', 'change', 'owner', 'parent_id', 'rating', 'sample', 'sample_height', 'sample_width',
    'score', 'tags', 'source', 'status', 'has_notes', 'comment_count',
    'filename', 'mimetype', 'scraped_at',
)


def _api_item(**overrides):
    """One entry shaped the way the post index API returns it."""
    item = {
        'preview_url': 'https://safebooru.org/thumbnails/843/thumbnail_0e76.jpg',
        'sample_url': 'https://safebooru.org/samples/843/sample_0e76.jpg',
        'file_url': 'https://safebooru.org/images/843/0e76.png',
        'directory': 843, 'hash': '058f0c8482a374a3bb4c3ba0723b2863',
        'width': 2700, 'height': 3500, 'id': 7050375, 'image': '0e76.png',
        'change': 1786954331, 'owner': 'someone', 'parent_id': 0, 'rating': 'safe',
        'sample': 1, 'sample_height': 1000, 'sample_width': 771, 'score': None,
        'tags': '1girl blue_archive brown_eyes solo', 'source': 'https://example/src',
        'status': 'active', 'has_notes': False, 'comment_count': 0,
    }
    item.update(overrides)
    return item


@pytest.mark.unittest
class TestTagFormat:
    def test_stored_form_is_space_wrapped(self):
        # What makes LIKE '% tag %' match a whole tag instead of a prefix.
        assert format_tags(['1girl', 'solo']) == ' 1girl solo '

    def test_empty_tag_list(self):
        # ' '.join(['', '']) is a single space - what a post with no tags looks like in the
        # published rows, so worth pinning rather than assuming.
        assert format_tags([]) == ' '

    def test_parse_strips_the_wrapping(self):
        assert parse_tags(' 1girl solo ') == ['1girl', 'solo']

    def test_parse_handles_api_form_too(self):
        assert parse_tags('1girl solo') == ['1girl', 'solo']

    @pytest.mark.parametrize('value', [None, '', '   '])
    def test_parse_of_nothing(self, value):
        assert parse_tags(value) == []

    def test_duplicates_are_dropped_in_order(self):
        assert parse_tags('a b a c b') == ['a', 'b', 'c']

    def test_html_entities_are_unescaped(self):
        assert parse_tags('a&amp;b c') == ['a&b', 'c']

    def test_round_trip_is_stable(self):
        stored = ' 1girl blue_archive solo '
        assert format_tags(parse_tags(stored)) == stored


@pytest.mark.unittest
class TestBuildRow:
    def test_produces_exactly_the_published_columns(self):
        row = build_row(_api_item(), ['1girl', 'solo'])
        assert set(row) == set(PUBLISHED_COLUMNS)

    def test_the_three_derived_columns(self):
        row = build_row(_api_item(), ['1girl'])
        assert row['filename'] == '0e76.png'          # the API's "image"
        assert row['mimetype'] == 'image/png'          # guessed from file_url
        assert isinstance(row['scraped_at'], float)    # now

    def test_tags_use_the_stored_form_not_the_api_form(self):
        row = build_row(_api_item(), ['1girl', 'solo'])
        assert row['tags'] == ' 1girl solo '

    def test_normalised_tags_win_over_the_api_string(self):
        # The API string is carried through by **item, so the explicit tags must come after it.
        row = build_row(_api_item(tags='raw tags here'), ['normalised'])
        assert row['tags'] == ' normalised '

    def test_mimetype_without_a_url(self):
        assert build_row(_api_item(file_url=None), [])['mimetype'] is None

    def test_webm_and_webp(self):
        assert build_row(_api_item(file_url='https://x/a.webm'), [])['mimetype'] == 'video/webm'
        assert build_row(_api_item(file_url='https://x/a.webp'), [])['mimetype'] == 'image/webp'

    def test_fits_the_published_arrow_schema(self):
        schema = pa.schema([
            ('preview_url', pa.string()), ('sample_url', pa.string()), ('file_url', pa.string()),
            ('directory', pa.int64()), ('hash', pa.string()), ('width', pa.int64()),
            ('height', pa.int64()), ('id', pa.int64()), ('image', pa.string()),
            ('change', pa.int64()), ('owner', pa.string()), ('parent_id', pa.int64()),
            ('rating', pa.string()), ('sample', pa.int64()), ('sample_height', pa.int64()),
            ('sample_width', pa.int64()), ('score', pa.float64()), ('tags', pa.string()),
            ('source', pa.string()), ('status', pa.string()), ('has_notes', pa.bool_()),
            ('comment_count', pa.int64()), ('filename', pa.string()), ('mimetype', pa.string()),
            ('scraped_at', pa.float64()),
        ])
        row = build_row(_api_item(), ['1girl'])
        table = pa.Table.from_pylist([{c: row.get(c) for c in schema.names}], schema=schema)
        assert table.num_rows == 1 and table.schema == schema


@pytest.mark.unittest
class TestSignature:
    def _row(self, **overrides):
        item = _api_item(**{k: v for k, v in overrides.items() if k != 'tags_list'})
        return build_row(item, overrides.get('tags_list', ['1girl', 'solo']))

    def test_scraped_at_is_ignored(self):
        # It is set to now on every fetch; triggering on it would rewrite every row every run.
        row = self._row()
        assert row_signature(row) == row_signature(dict(row, scraped_at=row['scraped_at'] + 1000))

    @pytest.mark.parametrize('field,value', [('score', 999), ('change', 1799999999),
                                             ('comment_count', 5)])
    def test_drifting_fields_are_ignored(self, field, value):
        row = self._row()
        assert row_signature(row) == row_signature(dict(row, **{field: value}))

    def test_tag_order_is_ignored(self):
        assert row_signature(self._row(tags_list=['1girl', 'solo'])) == \
               row_signature(self._row(tags_list=['solo', '1girl']))

    def test_a_real_tag_edit_registers(self):
        assert row_signature(self._row(tags_list=['1girl', 'solo'])) != \
               row_signature(self._row(tags_list=['1girl', 'solo', 'smile']))

    @pytest.mark.parametrize('field,value', [
        ('file_url', 'https://safebooru.org/images/843/other.png'),
        ('hash', 'ffffffffffffffffffffffffffffffff'),
        ('rating', 'questionable'),
        ('width', 100),
        ('status', 'deleted'),
        ('parent_id', 42),
    ])
    def test_meaningful_changes_register(self, field, value):
        assert row_signature(self._row()) != row_signature(self._row(**{field: value}))

    def test_table_and_row_signatures_agree(self):
        rows = [build_row(_api_item(id=1), ['a', 'b']),
                build_row(_api_item(id=2), ['c']),
                build_row(_api_item(id=3, file_url=None), [])]
        columns = sorted({k for r in rows for k in r})
        table = pa.table({c: [r.get(c) for r in rows] for c in columns})
        sigs = table_signatures(table)
        for row in rows:
            assert sigs[row['id']] == row_signature(row)

    def test_table_signatures_are_tag_order_blind(self):
        stored = build_row(_api_item(id=1), ['solo', '1girl'])
        columns = sorted(stored)
        table = pa.table({c: [stored.get(c)] for c in columns})
        assert table_signatures(table)[1] == row_signature(build_row(_api_item(id=1),
                                                                     ['1girl', 'solo']))

    def test_every_trigger_field_exists_in_a_built_row(self):
        row = build_row(_api_item(), ['a'])
        missing = [f for f in _UPDATE_TRIGGER_FIELDS if f not in row]
        assert not missing, f'trigger fields absent from the row: {missing}'


@pytest.mark.unittest
class TestShardNumber:
    @pytest.mark.parametrize('path,expected', [
        ('tables/safebooru-1.parquet', 1),
        ('tables/safebooru-12.parquet', 12),
        ('datasets/deepghs/safebooru_index/tables/safebooru-7.parquet', 7),
        ('tables/other.parquet', -1),
        ('safebooru-3.parquet', 3),
    ])
    def test_extraction(self, path, expected):
        assert shard_number(path) == expected

    def test_sorting_is_numeric_not_lexical(self):
        paths = ['tables/safebooru-2.parquet', 'tables/safebooru-10.parquet',
                 'tables/safebooru-1.parquet']
        assert sorted(paths, key=shard_number)[-1] == 'tables/safebooru-10.parquet'
