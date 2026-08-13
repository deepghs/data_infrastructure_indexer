import datetime

import pyarrow as pa
import pytest

from inf.aibooru.index import (_UPDATE_TRIGGER_FIELDS, build_row, is_recordable, row_signature,
                              table_signatures, to_timestamp)

#: The published column set, in order, as it stands on the hub. This is a contract: the job
#: overwrites aibooru.parquet in place, so a build_row that drifts from this would silently change
#: the schema consumers read.
PUBLISHED_COLUMNS = (
    'id', 'uploader_id', 'approver_id', 'up_score', 'down_score', 'score', 'fav_count',
    'source', 'md5', 'rating', 'tags', 'file_ext', 'file_size', 'width', 'height',
    'parent_id', 'has_children', 'has_active_children', 'has_visible_children', 'pixiv_id',
    'bit_flags', 'views', 'has_large', 'file_url', 'large_file_url', 'preview_file_url',
    'created_at', 'updated_at',
)


def _api_item(**overrides):
    """One entry shaped the way /posts.json returns it."""
    item = {
        'id': 173252, 'uploader_id': 12, 'approver_id': None,
        'up_score': 3, 'down_score': 0, 'score': 3, 'fav_count': 1,
        'source': 'https://example/src', 'md5': 'd41d8cd98f00b204e9800998ecf8427e',
        'rating': 'g', 'tag_string': 'ai_generated 1girl solo',
        'file_ext': 'png', 'file_size': 1234567,
        'image_width': 1024, 'image_height': 1536,
        'parent_id': None, 'has_children': False, 'has_active_children': False,
        'has_visible_children': False, 'pixiv_id': None, 'bit_flags': 0, 'views': 7,
        'has_large': True,
        'file_url': 'https://cdn.aibooru.download/original/d4/1d/x.png',
        'large_file_url': 'https://cdn.aibooru.download/sample/d4/1d/x.jpg',
        'preview_file_url': 'https://cdn.aibooru.download/preview/d4/1d/x.jpg',
        'created_at': '2026-08-13T04:12:11.123-04:00',
        'updated_at': '2026-08-13T05:00:00.000-04:00',
        'is_deleted': False,
    }
    item.update(overrides)
    return item


@pytest.mark.unittest
class TestToTimestamp:
    def test_parses_what_the_api_sends(self):
        expected = datetime.datetime(
            2026, 8, 13, 4, 12, 11, 123000,
            tzinfo=datetime.timezone(datetime.timedelta(hours=-4))).timestamp()
        assert to_timestamp('2026-08-13T04:12:11.123-04:00') == pytest.approx(expected)

    def test_matches_dateparser(self):
        # The prototype used dateparser; the fast path must agree with it or stored timestamps
        # would shift for no reason.
        import dateparser
        for value in ('2026-08-13T04:12:11.123-04:00', '2025-01-02T03:04:05.000+09:00',
                      '2024-12-31T23:59:59.999+00:00'):
            assert to_timestamp(value) == pytest.approx(dateparser.parse(value).timestamp())

    @pytest.mark.parametrize('value', [None, '', 0])
    def test_missing_reads_as_none(self, value):
        assert to_timestamp(value) is None

    def test_unparseable_falls_back_and_stays_none(self):
        assert to_timestamp('not a timestamp at all') is None

    def test_returns_a_float(self):
        assert isinstance(to_timestamp('2026-08-13T04:12:11.123-04:00'), float)

    def test_round_trips_through_datetime(self):
        ts = to_timestamp('2026-08-13T04:12:11.000-04:00')
        back = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
        assert back.strftime('%Y-%m-%d %H:%M') == '2026-08-13 08:12'


@pytest.mark.unittest
class TestIsRecordable:
    def test_ordinary_post(self):
        assert is_recordable(_api_item())

    def test_deleted_is_out(self):
        # No is_deleted column exists to mark it with.
        assert not is_recordable(_api_item(is_deleted=True))

    @pytest.mark.parametrize('md5', [None, ''])
    def test_missing_md5_is_out(self, md5):
        assert not is_recordable(_api_item(md5=md5))

    def test_md5_absent_entirely(self):
        item = _api_item()
        del item['md5']
        assert not is_recordable(item)


@pytest.mark.unittest
class TestBuildRow:
    def test_produces_exactly_the_published_columns(self):
        assert tuple(build_row(_api_item())) == PUBLISHED_COLUMNS

    def test_renamed_fields(self):
        row = build_row(_api_item())
        assert row['tags'] == 'ai_generated 1girl solo'      # tag_string
        assert row['width'] == 1024                          # image_width
        assert row['height'] == 1536                         # image_height

    def test_timestamps_are_floats(self):
        row = build_row(_api_item())
        assert isinstance(row['created_at'], float)
        assert isinstance(row['updated_at'], float)

    def test_absent_api_fields_become_none(self):
        row = build_row({'id': 5})
        assert row['md5'] is None and row['tags'] is None and row['created_at'] is None
        assert tuple(row) == PUBLISHED_COLUMNS

    def test_no_extra_keys_leak_from_the_api(self):
        row = build_row(_api_item(some_new_api_field='x', is_deleted=False))
        assert 'some_new_api_field' not in row
        assert 'is_deleted' not in row

    def test_fits_the_published_arrow_schema(self):
        # The types the hub table uses today.
        schema = pa.schema([
            ('id', pa.int64()), ('uploader_id', pa.int64()), ('approver_id', pa.float64()),
            ('up_score', pa.int64()), ('down_score', pa.int64()), ('score', pa.int64()),
            ('fav_count', pa.int64()), ('source', pa.string()), ('md5', pa.string()),
            ('rating', pa.string()), ('tags', pa.string()), ('file_ext', pa.string()),
            ('file_size', pa.int64()), ('width', pa.int64()), ('height', pa.int64()),
            ('parent_id', pa.float64()), ('has_children', pa.bool_()),
            ('has_active_children', pa.bool_()), ('has_visible_children', pa.bool_()),
            ('pixiv_id', pa.float64()), ('bit_flags', pa.int64()), ('views', pa.int64()),
            ('has_large', pa.bool_()), ('file_url', pa.string()),
            ('large_file_url', pa.string()), ('preview_file_url', pa.string()),
            ('created_at', pa.float64()), ('updated_at', pa.float64()),
        ])
        assert list(schema.names) == list(PUBLISHED_COLUMNS)
        table = pa.Table.from_pylist([build_row(_api_item())], schema=schema)
        assert table.num_rows == 1
        assert table.schema == schema

    def test_integer_parent_id_fits_the_float_column(self):
        # The API sends an int; the column is double. Arrow must accept the widening.
        schema = pa.schema([('id', pa.int64()), ('parent_id', pa.float64())])
        row = build_row(_api_item(parent_id=99))
        table = pa.Table.from_pylist([{k: row[k] for k in ('id', 'parent_id')}], schema=schema)
        assert table.to_pylist()[0]['parent_id'] == 99.0


@pytest.mark.unittest
class TestSignatures:
    def test_same_item_same_signature(self):
        assert row_signature(build_row(_api_item())) == row_signature(build_row(_api_item()))

    @pytest.mark.parametrize('field,value', [
        ('tag_string', 'ai_generated 1girl solo smile'),
        ('md5', 'ffffffffffffffffffffffffffffffff'),
        ('file_url', 'https://cdn.aibooru.download/original/aa/bb/y.png'),
        ('rating', 'e'),
        ('file_size', 999),
        ('image_width', 2048),
        ('parent_id', 12),
    ])
    def test_meaningful_changes_trigger(self, field, value):
        assert row_signature(build_row(_api_item())) != \
               row_signature(build_row(_api_item(**{field: value})))

    @pytest.mark.parametrize('field,value', [
        ('score', 999), ('up_score', 50), ('down_score', 3), ('fav_count', 77),
        ('views', 100000), ('updated_at', '2027-01-01T00:00:00.000-04:00'),
    ])
    def test_drifting_fields_do_not_trigger(self, field, value):
        # views and updated_at move constantly; triggering on them would rewrite the table daily.
        assert row_signature(build_row(_api_item())) == \
               row_signature(build_row(_api_item(**{field: value})))

    def test_table_and_row_signatures_agree(self):
        rows = [build_row(_api_item(id=1)),
                build_row(_api_item(id=2, tag_string='other tags')),
                build_row(_api_item(id=3, md5=None, file_url=None))]
        columns = ['id'] + list(_UPDATE_TRIGGER_FIELDS)
        table = pa.table({c: [r.get(c) for r in rows] for c in columns})
        sigs = table_signatures(table)
        for row in rows:
            assert sigs[row['id']] == row_signature(row)
