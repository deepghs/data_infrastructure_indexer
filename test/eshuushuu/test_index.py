import datetime

import pyarrow as pa
import pytest

from inf.eshuushuu.index import (_UPDATE_TRIGGER_FIELDS, _drop_duplicate_ids, build_row,
                                 row_signature, split_tags, table_signatures, to_timestamp)

PUBLISHED_COLUMNS = (
    'id', 'username', 'user_id', 'original_filename', 'filename', 'ext', 'src_filename',
    'file_url', 'cdn_url', 'thumbnail_url', 'medium_url', 'large_url', 'md5_hash', 'file_size',
    'width', 'height', 'mimetype', 'rating', 'score', 'num_ratings', 'favorites', 'posts',
    'status', 'caption', 'source_url', 'misc_metadata', 'replacement_id', 'created_at',
    'tags', 'tag_ids', 'tags_artist', 'tags_character', 'tags_source', 'tags_theme',
)


def _api_image(**overrides):
    """One entry shaped the way /api/v1/images returns it."""
    item = {
        'filename': '2026-08-13-1117538', 'ext': 'png',
        'original_filename': 'imported-pixiv.png',
        'md5_hash': '5798490cc7d34569d794facb87f5cdc4', 'filesize': 1536863,
        'width': 1273, 'height': 900, 'caption': '', 'miscmeta': None,
        'source_url': 'https://www.pixiv.net/artworks/148310871', 'status': 1,
        'rating': 0.0, 'image_id': 1117538, 'user_id': 846201,
        'user': {'user_id': 846201, 'username': 'Housekino'},
        'date_added': '2026-08-13T04:39:44Z', 'locked': 0, 'posts': 0, 'favorites': 0,
        'bayesian_rating': 0.0, 'num_ratings': 0, 'replacement_id': None,
        'tags': [
            {'tag_id': 52365, 'title': 'Hiten', 'type': 3, 'type_name': 'Artist',
             'usage_count': 204},
            {'tag_id': 245432, 'title': 'bag', 'type': 1, 'type_name': 'Theme',
             'usage_count': 5394},
            {'tag_id': 33, 'title': 'barefoot', 'type': 1, 'type_name': 'Theme',
             'usage_count': 50878},
            {'tag_id': 210, 'title': 'Chobits', 'type': 2, 'type_name': 'Source',
             'usage_count': 900},
            {'tag_id': 66628, 'title': 'Chii', 'type': 4, 'type_name': 'Character',
             'usage_count': 300},
        ],
        'url': 'https://cdn.e-shuushuu.net/fullsize/2026-08-13-1117538.png',
        'thumbnail_url': 'https://cdn.e-shuushuu.net/thumbs/2026-08-13-1117538.webp',
        'medium_url': None, 'large_url': None,
    }
    item.update(overrides)
    return item


@pytest.mark.unittest
class TestToTimestamp:
    def test_keeps_the_seconds(self):
        # The old table truncated these to the minute; this one must not.
        ts = to_timestamp('2026-08-13T04:39:44Z')
        assert datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).second == 44

    def test_value(self):
        expected = datetime.datetime(2026, 8, 13, 4, 39, 44,
                                     tzinfo=datetime.timezone.utc).timestamp()
        assert to_timestamp('2026-08-13T04:39:44Z') == pytest.approx(expected)

    @pytest.mark.parametrize('value', [None, '', 'nonsense'])
    def test_unusable(self, value):
        assert to_timestamp(value) is None


@pytest.mark.unittest
class TestSplitTags:
    def test_flat_forms_cover_every_tag(self):
        out = split_tags(_api_image())
        assert out['tags'] == ['Hiten', 'bag', 'barefoot', 'Chobits', 'Chii']
        assert out['tag_ids'] == [52365, 245432, 33, 210, 66628]

    def test_grouped_by_category(self):
        out = split_tags(_api_image())
        assert out['tags_artist'] == ['Hiten']
        assert out['tags_theme'] == ['bag', 'barefoot']
        assert out['tags_source'] == ['Chobits']
        assert out['tags_character'] == ['Chii']

    def test_every_category_column_exists_even_when_empty(self):
        out = split_tags({'tags': []})
        for name in ('tags_artist', 'tags_character', 'tags_source', 'tags_theme'):
            assert out[name] == []

    def test_unknown_category_still_reaches_the_flat_list(self):
        out = split_tags({'tags': [{'tag_id': 1, 'title': 'x', 'type_name': 'Something'}]})
        assert out['tags'] == ['x'] and out['tag_ids'] == [1]

    def test_no_tags_at_all(self):
        out = split_tags({})
        assert out['tags'] == [] and out['tag_ids'] == []


@pytest.mark.unittest
class TestBuildRow:
    def test_produces_exactly_the_published_columns(self):
        assert tuple(build_row(_api_image())) == PUBLISHED_COLUMNS

    def test_identity_and_names(self):
        row = build_row(_api_image())
        assert row['id'] == 1117538
        assert row['username'] == 'Housekino' and row['user_id'] == 846201
        assert row['src_filename'] == '2026-08-13-1117538.png'

    def test_file_url_keeps_the_pre_cdn_form(self):
        # It still resolves and is what the previous table recorded; the CDN url sits beside it.
        row = build_row(_api_image())
        assert row['file_url'] == 'https://e-shuushuu.net/images/2026-08-13-1117538.png'
        assert row['cdn_url'].startswith('https://cdn.e-shuushuu.net/fullsize/')

    def test_exact_file_size_not_a_formatted_string(self):
        assert build_row(_api_image())['file_size'] == 1536863

    def test_score_is_the_bayesian_rating(self):
        row = build_row(_api_image(bayesian_rating=7.61143, rating=7.0))
        assert row['score'] == pytest.approx(7.61143)
        assert row['rating'] == 7.0

    def test_empty_strings_become_none(self):
        row = build_row(_api_image(caption='', miscmeta='', source_url=''))
        assert row['caption'] is None
        assert row['misc_metadata'] is None
        assert row['source_url'] is None

    def test_mimetype(self):
        assert build_row(_api_image())['mimetype'] == 'image/png'
        assert build_row(_api_image(ext='jpeg'))['mimetype'] == 'image/jpeg'

    def test_missing_pieces_do_not_raise(self):
        row = build_row({'image_id': 5})
        assert row['id'] == 5 and row['src_filename'] is None and row['file_url'] is None
        assert tuple(row) == PUBLISHED_COLUMNS

    def test_fits_an_arrow_table(self):
        rows = [build_row(_api_image(image_id=i)) for i in (1, 2, 3)]
        table = pa.Table.from_pylist(rows)
        assert table.num_rows == 3
        assert set(table.schema.names) == set(PUBLISHED_COLUMNS)


@pytest.mark.unittest
class TestDropDuplicateIds:
    def _table(self, ids, marks):
        return pa.table({'id': ids, 'mark': marks})

    def test_keeps_the_last_occurrence(self):
        out = _drop_duplicate_ids(self._table([3, 1, 2, 1, 3], ['a', 'b', 'c', 'B', 'A']))
        by_id = {r['id']: r['mark'] for r in out.to_pylist()}
        assert out.num_rows == 3
        assert by_id == {1: 'B', 2: 'c', 3: 'A'}

    def test_clean_table_is_returned_untouched(self):
        table = self._table([1, 2, 3], ['a', 'b', 'c'])
        assert _drop_duplicate_ids(table).num_rows == 3

    def test_empty_table(self):
        table = pa.table({'id': pa.array([], type=pa.int64())})
        assert _drop_duplicate_ids(table).num_rows == 0

    def test_all_rows_the_same_id(self):
        out = _drop_duplicate_ids(self._table([7, 7, 7], ['a', 'b', 'c']))
        assert out.num_rows == 1 and out.to_pylist()[0]['mark'] == 'c'

    def test_ids_stay_unique_afterwards(self):
        out = _drop_duplicate_ids(self._table([1, 1, 2, 2, 3], list('abcde')))
        ids = out.column('id').to_pylist()
        assert len(ids) == len(set(ids))


@pytest.mark.unittest
class TestSignature:
    def test_same_image_same_signature(self):
        assert row_signature(build_row(_api_image())) == row_signature(build_row(_api_image()))

    @pytest.mark.parametrize('key,value', [
        ('favorites', 99), ('num_ratings', 12), ('rating', 9.5),
        ('bayesian_rating', 8.8), ('posts', 4),
    ])
    def test_drifting_fields_are_ignored(self, key, value):
        assert row_signature(build_row(_api_image())) == \
               row_signature(build_row(_api_image(**{key: value})))

    @pytest.mark.parametrize('key,value', [
        ('md5_hash', 'f' * 32),
        ('filesize', 999),
        ('width', 100),
        ('status', 2),
        ('source_url', 'https://other'),
        ('caption', 'now with words'),
    ])
    def test_meaningful_changes_register(self, key, value):
        assert row_signature(build_row(_api_image())) != \
               row_signature(build_row(_api_image(**{key: value})))

    def test_tag_order_is_ignored(self):
        item = _api_image()
        reordered = _api_image(tags=list(reversed(item['tags'])))
        assert row_signature(build_row(item)) == row_signature(build_row(reordered))

    def test_a_new_tag_registers(self):
        item = _api_image()
        extra = _api_image(tags=item['tags'] + [
            {'tag_id': 999, 'title': 'zzz', 'type': 1, 'type_name': 'Theme', 'usage_count': 1}])
        assert row_signature(build_row(item)) != row_signature(build_row(extra))

    def test_table_and_row_signatures_agree(self):
        rows = [build_row(_api_image(image_id=1)),
                build_row(_api_image(image_id=2, width=42)),
                build_row(_api_image(image_id=3, tags=[]))]
        table = pa.Table.from_pylist(rows)
        sigs = table_signatures(table)
        for row in rows:
            assert sigs[row['id']] == row_signature(row)

    def test_every_trigger_field_is_present_in_a_row(self):
        row = build_row(_api_image())
        missing = [f for f in _UPDATE_TRIGGER_FIELDS if f not in row]
        assert not missing, f'trigger fields absent from the row: {missing}'
