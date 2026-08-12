import pyarrow as pa
import pytest

from inf.atfbooru.index import build_row, drop_urlless_rows, has_file_url


@pytest.mark.unittest
class TestHasFileUrl:
    @pytest.mark.parametrize('item', [
        {'id': 1, 'file_url': 'https://booru.allthefallen.moe/data/abc.png'},
        {'id': 2, 'file_url': 'x'},
    ])
    def test_present(self, item):
        assert has_file_url(item)

    @pytest.mark.parametrize('item', [
        {'id': 1},
        {'id': 2, 'file_url': None},
        {'id': 3, 'file_url': ''},
        {'id': 4, 'file_url': '   '},
    ])
    def test_absent(self, item):
        assert not has_file_url(item)

    def test_banned_post_shape(self):
        # What the API actually returns for a banned post: metadata, no file, no checksum.
        item = {'id': 712315, 'is_banned': True, 'is_deleted': False,
                'file_url': None, 'md5': None, 'rating': 'e'}
        assert not has_file_url(item)

    def test_deleted_post_is_not_affected(self):
        # Deletion does not withhold the file; only a ban does.
        item = {'id': 100, 'is_banned': False, 'is_deleted': True,
                'file_url': 'https://booru.allthefallen.moe/data/deleted.jpg'}
        assert has_file_url(item)


@pytest.mark.unittest
class TestDropUrllessRows:
    def test_drops_null_and_empty(self):
        table = pa.table({
            'id': [1, 2, 3, 4],
            'file_url': ['https://x/a.png', None, '', 'https://x/b.jpg'],
        })
        out = drop_urlless_rows(table)
        assert out.num_rows == 2
        assert out.column('id').to_pylist() == [1, 4]

    def test_keeps_everything_when_all_present(self):
        table = pa.table({'id': [1, 2], 'file_url': ['a', 'b']})
        assert drop_urlless_rows(table).num_rows == 2

    def test_table_without_the_column_is_untouched(self):
        table = pa.table({'id': [1, 2]})
        assert drop_urlless_rows(table).num_rows == 2

    def test_preserves_schema(self):
        table = pa.table({'id': [1, 2], 'file_url': ['a', None], 'rating': ['e', 'g']})
        out = drop_urlless_rows(table)
        assert out.schema.names == table.schema.names

    def test_empty_table(self):
        table = pa.table({'id': pa.array([], type=pa.int64()),
                          'file_url': pa.array([], type=pa.string())})
        assert drop_urlless_rows(table).num_rows == 0


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
