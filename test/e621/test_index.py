import pytest

from inf.e621 import index as ix

pytestmark = pytest.mark.unittest

NEW_POST = {
    'id': 100,
    'created_at': '2026-11-20T10:00:00.123-05:00',
    'updated_at': '2026-11-21T12:30:00.000-05:00',
    'change_seq': 777,
    'uploader_id': 5,
    'uploader_name': 'someone',
    'approver_id': 6,
    'rating': 'e',
    'description': 'desc',
    'locked_tags': [],
    'sources': ['https://example.com/a'],
    'pools': [3],
    'files': {
        'original': {'width': 800, 'height': 600, 'url': 'https://static1.e621.net/data/ab/cd/abcd.png'},
        'sample': {'width': 800, 'height': 600, 'jpg': 'https://static1.e621.net/data/sample/ab/cd/abcd.jpg',
                   'webp': 'https://static1.e621.net/data/sample/ab/cd/abcd.webp'},
        'preview': {'width': 150, 'height': 112, 'jpg': 'https://static1.e621.net/data/preview/ab/cd/abcd.jpg',
                    'webp': 'https://static1.e621.net/data/preview/ab/cd/abcd.webp'},
        'meta': {'ext': 'png', 'size': 12345, 'md5': 'abcd', 'duration': None},
    },
    'stats': {'score': {'up': 4, 'down': -1, 'total': 3}, 'fav_count': 9, 'comment_count': 2,
              'is_favorited': False, 'vote': 0},
    'flags': {'pending': False, 'flagged': False, 'note_locked': False, 'status_locked': False,
              'rating_locked': False, 'deleted': False},
    'relationships': {'parent_id': None, 'children': [101]},
    'has': {'sample': True, 'children': True, 'active_children': True, 'notes': False},
    'tags': ['2026', 'wolf', 'artist_x', 'hi_res', 'solo'],
}
CATEGORIES = {'solo': 0, '2026': 7, 'artist_x': 1, 'wolf': 5, 'hi_res': 7}


def _categories(known=None, fetch=None):
    return ix.TagCategories(dict(CATEGORIES if known is None else known), fetch or (lambda name: None))


class TestPostsOf:
    def test_old_shape(self):
        assert ix._posts_of({'posts': [{'id': 1}]}) == [{'id': 1}]

    def test_new_shape(self):
        assert ix._posts_of([{'id': 1}, {'id': 2}]) == [{'id': 1}, {'id': 2}]

    def test_empty_list(self):
        assert ix._posts_of([]) == []

    @pytest.mark.parametrize('payload', [{'success': False}, 'blocked', None, {'posts': 3}])
    def test_unexpected(self, payload):
        with pytest.raises(ValueError):
            ix._posts_of(payload)


class TestTagCategories:
    def test_known_is_not_fetched(self):
        calls = []
        cats = _categories(fetch=lambda n: calls.append(n))
        assert cats.category_of('wolf') == 5
        assert calls == []

    def test_fetch_once_and_remember(self):
        calls = []

        def fetch(name):
            calls.append(name)
            return {'id': 9, 'name': name, 'category': 4, 'post_count': 12}

        cats = _categories(fetch=fetch)
        assert cats.category_of('new_char') == 4
        assert cats.category_of('new_char') == 4
        assert calls == ['new_char']
        assert cats.fetched['new_char']['post_count'] == 12

    def test_unresolved_falls_back_to_general(self):
        cats = _categories()
        assert cats.category_of('ghost') == 0
        assert cats.category_of('ghost') == 0
        assert cats.unresolved == ['ghost']

    def test_split_groups_in_site_order(self):
        grouped = _categories().split(['2026', 'wolf', 'artist_x', 'hi_res', 'solo'])
        assert list(grouped) == ['general', 'artist', 'species', 'meta']
        assert grouped == {'general': ['solo'], 'artist': ['artist_x'], 'species': ['wolf'],
                           'meta': ['2026', 'hi_res']}


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def get(self, url, params=None):
        self.calls.append((url, params))
        return _Resp(self.payload)


class TestTagFetcher:
    def test_exact_match_only(self):
        session = _Session([{'name': 'wolf_ears', 'category': 0}, {'name': 'wolf', 'category': 5}])
        assert ix._tag_fetcher(session)('wolf')['category'] == 5
        assert session.calls[0][1]['search[name_matches]'] == 'wolf'

    def test_no_exact_match(self):
        assert ix._tag_fetcher(_Session([{'name': 'wolf_ears'}]))('wolf') is None

    def test_dict_wrapped(self):
        assert ix._tag_fetcher(_Session({'tags': [{'name': 'wolf', 'category': 5}]}))('wolf')['category'] == 5

    def test_empty_search_answer(self):
        assert ix._tag_fetcher(_Session({'tags': []}))('wolf') is None


class TestStoredTime:
    def test_winter_zone_is_converted(self):
        assert ix._stored_time('2026-11-20T10:00:00.123-05:00') == '2026-11-20T16:00:00.123+01:00'

    def test_summer_zone_offset(self):
        assert ix._stored_time('2026-07-01T10:00:00.000-04:00') == '2026-07-01T16:00:00.000+02:00'

    def test_empty(self):
        assert ix._stored_time(None) is None


class TestLegacyPost:
    def test_old_shape_passes_through(self):
        old = {'id': 1, 'file': {}}
        assert ix._legacy_post(old, _categories()) is old

    def test_image_with_sample(self):
        post = ix._legacy_post(NEW_POST, _categories())
        assert post['file'] == {'width': 800, 'height': 600, 'ext': 'png', 'size': 12345, 'md5': 'abcd',
                                'url': 'https://static1.e621.net/data/ab/cd/abcd.png'}
        assert post['preview']['url'].endswith('abcd.jpg') and post['preview']['alt'].endswith('abcd.webp')
        assert post['sample']['has'] is True
        assert post['sample']['url'].endswith('abcd.jpg') and post['sample']['alt'].endswith('abcd.webp')
        assert post['sample']['alternates'] == {}
        assert post['score'] == {'up': 4, 'down': -1, 'total': 3}
        assert post['created_at'] == '2026-11-20T16:00:00.123+01:00'

    def test_no_sample_has_no_sample_urls(self):
        post = dict(NEW_POST, has=dict(NEW_POST['has'], sample=False))
        legacy = ix._legacy_post(post, _categories())
        assert legacy['sample']['has'] is False
        assert legacy['sample']['url'] is None and legacy['sample']['alt'] is None

    def test_video_alternates_and_duration(self):
        video = {'720p': {'url': 'https://static1.e621.net/data/v.mp4'}}
        files = dict(NEW_POST['files'], video=video, meta={'ext': 'webm', 'size': 5, 'md5': 'm', 'duration': 12.5})
        legacy = ix._legacy_post(dict(NEW_POST, files=files), _categories())
        assert legacy['sample']['alternates'] == video
        assert legacy['duration'] == 12.5


class TestRowOf:
    def test_row(self):
        row, tags_info = ix._row_of(ix._legacy_post(NEW_POST, _categories()))
        assert row['id'] == 100
        assert row['mimetype'] == 'image/png'
        assert row['tags'] == ['solo', 'artist_x', 'wolf', '2026', 'hi_res']
        assert tags_info['meta'] == ['2026', 'hi_res']
        assert (row['score'], row['up_score'], row['down_score'], row['fav_count']) == (3, 4, -1, 9)
        assert row['is_pending'] is False and row['is_deleted'] is False
        assert row['has_children'] is True and row['children'] == [101]
        assert row['sample_alternates'] == {'__dummy': None}

    def test_column_set_matches_stored_table(self):
        row, _ = ix._row_of(ix._legacy_post(NEW_POST, _categories()))
        assert set(row) == STORED_COLUMNS


STORED_COLUMNS = set(
    'id mimetype file_ext width height md5 file_url file_size rating tags uploader_id approver_id score up_score '
    'down_score fav_count preview_width preview_height preview_url preview_alt sample_has sample_width '
    'sample_height sample_url sample_alt sample_alternates is_pending is_flagged is_note_locked is_status_locked '
    'is_rating_locked is_deleted parent_id has_children has_active_children children created_at updated_at '
    'locked_tags change_seq sources pools uploader_name description comment_count is_favorited vote has_notes '
    'duration'.split()
)
