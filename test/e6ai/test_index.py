import pyarrow as pa
import pytest

from inf.e6ai.index import (_UNORDERED_TRIGGER_FIELDS, _UPDATE_TRIGGER_FIELDS, build_row,
                            has_file_url, parquet_safe, row_signature, table_signatures,
                            tags_by_category)
from inf.utils.upsert import adds_anything

#: The published column set, as it stands on the hub. A contract: the job overwrites e6ai.parquet
#: in place, and the column set is the flattening of an API shape that has already drifted once.
PUBLISHED_COLUMNS = (
    'id', 'mimetype', 'file_ext', 'width', 'height', 'md5', 'file_url', 'file_size', 'rating',
    'tags', 'uploader_id', 'approver_id', 'score', 'up_score', 'down_score', 'fav_count',
    'preview_width', 'preview_height', 'preview_url', 'sample_has', 'sample_height',
    'sample_width', 'sample_url', 'sample_alternates', 'is_pending', 'is_flagged',
    'is_note_locked', 'is_status_locked', 'is_rating_locked', 'is_deleted', 'parent_id',
    'has_children', 'has_active_children', 'children', 'created_at', 'updated_at',
    'locked_tags', 'change_seq', 'sources', 'pools', 'description', 'comment_count',
    'is_favorited', 'has_notes', 'duration', 'preview_alt', 'sample_alt', 'uploader_name',
)


def _api_post(**overrides):
    """One entry shaped the way /posts.json returns it, nested objects and all."""
    post = {
        'id': 179782,
        'created_at': '2026-08-13T04:12:11.123-04:00',
        'updated_at': '2026-08-13T05:00:00.000-04:00',
        'file': {'width': 1024, 'height': 1536, 'ext': 'png', 'size': 1234567,
                 'md5': 'd41d8cd98f00b204e9800998ecf8427e',
                 'url': 'https://static1.e6ai.net/data/d4/1d/x.png'},
        'preview': {'width': 150, 'height': 225, 'alt': 'a preview',
                    'url': 'https://static1.e6ai.net/data/preview/d4/1d/x.jpg'},
        'sample': {'has': True, 'height': 1200, 'width': 800, 'alt': 'a sample',
                   'url': 'https://static1.e6ai.net/data/sample/d4/1d/x.jpg',
                   'alternates': {}},
        'score': {'up': 5, 'down': -1, 'total': 4},
        'tags': {'general': ['solo', 'anthro'], 'artist': ['someone'], 'copyright': [],
                 'character': ['charname'], 'species': ['canine'], 'invalid': [],
                 'meta': ['ai_generated'], 'lore': []},
        'locked_tags': [],
        'change_seq': 12345,
        'flags': {'pending': False, 'flagged': False, 'note_locked': False,
                  'status_locked': False, 'rating_locked': False, 'deleted': False},
        'rating': 'e',
        'fav_count': 3,
        'sources': ['https://example/a'],
        'pools': [],
        'relationships': {'parent_id': None, 'has_children': False,
                          'has_active_children': False, 'children': []},
        'approver_id': None,
        'uploader_id': 42,
        'uploader_name': 'someuploader',
        'description': '',
        'comment_count': 0,
        'is_favorited': False,
        'has_notes': False,
        'duration': None,
    }
    post.update(overrides)
    return post


@pytest.mark.unittest
class TestParquetSafe:
    def test_empty_dict_gets_a_placeholder(self):
        # Arrow cannot infer a struct type for {}, and the API sends exactly that for most posts.
        assert parquet_safe({}) == {'__dummy': None}

    def test_nested_empty_dicts(self):
        assert parquet_safe({'a': {}, 'b': {'c': {}}}) == \
               {'a': {'__dummy': None}, 'b': {'c': {'__dummy': None}}}

    def test_populated_dict_is_kept(self):
        assert parquet_safe({'has': True, 'n': 1}) == {'has': True, 'n': 1}

    def test_lists_are_walked(self):
        assert parquet_safe([{}, {'a': {}}]) == [{'__dummy': None}, {'a': {'__dummy': None}}]

    def test_list_type_is_preserved(self):
        assert isinstance(parquet_safe(({},)), tuple)

    def test_scalars_pass_through(self):
        for value in (1, 'a', None, True, 1.5):
            assert parquet_safe(value) == value

    def test_real_alternates_shape(self):
        alternates = {'has': True, 'original': {'fps': 24.0, 'codec': 'av01', 'size': 13034945,
                                                'width': 3880, 'height': 2128, 'url': 'https://x'},
                      'samples': {}, 'variants': {}}
        out = parquet_safe(alternates)
        assert out['original']['fps'] == 24.0
        assert out['samples'] == {'__dummy': None}
        assert out['variants'] == {'__dummy': None}


@pytest.mark.unittest
class TestHasFileUrl:
    def test_present(self):
        assert has_file_url(_api_post())

    def test_none(self):
        assert not has_file_url(_api_post(file={'url': None, 'ext': 'png'}))

    def test_empty(self):
        assert not has_file_url(_api_post(file={'url': '   '}))

    def test_file_object_absent(self):
        post = _api_post()
        del post['file']
        assert not has_file_url(post)


@pytest.mark.unittest
class TestBuildRow:
    def test_covers_every_published_column(self):
        produced = set(build_row(_api_post()))
        missing = set(PUBLISHED_COLUMNS) - produced
        assert not missing, f'columns never produced: {sorted(missing)}'

    def test_flattens_the_nested_objects(self):
        row = build_row(_api_post())
        assert row['file_ext'] == 'png' and row['md5'].startswith('d41d8')
        assert row['width'] == 1024 and row['height'] == 1536
        assert row['preview_width'] == 150 and row['sample_width'] == 800
        assert row['score'] == 4 and row['up_score'] == 5 and row['down_score'] == -1
        assert row['is_deleted'] is False and row['is_pending'] is False
        assert row['parent_id'] is None and row['children'] == []

    def test_tags_are_a_flat_list_of_every_category(self):
        row = build_row(_api_post())
        assert isinstance(row['tags'], list)
        assert set(row['tags']) == {'solo', 'anthro', 'someone', 'charname', 'canine',
                                    'ai_generated'}

    def test_mimetype_comes_from_the_file_url(self):
        assert build_row(_api_post())['mimetype'] == 'image/png'
        post = _api_post()
        post['file'] = dict(post['file'], url='https://x/y.webm')
        assert build_row(post)['mimetype'] == 'video/webm'

    def test_mimetype_is_none_without_a_url(self):
        post = _api_post()
        post['file'] = dict(post['file'], url=None)
        assert build_row(post)['mimetype'] is None

    def test_empty_alternates_become_typeable(self):
        assert build_row(_api_post())['sample_alternates'] == {'__dummy': None}

    def test_input_is_not_mutated(self):
        # The prototype popped straight out of the API dict; this must not.
        post = _api_post()
        build_row(post)
        assert 'file' in post and 'tags' in post and 'relationships' in post

    def test_timestamps_stay_strings(self):
        # Unlike the danbooru-derived tables, this schema stores them as text.
        row = build_row(_api_post())
        assert isinstance(row['created_at'], str)
        assert isinstance(row['updated_at'], str)

    def test_missing_nested_objects_do_not_raise(self):
        row = build_row({'id': 7})
        assert row['id'] == 7 and row['tags'] == [] and row['file_url'] is None


@pytest.mark.unittest
class TestTagsByCategory:
    def test_groups_by_category_id(self):
        grouped = tags_by_category(_api_post())
        assert grouped[0] == ['solo', 'anthro']    # general
        assert grouped[1] == ['someone']           # artist
        assert grouped[7] == ['ai_generated']      # meta

    def test_unknown_category_is_ignored(self):
        assert tags_by_category({'tags': {'nonsense': ['x']}}) == {}

    def test_no_tags(self):
        assert tags_by_category({}) == {}


@pytest.mark.unittest
class TestSignatures:
    def test_same_post_same_signature(self):
        assert row_signature(build_row(_api_post())) == row_signature(build_row(_api_post()))

    def test_tag_order_does_not_matter(self):
        # The API does not hold tag order stable; comparing as ordered would mark a row changed on
        # every run forever.
        post = _api_post()
        reordered = _api_post(tags={'meta': ['ai_generated'], 'general': ['anthro', 'solo'],
                                    'artist': ['someone'], 'character': ['charname'],
                                    'species': ['canine'], 'copyright': [], 'invalid': [],
                                    'lore': []})
        assert set(build_row(post)['tags']) == set(build_row(reordered)['tags'])
        assert row_signature(build_row(post)) == row_signature(build_row(reordered))

    def test_tag_content_does_matter(self):
        post = _api_post()
        extra = _api_post(tags=dict(post['tags'], general=['solo', 'anthro', 'smile']))
        assert row_signature(build_row(post)) != row_signature(build_row(extra))

    @pytest.mark.parametrize('key,value', [
        ('rating', 's'),
        ('description', 'now with words'),
        ('sources', ['https://example/b']),
    ])
    def test_meaningful_changes_trigger(self, key, value):
        assert row_signature(build_row(_api_post())) != \
               row_signature(build_row(_api_post(**{key: value})))

    @pytest.mark.parametrize('key,value', [
        ('fav_count', 999),
        ('comment_count', 12),
        ('change_seq', 999999),
        ('updated_at', '2027-01-01T00:00:00.000-04:00'),
        ('is_favorited', True),
    ])
    def test_drifting_fields_do_not_trigger(self, key, value):
        assert row_signature(build_row(_api_post())) == \
               row_signature(build_row(_api_post(**{key: value})))

    def test_score_drift_does_not_trigger(self):
        assert row_signature(build_row(_api_post())) == \
               row_signature(build_row(_api_post(score={'up': 99, 'down': -9, 'total': 90})))

    def test_table_and_row_signatures_agree(self):
        rows = [build_row(_api_post(id=1)),
                build_row(_api_post(id=2, rating='s')),
                build_row(_api_post(id=3))]
        columns = ['id'] + list(_UPDATE_TRIGGER_FIELDS)
        table = pa.table({c: [r.get(c) for r in rows] for c in columns})
        sigs = table_signatures(table)
        for row in rows:
            assert sigs[row['id']] == row_signature(row)

    def test_table_signatures_are_order_blind_too(self):
        # The stored side must normalise identically, or every row reads as stale.
        stored = build_row(_api_post(id=1))
        stored['tags'] = list(reversed(stored['tags']))
        columns = ['id'] + list(_UPDATE_TRIGGER_FIELDS)
        table = pa.table({c: [stored.get(c)] for c in columns})
        assert table_signatures(table)[1] == row_signature(build_row(_api_post(id=1)))


@pytest.mark.unittest
class TestAddsAnything:
    def _stored_and_sig(self, **overrides):
        row = build_row(_api_post(**overrides))
        return row, row_signature(row)

    def test_a_field_the_api_stopped_sending_adds_nothing(self):
        # Measured on the live site: sample_url goes value -> None on 18% of re-fetched posts.
        stored, sig = self._stored_and_sig()
        post = _api_post()
        post['sample'] = dict(post['sample'], url=None)
        fetched = build_row(post)
        assert row_signature(fetched) != sig          # naive comparison sees a change
        assert not adds_anything(stored, fetched, sig, _UPDATE_TRIGGER_FIELDS,
                                 _UNORDERED_TRIGGER_FIELDS)

    def test_a_deleted_post_losing_its_urls_adds_nothing_but_the_flag(self):
        stored, sig = self._stored_and_sig()
        post = _api_post(flags={'pending': False, 'flagged': False, 'note_locked': False,
                                'status_locked': False, 'rating_locked': False, 'deleted': True})
        post['file'] = dict(post['file'], url=None, md5=None)
        fetched = build_row(post)
        # is_deleted flipping is a real change, so this one does need writing.
        assert adds_anything(stored, fetched, sig, _UPDATE_TRIGGER_FIELDS,
                             _UNORDERED_TRIGGER_FIELDS)

    def test_a_real_tag_edit_still_registers(self):
        stored, sig = self._stored_and_sig()
        post = _api_post()
        post['tags'] = dict(post['tags'], general=['solo', 'anthro', 'newtag'])
        assert adds_anything(stored, build_row(post), sig, _UPDATE_TRIGGER_FIELDS,
                             _UNORDERED_TRIGGER_FIELDS)

    def test_unknown_stored_row_is_treated_as_a_change(self):
        _, sig = self._stored_and_sig()
        assert adds_anything(None, build_row(_api_post()), sig, _UPDATE_TRIGGER_FIELDS)
