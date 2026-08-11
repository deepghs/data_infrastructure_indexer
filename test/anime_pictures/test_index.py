import json

import pytest

from inf.anime_pictures.index import (_TABLE_COLUMNS, _as_float, _as_int, _as_timestamp,
                                      _hex_color, build_row)

#: A payload shaped like the live API, including the two fields it no longer sends.
SAMPLE = {
    'file_url': '926072-2200x3628-original-lethe rin.png',
    'post': {
        'id': 926072, 'width': 2200, 'height': 3628, 'size': 11416887,
        'md5': '2d039f34977dd5285ac612c8a63940dc', 'md5_pixels': 'abc',
        'erotics': 1, 'ext': '.png', 'status': 0, 'status_type': 0,
        'spoiler': False, 'have_alpha': True, 'color': [185, 173, 172],
        'artefacts_degree': 8.122037921611959, 'smooth_degree': 38.27032415815451,
        'tags_count': 32, 'small_preview': 's.jpg', 'medium_preview': 'm.jpg',
        'big_preview': 'b.jpg', 'score': 5, 'score_number': 3, 'download_count': 12,
        'pubtime': '2026-08-11T11:49:41.981492', 'datetime': '2026-08-11T11:40:00.000000',
    },
    'tags': [{'tag': {'id': 94, 'tag': 'original', 'type': 6}},
             {'tag': {'id': 12, 'tag': 'long hair', 'type': 7}}],
    'favorites_users': [{'id': 1}, {'id': 2}],
    'user': {'id': 7, 'name': 'someone'},
    'moderator': None,
}


@pytest.mark.unittest
class TestBuildRow:
    def test_schema_matches_the_stored_table_exactly(self):
        # The published table must stay a continuation of the existing one, not a new shape.
        assert list(build_row(SAMPLE).keys()) == _TABLE_COLUMNS

    def test_fractional_metrics_survive(self):
        # These come back fractional; an integer coercion silently nulls them, which is what
        # happened on the first attempt.
        row = build_row(SAMPLE)
        assert row['artifacts_degree'] == pytest.approx(8.122037921611959)
        assert row['smooth_degree'] == pytest.approx(38.27032415815451)

    def test_tags_are_stored_as_a_json_string(self):
        assert json.loads(build_row(SAMPLE)['tags']) == ['original', 'long hair']

    def test_missing_moderator_is_tolerated(self):
        row = build_row(SAMPLE)
        assert row['moderator_id'] is None and row['moderator_name'] is None

    def test_absent_fields_are_none_not_errors(self):
        # position and redirect_id are gone from the API; both are null in every stored row.
        row = build_row(SAMPLE)
        assert row['position'] is None and row['redirect_id'] is None

    def test_redirect_id_is_recorded_when_supplied(self):
        assert build_row(SAMPLE, redirect_id=517734)['redirect_id'] == 517734

    def test_colour_is_rendered_as_hex(self):
        assert build_row(SAMPLE)['color'].startswith('#')

    def test_file_url_is_built_from_the_filename(self):
        row = build_row(SAMPLE)
        assert row['filename'] == SAMPLE['file_url']
        assert row['file_url'].startswith('https://api.anime-pictures.net/pictures/download_image/')
        assert ' ' not in row['file_url'], 'the filename must be url-quoted'

    def test_empty_payload_does_not_raise(self):
        row = build_row({})
        assert list(row.keys()) == _TABLE_COLUMNS
        assert row['id'] is None


@pytest.mark.unittest
class TestCoercion:
    def test_as_int_rejects_bool_and_fractions(self):
        assert _as_int(True) is None
        assert _as_int(8.5) is None
        assert _as_int(8.0) == 8
        assert _as_int('42') == 42
        assert _as_int('abc') is None

    def test_as_float_keeps_fractions(self):
        assert _as_float(8.122) == pytest.approx(8.122)
        assert _as_float('3.5') == pytest.approx(3.5)
        assert _as_float(True) is None
        assert _as_float(None) is None
        assert _as_float('abc') is None

    def test_as_timestamp_parses_and_tolerates_blanks(self):
        assert _as_timestamp('2026-08-11T11:49:41.981492') > 0
        assert _as_timestamp(None) is None
        assert _as_timestamp('') is None

    def test_hex_color_needs_a_triple(self):
        assert _hex_color([255, 0, 0]).startswith('#')
        for bad in (None, [], [1, 2], 'red', [1, 2, 3, 4]):
            assert _hex_color(bad) is None
