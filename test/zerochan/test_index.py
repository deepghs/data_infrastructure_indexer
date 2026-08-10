import json

import pytest

from inf.zerochan.index import loads_zerochan_json

#: A real body zerochan served for post 1054049, trimmed to the shape that matters. The tag
#: `Kokonose "Konoha" Haruka` carries quotes the site never escapes, so strict parsing fails.
MALFORMED = '''{
  "id": 1054049,
  "full": "https://static.zerochan.net/a.full.1054049.jpg",
  "width": 1000,
  "height": 1400,
  "size": 1244160,
  "hash": "deadbeef",
  "primary": "Kagerou Project",
  "tags": [
    "Female",
    "Kokonose "Konoha" Haruka",
    "Mobile Wallpaper"
  ]
}'''

WELL_FORMED = '''{
  "id": 4716276,
  "full": "https://static.zerochan.net/b.full.4716276.jpg",
  "width": 2364,
  "height": 3210,
  "size": 900,
  "hash": "cafebabe",
  "primary": "Fate/Grand Order",
  "tags": ["Female", "Long Hair"]
}'''


@pytest.mark.unittest
class TestLoadsZerochanJson:
    def test_rejects_strict_parsing(self):
        # Guards the premise: if the site ever starts escaping properly this test fails and the
        # repair path can be reconsidered rather than carried forever.
        with pytest.raises(json.JSONDecodeError):
            json.loads(MALFORMED)

    def test_repairs_unescaped_quotes_without_loss(self):
        got = loads_zerochan_json(MALFORMED)
        assert got['id'] == 1054049
        assert got['size'] == 1244160
        assert got['primary'] == 'Kagerou Project'
        # The whole point: the tag survives with its quotes, rather than being truncated at the
        # first one or dropped entirely.
        assert got['tags'] == ['Female', 'Kokonose "Konoha" Haruka', 'Mobile Wallpaper']

    def test_leaves_valid_bodies_untouched(self):
        assert loads_zerochan_json(WELL_FORMED) == json.loads(WELL_FORMED)

    def test_raises_when_body_is_not_an_object(self):
        # An HTML error page must not silently become an empty record.
        with pytest.raises(json.JSONDecodeError):
            loads_zerochan_json('<html><body>nope</body></html>')


@pytest.mark.unittest
class TestZerochanSessionPieces:
    def test_challenge_image_is_matched_by_shape_not_name(self):
        from inf.zerochan.base import _CHALLENGE_IMG
        # The nonce changes every response and the filename has already changed once, from
        # xbotcheck-image.svg to this, so matching on either would break again.
        body = ('<body><h1>Checking browser...</h1>'
                '<img src="/totally-innocent-logo-image.svg?iRVmyolJ" onload="location.reload()">')
        matched = _CHALLENGE_IMG.search(body)
        assert matched
        assert matched.group('path') == '/totally-innocent-logo-image.svg?iRVmyolJ'

    def test_ladder_is_available(self):
        from inf.zerochan.base import IMPERSONATE_LADDER
        assert IMPERSONATE_LADDER

    def test_paced_session_spaces_requests(self):
        import time
        from inf.zerochan.base import PacedSession

        class FakeResponse:
            status_code = 200

        class FakeSession:
            def __init__(self):
                self.stamps = []

            def request(self, method, url, **kwargs):
                self.stamps.append(time.time())
                return FakeResponse()

        fake = FakeSession()
        paced = PacedSession(fake, min_interval=0.2)
        for _ in range(3):
            paced.get('https://example.invalid/')
        gaps = [b - a for a, b in zip(fake.stamps, fake.stamps[1:])]
        assert all(g >= 0.19 for g in gaps), gaps


@pytest.mark.unittest
class TestLoadsZerochanJsonHostileInput:
    def test_html_body_is_rejected_without_repair(self):
        # Some ids answer 200 with the page of a different, merged post. Repairing that yields a
        # list, and a list would sail past a caller expecting a record.
        html = '<!DOCTYPE html>\r\n<html lang="en"><head><title>Not Found</title></head></html>'
        with pytest.raises(json.JSONDecodeError):
            loads_zerochan_json(html)

    def test_deeply_nested_body_raises_decode_error_not_valueerror(self):
        # json_repair reports a blown recursion limit as a bare ValueError, and JSONDecodeError
        # subclasses ValueError rather than the reverse - so a caller catching JSONDecodeError
        # would miss it and the whole run would die on one bad body. Regression guard for that.
        hostile = '{"a":' + '[' * 5000
        with pytest.raises(json.JSONDecodeError):
            loads_zerochan_json(hostile)

    def test_empty_body_raises_decode_error(self):
        for body in ('', '   ', None):
            with pytest.raises(json.JSONDecodeError):
                loads_zerochan_json(body)

    def test_json_array_is_rejected(self):
        # A valid JSON array parses fine but is not a record.
        with pytest.raises(json.JSONDecodeError):
            loads_zerochan_json('[1, 2, 3]')
