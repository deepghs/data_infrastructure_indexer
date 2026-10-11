import pytest

from inf.rule34 import index as rule34_index
from inf.rule34 import tags as rule34_tags
from inf.rule34.tags import _get_tag_aliases_by_page, _get_tags_by_page

_TAGS_XML = ('<?xml version="1.0" encoding="UTF-8"?><tags type="array">'
             '<tag type="4" count="2" name="kogurein" ambiguous="false" id="12668"/>'
             '<tag type="3" count="1" name="kyrie_(quilt)" ambiguous="false" id="20163"/></tags>')


class _Response:
    status_code = 200

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class _RecordingSession:
    def __init__(self, text):
        self.text = text
        self.urls = []

    def request(self, method, url, **kwargs):
        self.urls.append(url)
        return _Response(self.text)


@pytest.mark.unittest
class TestApiHost:
    def test_api_host_is_not_the_challenged_main_domain(self):
        assert rule34_tags.__api_url__ == 'https://api.rule34.xxx'
        assert rule34_tags.__api_url__ != rule34_tags.__site_url__

    def test_index_uses_the_same_api_host_as_tags(self):
        assert rule34_index.__api_url__ == rule34_tags.__api_url__

    def test_tag_dapi_requests_the_api_host(self):
        session = _RecordingSession(_TAGS_XML)
        rows = _get_tags_by_page(0, user_id='1', api_key='k', session=session)
        assert session.urls == ['https://api.rule34.xxx/index.php']
        assert [r['name'] for r in rows] == ['kogurein', 'kyrie_(quilt)']

    def test_alias_list_stays_on_the_main_domain(self):
        # This is an HTML page with no dapi equivalent; the api host answers it with an empty body.
        session = _RecordingSession('<html><body><div id="aliases"><table></table></div></body></html>')
        _get_tag_aliases_by_page(0, user_id='1', api_key='k', session=session)
        assert session.urls == ['https://rule34.xxx/index.php']
