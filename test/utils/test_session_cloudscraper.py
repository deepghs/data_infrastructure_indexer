import cloudscraper
import pytest

from inf.utils.session import get_cloudscraper_session


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_request(self, method, url, *args, **kwargs):
        calls.append(kwargs)
        return 'response'

    monkeypatch.setattr(cloudscraper.CloudScraper, 'request', fake_request)
    return calls


@pytest.mark.unittest
class TestCloudscraperSession:
    def test_requests_carry_a_timeout_by_default(self, recorded):
        session = get_cloudscraper_session(timeout=(3, 7))
        session.request('GET', 'https://example.test/')
        assert recorded[-1]['timeout'] == (3, 7)

    def test_an_explicit_timeout_wins(self, recorded):
        session = get_cloudscraper_session(timeout=(3, 7))
        session.request('GET', 'https://example.test/', timeout=1)
        assert recorded[-1]['timeout'] == 1

    def test_default_timeout_is_set_when_none_is_given(self, recorded):
        get_cloudscraper_session().request('GET', 'https://example.test/')
        assert recorded[-1]['timeout'] == (15, 60)

    def test_headers_include_a_user_agent_and_extras(self):
        session = get_cloudscraper_session(headers={'Referer': 'https://example.test/'})
        assert session.headers['User-Agent'] and session.headers['Referer'] == 'https://example.test/'

    def test_the_stock_wrapping_really_loses_the_timeout(self):
        # The reason the helper exists: wrapping a requests session leaves cloudscraper's own adapter,
        # which has no timeout and no retries.
        from inf.utils.session import get_requests_session
        stock = cloudscraper.session(get_requests_session(timeout=30, max_retries=3))
        assert getattr(stock.get_adapter('https://example.test/'), 'timeout', None) is None
