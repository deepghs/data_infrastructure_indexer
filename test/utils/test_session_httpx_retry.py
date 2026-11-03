import httpx
import pytest

from inf.utils import session as session_module
from inf.utils.session import REQUEST_ERRORS, srequest


class _Response:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def raise_for_status(self):
        pass


class _Session:
    """A session that fails the given number of times with ``error`` and then answers."""

    def __init__(self, error, failures):
        self.error, self.failures, self.calls = error, failures, 0

    def request(self, method, url, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return _Response()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(session_module.time, 'sleep', lambda seconds: None)


@pytest.mark.unittest
class TestHttpxErrorsAreRetried:
    @pytest.mark.parametrize('error', [
        httpx.ReadTimeout('The read operation timed out'),
        httpx.RemoteProtocolError('Server disconnected'),
        httpx.ConnectError('connection refused'),
        httpx.ConnectTimeout('connect timed out'),
    ])
    def test_a_transient_httpx_error_is_retried_until_the_request_goes_through(self, error):
        session = _Session(error, failures=2)
        assert srequest(session, 'GET', 'https://example.invalid/').status_code == 200
        assert session.calls == 3

    def test_the_public_alias_covers_httpx_too(self):
        assert issubclass(httpx.ReadTimeout, REQUEST_ERRORS)
        assert issubclass(httpx.RemoteProtocolError, REQUEST_ERRORS)

    def test_an_error_that_never_clears_is_not_hidden(self):
        session = _Session(httpx.ReadTimeout('slow'), failures=10 ** 6)
        with pytest.raises(AssertionError):
            srequest(session, 'GET', 'https://example.invalid/', max_retries=3)
        assert session.calls == 3
