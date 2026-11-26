import pytest

from inf.utils import session as session_module
from inf.utils.session import REQUEST_ERRORS

pytestmark = pytest.mark.unittest


class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass


def test_browser_session_is_impersonating():
    session = session_module.get_browser_session('safari17_0', timeout=7)
    assert type(session).__module__.startswith('curl_cffi')


def test_open_browser_session_retries_then_succeeds(monkeypatch):
    attempts = []

    def fake_srequest(session, method, url, **kwargs):
        attempts.append(url)
        if len(attempts) < 3:
            raise REQUEST_ERRORS[0]('blocked')
        return _Resp()

    monkeypatch.setattr(session_module, 'srequest', fake_srequest)
    monkeypatch.setattr(session_module.time, 'sleep', lambda _: None)
    session = session_module.open_browser_session('https://example.invalid/probe', attempts=5)
    assert session is not None
    assert len(attempts) == 3


def test_open_browser_session_gives_up(monkeypatch):
    def fake_srequest(session, method, url, **kwargs):
        raise REQUEST_ERRORS[0]('blocked')

    monkeypatch.setattr(session_module, 'srequest', fake_srequest)
    monkeypatch.setattr(session_module.time, 'sleep', lambda _: None)
    with pytest.raises(REQUEST_ERRORS[0]):
        session_module.open_browser_session('https://example.invalid/probe', attempts=2)
