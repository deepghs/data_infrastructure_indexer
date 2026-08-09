import pytest

from inf.konachan.base import IMPERSONATE_LADDER, PREFERRED_IMPERSONATES, get_konachan_session
from inf.utils.impersonate import build_ladder, supported_impersonates


@pytest.mark.unittest
class TestKonachanLadder:
    def test_ladder_is_not_empty(self):
        assert IMPERSONATE_LADDER

    def test_ladder_excludes_chrome_and_edge(self):
        # Every Chrome and Edge fingerprint tested from a GitHub runner was refused, and
        # chrome120 passing one endpoint while failing the other is worse than a clean refusal:
        # it lets a run start and then die partway through.
        for name in IMPERSONATE_LADDER:
            assert not name.startswith(('chrome', 'edge')), f'{name} is known to be refused'

    def test_ladder_only_offers_what_the_build_has(self):
        supported = supported_impersonates()
        if not supported:
            pytest.skip('this curl_cffi build cannot report its impersonation targets')
        # Naming a target the build has never heard of raises at session construction, which
        # downstream looks like a network fault rather than a configuration error.
        assert set(IMPERSONATE_LADDER) <= supported

    def test_session_uses_a_ladder_fingerprint(self):
        session = get_konachan_session()
        assert session.headers['Referer'] == 'https://konachan.com/'

    def test_explicit_fingerprint_is_honoured(self):
        session = get_konachan_session(impersonate=IMPERSONATE_LADDER[0])
        assert session is not None


@pytest.mark.unittest
class TestBuildLadder:
    def test_unsupported_names_are_dropped(self):
        supported = supported_impersonates()
        if not supported:
            pytest.skip('this curl_cffi build cannot report its impersonation targets')
        real = sorted(supported)[0]
        ladder = build_ladder([real, 'netscape_4_0'], site='test')
        assert ladder == [real]

    def test_falls_back_rather_than_returning_nothing(self):
        supported = supported_impersonates()
        if not supported:
            pytest.skip('this curl_cffi build cannot report its impersonation targets')
        # A wanted list with nothing available must still yield something usable, otherwise a
        # build refresh that renames every target takes the job down instead of degrading it.
        assert build_ladder(['netscape_4_0'], site='test')


@pytest.mark.unittest
def test_preferred_list_is_ordered_newest_first():
    assert PREFERRED_IMPERSONATES.index('safari17_0') < PREFERRED_IMPERSONATES.index('safari15_5')
