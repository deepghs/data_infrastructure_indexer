import hashlib

import pytest

from inf.atfbooru.base import clear_challenge, is_challenge, solve_pow

#: Shaped like the real challenge page, trimmed to what the parser reads.
CHALLENGE_PAGE = '''<!DOCTYPE html><html><head><title>booru.allthefallen.moe | Verification</title>
</head><body><script>
        const host = "booru.allthefallen.moe";
        const post_to = "booru.allthefallen.moe";
        const challenge_id = "E4Z0rsjI";
        const challenge_generated = "1786519064";
        const challenge_cookie_expires = "1787123864";
        const powSeed = "WsgEfDahW7kS0nDrXETimtrgd8A=";
        const powPrefix = "0".repeat(5);
        const delay = 5;
        const lifetime = 120;
</script></body></html>'''


class _Resp:
    def __init__(self, text='', headers=None, status_code=200):
        self.text = text
        self.headers = headers or {}
        self.status_code = status_code


@pytest.mark.unittest
class TestSolvePow:
    def test_digest_has_the_required_prefix(self):
        nonce, digest = solve_pow('seed', 3)
        assert digest.startswith('000')
        assert hashlib.sha1(f'seed:{nonce}'.encode()).hexdigest() == digest

    def test_zero_zeros_is_satisfied_immediately(self):
        nonce, _ = solve_pow('seed', 0)
        assert nonce == 0

    def test_candidate_format_is_seed_colon_nonce(self):
        # If the separator or ordering were wrong the server would reject every answer while the
        # solver still looked like it was working.
        nonce, digest = solve_pow('abc', 2)
        assert hashlib.sha1(f'abc:{nonce}'.encode()).hexdigest() == digest


@pytest.mark.unittest
class TestIsChallenge:
    def test_json_response_is_never_a_challenge(self):
        assert not is_challenge(_Resp('[]', {'content-type': 'application/json; charset=utf-8'}))

    def test_verification_page_is_detected(self):
        assert is_challenge(_Resp(CHALLENGE_PAGE, {'content-type': 'text/html'}))

    def test_unrelated_html_is_not_a_challenge(self):
        # Only the verification page carries powSeed; a plain error page must not be mistaken
        # for one, or the client would post answers into the void.
        assert not is_challenge(_Resp('<html><title>Not Found</title></html>',
                                      {'content-type': 'text/html'}))

    def test_empty_body_is_not_a_challenge(self):
        assert not is_challenge(_Resp('', {'content-type': 'text/html'}))


class _FakeSession:
    """Records the answer POST so the payload can be inspected."""

    def __init__(self, cookie_after_post='atf-anti-bot'):
        self.cookies = {}
        self.posted = None
        self._cookie = cookie_after_post

    def get(self, url, **kwargs):
        return _Resp('svg', {'content-type': 'image/svg+xml'})

    def post(self, url, json=None, headers=None, **kwargs):
        self.posted = {'url': url, 'json': json, 'headers': headers}
        if self._cookie:
            self.cookies[self._cookie] = 'value'
        return _Resp('', {}, 200)


@pytest.mark.unittest
class TestClearChallenge:
    def test_posts_every_field_the_page_declared(self):
        session = _FakeSession()
        assert clear_challenge(session, _Resp(CHALLENGE_PAGE, {'content-type': 'text/html'}),
                               honour_delay=False)
        payload = session.posted['json']
        assert payload['challenge_id'] == 'E4Z0rsjI'
        assert payload['challenge_generated'] == '1786519064'
        assert payload['challenge_cookie_expires'] == '1787123864'
        # The server verifies the digest, so the pair must be internally consistent.
        assert hashlib.sha1(
            f"WsgEfDahW7kS0nDrXETimtrgd8A=:{payload['pow_nonce']}".encode()
        ).hexdigest() == payload['pow_hash']
        assert payload['pow_hash'].startswith('00000')

    def test_sends_the_header_the_endpoint_requires(self):
        session = _FakeSession()
        clear_challenge(session, _Resp(CHALLENGE_PAGE, {'content-type': 'text/html'}),
                        honour_delay=False)
        assert session.posted['headers']['X-Verification-Challenge'] == '1'

    def test_reports_failure_when_no_cookie_is_granted(self):
        session = _FakeSession(cookie_after_post=None)
        assert not clear_challenge(session, _Resp(CHALLENGE_PAGE, {'content-type': 'text/html'}),
                                   honour_delay=False)

    def test_page_without_the_constants_is_refused(self):
        session = _FakeSession()
        assert not clear_challenge(session, _Resp('<html>nothing here</html>',
                                                  {'content-type': 'text/html'}),
                                   honour_delay=False)
        assert session.posted is None, 'must not post an answer it cannot compute'
