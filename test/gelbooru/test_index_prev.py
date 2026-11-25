import json

import pytest

from inf.gelbooru import index_prev as ip


class Reply:
    """What the post listing answers: a body that may or may not be JSON."""

    def __init__(self, text: str):
        self.text = text

    def json(self):
        return json.loads(self.text)


@pytest.fixture
def answer(monkeypatch):
    """Make the listing answer with ``answer.text``; the requests it received are in ``answer.calls``."""

    class State:
        text = '{}'
        calls = []

    state = State()
    state.calls = []

    def fake_srequest(session, method, url, **kwargs):
        state.calls.append(kwargs.get('params'))
        return Reply(state.text)

    monkeypatch.setattr(ip, 'srequest', fake_srequest)
    return state


@pytest.mark.unittest
class TestGetPostsByPage:
    def test_posts_are_returned(self, answer):
        answer.text = json.dumps({'@attributes': {'limit': 100}, 'post': [{'id': 3}, {'id': 2}]})
        assert ip._get_posts_by_page(p=0, session=object()) == [{'id': 3}, {'id': 2}]

    def test_an_empty_listing_has_no_post_key(self, answer):
        answer.text = json.dumps({'@attributes': {'limit': 100, 'count': 0}})
        assert ip._get_posts_by_page(p=4, session=object()) == []

    def test_a_page_that_is_too_deep_is_empty_instead_of_a_crash(self, answer):
        # the site answers HTTP 200 with this sentence, not JSON, for every page past _MAX_POST_PID
        answer.text = 'Too deep! Pull it back some. Holy fuck.'
        assert ip._get_posts_by_page(p=ip._MAX_POST_PID + 1, session=object()) == []

    def test_any_other_non_json_answer_still_fails_loudly(self, answer):
        # an empty or HTML body means the upstream is blocking us; that must not look like "no new posts"
        answer.text = '<html>blocked</html>'
        with pytest.raises(ValueError):
            ip._get_posts_by_page(p=0, session=object())

    def test_request_shape(self, answer):
        answer.text = json.dumps({'post': []})
        ip._get_posts_by_page(p=7, tags=['id:<100'], session=object(), user_id='u', api_key='k')
        params, = answer.calls
        assert params['pid'] == '7'
        assert params['limit'] == '100'
        assert params['tags'] == 'id:<100'
        assert params['user_id'] == 'u' and params['api_key'] == 'k'


@pytest.mark.unittest
class TestPageDepth:
    def test_the_walk_stays_inside_what_the_site_serves(self):
        # probed on 2026-11-25: pid 10 is the last page served at 100 posts a page
        assert ip._MAX_POST_PID == 10


@pytest.mark.unittest
class TestFlushOnFailure:
    def test_everything_passes_through_and_nothing_is_flushed_on_success(self):
        flushed = []
        assert list(ip._flush_on_failure(iter([1, 2, 3]), lambda: flushed.append(1))) == [1, 2, 3]
        assert flushed == []

    def test_a_failing_walk_is_flushed_once_and_the_error_still_surfaces(self):
        flushed, seen = [], []

        def walk():
            yield 1
            yield 2
            raise RuntimeError('upstream went away')

        with pytest.raises(RuntimeError, match='upstream went away'):
            for item in ip._flush_on_failure(walk(), lambda: flushed.append(list(seen))):
                seen.append(item)

        # the flush ran after both items were consumed, so it publishes both
        assert flushed == [[1, 2]]

    def test_stopping_early_is_not_a_failure(self):
        flushed = []
        gen = ip._flush_on_failure(iter([1, 2, 3]), lambda: flushed.append(1))
        assert next(gen) == 1
        gen.close()
        assert flushed == []
