import pytest

from inf.cosplaytele import index as ix

POST_HTML = """
<html><body>
<div class="page-title">
  <h1 class="entry-title"> Some Cosplayer cosplay Heroine </h1>
  <span class="posted-on">
    <time class="published" datetime="2026-05-14T08:19:55+00:00"></time>
    <time class="updated" datetime="2026-05-15T01:00:00+00:00"></time>
  </span>
  <div class="entry-category">
    <a href="https://cosplaytele.com/category/anime/">Cosplay Anime/Manga</a>
    <a href="https://cosplaytele.com/tag/some-cosplayer/">Some Cosplayer (some)</a>
    <a href="https://example.com/tag/other/">Elsewhere</a>
  </div>
</div>
<article><div class="entry-content">
  <blockquote><p>Cosplayer : <a href="https://cosplaytele.com/tag/some-cosplayer/">Some Cosplayer</a></p>
  <p>Password : <input value="cosplaytele"></p></blockquote>
  <a href="https://sorafolder.com/d/abc">dl</a>
  <a href="https://gofile.io/d/xyz">dl2</a>
  <a href="https://cosplaytele.com/other-post/">internal</a>
  <div class="gallery"><a href="https://imgur.com/skipme">x</a></div>
</div></article>
</body></html>
"""

LIST_HTML = """
<div id="post-list">
  <div class="post-item"><a href="https://cosplaytele.com/post-one/"><img src="/t1.jpg"></a></div>
  <div class="post-item"><a href="https://cosplaytele.com/post-two/"><img src="/t2.jpg"></a></div>
</div>
"""


class FakeResp:
    def __init__(self, text='', status_code=200, url='https://cosplaytele.com/x/'):
        self.text, self.status_code, self.url = text, status_code, url

    def raise_for_status(self):
        assert self.status_code < 400


@pytest.fixture
def serve(monkeypatch):
    def _serve(resp):
        monkeypatch.setattr(ix, 'srequest', lambda session, method, url, **kw: resp)

    return _serve


@pytest.mark.unittest
class TestCosplayteleIndex:
    def test_category_check(self):
        assert ix._category_check('https://cosplaytele.com/tag/foo/') == ('tag', 'foo')
        assert ix._category_check('https://cosplaytele.com/category/anime/') == ('category', 'anime')
        assert ix._category_check('https://cosplaytele.com/tag/') == (None, None)
        assert ix._category_check('https://example.com/tag/foo/') == (None, None)

    def test_post_page(self, serve):
        serve(FakeResp(POST_HTML))
        info = ix._get_info_from_post_page('https://cosplaytele.com/x/', session=None)
        assert info['title'] == 'Some Cosplayer cosplay Heroine'
        assert info['unzip_password'] == 'cosplaytele'
        assert info['created_at'] < info['updated_at']
        assert [(d['site'], d['url']) for d in info['download_urls']] == [
            ('sorafolder', 'https://sorafolder.com/d/abc'),
            ('gofile', 'https://gofile.io/d/xyz'),
        ]
        tags = {(t['type'], t['tag']): t for t in info['tags']}
        assert set(tags) == {('category', 'anime'), ('tag', 'some-cosplayer')}
        # The header name survives, and the blockquote label supplies the category.
        assert tags[('tag', 'some-cosplayer')]['name'] == 'Some Cosplayer (some)'
        assert tags[('tag', 'some-cosplayer')]['category'] == 'cosplayer'
        assert tags[('category', 'anime')]['category'] is None

    def test_listpage(self, serve):
        serve(FakeResp(LIST_HTML))
        assert list(ix._get_posts_from_listpage(1, session=None)) == [
            {'post_id': 'post-one', 'post_url': 'https://cosplaytele.com/post-one/',
             'thumbnail_url': 'https://cosplaytele.com/t1.jpg'},
            {'post_id': 'post-two', 'post_url': 'https://cosplaytele.com/post-two/',
             'thumbnail_url': 'https://cosplaytele.com/t2.jpg'},
        ]

    def test_listpage_past_end(self, serve):
        serve(FakeResp(status_code=404))
        assert list(ix._get_posts_from_listpage(999, session=None)) == []
