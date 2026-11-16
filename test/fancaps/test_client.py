import pytest

from inf.fancaps import client as c

LIST_HTML = '''<html><body><div>Counting 42</div><div class="single_post_content">
<div class="row"><div class="col"><h4><a href="/anime/showimages.php?5833-B_Gata_H_Kei">B Gata H Kei</a></h4>
</div></div>
<div class="row"><div class="col"><h4><a href="/anime/showimages.php?46849-Bunny_Drop">Bunny Drop</a></h4></div></div>
<div class="row"><div class="col"></div></div></div></body></html>'''

EPISODES_HTML = '''<html><body><div>Counting 42</div><div class="single_post_content">
<div class="post_title"><h2>Bunny</h2></div>
<div class="row"><div id="ad"></div></div>
<div class="row"><div class="col"><a href="/anime/episodeimages.php?46850-Bunny_Drop/Episode_1">
<h3> Images From Episode 1 </h3></a></div>
<div class="col">
<a class="btn" href="https://fancaps.net/anime/episodeimages.php?46850-Bunny_Drop/Episode_1">view</a></div>
</div>
<div class="row"><div class="col"><a href="/x"><h3> Images From Special </h3></a></div>
<div class="col"><a class="btn" href="/anime/episodeimages.php?46853-Bunny_Drop/Special">view</a></div></div>
</div></body></html>'''

IMAGES_HTML = '''<html><body><div>Counting 42</div><div class="single_post_area">
<div class="topImages"><div><a href="https://fancaps.net/anime/picture.php?/999">
<img src="https://ant.fancaps.net/999.jpg"></a></div></div>
<div><a href="https://fancaps.net/anime/picture.php?/101"><img src="https://ant.fancaps.net/101.jpg"></a></div>
<div><a href="https://fancaps.net/anime/picture.php?/102"><img src="/thumb/102.jpg"></a></div>
<div><a href="https://fancaps.net/anime/picture.php?/101"><img src="https://ant.fancaps.net/101.jpg"></a></div>
</div></body></html>'''


def page(text, url='https://fancaps.net/anime/x.php?a=1'):
    return c.Page(text, url)


@pytest.mark.unittest
class TestParsing:
    def test_show_list(self):
        shows = c.parse_show_list(page(LIST_HTML, 'https://fancaps.net/anime/showList.php?b=1'))
        assert shows == [
            c.Show(5833, 'B Gata H Kei', 'https://fancaps.net/anime/showimages.php?5833-B_Gata_H_Kei', 'B_Gata_H_Kei'),
            c.Show(46849, 'Bunny Drop', 'https://fancaps.net/anime/showimages.php?46849-Bunny_Drop', 'Bunny_Drop'),
        ]

    def test_episode_list_skips_ad_rows_and_reads_the_button_link(self):
        episodes = c.parse_episode_list(page(EPISODES_HTML, 'https://fancaps.net/anime/showimages.php?1-x=&page=1'))
        assert episodes == [
            c.Episode(46850, 'Episode 1', 'https://fancaps.net/anime/episodeimages.php?46850-Bunny_Drop/Episode_1'),
            c.Episode(46853, 'Special', 'https://fancaps.net/anime/episodeimages.php?46853-Bunny_Drop/Special'),
        ]

    def test_an_unexpected_row_title_is_an_error_not_a_skip(self):
        broken = EPISODES_HTML.replace('Images From Episode 1', 'Something else')
        with pytest.raises(c.SiteError):
            c.parse_episode_list(page(broken))

    def test_images_exclude_the_featured_block_and_resolve_relative_thumbnails(self):
        images = c.parse_episode_images(page(IMAGES_HTML, 'https://fancaps.net/anime/episodeimages.php?1-x/y&page=2'))
        assert images == [(101, 'https://ant.fancaps.net/101.jpg'), (102, 'https://fancaps.net/thumb/102.jpg'),
                          (101, 'https://ant.fancaps.net/101.jpg')]

    def test_page_urls_are_written_the_way_requests_writes_them(self):
        assert c.with_page('https://x/a.php?5-name', 3) == 'https://x/a.php?5-name&page=3'
        assert c.with_page('https://x/a.php', 1) == 'https://x/a.php?page=1'

    def test_validity_needs_the_counter_the_closing_tag_and_the_layout_marker(self):
        ok = '<html>Counting <b>1</b></html>\n'
        assert c.is_valid(ok)
        assert not c.is_valid(ok.replace('</html>', ''))                 # truncated
        assert not c.is_valid('<html>Just a moment...</html>')           # a challenge page
        assert not c.is_valid(ok, must=('post_title',))
        assert c.is_valid(ok.replace('Counting', 'Counting post_title'), must=('post_title',))


class FakeTransport(c.Transport):
    """Scripted responses; sessions are never built."""

    def __init__(self, script, ladder=('requests', 'cloudscraper', 'curl:safari17_0')):
        super().__init__(ladder=ladder)
        self.script, self.calls = list(script), []

    def request(self, url):
        self.calls.append((self.strategy, url))
        item = self.script.pop(0) if self.script else (200, GOOD, url)
        if isinstance(item, Exception):
            raise item
        return item if len(item) == 3 else (item[0], item[1], url)


GOOD = '<html>Counting <i>post_title single_post_area</i></html>'


@pytest.fixture()
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(c.time, 'sleep', lambda s: sleeps.append(s))
    return sleeps


@pytest.mark.unittest
class TestFetcher:
    def test_a_good_page_is_returned(self, no_sleep):
        f = c.Fetcher(FakeTransport([(200, GOOD)]))
        assert f.get('https://fancaps.net/a', must=('post_title',)).text == GOOD
        assert f.requests == 1 and f.retried == 0

    def test_truncated_empty_and_reset_responses_are_retried_never_read_as_empty_pages(self, no_sleep):
        script = [(200, ''), (200, GOOD.replace('</html>', '')), (500, ''), OSError('reset'), (200, GOOD)]
        f = c.Fetcher(FakeTransport(script))
        assert f.get('https://fancaps.net/a').text == GOOD
        assert f.requests == 5 and f.retried == 4 and len(no_sleep) == 4

    def test_404_and_410_are_permanent(self, no_sleep):
        for code in (404, 410):
            f = c.Fetcher(FakeTransport([(code, '')]))
            with pytest.raises(c.SiteGone):
                f.get('https://fancaps.net/a')
            assert f.requests == 1

    def test_retries_run_out_with_an_error(self, no_sleep):
        f = c.Fetcher(FakeTransport([(500, '')] * 10), retries=3)
        with pytest.raises(c.SiteError, match='failed after 3 attempts'):
            f.get('https://fancaps.net/a')
        assert f.requests == 3

    def test_a_challenge_moves_every_thread_to_the_next_strategy(self, no_sleep):
        t = FakeTransport([(403, 'Just a moment'), (200, GOOD)])
        f = c.Fetcher(t)
        assert f.get('https://fancaps.net/a').text == GOOD
        assert [s for s, _ in t.calls] == ['requests', 'cloudscraper']
        assert t.rotations == 1 and t.strategy == 'cloudscraper'

    def test_two_threads_refused_at_once_move_the_ladder_one_step(self, no_sleep):
        t = FakeTransport([])
        generation = t.generation
        t.rotate(generation)
        t.rotate(generation)              # the second thread saw the same generation
        assert t.rotations == 1

    def test_a_site_that_refuses_every_strategy_is_a_block(self, no_sleep):
        t = FakeTransport([(403, 'Just a moment')] * 100)
        f = c.Fetcher(t, retries=100, max_blocked_rounds=2)
        with pytest.raises(c.SiteBlocked):
            f.get('https://fancaps.net/a')

    def test_ladder_drops_fingerprints_the_build_lacks(self, monkeypatch):
        monkeypatch.setattr(c, 'supported_impersonates', lambda: {'safari17_0'})
        assert c.build_ladder(['curl:safari17_0', 'curl:nope', 'requests']) == ['curl:safari17_0', 'requests']
