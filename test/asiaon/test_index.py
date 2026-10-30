import shutil

import pandas as pd
import pytest
from requests import HTTPError, Response

from inf.asiaon import client as cl
from inf.asiaon import index as ix

DETAIL = {
    'databaseId': 77, 'title': 'Some Set Vol.1',
    'featuredImage': {'node': {'sourceUrl': 'https://img.example/x.jpg'}},
    'date': '2026-05-14T08:19:55', 'modified': '2026-05-15T01:00:00', 'status': 'publish',
    'categories': {'nodes': [{'uri': '/category/by-country/korean/', 'name': 'Korean', 'databaseId': 5, 'count': 10}]},
    'tags': {'nodes': [{'uri': '/tag/asian/', 'name': 'Asian', 'databaseId': 6, 'count': 99}]},
    'detail': {'resolution': '5000x3337', 'model': 'Rang', 'fileSize': '51P/412.93MB'},
    'download': {'down': (
        '<p>Mediafire: <a href="https://www.mediafire.com/file/abc">dl</a></p>'
        '<p>Pixeldrain (part 1): <a href="https://pixeldrain.com/l/xyz">dl</a></p>'
        '<p>Broken: <a>nothing</a></p>'
        '<p>Password: asiaon.top</p>'
    )},
}

LIST_ITEM = {
    'databaseId': 77, 'title': 'Some Set Vol.1', 'uri': '/some-set-vol-1/', 'status': 'publish',
    'featuredImage': None, 'date': '2026-05-14T08:19:55', 'modified': '2026-05-15T01:00:00',
    'categories': {'nodes': [{'uri': '/category/by-country/korean/', 'name': 'Korean', 'databaseId': 5, 'count': 10}]},
}


def _client(monkeypatch, payload):
    c = cl.AOClient(session=type('S', (), {'headers': {}})())
    c.build_id = 'b1'
    monkeypatch.setattr(c, '_get_json', lambda url, params: payload)
    return c


@pytest.mark.unittest
class TestClientParsing:
    def test_detail_record(self, monkeypatch):
        c = _client(monkeypatch, {'pageProps': {'__TEMPLATE_QUERY_DATA__': {'post': DETAIL}}})
        r = c.get_info('some-set-vol-1')
        assert (r['id'], r['dbid'], r['status']) == ('some-set-vol-1', 77, 'publish')
        assert (r['width'], r['height'], r['resolution']) == (5000, 3337, '5000x3337')
        assert (r['pages'], r['file_size_rough'], r['model']) == (51, 412930000.0, 'Rang')  # decimal MB
        assert r['categories'][0] == {'id': 'by-country/korean', 'name': 'Korean', 'dbid': 5,
                                      'url': 'https://asiaon.top/category/by-country/korean/', 'total_count': 10}
        assert r['tags'][0]['id'] == 'asian'
        assert r['unzip_password'] == 'asiaon.top'
        assert r['thumbnail_url'] == 'https://img.example/x.jpg'

    def test_download_links_are_keyed_by_host_prefix_and_linkless_entries_skipped(self, monkeypatch):
        c = _client(monkeypatch, {'pageProps': {'__TEMPLATE_QUERY_DATA__': {'post': DETAIL}}})
        urls = c.get_info('some-set-vol-1')['download_urls']
        assert urls == {'mediafire': 'https://www.mediafire.com/file/abc', 'pixeldrain': 'https://pixeldrain.com/l/xyz'}

    def test_unparseable_resolution_and_size_become_none(self, monkeypatch):
        post = {**DETAIL, 'detail': {'resolution': 'big', 'model': None, 'fileSize': 'lots'},
                'download': {'down': None}}
        r = _client(monkeypatch, {'pageProps': {'__TEMPLATE_QUERY_DATA__': {'post': post}}}).get_info('x')
        assert (r['width'], r['height'], r['pages'], r['file_size_rough']) == (None, None, None, None)
        assert r['download_urls'] == {} and r['unzip_password'] is None

    def test_list_page(self, monkeypatch):
        c = _client(monkeypatch, {'pageProps': {'data': {'posts': {'nodes': [LIST_ITEM]}}}})
        (r,) = c.get_lst_page(2)
        assert (r['id'], r['dbid'], r['thumbnail_url']) == ('some-set-vol-1', 77, None)
        assert r['categories'][0]['id'] == 'by-country/korean'

    def test_empty_list_page_means_the_end(self, monkeypatch):
        c = _client(monkeypatch, {'pageProps': {'data': {'posts': {'nodes': []}}}})
        assert c.get_lst_page(999) == []

    def test_redirecting_first_candidate_falls_through_to_the_next(self, monkeypatch):
        c = cl.AOClient(session=type('S', (), {'headers': {}})())
        c.build_id = 'b1'
        answers = iter([{'pageProps': {'__N_REDIRECT': '/page/1'}},
                        {'pageProps': {'data': {'posts': {'nodes': [LIST_ITEM]}}}}])
        monkeypatch.setattr(c, '_get_json', lambda url, params: next(answers))
        assert len(c.get_lst_page(1)) == 1


@pytest.mark.unittest
class TestParseSize:
    @pytest.mark.parametrize('text, expected', [
        ('412.93MB', 412930000.0), ('51M', 51000000.0), ('425Mb', 425000000.0), ('1.5GB', 1500000000.0),
        ('1,234MB', 1234000000.0), ('439727MBB', 439727000000.0), (' 90 MB ', 90000000.0), ('50', 50.0),
    ])
    def test_decimal_units_as_published(self, text, expected):
        assert cl.parse_size(text) == expected

    @pytest.mark.parametrize('text', ['897.97MB/', '42.69MB`', '46.19MBz', '642.45MB+9', 'lots', '', '12XB'])
    def test_anything_but_a_size_is_none(self, text):
        assert cl.parse_size(text) is None


@pytest.mark.unittest
class TestTermCounting:
    def test_counts_and_updates_total(self):
        terms = {'a': {'id': 'a', 'name': 'A', 'dbid': 1, 'url': 'u', 'total_count': 5, 'count': 3}}
        ids = ix._count_terms(terms, [{'id': 'a', 'name': 'A', 'dbid': 1, 'url': 'u', 'total_count': 9},
                                      {'id': 'b', 'name': 'B', 'dbid': 2, 'url': 'u', 'total_count': 1}])
        assert ids == ['a', 'b']
        assert (terms['a']['count'], terms['a']['total_count']) == (4, 9) and terms['b']['count'] == 1


def _row(dbid, **extra):
    return {'id': f'post-{dbid}', 'dbid': dbid, 'title': f't{dbid}', 'page_url': f'https://asiaon.top/post-{dbid}/',
            'thumbnail_url': None, 'created_at': float(dbid), 'updated_at': float(dbid), 'status': 'publish',
            'categories': ['by-country/korean'], 'tags': ['asian'], 'resolution': None, 'width': None,
            'height': None, 'model': None, 'file_size_str': None, 'pages': None, 'file_size_rough': None,
            'download_urls': {'mediafire': f'https://mf/{dbid}', 'pixeldrain': None},
            'mediafire_download_url': f'https://mf/{dbid}', 'pixeldrain_download_url': None,
            'unzip_password': 'asiaon.top', **extra}


TERM = {'name': 'N', 'dbid': 1, 'url': 'u', 'total_count': 1, 'count': 1}


class FakeClient:
    """Listing newest first, sixteen per page in the real site and two here."""

    def __init__(self, dbids, gone=()):
        self.dbids, self.gone, self.pages_asked = sorted(dbids, reverse=True), set(gone), []

    def init(self):
        pass

    def get_lst_page(self, page_id):
        self.pages_asked.append(page_id)
        chunk = self.dbids[(page_id - 1) * 2:page_id * 2]
        return [{'dbid': d, 'id': f'post-{d}', 'page_url': f'https://asiaon.top/post-{d}/'} for d in chunk]

    def get_info(self, id_):
        dbid = int(id_.split('-')[1])
        if dbid in self.gone:
            resp = Response()
            resp.status_code = 404
            raise HTTPError(response=resp)
        urls = {'mediafire': f'https://mf/{dbid}', 'vikingfile': f'https://vk/{dbid}'}
        return {
            **_row(dbid, title=f'new{dbid}'),
            'categories': [{**TERM, 'id': 'by-country/korean', 'name': 'Korean'}],
            'tags': [{**TERM, 'id': 'asian', 'name': 'Asian'}],
            'download_urls': urls,
        }


@pytest.mark.unittest
class TestSync:
    def _run(self, tmp_path, monkeypatch, stored, listed, gone=(), **kwargs):
        tables = {
            'posts.parquet': pd.DataFrame([_row(d) for d in stored]),
            'tags.parquet': pd.DataFrame([{**TERM, 'id': 'asian', 'name': 'Asian', 'count': len(stored)}]),
            'categories.parquet': pd.DataFrame([{**TERM, 'id': 'by-country/korean', 'name': 'Korean',
                                                 'count': len(stored)}]),
        }
        uploads, fake = [], FakeClient(listed, gone)

        class HfClient:
            def repo_exists(self, **kw):
                return True

            def __getattr__(self, name):
                raise AssertionError(f'unexpected hub call: {name}')

        monkeypatch.setenv('HF_TOKEN', 'x')
        monkeypatch.setattr(ix, 'get_hf_client', lambda **kw: HfClient())
        monkeypatch.setattr(ix, 'get_hf_fs', lambda **kw: None)
        monkeypatch.setattr(ix, '_read_table', lambda c, f, repo, name: tables.get(name))
        monkeypatch.setattr(ix, 'delete_detached_cache', lambda: None)
        monkeypatch.setattr(ix, 'AOClient', lambda: fake)

        def fake_upload(local_directory, message, **kw):
            dest = tmp_path / f'up{len(uploads)}'
            shutil.copytree(local_directory, dest)
            uploads.append((message, dest))

        monkeypatch.setattr(ix, 'safe_upload_directory_as_directory', fake_upload)
        ix.sync('repo', deploy_span=10 ** 9, **kwargs)
        return uploads, fake

    def test_adds_only_unseen_posts_keeps_old_rows_and_stops_after_the_empty_stretch(self, tmp_path, monkeypatch):
        stored = list(range(1, 21))
        uploads, fake = self._run(tmp_path, monkeypatch, stored, stored + [21, 22, 23], max_empty_pages=3)
        # the first post is deployed at once, the rest in the final commit
        assert [m for m, _ in uploads] == ['Add 1 post', 'Add 2 posts']
        posts = pd.read_parquet(uploads[-1][1] / 'posts.parquet')
        assert len(posts) == 23 and list(posts['dbid'][:3]) == [23, 22, 21]
        assert posts[posts['dbid'] == 5].iloc[0]['title'] == 't5'  # untouched
        assert posts[posts['dbid'] == 22].iloc[0]['vikingfile_download_url'] == 'https://vk/22'
        tags = pd.read_parquet(uploads[-1][1] / 'tags.parquet')
        assert tags.iloc[0]['count'] == 23 and tags.iloc[0]['total_count'] == 1
        assert fake.pages_asked == [1, 2, 3, 4, 5]  # two pages with new posts, then three empty ones
        assert (uploads[-1][1] / 'README.md').read_text().startswith('---\nlicense: other')

    def test_download_urls_struct_keeps_alphabetical_field_order(self, tmp_path, monkeypatch):
        import pyarrow.parquet as pq
        stored = [1, 2]
        uploads, _ = self._run(tmp_path, monkeypatch, stored, stored + [3], max_empty_pages=2)
        fields = [f.name for f in pq.read_schema(uploads[-1][1] / 'posts.parquet').field('download_urls').type]
        assert fields == sorted(fields) == ['mediafire', 'pixeldrain', 'vikingfile']

    def test_nothing_new_uploads_nothing(self, tmp_path, monkeypatch):
        stored = list(range(1, 11))
        uploads, _ = self._run(tmp_path, monkeypatch, stored, stored, max_empty_pages=2)
        assert uploads == []

    def test_the_end_of_the_listing_stops_a_full_walk(self, tmp_path, monkeypatch):
        uploads, fake = self._run(tmp_path, monkeypatch, [1, 2], [1, 2, 3, 4, 5], max_empty_pages=0)
        assert len(pd.read_parquet(uploads[-1][1] / 'posts.parquet')) == 5
        assert fake.pages_asked[-1] == 4  # page 4 came back empty

    def test_a_post_that_404s_is_skipped(self, tmp_path, monkeypatch):
        uploads, _ = self._run(tmp_path, monkeypatch, [1, 2], [1, 2, 3, 4], gone={3}, max_empty_pages=2)
        assert sorted(pd.read_parquet(uploads[-1][1] / 'posts.parquet')['dbid']) == [1, 2, 4]

    def test_max_posts_bounds_a_trial(self, tmp_path, monkeypatch):
        uploads, _ = self._run(tmp_path, monkeypatch, [1, 2], [1, 2, 3, 4, 5, 6], max_posts=2)
        assert len(pd.read_parquet(uploads[-1][1] / 'posts.parquet')) == 4

    def test_dry_run_uploads_nothing(self, tmp_path, monkeypatch):
        uploads, _ = self._run(tmp_path, monkeypatch, [1, 2], [1, 2, 3], dry_run=True)
        assert uploads == []


@pytest.mark.unittest
class TestStartPage:
    def _run(self, tmp_path, monkeypatch, **kwargs):
        stored = list(range(1, 41))
        # newest first: dbids 61..51 unstored, then 40..1 stored (the gap sits above the stored stretch)
        listed = list(range(41, 61)) + stored
        return TestSync()._run(tmp_path, monkeypatch, stored, listed, **kwargs)

    def test_a_run_from_page_one_that_meets_stored_pages_never_reaches_older_gaps(self, tmp_path, monkeypatch):
        # Pages 1-10 hold 41..60 and are new; the gap below it is walked only because the walk goes on
        # until it has seen max_empty_pages empty ones.
        uploads, fake = self._run(tmp_path, monkeypatch, max_empty_pages=3)
        assert len(pd.read_parquet(uploads[-1][1] / 'posts.parquet')) == 60 and fake.pages_asked[0] == 1

    def test_start_page_skips_ahead_to_where_the_last_run_stopped(self, tmp_path, monkeypatch):
        uploads, fake = self._run(tmp_path, monkeypatch, max_empty_pages=3, start_page=6)
        assert fake.pages_asked[0] == 6
        posts = pd.read_parquet(uploads[-1][1] / 'posts.parquet')
        assert len(posts) == 50  # pages 6-10 hold the ten oldest unstored posts, 41..50


@pytest.mark.unittest
def test_a_run_cut_short_says_where_to_resume(tmp_path, monkeypatch, caplog):
    import logging as std_logging
    caplog.set_level(std_logging.WARNING)
    seen = []
    monkeypatch.setattr(ix.logging, 'warning', lambda msg, *a, **k: seen.append(msg))
    TestSync()._run(tmp_path, monkeypatch, [1, 2], [1, 2, 3, 4, 5, 6, 7, 8], max_empty_pages=2, max_posts=2)
    assert any('--start-page' in m for m in seen)
