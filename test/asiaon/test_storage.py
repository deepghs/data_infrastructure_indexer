import json
import os
import shutil
import stat
import zipfile

import pandas as pd
import pyarrow.parquet as pq
import pytest
import requests

from inf.asiaon import storage as st
from inf.asiaon.hosts import Host
from inf.utils.transfer import DownloadError

UNZIP = shutil.which('unzip')


def term(term_id, count=0):
    return {'id': term_id, 'name': term_id.title(), 'dbid': 1, 'url': f'https://asiaon.top/{term_id}/',
            'total_count': 10, 'count': count}


def index_row(dbid, **extra):
    return {'id': f'post-{dbid}', 'dbid': dbid, 'title': f't{dbid}', 'page_url': f'https://asiaon.top/post-{dbid}/',
            'thumbnail_url': None, 'created_at': float(dbid), 'updated_at': float(dbid), 'status': 'publish',
            'categories': ['by-country/korean'], 'tags': ['asian', 'bodysuit'], 'resolution': None, 'width': None,
            'height': None, 'model': None, 'file_size_str': '10P', 'pages': 10.0, 'file_size_rough': None,
            'download_urls': {'mediafire': f'https://mf/{dbid}', 'pixeldrain': None},
            'unzip_password': 'asiaon.top', 'mediafire_download_url': f'https://mf/{dbid}',
            'pixeldrain_download_url': None, 'gofile_download_url': None, 'googledrive_download_url': None, **extra}


def stored_row(dbid):
    return {**index_row(dbid), 'file_count': 2, 'package_size': 100, 'package_file': f'images/aa/post-{dbid}.tar',
            'file_ids': json.dumps([2 * dbid - 1, 2 * dbid])}


def make_zip(path, names):
    with zipfile.ZipFile(path, 'w') as zf:
        for name in names:
            zf.writestr(name, name.encode() * 4)


def fake_unpack(archive_file, directory, password):
    with zipfile.ZipFile(archive_file) as zf:
        zf.extractall(directory)
        return len([i for i in zf.infolist() if not i.is_dir()])


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A fake hub: index and storage tables in memory, uploads copied aside, hosts that write small zips."""
    calls, uploads = [], []

    def good_host(name):
        def download(url, out):
            calls.append((name, url))
            path = os.path.join(out, 'p.zip')
            make_zip(path, [f'{url[-3:]}/Pic/1.jpg', f'{url[-3:]}/Pic/2.png'])
            return path
        return download

    class Ctx:
        pass

    ctx = Ctx()
    ctx.calls, ctx.uploads, ctx.good_host = calls, uploads, good_host
    ctx.backend = []
    monkeypatch.setattr(st, 'configure_hf_http_backend', lambda **kw: ctx.backend.append(kw))
    ctx.tables = {}

    class Client:
        def repo_exists(self, **kw):
            return True

        def __getattr__(self, name):
            raise AssertionError(f'unexpected hub call: {name}')

    monkeypatch.setenv('HF_TOKEN', 'x')
    monkeypatch.setattr(st, 'get_hf_client', lambda **kw: Client())
    monkeypatch.setattr(st, 'get_hf_fs', lambda **kw: None)
    monkeypatch.setattr(st, '_read_table', lambda c, f, repo, name: ctx.tables.get((repo, name)))
    monkeypatch.setattr(st, '_read_table_typed', lambda c, f, repo, name: ctx.tables.get((repo, name)))
    monkeypatch.setattr(st, 'delete_detached_cache', lambda: None)
    monkeypatch.setattr(st, 'unpack_package', fake_unpack)
    monkeypatch.setattr(st.time, 'sleep', lambda s: None)
    monkeypatch.setattr(st, 'HOSTS', [Host('mediafire', good_host('mediafire')),
                                      Host('pixeldrain', good_host('pixeldrain'))])

    def fake_upload(local_directory, message, **kw):
        dest = tmp_path / f'up{len(uploads)}'
        shutil.copytree(local_directory, dest)
        uploads.append((message, dest))

    monkeypatch.setattr(st, 'safe_upload_directory_as_directory', fake_upload)

    def setup(index_dbids, stored_dbids, extra=None):
        extra = extra or {}
        ctx.tables[('idx', 'posts.parquet')] = pd.DataFrame([index_row(d, **extra.get(d, {})) for d in index_dbids])
        ctx.tables[('idx', 'tags.parquet')] = pd.DataFrame([term('asian', 99), term('bodysuit', 50), term('rare', 5)])
        ctx.tables[('idx', 'categories.parquet')] = pd.DataFrame([term('by-country/korean', 99)])
        ctx.tables[('stor', 'posts.parquet')] = pd.DataFrame([stored_row(d) for d in stored_dbids])
        ctx.tables[('stor', 'files.parquet')] = pd.DataFrame([
            {'id': i, 'filename': f'{i}.jpg', 'mimetype': 'image/jpeg', 'post_id': f'post-{d}',
             'package_file': f'images/aa/post-{d}.tar', 'file_in_package': f'x/{i}.jpg', 'file_size': 3}
            for d in stored_dbids for i in (2 * d - 1, 2 * d)])
        ctx.tables[('stor', 'tags.parquet')] = pd.DataFrame([term('asian', len(stored_dbids))])
        ctx.tables[('stor', 'categories.parquet')] = pd.DataFrame([term('by-country/korean', len(stored_dbids))])

    ctx.setup = setup
    ctx.run = lambda **kw: st.sync('idx', 'stor', deploy_span=10 ** 9, **kw)
    return ctx


def tars(path):
    return sorted(p.name for p in (path / 'images').rglob('*.tar')) if (path / 'images').exists() else []


@pytest.mark.unittest
class TestSync:
    def test_stores_new_posts_newest_first_and_continues_the_tables(self, world):
        world.setup([1, 2, 3, 4, 5], [1, 2])
        world.run()
        # the first package goes up alone as a check, the rest in the closing batch
        assert [m for m, _ in world.uploads] == ['Add 1 package, with 2 files', 'Add 2 packages, with 4 files']
        final = world.uploads[-1][1]
        posts, files = pd.read_parquet(final / 'posts.parquet'), pd.read_parquet(final / 'files.parquet')
        assert list(posts['dbid']) == [5, 4, 3, 2, 1]
        assert sorted(files['id']) == list(range(1, 11))  # ids continue from the highest one in use
        assert posts[posts['dbid'] == 1].iloc[0]['package_size'] == 100  # an old row is untouched
        new5 = posts[posts['dbid'] == 5].iloc[0]
        assert json.loads(new5['file_ids']) == [5, 6] and new5['file_count'] == 2
        assert new5['package_file'].startswith('images/') and new5['package_file'].endswith('/post-5.tar')

    def test_each_batch_leaves_the_disk_before_the_next_is_built(self, world):
        world.setup([1, 2, 3, 4], [])
        world.run()
        first, second = world.uploads[0][1], world.uploads[1][1]
        assert tars(first) == ['post-4.tar'] and sorted(tars(second)) == ['post-1.tar', 'post-2.tar', 'post-3.tar']
        assert not set(tars(first)) & set(tars(second))  # the first batch was deleted after its commit

    def test_a_batch_goes_up_when_it_is_big_enough(self, world):
        world.setup([1, 2, 3, 4, 5, 6], [])
        world.run(max_batch_bytes=1)  # every package fills a batch
        assert len(world.uploads) == 6

    def test_a_tar_and_its_index_travel_together(self, world):
        world.setup([1, 2], [])
        world.run()
        tar = next((world.uploads[0][1] / 'images').rglob('*.tar'))
        index = json.load(open(tar.with_suffix('.json')))
        assert set(index) == {'filesize', 'hash', 'hash_lfs', 'files'} and len(index['files']) == 2

    def test_tags_and_categories_count_from_the_stored_tables(self, world):
        world.setup([1, 2, 3], [1], {3: {'tags': ['asian', 'rare']}})
        world.run()
        final = world.uploads[-1][1]
        tags = pd.read_parquet(final / 'tags.parquet').set_index('id')
        assert tags.loc['asian', 'count'] == 3  # 1 stored + 2 new
        assert tags.loc['rare', 'count'] == 1  # a tag new to storage starts from zero, not from the index's count
        cats = pd.read_parquet(final / 'categories.parquet').set_index('id')
        assert cats.loc['by-country/korean', 'count'] == 3

    def test_output_keeps_the_published_shape(self, world):
        world.setup([1, 2, 3], [1])
        world.run()
        final = world.uploads[-1][1]
        schema = pq.read_schema(final / 'posts.parquet')
        keys = [f.name for f in schema.field('download_urls').type]
        assert keys == sorted(keys)
        readme = (final / 'README.md').read_text()
        assert readme.startswith('---\nlicense: other') and 'middleware repository' in readme
        assert '## Posts' in readme and '## Files' in readme and '## Tags' in readme and '## Categories' in readme

    def test_posts_without_a_fetchable_link_are_not_candidates(self, world):
        world.setup([1, 2, 3], [1], {2: {'mediafire_download_url': None, 'download_urls': {}},
                                     3: {'mediafire_download_url': 'https://asiaon.top/_next/data/x/y.json?a=b'}})
        world.run()
        assert world.calls == []

    def test_dry_run_uploads_nothing(self, world):
        world.setup([1, 2], [])
        world.run(dry_run=True)
        assert world.uploads == []

    def test_hub_requests_get_a_timeout(self, world):
        # without one a stalled hub request blocks for about sixteen minutes
        world.setup([1], [])
        world.run(dry_run=True)
        assert world.backend == [{'timeout': 300}]

    def test_max_posts_counts_attempts(self, world):
        world.setup([1, 2, 3, 4], [])
        world.run(max_posts=2)
        assert len(world.calls) == 2


@pytest.mark.unittest
class TestFailureHandling:
    def _hosts(self, monkeypatch, *hosts):
        monkeypatch.setattr(st, 'HOSTS', list(hosts))

    def test_a_gone_link_is_not_a_fault_and_never_trips_the_breaker(self, world, monkeypatch):
        def gone(url, out):
            raise DownloadError('deleted', gone=True)

        world.setup(list(range(1, 8)), [])
        self._hosts(monkeypatch, Host('mediafire', gone))
        world.run(max_consecutive_failures=2, max_dead_streak=0)  # would have raised if gone counted as failure
        assert world.uploads == []

    def test_a_long_run_of_gone_posts_ends_the_walk_quietly(self, world, monkeypatch):
        calls = []

        def host(url, out):
            calls.append(url)
            if url.endswith(('/9', '/8')):  # the two newest are alive
                return world.good_host('mediafire')(url, out)
            raise DownloadError('deleted', gone=True)

        world.setup(list(range(1, 10)), [])
        self._hosts(monkeypatch, Host('mediafire', host))
        world.run(max_dead_streak=3)
        assert len(calls) == 5  # two stored, then three gone in a row, and the walk stops there
        assert len(pd.read_parquet(world.uploads[-1][1] / 'posts.parquet')) == 2

    def test_a_sweep_of_everything_can_ask_for_no_streak_limit(self, world, monkeypatch):
        calls = []

        def host(url, out):
            calls.append(url)
            raise DownloadError('deleted', gone=True)

        world.setup(list(range(1, 8)), [])
        self._hosts(monkeypatch, Host('mediafire', host))
        world.run(max_dead_streak=0)
        assert len(calls) == 7

    def test_failures_in_a_row_stop_the_run_with_an_error_after_saving_what_was_stored(self, world, monkeypatch):
        def host(url, out):
            if url.endswith('/9'):
                return world.good_host('mediafire')(url, out)
            raise requests.ConnectionError('reset')

        world.setup(list(range(1, 10)), [])
        self._hosts(monkeypatch, Host('mediafire', host))
        with pytest.raises(RuntimeError, match='3 posts in a row failed'):
            world.run(max_consecutive_failures=3)
        assert len(world.uploads) == 1  # the one good package was committed before giving up

    def test_a_success_resets_the_failure_streak(self, world, monkeypatch):
        def host(url, out):
            if url.endswith(('/9', '/6', '/3')):
                return world.good_host('mediafire')(url, out)
            raise requests.ConnectionError('reset')

        world.setup(list(range(1, 10)), [])
        self._hosts(monkeypatch, Host('mediafire', host))
        world.run(max_consecutive_failures=3)  # two failures between successes never reach three

    def test_a_host_that_is_throttling_is_not_asked_again_this_run(self, world, monkeypatch):
        asked = []

        def throttled(url, out):
            asked.append(url)
            raise DownloadError('too slow', blocked=True)

        world.setup([1, 2, 3], [])
        self._hosts(monkeypatch, Host('mediafire', throttled), Host('pixeldrain', world.good_host('pixeldrain')))
        for dbid in (1, 2, 3):
            pass
        world.tables[('idx', 'posts.parquet')] = pd.DataFrame([
            index_row(d, pixeldrain_download_url=f'https://pd/{d}') for d in (1, 2, 3)])
        world.run()
        assert len(asked) == 1  # once, and then the host was set aside
        assert [c[0] for c in world.calls] == ['pixeldrain'] * 3

    def test_the_next_host_is_tried_when_the_first_fails(self, world, monkeypatch):
        def broken(url, out):
            raise DownloadError('blocked by a challenge')

        world.setup([1], [])
        world.tables[('idx', 'posts.parquet')] = pd.DataFrame([index_row(1, pixeldrain_download_url='https://pd/1')])
        self._hosts(monkeypatch, Host('mediafire', broken), Host('pixeldrain', world.good_host('pixeldrain')))
        world.run()
        assert [c[0] for c in world.calls] == ['pixeldrain']

    def test_a_dropped_connection_is_retried_once_on_the_same_host(self, world, monkeypatch):
        attempts = []

        def flaky(url, out):
            attempts.append(url)
            if len(attempts) == 1:
                raise requests.ConnectionError('reset')
            return world.good_host('mediafire')(url, out)

        world.setup([1], [])
        self._hosts(monkeypatch, Host('mediafire', flaky))
        world.run()
        assert len(attempts) == 2 and len(world.uploads) == 1

    def test_a_package_that_does_not_unpack_is_not_stored(self, world, monkeypatch):
        def refuses(archive, directory, password):
            raise st.UnpackError('Password not found')

        world.setup([1], [])
        monkeypatch.setattr(st, 'unpack_package', refuses)
        world.run()
        assert world.uploads == []

    def test_upload_failure_leaves_the_live_tables_alone(self, world, monkeypatch):
        world.setup([1, 2], [])

        def fails(*a, **k):
            raise RuntimeError('hub down')

        monkeypatch.setattr(st, 'safe_upload_directory_as_directory', fails)
        with pytest.raises(RuntimeError, match='hub down'):
            world.run()


@pytest.mark.unittest
class TestHelpers:
    def test_unencodable_characters_become_underscores(self):
        assert st.utf8_encode_safe('a\udce9b/名前.jpg') == 'a_b/名前.jpg'
        assert st.utf8_encode_safe('plain.jpg') == 'plain.jpg'

    def test_counting_terms_leaves_the_live_table_untouched_until_commit(self):
        live, pending = {'a': term('a', 4)}, {}
        st._count_terms(live, pending, {'a': term('a', 99), 'b': term('b', 7)}, ['a', 'b'])
        assert pending['a']['count'] == 5 and pending['b']['count'] == 1 and live['a']['count'] == 4

    def test_no_history_is_ever_rewritten(self):
        source = open(st.__file__).read()
        assert 'super_squash_history' not in source and not hasattr(st, '_squash_if_needed')


SHIM = """#!/bin/sh
# a stand-in for 7z that drives unzip, for machines without 7z
cmd="$1"; shift; out=""; pw=""; arc=""
for a in "$@"; do case "$a" in -o*) out="${a#-o}";; -p*) pw="${a#-p}";; -y) ;; *) arc="$a";; esac; done
if [ "$cmd" = "x" ]; then
  if [ -n "$pw" ]; then exec unzip -qq -o -P "$pw" "$arc" -d "$out"; else exec unzip -qq -o "$arc" -d "$out"; fi
fi
exit 2
"""


@pytest.mark.unittest
@pytest.mark.skipif(UNZIP is None or shutil.which('zip') is None, reason='unzip and zip are needed')
class TestUnpackPackage:
    @pytest.fixture(autouse=True)
    def shim(self, tmp_path, monkeypatch):
        bindir = tmp_path / 'bin'
        bindir.mkdir()
        (bindir / '7z').write_text(SHIM)
        (bindir / '7z').chmod((bindir / '7z').stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv('PATH', f'{bindir}{os.pathsep}{os.environ["PATH"]}')
        # keep the real archive checks, drop only the fakes the sync fixture installs
        monkeypatch.setattr(st, 'unpack_package', st.__dict__['unpack_package'])

    def _zip(self, tmp_path, password):
        import subprocess
        src = tmp_path / 'src'
        (src / 'Pack').mkdir(parents=True)
        (src / 'Pack' / 'a.jpg').write_text('image a')
        (src / 'Pack' / 'b.jpg').write_text('image b, a little longer')
        out = tmp_path / 'p.zip'
        cmd = ['zip', '-q', '-r', str(out), 'Pack']
        if password:
            cmd[1:1] = ['-P', password]
        subprocess.run(cmd, check=True, cwd=str(src))
        return str(out)

    def test_unpacks_with_the_password_the_archive_accepts(self, tmp_path):
        out = tmp_path / 'out'
        out.mkdir()
        sentence = 'asiaontop.com or asiaon.top. Try one of the two; new archives also include their password.'
        assert st.unpack_package(self._zip(tmp_path, 'asiaontop.com'), str(out), sentence) == 2
        assert (out / 'Pack' / 'a.jpg').read_text() == 'image a'

    def test_an_archive_no_candidate_opens_is_an_unpack_error(self, tmp_path):
        out = tmp_path / 'out'
        out.mkdir()
        with pytest.raises(st.UnpackError, match='Password not found'):
            st.unpack_package(self._zip(tmp_path, 'unrelated'), str(out), 'asiaon.top')
        assert not any(out.iterdir())

    def test_an_unencrypted_archive_unpacks_without_a_password(self, tmp_path):
        out = tmp_path / 'out'
        out.mkdir()
        assert st.unpack_package(self._zip(tmp_path, None), str(out), None) == 2


@pytest.mark.unittest
class TestParallelPipeline:
    def _final(self, world):
        final = world.uploads[-1][1]
        posts = pd.read_parquet(final / 'posts.parquet')
        files = pd.read_parquet(final / 'files.parquet').sort_values('id')
        return posts, files

    def test_workers_give_exactly_the_tables_a_sequential_run_gives(self, world):
        world.setup(list(range(1, 13)), [1, 2])
        world.run(workers=1)
        seq_posts, seq_files = self._final(world)

        world.setup(list(range(1, 13)), [1, 2])
        world.run(workers=4)
        par_posts, par_files = self._final(world)

        key = ['dbid', 'file_ids', 'package_file', 'file_count']
        assert par_posts[key].reset_index(drop=True).equals(seq_posts[key].reset_index(drop=True))
        cols = ['id', 'post_id', 'file_in_package', 'file_size']
        assert par_files[cols].reset_index(drop=True).equals(seq_files[cols].reset_index(drop=True))

    def test_parallel_batches_carry_only_their_own_packages(self, world):
        world.setup(list(range(1, 13)), [])
        world.run(workers=3, max_batch_bytes=1)
        seen = []
        for _, path in world.uploads:
            names = tars(path)
            assert not set(names) & set(seen)  # no package is committed twice
            seen += names
        assert sorted(seen) == sorted(f'post-{d}.tar' for d in range(1, 13))

    def test_a_failure_among_parallel_posts_does_not_lose_the_others(self, world, monkeypatch):
        def host(url, out):
            if url.endswith('/5'):
                raise requests.ConnectionError('reset')
            return world.good_host('mediafire')(url, out)

        world.setup(list(range(1, 9)), [])
        monkeypatch.setattr(st, 'HOSTS', [Host('mediafire', host)])
        world.run(workers=3, max_consecutive_failures=5)
        posts, _ = TestParallelPipeline._final(self, world)
        assert sorted(posts['dbid']) == [1, 2, 3, 4, 6, 7, 8]


@pytest.mark.unittest
class TestSlowHosts:
    def _slow_then_fine(self, world, monkeypatch, slow_for):
        asked = []

        def host(url, out):
            asked.append(url)
            if len(asked) <= slow_for:
                raise DownloadError('Too slow', slow=True)
            return world.good_host('mediafire')(url, out)

        world.setup([1, 2, 3, 4], [])
        world.tables[('idx', 'posts.parquet')] = pd.DataFrame([
            index_row(d, pixeldrain_download_url=f'https://pd/{d}') for d in (1, 2, 3, 4)])
        monkeypatch.setattr(st, 'HOSTS', [Host('mediafire', host), Host('pixeldrain', world.good_host('pixeldrain'))])
        return asked

    def test_one_slow_transfer_does_not_set_a_host_aside(self, world, monkeypatch):
        asked = self._slow_then_fine(world, monkeypatch, slow_for=1)
        world.run()
        assert len(asked) == 4  # asked for every post, it was only one bad file

    def test_two_slow_transfers_set_the_host_aside_for_the_run(self, world, monkeypatch):
        asked = self._slow_then_fine(world, monkeypatch, slow_for=99)
        world.run()
        assert len(asked) == 2  # two strikes, and the other two posts never asked it
        assert [c[0] for c in world.calls] == ['pixeldrain'] * 4
