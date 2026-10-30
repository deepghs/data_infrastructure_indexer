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
    ctx.gone = {}
    monkeypatch.setattr(st, 'read_gone', lambda c, f, repo: ctx.gone)
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


def package_commits(world):
    return [u for u in world.uploads if u[0].startswith('Add')]


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

    def test_posts_can_be_limited_by_creation_date(self, world):
        world.setup([1, 2, 3, 4, 5, 6], [])
        world.run(created_before=5.0, created_after=2.0)  # created_at is the dbid in these fixtures
        assert sorted(c[1][-1] for c in world.calls) == ['2', '3', '4']

    def test_the_cli_turns_a_date_into_a_utc_timestamp(self):
        from datetime import datetime
        assert st._epoch(datetime(2026, 5, 16)) == 1778889600.0 and st._epoch(None) is None

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
        assert package_commits(world) == []  # nothing to store; the dead links are only recorded

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

    def test_posts_left_without_an_answer_by_a_host_that_is_set_aside_are_deferred_not_failed(self, world, monkeypatch):
        # The incident: gofile answers 429 and is set aside; afterwards every post whose pixeldrain link is dead
        # and whose gofile link was skipped used to count as a fault, and ten in a row ended the run.
        def gofile(url, out):
            raise DownloadError('rate limited', blocked=True)

        def pixeldrain(url, out):
            raise DownloadError('file does not exist', gone=True)

        world.setup(list(range(1, 16)), [])
        world.tables[('idx', 'posts.parquet')] = pd.DataFrame([
            index_row(d, mediafire_download_url=None, download_urls={}, pixeldrain_download_url=f'https://pd/{d}',
                      gofile_download_url=f'https://gf/{d}') for d in range(1, 16)])
        self._hosts(monkeypatch, Host('pixeldrain', pixeldrain), Host('gofile', gofile))
        world.run(max_consecutive_failures=3, max_dead_streak=0)  # raised before the fix
        assert package_commits(world) == []

    def test_a_post_every_asked_host_says_is_gone_is_a_dead_link_when_nothing_was_skipped(self, world, monkeypatch):
        def gone(url, out):
            raise DownloadError('deleted', gone=True)

        world.setup([1, 2, 3], [])
        world.tables[('idx', 'posts.parquet')] = pd.DataFrame([
            index_row(d, pixeldrain_download_url=f'https://pd/{d}') for d in (1, 2, 3)])
        self._hosts(monkeypatch, Host('mediafire', gone), Host('pixeldrain', gone))
        with pytest.raises(st.PackageUnavailable):
            st.fetch_package(index_row(1, pixeldrain_download_url='https://pd/1'), 'unused', set())

    def test_one_fault_among_dead_links_is_still_a_fault(self, world, monkeypatch):
        def gone(url, out):
            raise DownloadError('deleted', gone=True)

        def broken(url, out):
            raise DownloadError('a challenge page')

        self._hosts(monkeypatch, Host('mediafire', broken), Host('pixeldrain', gone))
        with pytest.raises(st.PackageFailed):
            st.fetch_package(index_row(1, pixeldrain_download_url='https://pd/1'), 'unused', set())

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
class TestBatchesDoNotWaitForTheNextStoredPost:
    def test_built_packages_are_committed_during_a_stretch_of_dead_links(self, world, monkeypatch):
        # The incident: 8 packages were built in the first minutes, then 1,800 dead links followed. Nothing was
        # committed until the end of the run, and cancelling the run lost all of it.
        asked = []

        def host(url, out):
            asked.append(url)
            if url.endswith(('/20', '/19', '/18')):  # the newest three are alive, everything older is dead
                return world.good_host('mediafire')(url, out)
            raise DownloadError('deleted', gone=True)

        world.setup(list(range(1, 21)), [])
        monkeypatch.setattr(st, 'HOSTS', [Host('mediafire', host)])
        clock = {'t': 1000.0}

        class FakeTime:  # only the storage module sees it; the zip writer and everything else keep the real clock
            @staticmethod
            def time():
                clock['t'] += 40.0  # every look at the clock is forty seconds later
                return clock['t']

            @staticmethod
            def sleep(seconds):
                pass

        monkeypatch.setattr(st, 'time', FakeTime)
        seen_at_upload = []
        original = st.safe_upload_directory_as_directory

        def recording_upload(**kw):
            seen_at_upload.append(len(asked))
            return original(**kw)

        monkeypatch.setattr(st, 'safe_upload_directory_as_directory', recording_upload)
        st.sync('idx', 'stor', deploy_span=300, max_dead_streak=0)  # a real interval; world.run never lets one expire
        total = len(asked)
        assert total == 20
        # the commit that carries the last two packages went up while the walk was still running, not after it
        carrying = [i for i, (_, path) in enumerate(world.uploads) if 'post-18.tar' in tars(path)]
        assert len(carrying) == 1 and seen_at_upload[carrying[0]] < total

    def test_nothing_is_committed_early_when_nothing_is_pending(self, world, monkeypatch):
        def fault(url, out):
            raise requests.ConnectionError('reset')  # a fault is neither stored nor recorded, so nothing is pending

        world.setup(list(range(1, 6)), [])
        monkeypatch.setattr(st, 'HOSTS', [Host('mediafire', fault)])
        world.run(max_dead_streak=0, max_consecutive_failures=10)
        assert world.uploads == []


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

        # File ids follow the order posts finish in, so only what each post holds has to agree.
        def content(posts, files):
            by_post = {}
            for row in files.to_dict('records'):
                by_post.setdefault(row['post_id'], []).append((row['file_in_package'], row['file_size']))
            return {row['id']: (row['package_file'], row['file_count'], sorted(by_post[row['id']]))
                    for row in posts.to_dict('records')}

        assert content(par_posts, par_files) == content(seq_posts, seq_files)
        assert sorted(par_files['id']) == sorted(seq_files['id'])

    def test_a_slow_post_does_not_hold_back_the_ones_that_are_done(self, world, monkeypatch):
        import threading
        release = threading.Event()

        def host(url, out):
            if url.endswith('/1'):
                release.wait(20)
            return world.good_host('mediafire')(url, out)

        monkeypatch.setattr(st, 'HOSTS', [Host('mediafire', host)])
        monkeypatch.setattr(st, '_WAIT_SLICE', 0.05)
        world.setup([1, 2, 3, 4], [])
        def deploy_while_blocked():
            seen = []
            real_upload = st.safe_upload_directory_as_directory

            def watch(*args, **kwargs):
                seen.append(1)
                release.set()  # the first commit arrives while post 1 is still downloading
                return real_upload(*args, **kwargs)

            monkeypatch.setattr(st, 'safe_upload_directory_as_directory', watch)
            st.sync('idx', 'stor', deploy_span=0, workers=3)
            return seen

        assert deploy_while_blocked()

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


def gone_file(path):
    return json.load(open(path / 'meta' / 'gone.json', encoding='utf-8'))


def gone_entry(host, url, kind='deleted', at=1):
    return {'url': url, 'kind': kind, 'reason': 'deleted', 'at': at}


@pytest.mark.unittest
class TestGoneRecord:
    def _gone_host(self, kind='deleted'):
        def host(url, out):
            raise DownloadError('file removed', gone=True, kind=kind)
        return host

    def _hosts(self, monkeypatch, *hosts):
        monkeypatch.setattr(st, 'HOSTS', list(hosts))

    def test_dead_posts_are_recorded_with_several_fields_per_host(self, world, monkeypatch):
        world.setup([1, 2], [])
        self._hosts(monkeypatch, Host('mediafire', self._gone_host('takedown')))
        world.run(max_dead_streak=0)
        record = gone_file(world.uploads[-1][1])
        assert record['version'] == 1 and sorted(record['posts']) == ['post-1', 'post-2']
        entry = record['posts']['post-2']
        assert entry['dbid'] == 2
        host = entry['hosts']['mediafire']
        assert (host['url'], host['kind']) == ('https://mf/2', 'takedown')
        assert 'removed' in host['reason'] and host['at'] > 0

    def test_a_commit_that_only_records_dead_links_does_not_rewrite_the_tables(self, world, monkeypatch):
        world.setup([1, 2, 3], [1])
        self._hosts(monkeypatch, Host('mediafire', self._gone_host()))
        world.run(max_dead_streak=0)
        assert [m for m, _ in world.uploads] == ['Record 1 post as gone', 'Record 1 post as gone']
        for _, path in world.uploads:
            assert sorted(p.name for p in path.rglob('*') if p.is_file()) == ['gone.json']  # no tables, no tars
        assert sorted(gone_file(world.uploads[-1][1])['posts']) == ['post-2', 'post-3']

    def test_the_next_run_skips_recorded_posts_without_asking_anyone(self, world, monkeypatch):
        world.setup([1, 2, 3], [])
        world.gone.update({f'post-{d}': {'dbid': d, 'hosts': {'mediafire': gone_entry('mediafire', f'https://mf/{d}')}}
                           for d in (1, 2, 3)})
        asked = []
        monkeypatch.setattr(st, 'HOSTS', [Host('mediafire', lambda url, out: asked.append(url))])
        world.run(max_dead_streak=0)
        assert asked == [] and world.uploads == []

    def test_only_the_recorded_host_is_skipped_the_others_are_still_tried(self, world, monkeypatch):
        world.setup([1], [])
        world.tables[('idx', 'posts.parquet')] = pd.DataFrame([index_row(1, pixeldrain_download_url='https://pd/1')])
        world.gone['post-1'] = {'dbid': 1, 'hosts': {'mediafire': gone_entry('mediafire', 'https://mf/1')}}
        asked = []

        def mediafire(url, out):
            asked.append('mediafire')
            raise AssertionError('a host recorded as gone must not be asked')

        self._hosts(monkeypatch, Host('mediafire', mediafire), Host('pixeldrain', world.good_host('pixeldrain')))
        world.run()
        assert asked == [] and [c[0] for c in world.calls] == ['pixeldrain']
        # a commit with nothing new to record leaves the gone file in the repository exactly as it was
        assert not (world.uploads[-1][1] / 'meta').exists()
        assert list(world.gone['post-1']['hosts']) == ['mediafire']

    def test_a_different_link_for_the_same_post_is_tried_again(self, world, monkeypatch):
        world.setup([1], [])
        world.gone['post-1'] = {'dbid': 1, 'hosts': {'mediafire': gone_entry('mediafire', 'https://mf/OLD')}}
        self._hosts(monkeypatch, Host('mediafire', world.good_host('mediafire')))
        world.run()
        assert [c[0] for c in world.calls] == ['mediafire'] and len(world.uploads) == 1

    def test_records_are_never_removed_and_new_ones_are_added(self, world, monkeypatch):
        world.setup([1, 2], [])
        world.gone['old-post'] = {'dbid': 99, 'hosts': {'mega': gone_entry('mega', 'https://mega/x', 'takedown')}}
        self._hosts(monkeypatch, Host('mediafire', self._gone_host()))
        world.run(max_dead_streak=0)
        record = gone_file(world.uploads[-1][1])['posts']
        assert sorted(record) == ['old-post', 'post-1', 'post-2']
        assert record['old-post']['hosts']['mega']['kind'] == 'takedown'

    def test_a_stored_post_still_records_the_dead_host_it_passed_over_in_the_same_commit(self, world, monkeypatch):
        world.setup([1], [])
        world.tables[('idx', 'posts.parquet')] = pd.DataFrame([index_row(1, pixeldrain_download_url='https://pd/1')])
        self._hosts(monkeypatch, Host('mediafire', self._gone_host()),
                    Host('pixeldrain', world.good_host('pixeldrain')))
        world.run()
        message, path = world.uploads[-1]
        assert message == 'Add 1 package, with 2 files, record 1 post as gone'
        posts = pd.read_parquet(path / 'posts.parquet')
        assert list(posts['id']) == ['post-1']  # the package and the verdict about its dead link: one commit
        assert list(gone_file(path)['posts']['post-1']['hosts']) == ['mediafire']

    def test_faults_and_throttling_are_never_recorded_as_dead(self, world, monkeypatch):
        def fault(url, out):
            raise requests.ConnectionError('reset')

        def blocked(url, out):
            raise DownloadError('quota', blocked=True)

        world.setup([1, 2], [])
        self._hosts(monkeypatch, Host('mediafire', fault))
        world.run(max_consecutive_failures=5)
        assert world.uploads == []
        world.setup([1, 2], [])
        self._hosts(monkeypatch, Host('mediafire', blocked))
        world.run(max_consecutive_failures=5, max_dead_streak=0)
        assert world.uploads == []

    def test_dead_links_are_committed_on_schedule_during_a_long_stretch_of_them(self, world, monkeypatch):
        # Not only once every post has been asked: a record that waited for the end would be lost with the run.
        asked = []

        def host(url, out):
            asked.append(url)
            raise DownloadError('deleted', gone=True)

        world.setup(list(range(1, 31)), [])
        monkeypatch.setattr(st, 'HOSTS', [Host('mediafire', host)])
        clock = {'t': 1000.0}

        class FakeTime:
            @staticmethod
            def time():
                clock['t'] += 40.0
                return clock['t']

            @staticmethod
            def sleep(seconds):
                pass

        monkeypatch.setattr(st, 'time', FakeTime)
        seen = []
        original = st.safe_upload_directory_as_directory

        def recording(**kw):
            seen.append(len(asked))
            return original(**kw)

        monkeypatch.setattr(st, 'safe_upload_directory_as_directory', recording)
        st.sync('idx', 'stor', deploy_span=300, max_dead_streak=0)
        assert len(seen) >= 3 and seen[0] < seen[-1] <= 30  # several commits while the walk was still going
        assert len(gone_file(world.uploads[-1][1])['posts']) == 30


@pytest.mark.unittest
class TestEveryCommitIsConsistent:
    """The state at every commit, taken as the dataset's files accumulate, must hold together."""

    def _snapshots(self, world):
        files = {}
        for message, path in world.uploads:
            for p in path.rglob('*'):
                if p.is_file():
                    files[str(p.relative_to(path))] = p
            yield message, dict(files)

    def test_tables_tars_and_the_gone_record_agree_in_every_commit(self, world, monkeypatch):
        def host(url, out):
            if url[-1] in '13579':
                raise DownloadError('deleted', gone=True, kind='takedown')
            return world.good_host('mediafire')(url, out)

        world.setup(list(range(1, 25)), [1, 2])
        monkeypatch.setattr(st, 'HOSTS', [Host('mediafire', host)])
        clock = {'t': 1000.0}

        class FakeTime:
            @staticmethod
            def time():
                clock['t'] += 40.0
                return clock['t']

            @staticmethod
            def sleep(seconds):
                pass

        monkeypatch.setattr(st, 'time', FakeTime)
        st.sync('idx', 'stor', deploy_span=300, max_dead_streak=0, workers=3, max_batch_bytes=1500)
        assert len(world.uploads) >= 3
        previous_gone = set()
        for message, snap in self._snapshots(world):
            if 'posts.parquet' not in snap:
                # a commit that only carries verdicts: the tables of the hub are untouched by it
                assert set(snap) == {'meta/gone.json'}
                continue
            posts = pd.read_parquet(snap['posts.parquet'])
            files = pd.read_parquet(snap['files.parquet'])
            tar_names = {k for k in snap if k.endswith('.tar')}
            # every stored post has its tar and its index in the snapshot, and exactly its file rows
            for row in posts.itertuples():
                if row.id in ('post-1', 'post-2'):
                    continue  # the two that were stored before this run
                assert row.package_file in tar_names and row.package_file[:-4] + '.json' in snap
                ids = json.loads(row.file_ids)
                mine = files[files.post_id == row.id]
                assert sorted(mine['id']) == sorted(ids) and len(ids) == row.file_count
            # no file row without its post, ids stay unique and contiguous
            assert set(files.post_id) <= set(posts['id']) and files['id'].is_unique
            assert sorted(files['id']) == list(range(1, len(files) + 1))
            # the gone record only grows, and a post is never both stored and recorded dead on every host
            record = gone_file(snap['meta/gone.json'].parent.parent)['posts'] if 'meta/gone.json' in snap else {}
            assert previous_gone <= set(record)
            previous_gone = set(record)
            assert not (set(record) & set(posts['id']))
        final = gone_file(world.uploads[-1][1])['posts']
        # every odd post is dead, every even one stored, and the odd ones were recorded as they were met
        assert sorted(int(k.split('-')[1]) for k in final) == [d for d in range(3, 25) if d % 2 == 1]
