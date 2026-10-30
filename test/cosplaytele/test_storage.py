import base64
import json
import os
import shutil
import zipfile
from hashlib import md5

import pandas as pd
import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

from inf.cosplaytele import storage as st
from inf.cosplaytele.hosts import Host, base, sorafolder


class FakeResponse:
    def __init__(self, chunks, length=None, status_code=200):
        self._chunks = chunks
        self.headers = {} if length is None else {'Content-Length': str(length)}
        self.status_code = status_code
        self.closed = False

    def iter_content(self, chunk_size=0):
        yield from self._chunks

    def close(self):
        self.closed = True


def _encrypt_like_cryptojs(plain: str, salt: bytes = b'12345678') -> str:
    key = sorafolder._LINK_KEY
    derived = prev = b''
    while len(derived) < 48:
        prev = md5(prev + key + salt).digest()
        derived += prev
    padder = PKCS7(128).padder()
    enc = Cipher(algorithms.AES(derived[:32]), modes.CBC(derived[32:48]), backend=default_backend()).encryptor()
    ct = enc.update(padder.update(plain.encode()) + padder.finalize()) + enc.finalize()
    return base64.b64encode(b'Salted__' + salt + ct).decode()


@pytest.mark.unittest
class TestSorafolder:
    def test_decrypt_roundtrip(self):
        url = 'https://cdn.sorafolder.com/token?x=1'
        assert sorafolder.decrypt_link(_encrypt_like_cryptojs(url)) == url

    def test_page_constant_survives_apostrophe_entity(self):
        page = "const keyEncrypte = 'abc';\nconst fileName = 'YU - Stepmom&#39;s Scent.rar';\n"
        assert sorafolder._page_constant(page, 'fileName') == "YU - Stepmom's Scent.rar"
        assert sorafolder._page_constant(page, 'keyEncrypte') == 'abc'

    def test_page_constant_accepts_double_quotes(self):
        assert sorafolder._page_constant('const fileId = "x1";', 'fileId') == 'x1'

    def test_page_constant_missing_means_removed(self):
        with pytest.raises(base.DownloadError, match='removed'):
            sorafolder._page_constant('<html>gone</html>', 'keyEncrypte')

    def test_safe_local_name_never_keeps_remote_path(self):
        assert base.safe_local_name('../../etc/passwd.rar') == 'package.rar'
        assert base.safe_local_name('Pack.ZIP') == 'package.zip'
        assert base.safe_local_name('no extension') == 'package'
        assert base.safe_local_name('a.r;m -rf') == 'package'
        assert base.safe_local_name('x.sh') == 'package'


@pytest.mark.unittest
class TestStreamToFile:
    def test_complete(self, tmp_path):
        dst = str(tmp_path / 'a.bin')
        n = base.stream_to_file(FakeResponse([b'ab', b'cd'], length=4), dst)
        assert n == 4 and open(dst, 'rb').read() == b'abcd'

    def test_short_body_is_an_error_and_leaves_nothing(self, tmp_path):
        dst = str(tmp_path / 'a.bin')
        resp = FakeResponse([b'ab'], length=10)
        with pytest.raises(base.DownloadError, match='Incomplete'):
            base.stream_to_file(resp, dst)
        assert not os.path.exists(dst) and resp.closed

    def test_stalled_transfer_is_abandoned(self, tmp_path):
        ticks = iter(range(0, 10_000, 10))
        dst = str(tmp_path / 'a.bin')
        with pytest.raises(base.DownloadError, match='Too slow'):
            base.stream_to_file(FakeResponse([b'x'] * 100), dst, clock=lambda: next(ticks))
        assert not os.path.exists(dst)


def _make_zip(path, names):
    with zipfile.ZipFile(path, 'w') as zf:
        for name in names:
            zf.writestr(name, name.encode() * 3)


@pytest.mark.unittest
class TestUnpackAndBuild:
    def test_unpack_counts_files(self, tmp_path):
        _make_zip(tmp_path / 'p.zip', ['Post/Pic/1.jpg', 'Post/Pic/2.jpg', 'Post/Video/a.mp4'])
        out = tmp_path / 'out'
        out.mkdir()
        assert st.unpack_package(str(tmp_path / 'p.zip'), str(out), None) == 3

    def test_empty_or_corrupt_archive_is_rejected(self, tmp_path):
        (tmp_path / 'bad.zip').write_bytes(b'not an archive')
        out = tmp_path / 'out'
        out.mkdir()
        with pytest.raises(st.UnpackError):
            st.unpack_package(str(tmp_path / 'bad.zip'), str(out), None)
        _make_zip(tmp_path / 'empty.zip', [])
        with pytest.raises(st.UnpackError):
            st.unpack_package(str(tmp_path / 'empty.zip'), str(out), None)

    def test_build_package_layout_and_index(self, tmp_path):
        src, up = tmp_path / 'src', tmp_path / 'up'
        (src / 'Post' / 'Pic').mkdir(parents=True)
        for i in range(3):
            (src / 'Post' / 'Pic' / f'{i}.jpg').write_bytes(b'img%d' % i)
        package_file, size, count, members = st.build_package('some-post', str(src), str(up))
        prefix = md5(b'some-post').hexdigest()[:2]
        assert package_file == f'images/{prefix}/some-post.tar'
        assert count == 3 and sorted(members) == [f'Post/Pic/{i}.jpg' for i in range(3)]
        assert os.path.getsize(up / 'images' / prefix / 'some-post.tar') == size
        index = json.load(open(up / 'images' / prefix / 'some-post.json'))
        assert set(index) == {'filesize', 'hash', 'hash_lfs', 'files'} and len(index['files']) == 3


@pytest.mark.unittest
class TestTagUpdates:
    def test_new_and_existing_tags(self):
        origin = {('tag', 'a'): {'tag': 'a', 'name': 'A Long Name', 'type': 'tag', 'category': 'cosplayer'},
                  ('tag', 'b'): {'tag': 'b', 'name': 'B', 'type': 'tag', 'category': None}}
        live = {('tag', 'a'): {'tag': 'a', 'name': 'A', 'type': 'tag', 'category': None, 'count': 4}}
        up = st._tag_updates({'tags': json.dumps(['tag/a', 'tag/b'])}, live, origin)
        assert up[('tag', 'a')]['count'] == 5 and up[('tag', 'a')]['name'] == 'A Long Name'
        assert up[('tag', 'b')]['count'] == 1
        assert live[('tag', 'a')]['count'] == 4 and live[('tag', 'a')]['name'] == 'A'  # live table untouched


class TestFetchPackage:
    def _hosts(self, monkeypatch, *hosts):
        monkeypatch.setattr(st, 'HOSTS', list(hosts))

    @pytest.mark.unittest
    def test_falls_through_to_the_next_host_and_cleans_up(self, tmp_path, monkeypatch):
        seen = []

        def bad(url, out, session):
            seen.append('bad')
            raise base.DownloadError('gone')

        def junk(url, out, session):
            seen.append('junk')
            path = os.path.join(out, 'p.zip')
            open(path, 'wb').write(b'junk')
            return path

        def good(url, out, session):
            seen.append('good')
            path = os.path.join(out, 'p.zip')
            _make_zip(path, ['P/1.jpg'])
            return path

        self._hosts(monkeypatch, Host('a', bad), Host('b', junk), Host('c', good))
        post = {'id': 'x', 'a_download_url': 'u1', 'b_download_url': 'u2', 'c_download_url': 'u3'}
        host, unpack_dir = st.fetch_package(post, str(tmp_path), None)
        assert (host, seen) == ('c', ['bad', 'junk', 'good'])
        assert os.path.exists(os.path.join(unpack_dir, 'P', '1.jpg'))
        assert not os.path.exists(tmp_path / 'download' / 'b' / 'p.zip')

    @pytest.mark.unittest
    def test_hosts_without_a_link_are_skipped_and_none_left_is_an_error(self, tmp_path, monkeypatch):
        self._hosts(monkeypatch, Host('a', lambda *a: pytest.fail('should not be called')))
        with pytest.raises(RuntimeError, match='no supported host link'):
            st.fetch_package({'id': 'x', 'a_download_url': None}, str(tmp_path), None)


OLD_ROW = {'id': 'old-post', 'title': 'Old', 'page_url': 'u', 'thumbnail_url': 't', 'unzip_password': None,
           'tags': json.dumps(['tag/a']), 'sorafolder_download_url': 'https://s/old', 'created_at': 1.0,
           'updated_at': 1.0, 'file_count': 1, 'package_size': 10, 'package_file': 'images/aa/old-post.tar',
           'file_ids': json.dumps([7])}


@pytest.mark.unittest
class TestSync:
    def _run(self, tmp_path, monkeypatch, index_rows, fail_ids=(), **kwargs):
        index_tags = pd.DataFrame([{'tag': 'a', 'name': 'Alpha', 'type': 'tag', 'category': None},
                                   {'tag': 'b', 'name': 'Beta', 'type': 'tag', 'category': 'cosplayer'}])
        tables = {
            ('idx', 'posts.parquet'): pd.DataFrame(index_rows),
            ('idx', 'tags.parquet'): index_tags,
            ('stor', 'posts.parquet'): pd.DataFrame([OLD_ROW]),
            ('stor', 'files.parquet'): pd.DataFrame([{'id': 7, 'filename': '7.jpg', 'mimetype': 'image/jpeg',
                                                      'post_id': 'old-post', 'package_file': 'images/aa/old-post.tar',
                                                      'file_in_package': 'O/1.jpg', 'file_size': 3}]),
            ('stor', 'tags.parquet'): pd.DataFrame([{'tag': 'a', 'name': 'Alpha', 'type': 'tag',
                                                     'category': None, 'count': 1}]),
        }
        uploads = []

        class Client:
            def repo_exists(self, **kw):
                return True

        monkeypatch.setenv('HF_TOKEN', 'x')
        monkeypatch.setattr(st, 'get_hf_client', lambda **kw: Client())
        monkeypatch.setattr(st, 'get_hf_fs', lambda **kw: None)
        monkeypatch.setattr(st, '_read_table', lambda c, f, repo, name: tables.get((repo, name)))
        monkeypatch.setattr(st, 'get_download_session', lambda: None)
        monkeypatch.setattr(st, 'delete_detached_cache', lambda: None)
        monkeypatch.setattr(st, '_squash_if_needed', lambda *a: None)

        def fake_upload(local_directory, message, **kw):
            dest = tmp_path / f'upload{len(uploads)}'
            shutil.copytree(local_directory, dest)
            uploads.append((message, dest))

        monkeypatch.setattr(st, 'safe_upload_directory_as_directory', fake_upload)

        def download(url, out, session):
            if url in fail_ids:
                raise base.DownloadError('removed')
            path = os.path.join(out, 'p.zip')
            _make_zip(path, [f'{url[-3:]}/Pic/1.jpg', f'{url[-3:]}/Pic/2.png'])
            return path

        monkeypatch.setattr(st, 'HOSTS', [Host('sorafolder', download)])
        st.sync('idx', 'stor', hf_upload_interval=1, **kwargs)
        return uploads

    def _rows(self):
        base_row = {'title': 't', 'page_url': 'u', 'thumbnail_url': 't', 'unzip_password': None,
                    'tags': json.dumps(['tag/a', 'tag/b']), 'updated_at': 1.0}
        return [
            OLD_ROW,
            {**base_row, 'id': 'p-new1', 'sorafolder_download_url': 'https://s/n01', 'created_at': 30.0},
            {**base_row, 'id': 'p-new2', 'sorafolder_download_url': 'https://s/n02', 'created_at': 20.0},
            {**base_row, 'id': 'p-gofile-only', 'sorafolder_download_url': None, 'created_at': 25.0},
        ]

    def test_stores_new_posts_newest_first_and_continues_ids(self, tmp_path, monkeypatch):
        uploads = self._run(tmp_path, monkeypatch, self._rows())
        assert [m for m, _ in uploads] == ["Add package 'p-new1', with 2 files", "Add package 'p-new2', with 2 files"]
        final = uploads[-1][1]
        posts = pd.read_parquet(final / 'posts.parquet')
        files = pd.read_parquet(final / 'files.parquet')
        tags = pd.read_parquet(final / 'tags.parquet').set_index('tag')
        assert set(posts['id']) == {'old-post', 'p-new1', 'p-new2'}  # the link-less post is left out
        assert sorted(files['id']) == [7, 8, 9, 10, 11]  # ids continue from the highest one in use
        assert list(posts.sort_values('created_at')['id']) == ['old-post', 'p-new2', 'p-new1']
        assert posts[posts['id'] == 'old-post'].iloc[0]['package_size'] == 10  # old row untouched
        assert tags.loc['a', 'count'] == 3 and tags.loc['b', 'count'] == 2
        assert tags.loc['b', 'category'] == 'cosplayer'
        new1 = posts[posts['id'] == 'p-new1'].iloc[0]
        assert json.loads(new1['file_ids']) == [8, 9] and new1['file_count'] == 2
        # Each commit carries only its own package; the tables always cover everything stored so far.
        first = uploads[0][1]
        assert (first / new1['package_file']).exists() and (first / new1['package_file']).with_suffix('.json').exists()
        assert not (final / new1['package_file']).exists()
        assert (final / 'README.md').read_text().startswith('---\nlicense: other\n---')

    def test_a_failed_post_is_not_stored_and_does_not_block_the_rest(self, tmp_path, monkeypatch):
        uploads = self._run(tmp_path, monkeypatch, self._rows(), fail_ids={'https://s/n01'})
        assert [m for m, _ in uploads] == ["Add package 'p-new2', with 2 files"]
        files = pd.read_parquet(uploads[-1][1] / 'files.parquet')
        assert sorted(files['id']) == [7, 8, 9]  # no ids burnt on the failed post

    def test_a_long_run_of_failures_aborts_instead_of_downloading_everything(self, tmp_path, monkeypatch):
        rows = self._rows()
        base_row = rows[1]
        rows += [{**base_row, 'id': f'p-bad{i}', 'sorafolder_download_url': f'https://s/b{i:02d}',
                  'created_at': 100.0 + i} for i in range(5)]
        bad = {f'https://s/b{i:02d}' for i in range(5)}
        with pytest.raises(RuntimeError, match='3 posts in a row failed'):
            self._run(tmp_path, monkeypatch, rows, fail_ids=bad, max_consecutive_failures=3)

    def test_a_success_resets_the_failure_streak(self, tmp_path, monkeypatch):
        rows = self._rows()
        base_row = rows[1]
        rows += [{**base_row, 'id': 'p-bad-a', 'sorafolder_download_url': 'https://s/ba1', 'created_at': 100.0},
                 {**base_row, 'id': 'p-bad-b', 'sorafolder_download_url': 'https://s/bb1', 'created_at': 10.0}]
        # newest first: bad-a, new1, new2, bad-b -> never more than one failure in a row
        uploads = self._run(tmp_path, monkeypatch, rows, fail_ids={'https://s/ba1', 'https://s/bb1'},
                            max_consecutive_failures=2)
        assert len(uploads) == 2

    def test_max_posts_counts_failed_attempts_too(self, tmp_path, monkeypatch):
        rows = self._rows()
        uploads = self._run(tmp_path, monkeypatch, rows, fail_ids={'https://s/n01'}, max_posts=1)
        assert uploads == []  # the one attempt allowed was spent on the failing post

    def test_max_posts_and_dry_run(self, tmp_path, monkeypatch):
        assert len(self._run(tmp_path, monkeypatch, self._rows(), max_posts=1)) == 1
        assert self._run(tmp_path, monkeypatch, self._rows(), dry_run=True) is not None
