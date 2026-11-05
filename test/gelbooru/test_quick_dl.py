import hashlib
import io
import json
import os
import shutil
import tarfile

import pandas as pd
import pytest
from PIL import Image

from inf.gelbooru import quick_dl as q
from inf.utils.download import DownloadDigestMismatch


def make_image(seed: int, size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', size, (seed % 256, (seed * 7) % 256, (seed * 13) % 256)).save(buf, format='PNG')
    return buf.getvalue()


class HttpError(Exception):
    def __init__(self, status):
        super().__init__(f'HTTP {status}')
        self.response = type('R', (), {'status_code': status})()


class Web:
    """What the image host answers, per url: a list of outcomes consumed in order, the last repeating."""

    def __init__(self):
        self.script = {}
        self.calls = []
        self.referers = set()

    def set(self, url, *outcomes):
        self.script[url] = list(outcomes)

    def next(self, url):
        self.calls.append(url)
        outcomes = self.script.get(url)
        if not outcomes:
            return 'ok'
        return outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]


def row(post_id, host='img2', mimetype='image/png', md5=None, width=8, height=8):
    data = make_image(post_id)
    return {'id': post_id, 'file_url': f'https://{host}.gelbooru.com/images/aa/bb/{post_id}.png',
            'md5': md5 or hashlib.md5(data).hexdigest(), 'mimetype': mimetype, 'width': width, 'height': height,
            'rating': 'general'}


@pytest.fixture
def world(tmp_path, monkeypatch):
    class W:
        pass

    w = W()
    w.web, w.uploads, w.rows, w.hub = Web(), [], [], {}
    w.baseline_ids, w.baseline_bad = [1, 2, 3], [4]
    w.sleeps = []

    def put(repo, name, text_or_json):
        path = tmp_path / 'hub' / repo.replace('/', '_') / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(text_or_json))
        w.hub.setdefault(repo, {})[name] = str(path)

    w.put = put
    put('base/full', 'exist_ids.json', w.baseline_ids)
    put('base/full', 'bad_ids.json', w.baseline_bad)

    class Client:
        def repo_exists(self, **kw):
            return True

        def create_repo(self, **kw):
            pass

        def file_exists(self, repo_id, repo_type, filename):
            return filename in w.hub.get(repo_id, {})

    def fake_download(url, dst, session=None, expected_md5=None, headers=None, timeout=None, **kw):
        w.web.referers.add((headers or {}).get('Referer'))
        outcome = w.web.next(url)
        if isinstance(outcome, int):
            raise HttpError(outcome)
        if outcome == 'html':
            body = b'<!DOCTYPE html><html>hotlink</html>'
        else:
            post_id = int(os.path.splitext(os.path.basename(url))[0])
            body = make_image(post_id)
        with open(dst, 'wb') as f:
            f.write(body)
        actual = hashlib.md5(body).hexdigest()
        if expected_md5 and actual != expected_md5:
            os.remove(dst)
            raise DownloadDigestMismatch(url, expected_md5, actual)
        return len(body)

    def fake_upload(local_directory, repo_id, repo_type, path_in_repo, message, **kw):
        dest = tmp_path / f'commit{len(w.uploads)}'
        shutil.copytree(local_directory, dest)
        w.uploads.append((message, dest))
        for dirpath, _, files in os.walk(dest):
            for name in files:
                rel = os.path.relpath(os.path.join(dirpath, name), dest)
                w.hub.setdefault(repo_id, {})[rel] = os.path.join(dirpath, name)

    monkeypatch.setattr(q, 'get_hf_client', lambda **kw: Client())
    monkeypatch.setattr(q, 'safe_hf_hub_download', lambda c, repo_id, repo_type, filename: w.hub[repo_id][filename])
    monkeypatch.setattr(q, '_iter_index_rows', lambda repo, rev: iter(w.rows))
    monkeypatch.setattr(q, 'download_file', fake_download)
    monkeypatch.setattr(q, 'safe_upload_directory_as_directory', fake_upload)
    monkeypatch.setattr(q, 'configure_hf_http_backend', lambda **kw: None)
    monkeypatch.setattr(q, 'delete_detached_cache', lambda: None)
    monkeypatch.setattr(q.time, 'sleep', lambda s: w.sleeps.append(s))

    def run(**kw):
        args = dict(repository='stage/dl', src_repository='idx/gb', baseline_repository='base/full',
                    download_workers=1, min_free_disk=0, upload_time_span=0.001, initial_rate=1000, max_rate=1000,
                    retry_wait=0.0, cooldown=0.0, diagnose_wait=0.0, min_attempts_for_abort=5)
        args.update(kw)
        q.sync(**args)

    w.run = run
    return w


def check_commit(commit_dir):
    """A commit on its own must hold a whole, consistent state of the staging repository."""
    table = pd.read_parquet(commit_dir / 'table.parquet')
    meta = json.loads((commit_dir / 'meta.json').read_text())
    assert meta['max_volume_id'] >= 1
    members = {}
    for tar_path in (commit_dir / 'images').rglob('*.tar'):
        assert tar_path.with_suffix('.json').exists(), 'a tar without its sidecar'
        with tarfile.open(tar_path) as tar:
            members[os.path.relpath(tar_path, commit_dir).replace(os.sep, '/')] = sorted(tar.getnames())
    return table, meta, members


@pytest.mark.unittest
class TestHelpers:
    @pytest.mark.parametrize('url, expected', [
        ('https://img2.gelbooru.com/images/aa/bb/x.jpg', 'https://img4.gelbooru.com/images/aa/bb/x.jpg'),
        ('https://img3.gelbooru.com/images/aa/bb/x.jpg', 'https://img4.gelbooru.com/images/aa/bb/x.jpg'),
        ('http://img4.gelbooru.com/images/aa/bb/x.jpg', 'https://img4.gelbooru.com/images/aa/bb/x.jpg'),
        ('https://video-cdn.example/x.mp4', 'https://video-cdn.example/x.mp4'),
        ('', ''),
    ])
    def test_every_image_host_becomes_img4(self, url, expected):
        assert q.download_url(url) == expected

    def test_candidates_skip_covered_nonimages_and_rows_without_a_checksum_and_come_newest_first(self):
        rows = [row(5), row(6, mimetype='video/webm'), {**row(7), 'md5': None}, row(8), row(9), row(10)]
        out = q.scan_candidates(iter(rows), covered_ids={9})
        assert [r['id'] for r in out] == [10, 8, 5]
        assert all(r['file_url'].startswith('https://img4.gelbooru.com/') for r in out)

    def test_a_volume_stops_at_the_file_cap_and_at_the_byte_budget(self):
        rows = [{**row(i), 'width': 1000, 'height': 1000} for i in range(1, 30)]
        assert q.take_volume(rows, max_volume_files=5, max_volume_bytes=1 << 40) == 5
        per_post = q.estimate_bytes(rows[0])
        assert q.take_volume(rows, max_volume_files=100, max_volume_bytes=per_post * 3) == 3
        assert q.take_volume(rows, max_volume_files=100, max_volume_bytes=1) == 1  # one post at least

    def test_the_classifier_tells_a_suspicion_from_a_verdict(self):
        assert q.classify_error(HttpError(404)) == 'missing'
        assert q.classify_error(HttpError(410)) == 'missing'
        assert q.classify_error(HttpError(429)) == 'rate_limit'
        assert q.classify_error(HttpError(403)) == 'blocked'
        assert q.classify_error(DownloadDigestMismatch('u', 'a', 'b')) == 'blocked'
        assert q.classify_error(ConnectionError('reset')) == 'network'
        assert q.classify_error(q.UndecodableImage(1, 'u', ValueError('x'))) == 'unusable'


@pytest.mark.unittest
class TestVolumes:
    def test_one_volume_is_one_consistent_commit_and_stores_the_url_actually_used(self, world):
        world.rows = [row(i) for i in range(1, 9)]  # 1-4 are in the baseline
        world.run()
        assert len(world.uploads) == 1
        table, meta, members = check_commit(world.uploads[0][1])
        assert sorted(table['id']) == [5, 6, 7, 8]
        assert set(table['file_url']) == {f'https://img4.gelbooru.com/images/aa/bb/{i}.png' for i in range(5, 9)}
        assert members == {'images/0/001.tar': sorted(f'{i}.png' for i in range(5, 9))}
        assert meta == {'max_volume_id': 1, 'bad_image_ids': []}
        assert (world.uploads[0][1] / 'glob_exist_ids.json').exists()  # the baseline travels with the first commit
        assert json.loads((world.uploads[0][1] / 'glob_exist_ids.json').read_text()) == [1, 2, 3, 4]
        assert world.web.referers == {q.REFERER}

    def test_a_later_run_continues_the_volume_numbering_and_keeps_the_baseline_untouched(self, world):
        world.rows = [row(i) for i in range(5, 9)]
        world.run()
        world.rows = [row(i) for i in range(5, 12)]
        world.run()
        assert len(world.uploads) == 2
        table, meta, members = check_commit(world.uploads[1][1])
        assert sorted(table['id']) == list(range(5, 12))  # the table carries every row, not only the new volume
        assert meta['max_volume_id'] == 2
        assert not (world.uploads[1][1] / 'glob_exist_ids.json').exists()
        assert set(members) == {'images/0/002.tar'}  # only the new volume rides in the commit

    def test_a_small_hard_ceiling_seals_volumes_and_every_commit_stays_consistent(self, world):
        world.rows = [row(i) for i in range(5, 25)]
        world.run(max_volume_hard_bytes=3 * len(make_image(5)))
        assert len(world.uploads) >= 3
        seen = []
        for _, commit in world.uploads:
            table, meta, members = check_commit(commit)
            for tar_name, names in members.items():
                in_table = sorted(table[table['volume_file'] == tar_name]['filename'])
                assert in_table == names
                seen += names
        assert sorted(seen) == sorted(f'{i}.png' for i in range(5, 25))  # nothing lost, nothing twice

    def test_the_file_cap_limits_a_volume(self, world):
        world.rows = [row(i) for i in range(5, 17)]
        world.run(max_volume_files=5)
        sizes = [len(check_commit(c)[2][next(iter(check_commit(c)[2]))]) for _, c in world.uploads]
        assert sizes == [5, 5, 2]


@pytest.mark.unittest
class TestTheHostsAnswers:
    def test_a_404_that_clears_on_a_retry_is_stored_not_blamed(self, world):
        world.rows = [row(i) for i in range(5, 9)]
        world.web.set('https://img4.gelbooru.com/images/aa/bb/6.png', 404, 404, 'ok')
        world.run()
        table, meta, _ = check_commit(world.uploads[0][1])
        assert 6 in set(table['id']) and meta['bad_image_ids'] == []

    def test_a_html_200_is_retried_with_a_new_session_and_never_stored(self, world):
        world.rows = [row(i) for i in range(5, 8)]
        world.web.set('https://img4.gelbooru.com/images/aa/bb/6.png', 'html', 'ok')
        world.run()
        table, meta, members = check_commit(world.uploads[0][1])
        assert sorted(table['id']) == [5, 6, 7]

    def test_a_post_missing_everywhere_while_the_host_serves_is_recorded_bad_after_a_slow_second_look(self, world):
        world.rows = [row(i) for i in range(5, 9)]
        world.web.set('https://img4.gelbooru.com/images/aa/bb/6.png', 404)
        world.run(retries=3, diagnose_attempts=3)
        table, meta, _ = check_commit(world.uploads[0][1])
        assert sorted(table['id']) == [5, 7, 8]
        assert meta['bad_image_ids'] == [6]
        # three attempts in the volume, then three slow looks
        assert world.web.calls.count('https://img4.gelbooru.com/images/aa/bb/6.png') == 6

    def test_a_404_that_clears_on_the_slow_look_is_stored(self, world):
        world.rows = [row(i) for i in range(5, 9)]
        world.web.set('https://img4.gelbooru.com/images/aa/bb/6.png', 404, 404, 404, 'ok')
        world.run(retries=3, diagnose_attempts=3)
        table, meta, _ = check_commit(world.uploads[0][1])
        assert 6 in set(table['id']) and meta['bad_image_ids'] == []

    def test_when_the_control_download_fails_nothing_is_recorded_bad(self, world):
        # post 8 is the newest, so it is downloaded first and becomes the control; it then stops answering
        world.rows = [row(i) for i in range(5, 9)]
        world.web.set('https://img4.gelbooru.com/images/aa/bb/8.png', 'ok', 404)
        world.web.set('https://img4.gelbooru.com/images/aa/bb/6.png', 404)
        world.run(retries=2, diagnose_attempts=2)
        table, meta, _ = check_commit(world.uploads[0][1])
        assert 6 not in set(table['id'])
        assert meta['bad_image_ids'] == []  # the host was not serving, so 6 is not blamed

    def test_a_decoder_refusal_of_intact_bytes_is_permanent(self, world, monkeypatch):
        world.rows = [row(i) for i in range(5, 9)]

        real_open = Image.open

        def fussy_open(fp, *a, **k):
            if str(fp).endswith('7.png'):
                raise OSError('bad chunk crc')
            return real_open(fp, *a, **k)

        monkeypatch.setattr(q.Image, 'open', fussy_open)
        world.run()
        table, meta, _ = check_commit(world.uploads[0][1])
        assert sorted(table['id']) == [5, 6, 8] and meta['bad_image_ids'] == [7]

    def test_a_blocked_host_stops_the_run_before_anything_is_published(self, world):
        world.rows = [row(i) for i in range(5, 15)]
        for i in range(5, 15):
            world.web.set(f'https://img4.gelbooru.com/images/aa/bb/{i}.png', 500)
        world.run(retries=2)
        assert world.uploads == []

    def test_a_few_failures_in_a_big_volume_do_not_stop_it(self, world):
        world.rows = [row(i) for i in range(5, 35)]
        world.web.set('https://img4.gelbooru.com/images/aa/bb/9.png', 500)
        world.run(retries=2, max_failed_ratio=0.3)
        table, _, _ = check_commit(world.uploads[0][1])
        assert 9 not in set(table['id']) and len(table) == 29

    def test_a_dry_run_uploads_nothing(self, world):
        world.rows = [row(i) for i in range(5, 9)]
        world.run(dry_run=True)
        assert world.uploads == []
