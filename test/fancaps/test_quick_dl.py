import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from PIL import Image

from inf.fancaps import quick_dl as q


def make_jpeg(seed: int, size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', size, (seed % 256, (seed * 7) % 256, (seed * 13) % 256)).save(buf, format='JPEG')
    return buf.getvalue()


def index_row(image_id: int) -> dict:
    """A row with every column the real index has."""
    return {'id': image_id,
            'url': f'https://fancaps.net/anime/episodeimages.php?{image_id // 10}-slug/ep#{image_id}',
            'thumbnail_url': f'https://cdni.fancaps.net/file/fancaps-animeimages/thumb/{image_id}.jpg',
            'bangumi_id': image_id // 100, 'bangumi_name': f'show {image_id // 100}',
            'bangumi_url': f'https://fancaps.net/anime/showimages.php?{image_id // 100}-show',
            'episode_id': image_id // 10, 'episode_name': f'episode {image_id // 10}',
            'episode_url': f'https://fancaps.net/anime/episodeimages.php?{image_id // 10}-slug'}


def to_batch(rows):
    return pa.RecordBatch.from_pylist(rows)


class Response:
    def __init__(self, status=200, body=b'', headers=None):
        self.status_code = status
        self.content = body
        self.headers = headers if headers is not None else {'content-length': str(len(body))}


class Origin:
    """What the CDN answers, by image id: outcomes consumed in order, the last one repeating."""

    def __init__(self):
        self.script = {}
        self.calls = []
        self.by_session = {}

    def set(self, image_id, *outcomes):
        self.script[image_id] = list(outcomes)

    def answer(self, session_name, url):
        image_id = int(url.split('/')[-1].split('.')[0])
        self.calls.append(image_id)
        self.by_session.setdefault(session_name, []).append(image_id)
        outcomes = self.script.get(image_id)
        outcome = 'ok' if not outcomes else (outcomes.pop(0) if len(outcomes) > 1 else outcomes[0])
        body = make_jpeg(image_id)
        if isinstance(outcome, int):
            return Response(outcome, b'error code: 1015' if outcome == 429 else b'', {'retry-after': '3600'})
        if outcome == 'webp':
            return Response(200, b'RIFF....WEBP', {'content-type': 'image/webp'})
        if outcome == 'empty':
            return Response(200, b'', {'content-length': '0', 'content-type': 'image/jpeg'})
        if outcome == 'truncated':
            return Response(200, body[:-5], {'content-length': str(len(body))})
        if outcome == 'garbage':
            junk = b'\xff\xd8' + b'not really a jpeg'
            return Response(200, junk, {'content-length': str(len(junk))})
        return Response(200, body)


class FakeSession:
    def __init__(self, origin, name, ip):
        self.origin, self.name, self.ip = origin, name, ip

    def get(self, url, timeout=None, **kw):
        if url.startswith(q.EXIT_PROBE_URL):
            return Response(200, json.dumps({'ip': self.ip}).encode())
        return self.origin.answer(self.name, url)

    def close(self):
        pass


@pytest.fixture
def egress():
    class E:
        pass

    e = E()
    e.origin = Origin()
    e.ips = iter(f'10.0.0.{i}' for i in range(1, 200))
    e.forced = []  # addresses handed out first, to simulate two sessions landing on one address
    e.sessions = []

    e.by_url = {}

    def factory(proxy_url, fingerprint):
        # A pinned proxy session id keeps one exit address however many sessions use it.
        if proxy_url is None:
            ip = 'direct'
        else:
            if proxy_url not in e.by_url:
                e.by_url[proxy_url] = e.forced.pop(0) if e.forced else next(e.ips)
            ip = e.by_url[proxy_url]
        session = FakeSession(e.origin, ip, ip)
        e.sessions.append(session)
        return session

    e.factory = factory
    e.probe = lambda session, timeout=30.0: session.ip

    def pool(lanes=2, direct=False, **kw):
        kw.setdefault('rate_per_ip', 100000)
        return q.EgressPool(proxy_url='http://user:pass@proxy.example:1' if lanes else None, lanes=lanes, direct=direct,
                            session_factory=e.factory, probe=e.probe, reopen_timeout=5, **kw)

    e.pool = pool
    return e


@pytest.mark.unittest
class TestJudgeResponse:
    def test_a_complete_jpeg_passes(self):
        body = make_jpeg(1)
        q.judge_response(200, {'Content-Length': str(len(body)), 'Content-Type': 'image/jpeg'}, body)

    @pytest.mark.parametrize('status, kind', [(429, 'blocked'), (404, 'missing'), (410, 'missing'),
                                              (500, 'network'), (502, 'network'), (403, 'network')])
    def test_status_codes_map_to_what_to_do(self, status, kind):
        with pytest.raises(q.FetchError) as ei:
            q.judge_response(status, {}, b'')
        assert ei.value.kind == kind

    def test_a_cloudflare_403_is_a_block_but_a_plain_403_is_not(self):
        with pytest.raises(q.FetchError) as ei:
            q.judge_response(403, {'cf-mitigated': 'challenge'}, b'')
        assert ei.value.kind == 'blocked'
        with pytest.raises(q.FetchError) as ei:
            q.judge_response(403, {}, b'<html>error code: 1015</html>')
        assert ei.value.kind == 'blocked'

    def test_the_retry_after_of_a_block_is_carried(self):
        with pytest.raises(q.FetchError) as ei:
            q.judge_response(429, {'Retry-After': '1800'}, b'')
        assert ei.value.retry_after == 1800.0

    def test_a_200_that_is_not_a_jpeg_is_never_accepted(self):
        with pytest.raises(q.FetchError) as ei:
            q.judge_response(200, {'content-type': 'image/webp'}, b'RIFF1234WEBP')
        assert ei.value.kind == 'bad_body'

    def test_an_empty_200_is_a_bad_body(self):
        with pytest.raises(q.FetchError) as ei:
            q.judge_response(200, {'content-length': '0', 'content-type': 'image/jpeg'}, b'')
        assert ei.value.kind == 'bad_body' and 'empty' in str(ei.value)

    def test_a_truncated_body_is_rejected_by_its_announced_length(self):
        body = make_jpeg(2)
        with pytest.raises(q.FetchError) as ei:
            q.judge_response(200, {'content-length': str(len(body))}, body[:-3])
        assert ei.value.kind == 'bad_body'

    def test_a_polished_re_encode_is_rejected_but_the_original_is_kept(self):
        body = make_jpeg(3)
        with pytest.raises(q.FetchError):
            q.judge_response(200, {'cf-polished': f'ok, orig_size={len(body) * 3}'}, body)
        q.judge_response(200, {'cf-polished': f'ok, orig_size={len(body)}'}, body)

    def test_decode_size_reads_the_dimensions_and_refuses_garbage(self):
        assert q.decode_size(make_jpeg(1, size=(12, 7))) == (12, 7)
        with pytest.raises(Exception):
            q.decode_size(b'\xff\xd8not an image')


@pytest.mark.unittest
class TestHelpers:
    def test_the_url_carries_a_random_cache_buster_and_never_repeats(self):
        a, b = q.image_url(31449534), q.image_url(31449534)
        assert a.startswith('https://cdni.fancaps.net/file/fancaps-animeimages/31449534.jpg?cb=')
        assert a != b
        assert q.image_url(1, 'x').endswith('?cb=x')

    def test_volume_paths_match_the_other_staging_repositories(self):
        assert q._volume_paths(1) == ('images/0/001.tar', 'images/0/001.json')
        assert q._volume_paths(1234) == ('images/1/234.tar', 'images/1/234.json')

    def test_candidates_skip_the_baseline_and_covered_ids_and_come_lowest_id_first(self):
        batches = [to_batch([index_row(i) for i in (9, 8, 7)]), to_batch([index_row(i) for i in (6, 5, 4, 3)])]
        out = q.scan_candidates(iter(batches), baseline=np.array([3, 4, 8], dtype=np.int64), extra_covered={6})
        assert out.column('id').to_pylist() == [5, 7, 9]

    def test_an_id_listed_twice_is_downloaded_once(self):
        batches = [to_batch([index_row(i) for i in (5, 6, 6, 7)]), to_batch([index_row(5)])]
        out = q.scan_candidates(iter(batches), baseline=np.array([], dtype=np.int64), extra_covered=set())
        assert out.column('id').to_pylist() == [5, 6, 7]

    def test_nothing_left_gives_an_empty_table(self):
        out = q.scan_candidates(iter([to_batch([index_row(1)])]), baseline=np.array([1], dtype=np.int64),
                                extra_covered=set())
        assert out.num_rows == 0

    def test_a_volume_stops_at_the_file_cap_and_at_the_byte_budget(self):
        assert q.take_volume(100, max_volume_files=5, max_volume_bytes=1 << 40) == 5
        assert q.take_volume(100, max_volume_files=1000, max_volume_bytes=q.estimate_bytes() * 3) == 3
        assert q.take_volume(100, max_volume_files=1000, max_volume_bytes=1) == 1  # one image at least
        assert q.take_volume(2, max_volume_files=1000, max_volume_bytes=1 << 40) == 2
        assert q.take_volume(0, max_volume_files=1000, max_volume_bytes=1 << 40) == 0

    def test_the_size_scale_stretches_or_shrinks_a_volume(self):
        budget = q.estimate_bytes() * 10
        assert q.take_volume(100, 100, budget) == 10
        assert q.take_volume(100, 100, budget, scale=0.5) == 20
        assert q.take_volume(100, 100, budget, scale=2.0) == 5

    def test_the_pacer_keeps_requests_apart_whoever_sends_them(self):
        pacer = q.IpPacer(rate=50)
        t0 = time.time()
        for _ in range(6):
            pacer.acquire()
        assert time.time() - t0 >= 5 / 50 - 0.005


@pytest.mark.unittest
class TestEgressPool:
    def test_every_lane_gets_its_own_address(self, egress):
        pool = egress.pool(lanes=4).open()
        assert len({x.exit_ip for x in pool.lanes}) == 4

    def test_two_sessions_on_one_address_are_not_both_kept(self, egress):
        egress.forced = ['10.9.9.9', '10.9.9.9']
        pool = egress.pool(lanes=2).open()
        assert len({x.exit_ip for x in pool.lanes}) == 2  # the second lane re-rolled onto another address

    def test_a_lane_keeps_one_session_per_worker_all_on_the_same_address(self, egress):
        pool = egress.pool(lanes=2, sessions_per_lane=3).open()
        for lane in pool.lanes:
            assert lane.sessions.qsize() == 3
            assert len({s.ip for s in list(lane.sessions.queue)}) == 1

    def test_a_session_is_never_in_use_by_two_threads_at_once(self, egress):
        busy, violations, used = set(), [], set()

        class Slow(FakeSession):
            def get(self, url, timeout=None, **kw):
                if url.startswith(q.EXIT_PROBE_URL):
                    return super().get(url, timeout)
                if id(self) in busy:
                    violations.append(id(self))
                busy.add(id(self))
                used.add(id(self))
                time.sleep(0.003)
                busy.discard(id(self))
                return super().get(url, timeout)

        pool = q.EgressPool(proxy_url='http://user:pass@proxy.example:1', lanes=1, rate_per_ip=100000,
                            sessions_per_lane=3, session_factory=lambda url, fp: Slow(egress.origin, 'x', '10.1.1.1'),
                            probe=egress.probe, reopen_timeout=5).open()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=12) as tp:
            list(tp.map(lambda i: pool.lanes[0].fetch(q.image_url(i)), range(60)))
        assert not violations
        assert len(used) <= 3

    def test_the_pacer_is_keyed_by_the_real_address(self, egress):
        pool = egress.pool(lanes=2).open()
        assert pool.pacer('10.0.0.1') is pool.pacer('10.0.0.1')
        assert pool.pacer('10.0.0.1') is not pool.pacer('10.0.0.2')

    def test_a_pool_needs_some_route(self):
        with pytest.raises(ValueError):
            q.EgressPool(proxy_url=None, lanes=0, direct=False)

    def test_the_direct_route_is_one_slot_with_its_own_pacer(self, egress):
        pool = egress.pool(lanes=0, direct=True).open()
        assert [x.exit_ip for x in pool.lanes] == ['direct']

    def test_a_blocked_address_is_retired_and_the_slot_moves_to_a_fresh_one(self, egress):
        pool = egress.pool(lanes=2, max_retired=5).open()
        lane = pool.lanes[0]
        old, generation = lane.exit_ip, lane.generation
        lane.retire(generation, 3600, 'test')
        assert lane.exit_ip != old and not lane.dead
        assert pool._banned[old] > time.time()
        assert not pool.tripped

    def test_a_retirement_seen_by_two_workers_is_handled_once(self, egress):
        pool = egress.pool(lanes=2, max_retired=5).open()
        lane = pool.lanes[0]
        generation = lane.generation
        lane.retire(generation, 3600, 'first')
        lane.retire(generation, 3600, 'second worker, same address')
        assert pool.retired_total == 1

    def test_a_banned_address_is_never_handed_out_again(self, egress):
        pool = egress.pool(lanes=1, max_retired=50).open()
        lane = pool.lanes[0]
        banned = lane.exit_ip
        egress.ips = iter([banned, banned, 'fresh'])
        lane.retire(lane.generation, 3600, 'test')
        assert lane.exit_ip == 'fresh'

    def test_a_session_that_drifts_onto_an_address_in_use_is_re_rolled_without_banning_anyone(self, egress):
        pool = egress.pool(lanes=2, max_retired=5).open()
        a, b = pool.lanes
        old = a.exit_ip
        generation = a.generation
        a.sessions.queue[0].ip = b.exit_ip  # the proxy moved lane a's session onto lane b's address
        pool.verify_exit(a, a.sessions.queue[0], generation)
        assert a.exit_ip not in (old, b.exit_ip) and not a.dead
        assert old not in pool._banned and pool.retired_total == 0  # nobody was limited, nothing counts
        assert not pool.tripped

    def test_a_session_that_drifts_onto_a_free_address_is_followed(self, egress):
        pool = egress.pool(lanes=2).open()
        lane = pool.lanes[0]
        lane.sessions.queue[0].ip = '10.77.77.77'
        pool.verify_exit(lane, lane.sessions.queue[0], lane.generation)
        assert lane.exit_ip == '10.77.77.77' and pool._in_use.get('10.77.77.77') == 1

    def test_the_address_is_re_checked_every_so_many_requests(self, egress):
        pool = egress.pool(lanes=1, reprobe_every=5).open()
        probes = []
        pool._probe = lambda session, timeout=30.0: probes.append(1) or session.ip
        for i in range(20):
            pool.lanes[0].fetch(q.image_url(i))
        assert len(probes) == 4

    def test_a_slot_waits_for_an_address_instead_of_dying_when_none_is_free(self, egress):
        pool = egress.pool(lanes=1, max_retired=50, open_wait=5.0, max_address_tries=2).open()
        lane = pool.lanes[0]
        busy = lane.exit_ip
        egress.ips = iter([busy, busy, busy, busy, 'fresh'])  # two rounds of unusable addresses, then a free one
        pool._probe = lambda session, timeout=30.0: session.ip
        real_sleep = q.time.sleep
        q.time.sleep = lambda s: None
        try:
            lane.retire(lane.generation, 3600, 'test')
        finally:
            q.time.sleep = real_sleep
        assert lane.exit_ip == 'fresh' and not lane.dead

    def test_the_number_of_lanes_is_discovered_from_the_pool_not_configured(self, egress):
        import random
        rng = random.Random(1)
        addresses = [f'10.50.0.{i}' for i in range(30)]  # a pool of 30, however many slots anyone asked for
        egress.by_url = {}

        def factory(url, fp):
            if url not in egress.by_url:
                egress.by_url[url] = rng.choice(addresses)
            return FakeSession(egress.origin, egress.by_url[url], egress.by_url[url])

        pool = q.EgressPool(proxy_url='http://user:pass@proxy.example:1', lanes=0, max_lanes=500, rate_per_ip=100000,
                            session_factory=factory, probe=egress.probe, max_address_tries=6, open_wait=0).open()
        found = len({x.exit_ip for x in pool.lanes})
        assert found == len(pool.lanes)  # never two lanes on one address
        assert 18 <= len(pool.lanes) <= 24  # most of the pool, with a fifth of what was found held back
        assert len(pool.lanes) < 30

    def test_a_bigger_pool_gives_more_lanes_up_to_the_cap_and_a_replaced_pool_needs_nothing(self, egress):
        big = q.EgressPool(proxy_url='http://user:pass@proxy.example:1', lanes=0, max_lanes=20, rate_per_ip=100000,
                           session_factory=egress.factory, probe=egress.probe).open()
        assert len(big.lanes) == 20  # an inexhaustible pool stops at the cap
        egress.ips = iter(f'172.16.9.{i}' for i in range(1, 100))  # the whole set of addresses is swapped
        egress.by_url = {}
        again = q.EgressPool(proxy_url='http://user:pass@proxy.example:1', lanes=0, max_lanes=12,
                             rate_per_ip=100000, session_factory=egress.factory, probe=egress.probe).open()
        assert len(again.lanes) == 12 and all(x.exit_ip.startswith('172.16.9.') for x in again.lanes)

    def test_a_proxy_whose_exit_address_cannot_be_probed_still_works_paced_per_session(self, egress):
        pool = q.EgressPool(proxy_url='http://user:pass@proxy.example:1', lanes=3, rate_per_ip=100000,
                            session_factory=egress.factory, probe=lambda session, timeout=30.0: None,
                            max_address_tries=2).open()
        assert len(pool.lanes) == 3
        assert all(x.exit_ip.startswith('session-') for x in pool.lanes)
        assert len({x.exit_ip for x in pool.lanes}) == 3

    def test_a_proxy_url_that_cannot_carry_a_session_tag_is_used_as_it_is(self, egress):
        pool = q.EgressPool(proxy_url='http://proxy.example:8080', lanes=2, rate_per_ip=100000,
                            session_factory=egress.factory, probe=egress.probe, max_address_tries=2, open_wait=0).open()
        assert len(pool.lanes) >= 1  # one address behind the url, however many slots were asked for

    def test_the_rate_eases_off_after_a_limit_and_earns_itself_back_when_quiet(self, egress):
        pool = egress.pool(lanes=2, max_retired=50, rate_per_ip=2.5, rate_recover_every=0.05).open()
        pool.rate_per_ip = pool.rate_cap  # the fixture starts high; use the real ceiling
        pool.rate_cap = 2.5
        pool.rate_per_ip = 2.5
        lane = pool.lanes[0]
        lane.retire(lane.generation, 3600, 'limited')
        assert pool.rate_per_ip == pytest.approx(2.5 * 0.85)
        again = pool.lanes[1]
        again.retire(again.generation, 3600, 'limited again within a minute')
        assert pool.rate_per_ip == pytest.approx(2.5 * 0.85)  # one cut per minute, not one per limited address
        time.sleep(0.1)
        pool.pacer('10.0.0.1')
        assert pool.rate_per_ip > 2.5 * 0.85 and pool.rate_per_ip <= 2.5

    def test_the_rate_never_falls_below_the_floor_or_recovers_above_the_ceiling(self, egress):
        pool = egress.pool(lanes=2, max_retired=500, rate_per_ip=2.0, rate_recover_every=0.0, min_rate=1.5).open()
        pool.rate_cap, pool.rate_per_ip = 2.0, 2.0
        for _ in range(6):
            pool._last_rate_change = 0.0
            pool._note_block()
        assert pool.rate_per_ip == pytest.approx(1.5)
        for _ in range(40):
            pool._last_block = 0.0
            pool._last_rate_change = 0.0
            pool.pacer('x')
        assert pool.rate_per_ip == pytest.approx(2.0)

    def test_an_address_that_keeps_failing_is_swapped_for_another_without_blame(self, egress):
        pool = egress.pool(lanes=2, max_lane_failures=3, max_retired=50).open()
        lane = pool.lanes[0]
        old, generation = lane.exit_ip, lane.generation
        lane.note(False, generation)
        lane.note(False, generation)
        lane.note(True, generation)  # a success resets the streak
        lane.note(False, generation)
        lane.note(False, generation)
        assert lane.exit_ip == old
        lane.note(False, generation)
        assert lane.exit_ip != old and old not in pool._banned and pool.retired_total == 0

    def test_the_breaker_opens_when_too_many_addresses_go_at_once(self, egress):
        pool = egress.pool(lanes=4, max_retired=2).open()
        pool.lanes[0].retire(pool.lanes[0].generation, 3600, 'a')
        assert not pool.tripped
        pool.lanes[1].retire(pool.lanes[1].generation, 3600, 'b')
        assert pool.tripped

    def test_retirements_outside_the_window_do_not_count(self, egress):
        pool = egress.pool(lanes=4, max_retired=2, retire_window=0.05).open()
        pool.lanes[0].retire(pool.lanes[0].generation, 3600, 'a')
        time.sleep(0.1)
        pool.lanes[1].retire(pool.lanes[1].generation, 3600, 'b')
        assert not pool.tripped

    def test_a_banned_direct_route_cannot_come_back_and_leaves_the_pool_dead(self, egress):
        pool = egress.pool(lanes=0, direct=True).open()
        lane = pool.lanes[0]
        lane.retire(lane.generation, 3600, 'banned')
        assert lane.dead and pool.tripped

    def test_fetching_through_a_dead_lane_raises(self, egress):
        pool = egress.pool(lanes=0, direct=True).open()
        pool.lanes[0].dead = True
        with pytest.raises(q.EgressExhausted):
            pool.lanes[0].fetch(q.image_url(1))


@pytest.fixture
def world(tmp_path, monkeypatch, egress):
    class W:
        pass

    w = W()
    w.egress, w.uploads, w.rows, w.hub = egress, [], [], {}
    w.baseline_ids, w.baseline_bad = [1, 2, 3], [4]

    def put(repo, name, payload):
        path = tmp_path / 'hub' / repo.replace('/', '_') / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        w.hub.setdefault(repo, {})[name] = str(path)

    w.put = put
    put('base/full', 'meta.json', {'exist_ids': w.baseline_ids, 'bad_ids': w.baseline_bad})

    class Client:
        def repo_exists(self, **kw):
            return True

        def create_repo(self, **kw):
            pass

        def file_exists(self, repo_id, repo_type, filename):
            return filename in w.hub.get(repo_id, {})

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
    monkeypatch.setattr(q, '_iter_index_batches', lambda repo, rev: iter([to_batch(w.rows)] if w.rows else []))
    monkeypatch.setattr(q, 'safe_upload_directory_as_directory', fake_upload)
    monkeypatch.setattr(q, 'configure_hf_http_backend', lambda **kw: None)
    monkeypatch.setattr(q, 'delete_detached_cache', lambda: None)

    def run(lanes=2, pool=None, **kw):
        args = dict(repository='stage/dl', src_repository='idx/fc', baseline_repository='base/full',
                    workers_per_lane=2, min_free_disk=0, upload_time_span=0.001, retry_wait=0.0,
                    diagnose_wait=0.0, min_attempts_for_abort=5, pool=pool or egress.pool(lanes=lanes).open())
        args.update(kw)
        q.sync(**args)

    w.run = run
    return w


def check_commit(commit_dir, earlier=()):
    """
    The repository as of one commit must be consistent: the table, the meta and the tars it names agree.

    Uploads are incremental, a commit directory holds only the volume it adds, so the earlier commits'
    directories are merged in to see the whole state.
    """
    table = pd.read_parquet(commit_dir / 'table.parquet')
    meta = json.loads((commit_dir / 'meta.json').read_text())
    assert meta['max_volume_id'] >= 1
    members = {}
    for root in [*earlier, commit_dir]:
        for tar_path in (root / 'images').rglob('*.tar'):
            assert tar_path.with_suffix('.json').exists(), 'a tar without its sidecar'
            with tarfile.open(tar_path) as tar:
                members[os.path.relpath(tar_path, root).replace(os.sep, '/')] = sorted(tar.getnames())
    assert meta['max_volume_id'] == len(members), 'meta and the tars disagree about the volume count'
    assert set(table['volume_file']) <= set(members)
    for _, r in table.iterrows():
        assert r['filename'] in members[r['volume_file']]
    return table, meta, members


@pytest.mark.unittest
class TestSync:
    def test_the_first_commit_carries_the_baseline_and_only_missing_ids_are_downloaded(self, world):
        world.rows = [index_row(i) for i in (10, 9, 8, 3, 2, 4, 1)]
        world.run()
        assert len(world.uploads) == 1
        message, commit = world.uploads[0]
        table, meta, members = check_commit(commit)
        assert sorted(table['id']) == [8, 9, 10]
        assert sorted(world.egress.origin.calls) == [8, 9, 10]  # 1, 2, 3 and the bad 4 were never requested
        assert json.loads((commit / 'glob_exist_ids.json').read_text()) == [1, 2, 3, 4]
        assert 'with 3 images' in message
        assert (commit / 'README.md').exists()

    def test_rows_of_the_table_describe_the_stored_files(self, world):
        world.rows = [index_row(i) for i in (21, 22)]
        world.run()
        table, _, _ = check_commit(world.uploads[0][1])
        row = table[table['id'] == 21].iloc[0]
        assert row['filename'] == '21.jpg' and row['file_ext'] == 'jpg' and row['mimetype'] == 'image/jpeg'
        assert (row['width'], row['height']) == (8, 8)
        assert row['file_size'] == len(make_jpeg(21))
        assert row['md5'] == hashlib.md5(make_jpeg(21)).hexdigest()
        assert row['file_url'] == 'https://cdni.fancaps.net/file/fancaps-animeimages/21.jpg'
        assert row['bangumi_name'] == 'show 0' and row['episode_name'] == 'episode 2'

    def test_every_column_of_the_index_reaches_the_table_with_its_value(self, world):
        world.rows = [index_row(i) for i in (21, 22, 23)]
        world.run()
        table, _, _ = check_commit(world.uploads[0][1])
        index_columns = list(index_row(1))
        assert set(index_columns) <= set(table.columns), set(index_columns) - set(table.columns)
        for source in world.rows:
            stored = table[table['id'] == source['id']].iloc[0]
            for column, value in source.items():
                assert stored[column] == value, column

    def test_a_column_the_index_gains_later_is_carried_without_any_change_here(self, world):
        world.rows = [{**index_row(i), 'air_date': f'2026-11-{i % 28 + 1:02d}', 'score': i / 7} for i in (21, 22)]
        world.run()
        table, _, _ = check_commit(world.uploads[0][1])
        assert {'air_date', 'score'} <= set(table.columns)
        assert table.set_index('id').loc[21, 'air_date'] == '2026-11-22'

    def test_an_index_column_that_clashes_with_a_staging_column_is_refused(self, world):
        world.rows = [{**index_row(21), 'filename': 'clash.jpg'}]
        with pytest.raises(ValueError):
            world.run()
        assert world.uploads == []

    def test_a_table_that_lacks_index_columns_is_not_extended_silently(self, world):
        world.rows = [index_row(i) for i in range(10, 14)]
        world.run(max_volume_files=2, max_volumes=1)
        old = pd.read_parquet(world.uploads[0][1] / 'table.parquet').drop(columns=['episode_url'])
        old.to_parquet(world.uploads[0][1] / 'table.parquet', index=False)
        with pytest.raises(RuntimeError):
            world.run(max_volume_files=2)

    def test_downloads_start_from_the_lowest_id(self, world):
        world.rows = [index_row(i) for i in range(50, 20, -1)]
        world.run(lanes=1, workers_per_lane=1)
        assert world.egress.origin.calls == sorted(world.egress.origin.calls)
        assert world.egress.origin.calls[0] == 21

    def test_volumes_are_split_and_every_commit_is_consistent_and_grows(self, world):
        world.rows = [index_row(i) for i in range(10, 17)]
        world.run(max_volume_files=3)
        assert len(world.uploads) == 3
        sizes = []
        for index, (_, commit) in enumerate(world.uploads):
            table, meta, members = check_commit(commit, earlier=[c for _, c in world.uploads[:index]])
            sizes.append(len(table))
            assert meta['max_volume_id'] == len(sizes)
        assert sizes == [3, 6, 7]
        ids_of_volume = {}
        table, _, _ = check_commit(world.uploads[-1][1], earlier=[c for _, c in world.uploads[:-1]])
        for vol, group in table.groupby('volume_file'):
            ids_of_volume[vol] = (group['id'].min(), group['id'].max())
        ordered = [ids_of_volume[v] for v in sorted(ids_of_volume)]
        assert all(a[1] < b[0] for a, b in zip(ordered, ordered[1:]))  # lower ids in earlier volumes

    def test_a_second_run_continues_from_the_repository_and_downloads_nothing_twice(self, world):
        world.rows = [index_row(i) for i in range(10, 14)]
        world.run(max_volume_files=2, max_volumes=1)
        assert len(world.uploads) == 1
        first_calls = list(world.egress.origin.calls)
        world.run(max_volume_files=2)
        assert len(world.uploads) == 2
        table, meta, _ = check_commit(world.uploads[1][1], earlier=[world.uploads[0][1]])
        assert sorted(table['id']) == [10, 11, 12, 13] and meta['max_volume_id'] == 2
        assert not set(first_calls) & set(world.egress.origin.calls[len(first_calls):])
        assert not (world.uploads[1][1] / 'glob_exist_ids.json').exists()  # the baseline ships once

    def test_nothing_to_do_uploads_nothing(self, world):
        world.rows = [index_row(i) for i in (1, 2, 3)]
        world.run()
        assert world.uploads == []

    def test_the_readme_lists_the_newest_images_first_and_names_the_baseline(self, world):
        world.rows = [index_row(i) for i in range(10, 16)]
        world.run()
        text = (world.uploads[0][1] / 'README.md').read_text()
        assert 'base/full' in text and 'complement' in text
        assert 'volume_file' in text and 'hf_tar_file_download' in text
        first = re.search(r'^\|\s*15\s*\|', text, re.M)
        last = re.search(r'^\|\s*10\s*\|', text, re.M)
        assert first and last and first.start() < last.start()  # highest id first in the sample table
        assert '6 images' in text
        assert 'Every column of the index' in text and '`episode_url`' in text and '`thumbnail_url`' in text
        assert 'everything after that' not in text and 'what that collection lacks' in text
        assert 'GB of original bytes' in text

    def test_a_dry_run_uploads_nothing(self, world):
        world.rows = [index_row(i) for i in (10, 11)]
        world.run(dry_run=True)
        assert world.uploads == []

    def test_a_transient_server_error_is_retried(self, world):
        world.rows = [index_row(10), index_row(11)]
        world.egress.origin.set(10, 500, 502, 'ok')
        world.run()
        table, _, _ = check_commit(world.uploads[0][1])
        assert sorted(table['id']) == [10, 11]

    def test_a_webp_or_truncated_or_garbage_answer_is_retried_and_never_stored(self, world):
        world.rows = [index_row(i) for i in (10, 11, 12)]
        world.egress.origin.set(10, 'webp', 'ok')
        world.egress.origin.set(11, 'truncated', 'ok')
        world.egress.origin.set(12, 'garbage', 'ok')
        world.run()
        table, _, members = check_commit(world.uploads[0][1])
        assert sorted(table['id']) == [10, 11, 12]
        with tarfile.open(next((world.uploads[0][1] / 'images').rglob('*.tar'))) as tar:
            for name in tar.getnames():
                assert Image.open(io.BytesIO(tar.extractfile(name).read())).size == (8, 8)

    def test_a_missing_image_is_recorded_bad_only_after_the_slow_looks(self, world):
        world.rows = [index_row(i) for i in (10, 11, 12)]
        world.egress.origin.set(11, 404)
        world.run(retries=2, diagnose_attempts=2)
        table, meta, _ = check_commit(world.uploads[0][1])
        assert sorted(table['id']) == [10, 12]
        assert meta['bad_image_ids'] == [11]
        assert world.egress.origin.calls.count(11) == 2 + 2  # the attempts, then the slow looks

    def test_an_image_that_turns_up_on_the_second_look_is_stored_not_blacklisted(self, world):
        world.rows = [index_row(i) for i in (10, 11)]
        world.egress.origin.set(11, 404, 404, 'ok')
        world.run(retries=2, diagnose_attempts=2)
        table, meta, _ = check_commit(world.uploads[0][1])
        assert sorted(table['id']) == [10, 11] and meta['bad_image_ids'] == []

    def test_a_bad_id_is_never_asked_for_again(self, world):
        world.rows = [index_row(i) for i in (10, 11, 12, 13)]
        world.egress.origin.set(11, 404)
        world.run(retries=2, diagnose_attempts=1, max_volume_files=2, max_volumes=1)
        world.egress.origin.calls.clear()
        world.run(retries=2, diagnose_attempts=1, max_volume_files=2)
        assert 11 not in world.egress.origin.calls

    def test_a_blocked_address_is_retired_and_the_image_still_arrives_on_another(self, world):
        world.rows = [index_row(i) for i in (10, 11, 12, 13)]
        world.egress.origin.set(11, 429, 'ok')
        pool = world.egress.pool(lanes=2, max_retired=10).open()
        world.run(pool=pool, workers_per_lane=1)
        table, _, _ = check_commit(world.uploads[0][1])
        assert sorted(table['id']) == [10, 11, 12, 13]
        assert pool.retired_total == 1

    def test_when_the_breaker_opens_the_volume_is_not_filled_and_the_run_stops(self, world):
        world.rows = [index_row(i) for i in range(10, 20)]
        for i in range(10, 20):
            world.egress.origin.set(i, 429)
        pool = world.egress.pool(lanes=2, max_retired=2).open()
        world.run(pool=pool, workers_per_lane=1, retries=3)
        assert world.uploads == []
        assert pool.tripped

    def test_a_volume_failing_beyond_the_threshold_is_not_published(self, world):
        world.rows = [index_row(i) for i in range(10, 20)]
        for i in range(10, 20):
            world.egress.origin.set(i, 500)
        world.run(retries=2, max_failed_ratio=0.3)
        assert world.uploads == []

    def test_an_image_that_is_always_an_empty_200_is_recorded_bad_not_failed(self, world):
        world.rows = [index_row(i) for i in (10, 11, 12)]
        world.egress.origin.set(11, 'empty')
        world.run(retries=4)
        table, meta, _ = check_commit(world.uploads[0][1])
        assert sorted(table['id']) == [10, 12] and meta['bad_image_ids'] == [11]

    def test_a_run_of_deleted_ids_is_not_waited_on(self, world):
        world.rows = [index_row(i) for i in range(10, 60)]
        for i in range(20, 50):
            world.egress.origin.set(i, 404)
        started = time.time()
        world.run(retry_wait=5.0, diagnose_wait=0.0, workers_per_lane=2)
        assert time.time() - started < 5.0  # a backoff on each of the 30 would take minutes
        table, meta, _ = check_commit(world.uploads[0][1])
        assert meta['bad_image_ids'] == list(range(20, 50))
        assert sorted(table['id']) == list(range(10, 20)) + list(range(50, 60))

    def test_suspects_are_left_for_the_next_run_when_the_control_download_fails(self, world):
        world.rows = [index_row(i) for i in (10, 11, 12)]
        world.egress.origin.set(10, 'ok', 500)  # downloads once, then the CDN "stops serving"
        world.egress.origin.set(11, 404)
        world.egress.origin.set(12, 404)
        world.run(diagnose_wait=0.0)
        table, meta, _ = check_commit(world.uploads[0][1])
        assert sorted(table['id']) == [10] and meta['bad_image_ids'] == []
        world.egress.origin.script.clear()
        world.run(diagnose_wait=0.0)
        table, _, _ = check_commit(world.uploads[1][1], earlier=[world.uploads[0][1]])
        assert sorted(table['id']) == [10, 11, 12]


@pytest.mark.unittest
def test_sizes_read_in_gb_below_a_terabyte_and_tb_above():
    assert q._size_text(2 * 1024 ** 3) == '2.00 GB'
    assert q._size_text(3 * 1024 ** 4) == '3.000 TB'


@pytest.mark.unittest
class TestVolumeSizing:
    GiB = 1024 ** 3

    def test_default_budgets_plan_a_volume_between_5_and_10_gb_when_there_is_enough(self):
        from click.testing import CliRunner
        defaults = {p.name: p.default for p in q.cli.params}
        count = q.take_volume(10 ** 7, defaults['max_volume_files'], defaults['max_volume_bytes'])
        planned = count * q.estimate_bytes()
        assert 5 * self.GiB <= planned <= 10 * self.GiB
        # leaves room under a decimal 10 GB for tar headers and requests in flight
        assert defaults['max_volume_hard_bytes'] <= 9.7 * 10 ** 9
        assert 5 * 10 ** 9 <= planned
        assert count < defaults['max_volume_files']  # the bytes decide, not the file cap
        assert CliRunner().invoke(q.cli, ['--help']).exit_code == 0

    def test_a_last_volume_that_cannot_be_filled_is_planned_whole(self):
        assert q.take_volume(1234, 100000, 8 * self.GiB) == 1234


@pytest.mark.unittest
class TestInitialSizeScale:
    def test_a_fresh_repository_starts_from_the_builtin_estimate(self):
        assert q.initial_size_scale([]) == 1.0

    def test_a_resumed_run_plans_by_the_newest_stored_sizes(self):
        big = q.estimate_bytes() * 2
        sizes = [q.estimate_bytes() // 4] * 50 + [big] * 30000
        assert q.initial_size_scale(sizes) == pytest.approx(2.0)  # only the newest window counts
        assert q.take_volume(10 ** 7, 10 ** 6, 8 * 1024 ** 3, scale=q.initial_size_scale(sizes)) < \
            q.take_volume(10 ** 7, 10 ** 6, 8 * 1024 ** 3)

    def test_the_scale_is_bounded(self):
        assert q.initial_size_scale([1]) == 0.3
        assert q.initial_size_scale([10 ** 12]) == 3.0


@pytest.mark.unittest
class TestLaneFetch:
    def test_a_slot_that_sat_idle_is_checked_before_the_next_request_goes_out(self, egress):
        pool = egress.pool(lanes=1, reprobe_every=0, idle_recheck=0.05).open()
        lane = pool.lanes[0]
        probes = []
        pool._probe = lambda session, timeout=30.0: probes.append(1) or session.ip
        lane.fetch(q.image_url(1))
        assert probes == []  # busy: no re-check
        time.sleep(0.1)
        lane.fetch(q.image_url(2))
        assert probes == [1]

    def test_an_idle_slot_whose_session_moved_onto_a_used_address_never_sends_on_the_old_identity(self, egress):
        pool = egress.pool(lanes=2, reprobe_every=0, idle_recheck=0.05, max_retired=50).open()
        a, b = pool.lanes
        time.sleep(0.1)
        a.sessions.queue[0].ip = b.exit_ip  # the proxy moved lane a while it was quiet
        before = a.exit_ip
        egress.origin.calls.clear()
        a.fetch(q.image_url(7))
        assert a.exit_ip not in (before, b.exit_ip)
        assert egress.origin.by_session.get(b.exit_ip, []) == []  # nothing was sent on the other lane's address

    def test_a_request_waiting_on_a_drained_session_queue_moves_on_to_the_reopened_slot(self, egress):
        import threading
        from queue import Queue
        pool = egress.pool(lanes=1, reprobe_every=0, idle_recheck=1e9, session_wait=0.05).open()
        lane = pool.lanes[0]
        old = lane.sessions
        while not old.empty():  # every session of the slot is out: a new request has to wait
            old.get_nowait()
        out = []
        worker = threading.Thread(target=lambda: out.append(lane.fetch(q.image_url(3))[0]), daemon=True)
        worker.start()
        time.sleep(0.2)
        assert not out  # it is waiting
        fresh = Queue()
        fresh.put(FakeSession(egress.origin, '10.9.9.9', '10.9.9.9'))
        lane.generation += 1  # the slot was re-opened; the old queue will never be filled again
        lane.sessions, lane.exit_ip = fresh, '10.9.9.9'
        worker.join(timeout=5)
        assert out == [200]

    def test_a_session_returned_after_the_slot_moved_is_closed_not_queued(self, egress):
        closed = []

        class Tracked(FakeSession):
            def close(self):
                closed.append(self.ip)

        pool = q.EgressPool(proxy_url='http://user:pass@proxy.example:1', lanes=1, rate_per_ip=100000,
                            sessions_per_lane=1, probe=egress.probe, reprobe_every=0, idle_recheck=1e9,
                            session_factory=lambda url, fp: Tracked(egress.origin, 'x', '10.2.2.2')).open()
        lane = pool.lanes[0]
        orig = lane.sessions.queue[0]
        real_get = orig.get

        def moving_get(url, timeout=None, **kw):
            lane.generation += 1  # the slot is re-opened while the request is in flight
            return real_get(url, timeout)

        orig.get = moving_get
        assert lane.fetch(q.image_url(1))[0] == 200
        assert closed == ['10.2.2.2']


@pytest.mark.unittest
class TestStallWatchdog:
    def test_it_raises_the_alarm_when_nothing_moves_and_stays_quiet_while_things_do(self):
        alarms, counter = [], [0]
        with q.StallWatchdog(lambda: counter[0], stall_seconds=0.15, interval=0.03, on_stall=alarms.append):
            for _ in range(8):
                counter[0] += 1
                time.sleep(0.04)
            assert alarms == []
            time.sleep(0.4)
        assert len(alarms) >= 1 and alarms[0] >= 0.15

    def test_the_default_alarm_runs_without_error(self):
        q.StallWatchdog._dump(12.0)  # faulthandler writes the stacks to the real stderr; it must not raise


@pytest.mark.unittest
class TestChurn:
    @pytest.mark.timeout(90)
    def test_a_volume_survives_dozens_of_slots_being_retired_while_workers_are_in_flight(self, world):
        world.rows = [index_row(i) for i in range(1000, 1400)]
        for i in range(1000, 1400, 7):
            world.egress.origin.set(i, 429, 'ok')  # each of these retires the slot that asked first
        pool = world.egress.pool(lanes=8, max_retired=10 ** 6, sessions_per_lane=3, reprobe_every=7,
                                 idle_recheck=0.01, session_wait=0.05).open()
        world.run(pool=pool, workers_per_lane=3, max_volume_files=10 ** 6)
        table, _, _ = check_commit(world.uploads[0][1])
        assert sorted(table['id']) == list(range(1000, 1400))  # nothing lost, nothing stuck
        assert pool.retired_total >= 40 and not pool.tripped


@pytest.mark.unittest
class TestVolumeFill:
    def test_the_margin_plans_more_than_the_budget_holds_but_never_less(self):
        budget = q.estimate_bytes() * 10
        assert q.take_volume(10 ** 6, 10 ** 6, budget) == 10
        assert q.take_volume(10 ** 6, 10 ** 6, budget, margin=1.5) == 15
        assert q.take_volume(10 ** 6, 10 ** 6, budget, margin=0.1) == 10  # a margin below one is ignored

    @pytest.mark.parametrize('wrong_by', [10.0, 0.1], ids=['estimate-10x-too-big', 'estimate-10x-too-small'])
    def test_every_volume_but_the_last_fills_its_byte_budget_however_the_size_was_guessed(
            self, world, monkeypatch, wrong_by):
        rows = [index_row(i) for i in range(1000, 1400)]
        world.rows = rows
        real = sum(len(make_jpeg(r['id'])) for r in rows) / len(rows)
        monkeypatch.setattr(q, '_BYTES_PER_IMAGE', int(real * wrong_by))
        budget = int(real * 60)
        world.run(lanes=1, workers_per_lane=1, max_volume_bytes=budget, max_volume_hard_bytes=budget * 100,
                  max_volume_files=10 ** 6, plan_margin=1.2)
        table, _, _ = check_commit(world.uploads[-1][1], earlier=[c for _, c in world.uploads[:-1]])
        assert sorted(table['id']) == list(range(1000, 1400))  # everything arrived, each id once
        volumes = table.groupby('volume_file')['file_size'].sum().sort_index()
        assert len(volumes) >= 4
        for size in list(volumes)[:-1]:
            assert 0.9 * budget <= size <= 1.3 * budget, (size, budget, list(volumes))


@pytest.mark.unittest
class TestArrowState:
    def test_the_baseline_list_is_parsed_without_python_ints_whatever_the_spacing(self, tmp_path):
        for text, expected in (('[3,1,2,2]', [1, 2, 3]), ('[ 5, 4 ,\n 6 ]', [4, 5, 6]), ('[]', []), ('', [])):
            path = tmp_path / 'ids.json'
            path.write_text(text)
            out = q._parse_id_list(str(path))
            assert out.dtype == np.int64 and out.tolist() == expected

    def test_appending_a_volume_keeps_the_table_sorted_and_its_types(self):
        cols = q.table_columns_of(list(index_row(1)))
        first = [{**index_row(i), 'filename': f'{i}.jpg', 'volume_file': 'images/0/001.tar', 'file_size': 10,
                  'mimetype': 'image/jpeg', 'file_ext': 'jpg', 'width': 8, 'height': 8, 'md5': 'x', 'file_url': 'u'}
                 for i in (5, 3, 9)]
        staged = q._append_rows(None, first, cols)
        assert staged.column('id').to_pylist() == [3, 5, 9]
        later = [{**r, 'id': r['id'] - 2, 'volume_file': 'images/0/002.tar'} for r in first]  # ids 3, 1, 7: some older
        out = q._append_rows(staged, later, cols)
        assert out.column('id').to_pylist() == sorted([3, 5, 9, 3, 1, 7])
        assert out.schema.types == staged.schema.types and out.column_names == staged.column_names

    def test_the_resumed_state_is_an_arrow_table_not_python_objects(self, world):
        world.rows = [index_row(i) for i in range(10, 14)]
        world.run(max_volume_files=2, max_volumes=1)
        client = type('C', (), {'file_exists': lambda self, repo_id, repo_type, filename:
                                filename in world.hub.get(repo_id, {})})()
        staged, extra, bad, max_volume, baseline, is_new = q._load_state(client, 'stage/dl', 'base/full',
                                                                         'glob_exist_ids.json')
        assert isinstance(staged, pa.Table) and isinstance(extra, np.ndarray) and isinstance(baseline, np.ndarray)
        assert staged.num_rows == 2 and max_volume == 1 and not is_new
        assert extra.tolist() == sorted(extra.tolist())
