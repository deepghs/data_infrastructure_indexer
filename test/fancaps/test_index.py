import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from inf.fancaps import index as idx
from inf.fancaps import table as t
from inf.fancaps.client import Episode, SiteBlocked
from .fakes import FakeHub, FakeSite, fail, gone

REPO = 'idx/fc'


def make_site(shows=2, episodes=3, images=4):
    """``shows`` shows of ``episodes`` episodes, ``images`` images each, ids unique and rising."""
    structure, imgs, next_image, next_ep = {}, {}, 1000, 100
    for sid in range(1, shows + 1):
        eps = []
        for _ in range(episodes):
            imgs[next_ep] = list(range(next_image, next_image + images))
            next_image += images
            eps.append(next_ep)
            next_ep += 1
        structure[sid] = {'name': f'Show {sid}', 'episodes': eps}
    return FakeSite(structure, imgs)


def add_episode(site, sid, position, images=4):
    ep = max(e for s in site.shows.values() for e in s['episodes']) + 1
    first = max(i for v in site.images.values() for i in v) + 1
    site.images[ep] = list(range(first, first + images))
    site.shows[sid]['episodes'].insert(position, ep)
    return ep


def check_hub(hub, repo=REPO, site=None):
    """The invariants every published snapshot must satisfy, read from the files the hub holds right now."""
    files = hub.files[repo]
    meta = json.load(open(files['meta.json']))
    state = meta['fancaps_state']
    if 'fancaps.parquet' not in files:
        assert meta['exist_episode_ids'] == [] and meta['exist_bangumi_ids'] == []
        return None
    scan = t.scan_table(files['fancaps.parquet'])
    t.check_table(files['fancaps.parquet'], scan.rows)
    assert sorted(scan.episodes) == meta['exist_episode_ids']
    assert sorted({i.bangumi_id for i in scan.episodes.values()}) == meta['exist_bangumi_ids']
    bangumi = json.load(open(files['bangumi.json']))
    assert sorted(e['id'] for b in bangumi for e in b['episodes']) == meta['exist_episode_ids']
    assert [b['id'] for b in bangumi] == sorted(b['id'] for b in bangumi)
    readme = open(files['README.md']).read()
    assert f'{scan.rows:,}' in readme and f'{len(scan.episodes):,}' in readme
    listed = {e for s in state['shows'].values() for e in s['episodes']}
    pending = {int(k) for k in state['pending']}
    assert pending <= listed and not pending & set(scan.episodes)
    assert not {int(k) for k in state['failed']} & set(scan.episodes)
    assert all(len(v) == 3 for v in state['pending'].values())
    if site is not None:
        assert scan.rows == sum(len(site.images.get(e, [])) for e in scan.episodes)
    return scan


@pytest.fixture()
def world(tmp_path, monkeypatch):
    class W:
        pass

    w = W()
    w.hub = FakeHub(tmp_path)
    w.created = []
    w.exists = True

    class Client:
        def repo_exists(self, **kw):
            return w.exists

        def create_repo(self, **kw):
            w.created.append(kw['repo_id'])
            w.exists = True

        def file_exists(self, repo_id, repo_type, filename):
            return filename in w.hub.files.get(repo_id, {})

    monkeypatch.setattr(idx, 'get_hf_client', lambda **kw: Client())
    monkeypatch.setattr(idx, 'safe_hf_hub_download',
                        lambda c, repo_id, repo_type, filename: w.hub.files[repo_id][filename])
    monkeypatch.setattr(idx, 'safe_upload_directory_as_directory', w.hub.upload)
    monkeypatch.setattr(idx, 'configure_hf_http_backend', lambda **kw: None)
    monkeypatch.setattr(idx, 'delete_detached_cache', lambda: None)
    monkeypatch.setattr(idx, 'render_figures', lambda stats, directory: [])

    def run(site, **kw):
        args = dict(repository=REPO, workers=2, deploy_span=3600.0, max_time_limit=None, client=site)
        args.update(kw)
        idx.sync(**args)

    w.run = run
    w.check = lambda site=None: check_hub(w.hub, site=site)
    return w


@pytest.mark.unittest
class TestFirstRun:
    def test_everything_is_published_consistently(self, world):
        site = make_site()
        world.run(site)
        scan = world.check(site)
        assert scan.rows == 24 and sorted(scan.episodes) == list(range(100, 106))
        messages = [m for m, _ in world.hub.commits]
        assert messages[0] == 'Update the episode listing state'
        assert messages[-1] == 'Add 24 new images, with 6 new episodes'
        ids = pq.read_table(world.hub.files[REPO]['fancaps.parquet'])['id'].to_pylist()
        assert ids == sorted(ids, reverse=True) and len(set(ids)) == 24

    def test_thumbnail_and_urls_come_out_as_the_earlier_job_wrote_them(self, world):
        site = make_site(1, 1, 2)
        site.thumb = 'https://animethumbs.fancaps.net'
        world.run(site)
        row = pq.read_table(world.hub.files[REPO]['fancaps.parquet']).to_pylist()[0]
        assert row == {
            'id': 1001, 'url': 'https://cdni.fancaps.net/file/fancaps-animeimages/1001.jpg',
            'thumbnail_url': 'https://animethumbs.fancaps.net/1001.jpg', 'bangumi_id': 1, 'bangumi_name': 'Show 1',
            'bangumi_url': 'https://fancaps.net/anime/showimages.php?1-Show 1', 'episode_id': 100,
            'episode_name': 'Episode 100', 'episode_url': 'https://fancaps.net/anime/episodeimages.php?100-S/E100'}

    def test_a_second_run_finds_nothing_and_commits_nothing(self, world):
        site = make_site()
        world.run(site)
        commits = len(world.hub.commits)
        site.requests = {'list': 0, 'episode': 0}
        world.run(site)
        assert len(world.hub.commits) == commits and site.requests['episode'] == 0

    def test_a_new_repository_gets_created_and_json_goes_to_lfs(self, world, monkeypatch):
        written = {}

        class Fs:
            def read_text(self, path):
                return '*.bin filter=lfs'

            def write_text(self, path, text):
                written[path] = text

        monkeypatch.setattr(idx, 'get_hf_fs', lambda: Fs())
        world.exists = False
        world.run(make_site(1, 1, 1))
        assert world.created == [REPO]
        assert 'filter=lfs diff=lfs merge=lfs -text' in written[f'datasets/{REPO}/.gitattributes']
        assert world.check().rows == 1


@pytest.mark.unittest
class TestIncremental:
    def test_a_late_upload_in_the_middle_is_found_and_merged_in_order(self, world):
        site = make_site()
        world.run(site)
        new = add_episode(site, 1, 1, images=5)
        world.run(site)
        scan = world.check(site)
        assert new in scan.episodes and scan.rows == 29
        assert world.hub.commits[-1][0] == 'Add 5 new images, with 1 new episode'
        state = world.hub.meta()['fancaps_state']
        assert state['shows']['1']['episodes'] == site.shows[1]['episodes']

    def test_a_new_show_is_walked_and_read(self, world):
        site = make_site(1, 2, 3)
        world.run(site)
        site.shows[9] = {'name': 'Show 9', 'episodes': [900, 901]}
        site.images.update({900: [5000, 5001], 901: [5002]})
        world.run(site)
        assert {i.bangumi_id for i in world.check(site).episodes.values()} == {1, 9}

    def test_when_only_the_state_changed_only_meta_json_is_committed(self, world):
        site = make_site()
        world.run(site)
        add_episode(site, 2, 0, images=2)
        site.broken[max(e for s in site.shows.values() for e in s['episodes'])] = fail(0)
        world.run(site)
        message, directory = world.hub.commits[-1]
        assert message == 'Update the episode listing state'
        assert os.listdir(directory) == ['meta.json']
        world.check()

    def test_an_episode_listed_only_by_a_sibling_page_is_read_after_a_crash_without_titles(self, world):
        site = make_site(1, 2, 3)
        world.run(site)
        ep = add_episode(site, 1, 0, images=3)
        world.run(site, max_episodes=0)                       # listing state committed, nothing read
        meta = world.hub.meta()
        assert str(ep) in meta['fancaps_state']['pending']
        world.run(site)
        assert ep in world.check(site).episodes


@pytest.mark.unittest
class TestFailures:
    def test_a_failed_episode_is_never_recorded_as_done_and_is_retried(self, world):
        site = make_site()
        site.broken[103] = fail(103)
        world.run(site)
        scan = world.check(site)
        assert 103 not in scan.episodes and len(scan.episodes) == 5
        assert world.hub.meta()['fancaps_state']['failed']['103']['kind'] == 'error'
        assert str(103) in world.hub.meta()['fancaps_state']['pending']
        del site.broken[103]
        world.run(site)
        assert 103 in world.check(site).episodes
        assert world.hub.meta()['fancaps_state']['failed'] == {}

    def test_a_gone_episode_is_not_retried_until_the_next_complete_walk(self, world):
        site = make_site()
        site.broken[103] = gone(103)
        world.run(site)
        assert world.hub.meta()['fancaps_state']['failed']['103']['kind'] == 'gone'
        site.requests['episode'] = 0
        world.run(site)
        assert site.requests['episode'] == 0
        world.run(site, full_walk=True)
        assert site.requests['episode'] == 1

    def test_an_episode_failing_several_runs_in_a_row_waits_for_the_complete_walk(self, world):
        site = make_site(1, 2, 3)
        site.broken[100] = fail(100)
        for _ in range(3):
            world.run(site, max_attempts=3)
        assert world.hub.meta()['fancaps_state']['failed']['100']['attempts'] == 3
        site.requests['episode'] = 0
        world.run(site, max_attempts=3)
        assert site.requests['episode'] == 0
        world.run(site, max_attempts=3, full_walk=True)
        assert site.requests['episode'] == 1

    def test_an_episode_with_no_image_is_not_done_and_is_looked_at_again(self, world):
        site = make_site(1, 2, 3)
        site.images[100] = []
        world.run(site)
        assert 100 not in world.check(site).episodes
        assert world.hub.meta()['fancaps_state']['failed']['100']['kind'] == 'empty'
        site.images[100] = [7, 8, 9]
        world.run(site)
        assert 100 in world.check(site).episodes

    def test_an_episode_whose_images_are_all_known_elsewhere_is_set_aside(self, world):
        site = make_site(1, 2, 3)
        site.images[101] = list(site.images[100])
        world.run(site)
        scan = world.check()
        assert (100 in scan.episodes) != (101 in scan.episodes)
        other = 101 if 100 in scan.episodes else 100
        assert world.hub.meta()['fancaps_state']['failed'][str(other)]['kind'] == 'duplicate'

    def test_a_block_aborts_the_run_but_keeps_what_was_read(self, world):
        site = make_site(1, 4, 3)
        site.broken[103] = SiteBlocked('refused')
        with pytest.raises(SiteBlocked):
            world.run(site, workers=1)
        scan = world.check()
        assert scan is not None and 103 not in scan.episodes and {100, 101, 102} <= set(scan.episodes)


@pytest.mark.unittest
class TestBootstrapAndBudget:
    def seed(self, world, ids_by_episode, ghosts=()):
        tables = [t.episode_rows(1, 'Show 1', 'https://fancaps.net/anime/showimages.php?1-Show 1',
                                 Episode(ep, f'Episode {ep}',
                                         f'https://fancaps.net/anime/episodeimages.php?{ep}-S/E{ep}'),
                                 [(i, f'https://ant.fancaps.net/{i}.jpg') for i in ids])
                  for ep, ids in ids_by_episode.items()]
        df = pd.concat([x.to_pandas() for x in tables]).sort_values('id', ascending=False)
        path = world.hub.tmp / 'seed.parquet'
        df.to_parquet(path, index=False)
        meta = world.hub.tmp / 'seed-meta.json'
        done = sorted(list(ids_by_episode) + list(ghosts))
        meta.write_text(json.dumps({'exist_bangumi_ids': [], 'exist_episode_ids': done}))
        world.hub.put(REPO, 'fancaps.parquet', path)
        world.hub.put(REPO, 'meta.json', meta)

    def test_the_published_table_is_continued_and_phantom_ids_are_dropped(self, world):
        site = make_site(1, 3, 4)                                  # episodes 100, 101, 102
        self.seed(world, {100: site.images[100], 101: site.images[101]}, ghosts=[102])
        world.run(site)
        scan = world.check(site)
        assert sorted(scan.episodes) == [100, 101, 102] and scan.rows == 12
        meta = world.hub.meta()
        assert meta['exist_episode_ids'] == [100, 101, 102] and meta['exist_bangumi_ids'] == [1]

    def test_deploy_span_makes_several_commits_and_every_one_is_consistent(self, world):
        site = make_site(2, 3, 4)
        world.hub.on_commit = lambda: world.check()
        world.run(site, deploy_span=0.0, workers=1)
        assert len([m for m, _ in world.hub.commits if m.startswith('Add')]) > 1
        assert world.check(site).rows == 24

    def test_max_episodes_stops_early_and_the_next_run_finishes(self, world):
        site = make_site(2, 3, 4)
        world.run(site, max_episodes=2, workers=1)
        assert len(world.check(site).episodes) == 2
        world.run(site)
        assert len(world.check(site).episodes) == 6

    def test_an_expired_time_limit_schedules_nothing(self, world):
        site = make_site()
        world.run(site, max_time_limit=1e-9)
        assert site.requests['episode'] == 0

    def test_dry_run_uploads_nothing_and_writes_the_files_it_would_commit(self, world, tmp_path):
        site = make_site()
        out = tmp_path / 'out'
        world.run(site, dry_run=True, output_dir=str(out))
        assert world.hub.commits == []
        assert {'fancaps.parquet', 'meta.json', 'bangumi.json', 'README.md'} <= set(os.listdir(out))
        assert pq.ParquetFile(out / 'fancaps.parquet').metadata.num_rows == 24


@pytest.mark.unittest
class TestBangumiJson:
    def test_layout_and_natural_order_match_the_earlier_job(self):
        info = t.EpisodeInfo
        data = idx.build_bangumi_json({
            3: info(2, 'B', 'u2', 'Episode 10', 'e3'), 2: info(2, 'B', 'u2', 'Episode 2', 'e2'),
            1: info(1, 'A', 'u1', 'Episode 1', 'e1'),
        })
        assert data == [
            {'id': 1, 'title': 'A', 'url': 'u1', 'episodes': [{'id': 1, 'title': 'Episode 1', 'url': 'e1'}]},
            {'id': 2, 'title': 'B', 'url': 'u2', 'episodes': [{'id': 2, 'title': 'Episode 2', 'url': 'e2'},
                                                              {'id': 3, 'title': 'Episode 10', 'url': 'e3'}]},
        ]


@pytest.mark.unittest
def test_the_real_figures_and_card_are_produced(world, monkeypatch):
    from inf.fancaps import readme
    monkeypatch.setattr(idx, 'render_figures', readme.render_figures)
    site = make_site(3, 4, 6)
    world.run(site)
    directory = world.hub.commits[-1][1]
    assert sorted(os.listdir(directory / 'figures')) == sorted(name for name, _ in readme.FIGURES)
    card = (directory / 'README.md').read_text()
    assert card.startswith('---\nlicense: other') and '| screenshots | 72 |' in card and '| episodes | 12 |' in card
    assert np.all([os.path.getsize(directory / 'figures' / name) > 2000 for name, _ in readme.FIGURES])
