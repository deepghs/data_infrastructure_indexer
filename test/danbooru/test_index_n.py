import os
from typing import List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest


def _make_post(post_id: int, file_url: Optional[str] = 'https://x/img.jpg', **extra) -> dict:
    base = {
        'id': post_id,
        'file_url': file_url,
        'image_width': 800,
        'image_height': 600,
        'rating': 's',
        'file_size': 12345,
        'media_asset': {'foo': 'bar'},
        'source': 'http://upstream/',
        'tag_string': '1girl',
    }
    base.update(extra)
    return base


class _MockResponse:
    def __init__(self, items: List[dict]):
        self._items = items

    def json(self):
        return self._items


def _srequest_factory(pages: List[List[dict]]):
    state = {'idx': 0}

    def _srequest(session, method, url, params=None, auth=None, **kwargs):
        idx = state['idx']
        state['idx'] += 1
        if idx < len(pages):
            return _MockResponse(pages[idx])
        return _MockResponse([])

    return _srequest


def _drive_sync(
        tmp_path,
        monkeypatch,
        existing_df: Optional[pd.DataFrame],
        pages: List[List[dict]],
        captured: List,
):
    from inf.danbooru import index_n as mod

    existing_path = tmp_path / 'records.parquet'
    if existing_df is not None:
        existing_df.to_parquet(existing_path, engine='pyarrow', index=False)

    hf_fs = MagicMock()
    hf_fs.exists = MagicMock(return_value=existing_df is not None)

    hf_client = MagicMock()
    hf_client.repo_exists = MagicMock(return_value=True)

    def fake_download(*args, **kwargs):
        return str(existing_path)

    def fake_upload(*args, **kwargs):
        local_dir = kwargs['local_directory']
        parquet = os.path.join(local_dir, 'records.parquet')
        captured.append(pd.read_parquet(parquet))

    danbooru_source = MagicMock()
    danbooru_source.session = MagicMock()
    danbooru_source.auth = None

    monkeypatch.setenv('HF_TOKEN', 'fake-token')
    monkeypatch.setattr(mod, 'get_hf_client', lambda **kw: hf_client)
    monkeypatch.setattr(mod, 'get_hf_fs', lambda **kw: hf_fs)
    monkeypatch.setattr(mod, 'safe_hf_hub_download', fake_download)
    monkeypatch.setattr(mod, 'safe_upload_directory_as_directory', fake_upload)
    monkeypatch.setattr(mod, 'srequest', _srequest_factory(pages))
    monkeypatch.setattr(mod, 'DanbooruSource', lambda *args, **kwargs: danbooru_source)
    monkeypatch.setattr(mod, 'delete_detached_cache', lambda: None)
    monkeypatch.setattr(mod, 'Limiter', lambda *a, **kw: MagicMock())
    monkeypatch.setattr(mod, 'Rate', lambda *a, **kw: MagicMock())

    mod.sync(
        repository='fake/fake',
        upload_time_span=0.01,
        deploy_span=0.0,
        max_time_limit=60.0,
        sync_mode=True,
        site_username=None,
        site_apikey=None,
        site_golden=False,
        start_from_id=None,
    )


@pytest.mark.unittest
def test_index_n_empty_repo(tmp_path, monkeypatch):
    captured: List[pd.DataFrame] = []
    new_posts = [_make_post(i, file_url=f'https://x/{i}.jpg') for i in [105, 104, 103, 102, 101]]
    _drive_sync(tmp_path, monkeypatch, existing_df=None, pages=[new_posts], captured=captured)

    assert len(captured) >= 1
    final = captured[-1]
    assert len(final) == 5
    assert set(final['id'].tolist()) == {101, 102, 103, 104, 105}
    assert final['id'].tolist() == [105, 104, 103, 102, 101]
    assert 'media_asset' not in final.columns
    assert 'mimetype' in final.columns
    assert final['mimetype'].iloc[0] == 'image/jpeg'


@pytest.mark.unittest
def test_index_n_merge_with_existing(tmp_path, monkeypatch):
    existing = pd.DataFrame([
        {
            'id': i,
            'file_url': f'https://old/{i}.jpg',
            'image_width': 100,
            'image_height': 100,
            'rating': 's',
            'file_size': 999,
            'mimetype': 'image/jpeg',
            'source': 'http://old/',
            'tag_string': '1girl',
        }
        for i in range(1, 101)
    ])

    new_posts = (
        [_make_post(i, file_url=f'https://new/{i}.jpg') for i in [105, 104, 103, 102, 101]]
        + [_make_post(i, file_url=f'https://updated/{i}.jpg') for i in [3, 2, 1]]
    )
    captured: List[pd.DataFrame] = []
    _drive_sync(tmp_path, monkeypatch, existing_df=existing, pages=[new_posts], captured=captured)

    assert len(captured) >= 1
    final = captured[-1]
    assert len(final) == 105, f'expected 105 rows, got {len(final)}'
    assert final['id'].tolist()[:5] == [105, 104, 103, 102, 101]
    for i in [1, 2, 3]:
        row = final[final['id'] == i].iloc[0]
        assert row['file_url'] == f'https://updated/{i}.jpg', f'id {i} not updated'
    for i in [4, 50, 100]:
        row = final[final['id'] == i].iloc[0]
        assert row['file_url'] == f'https://old/{i}.jpg', f'id {i} should still be old'


@pytest.mark.unittest
def test_index_n_no_update_skips_upload(tmp_path, monkeypatch):
    existing = pd.DataFrame([
        {
            'id': i,
            'file_url': f'https://old/{i}.jpg',
            'image_width': 100,
            'image_height': 100,
            'rating': 's',
            'file_size': 999,
            'mimetype': 'image/jpeg',
            'source': 'http://old/',
            'tag_string': '1girl',
        }
        for i in [105, 104, 103, 102, 101]
    ])
    same_posts = [
        _make_post(i, file_url=f'https://old/{i}.jpg') for i in [105, 104, 103, 102, 101]
    ]
    captured: List[pd.DataFrame] = []
    _drive_sync(tmp_path, monkeypatch, existing_df=existing, pages=[same_posts], captured=captured)

    assert captured == [], 'no upload expected when nothing changed'


@pytest.mark.unittest
def test_index_n_no_smaller_row_count(tmp_path, monkeypatch):
    """Sanity invariant: an upload must never shrink the row count below what we started with."""
    existing = pd.DataFrame([
        {
            'id': i,
            'file_url': f'https://old/{i}.jpg',
            'image_width': 100,
            'image_height': 100,
            'rating': 's',
            'file_size': 999,
            'mimetype': 'image/jpeg',
            'source': 'http://old/',
            'tag_string': '1girl',
        }
        for i in range(1, 51)
    ])
    new_posts = [_make_post(i, file_url=f'https://new/{i}.jpg') for i in [60, 59]]
    captured: List[pd.DataFrame] = []
    _drive_sync(tmp_path, monkeypatch, existing_df=existing, pages=[new_posts], captured=captured)

    assert len(captured) >= 1
    for snapshot in captured:
        assert len(snapshot) >= len(existing), (
            f'upload of {len(snapshot)} rows would shrink remote (had {len(existing)})'
        )
    assert len(captured[-1]) == 52
