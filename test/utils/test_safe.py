import json
import os
import uuid
import zipfile
from pathlib import Path

import pytest
import requests
from huggingface_hub import HfApi

from inf.utils.safe import safe_hf_hub_download, safe_download_file_to_file, safe_download_archive_as_directory


class _FlakyClient:
    def __init__(self, fail_times: int = 1):
        self.fail_times = fail_times
        self.calls = 0
        self.force_download_values = []

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs):
        self.calls += 1
        self.force_download_values.append(kwargs.get('force_download', False))
        if self.calls <= self.fail_times:
            raise requests.ReadTimeout(f'synthetic timeout on call {self.calls}')

        local_dir = Path(kwargs['local_dir'])
        target_file = local_dir / filename
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(f'{repo_id}:{filename}')
        return str(target_file)


@pytest.mark.unittest
def test_safe_hf_hub_download_retries_on_transient_errors(tmp_path):
    hf_client = _FlakyClient(fail_times=1)
    downloaded = safe_hf_hub_download(
        hf_client,
        repo_id='demo/repo',
        repo_type='dataset',
        filename='meta.json',
        local_dir=tmp_path,
        max_retries=3,
        retry_wait_time=0.01,
    )

    assert Path(downloaded).read_text() == 'demo/repo:meta.json'
    assert hf_client.calls == 2
    assert hf_client.force_download_values == [False, True]


def _require_smoke_env():
    required_envs = ['HF_SAFE_SMOKE', 'HF_TOKEN', 'HF_ENDPOINT']
    missing = [name for name in required_envs if not os.environ.get(name)]
    if missing:
        pytest.skip(f'Smoke test requires environment variables: {missing!r}')


@pytest.mark.unittest
def test_safe_download_helpers_smoke(tmp_path):
    _require_smoke_env()

    hf_client = HfApi(token=os.environ['HF_TOKEN'])
    user_info = hf_client.whoami()
    repo_owner = user_info['name']
    repo_suffix = os.environ.get('HF_SAFE_SMOKE_SUFFIX') or uuid.uuid4().hex[:8]
    repo_id = f'{repo_owner}/data-infra-safe-smoke-{repo_suffix}'

    source_dir = tmp_path / 'source'
    source_dir.mkdir(parents=True, exist_ok=True)
    meta_file = source_dir / 'meta.json'
    meta_content = {
        'smoke': True,
        'repo': repo_id,
    }
    meta_file.write_text(json.dumps(meta_content, indent=4))

    payload_dir = source_dir / 'payload'
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_file = payload_dir / 'hello.txt'
    payload_file.write_text('hello from safe smoke test\n')

    archive_file = source_dir / 'payload.zip'
    with zipfile.ZipFile(archive_file, 'w') as zf:
        zf.write(payload_file, arcname='hello.txt')

    hf_client.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=False)
    try:
        hf_client.upload_file(
            path_or_fileobj=str(meta_file),
            path_in_repo='meta.json',
            repo_id=repo_id,
            repo_type='dataset',
            commit_message='Add smoke meta file',
        )
        hf_client.upload_file(
            path_or_fileobj=str(archive_file),
            path_in_repo='payload.zip',
            repo_id=repo_id,
            repo_type='dataset',
            commit_message='Add smoke archive file',
        )

        download_dir = tmp_path / 'download'
        download_dir.mkdir(parents=True, exist_ok=True)
        downloaded_meta = safe_hf_hub_download(
            hf_client,
            repo_id=repo_id,
            repo_type='dataset',
            filename='meta.json',
            local_dir=download_dir,
            max_retries=3,
            retry_wait_time=0.5,
        )
        assert json.loads(Path(downloaded_meta).read_text()) == meta_content

        downloaded_copy = tmp_path / 'copy' / 'meta.json'
        safe_download_file_to_file(
            local_file=str(downloaded_copy),
            repo_id=repo_id,
            repo_type='dataset',
            file_in_repo='meta.json',
            hf_client=hf_client,
            max_retries=3,
            retry_wait_time=0.5,
        )
        assert json.loads(downloaded_copy.read_text()) == meta_content

        unpacked_dir = tmp_path / 'unpacked'
        safe_download_archive_as_directory(
            local_directory=str(unpacked_dir),
            repo_id=repo_id,
            repo_type='dataset',
            file_in_repo='payload.zip',
            hf_client=hf_client,
            max_retries=3,
            retry_wait_time=0.5,
        )
        assert (unpacked_dir / 'hello.txt').read_text() == 'hello from safe smoke test\n'
    finally:
        hf_client.delete_repo(repo_id=repo_id, repo_type='dataset')
