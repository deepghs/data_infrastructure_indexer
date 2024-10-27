import os
from functools import partial

from hfutils.utils import get_requests_session
from huggingface_hub import HfApi, configure_http_backend


def repo_squash():
    configure_http_backend(partial(get_requests_session, timeout=180))
    hf_client = HfApi(token=os.environ.get('HF_TOKEN'))

    remote_repo = os.environ.get('REMOTE_REPOSITORY_GB')
    if len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= 500:
        hf_client.super_squash_history(repo_id=remote_repo, repo_type='dataset', commit_message='Squashed!')
    remote_repo = os.environ.get('REMOTE_REPOSITORY_GB_PUBLIC')
    if len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= 500:
        hf_client.super_squash_history(repo_id=remote_repo, repo_type='dataset', commit_message='Squashed!')
    remote_repo = os.environ.get('REMOTE_REPOSITORY_GB_PUBLIC_4M')
    if len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= 500:
        hf_client.super_squash_history(repo_id=remote_repo, repo_type='dataset', commit_message='Squashed!')


if __name__ == '__main__':
    repo_squash()
