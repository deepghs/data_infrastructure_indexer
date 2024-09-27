import os
from functools import partial

from hfutils.operate import get_hf_client
from hfutils.utils import get_requests_session
from huggingface_hub import configure_http_backend


def repo_squash():
    configure_http_backend(partial(get_requests_session, timeout=180))

    hf_client = get_hf_client()

    remote_repo = os.environ.get('REMOTE_REPOSITORY_YR')
    if len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= 500:
        hf_client.super_squash_history(repo_id=remote_repo, repo_type='dataset', commit_message='Squashed!')
    remote_repo = os.environ.get('REMOTE_REPOSITORY_YR_PUBLIC')
    if len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= 500:
        hf_client.super_squash_history(repo_id=remote_repo, repo_type='dataset', commit_message='Squashed!')
    remote_repo = os.environ.get('REMOTE_REPOSITORY_YR_PUBLIC_4M')
    if len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= 500:
        hf_client.super_squash_history(repo_id=remote_repo, repo_type='dataset', commit_message='Squashed!')


if __name__ == '__main__':
    repo_squash()
