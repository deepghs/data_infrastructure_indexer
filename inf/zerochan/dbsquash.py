import os

from hfutils.operate import get_hf_client


def repo_squash():
    hf_client = get_hf_client()
    remote_repo = os.environ.get('REMOTE_REPOSITORY_ZC')
    if len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= 500:
        hf_client.super_squash_history(repo_id=remote_repo, repo_type='dataset', commit_message='Squashed!')
    remote_repo = os.environ.get('REMOTE_REPOSITORY_ZC_PUBLIC')
    if len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= 500:
        hf_client.super_squash_history(repo_id=remote_repo, repo_type='dataset', commit_message='Squashed!')
    remote_repo = os.environ.get('REMOTE_REPOSITORY_ZC_PUBLIC_4M')
    if len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= 500:
        hf_client.super_squash_history(repo_id=remote_repo, repo_type='dataset', commit_message='Squashed!')


if __name__ == '__main__':
    repo_squash()
