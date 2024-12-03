from ditk import logging
import os
from functools import partial

from hfutils.utils import get_requests_session
from huggingface_hub import HfApi, configure_http_backend


def repo_squash():
    configure_http_backend(partial(get_requests_session, timeout=180))
    hf_client = HfApi(token=os.environ.get('HF_TOKEN'))

    for remote_repo in [
        os.environ.get('REMOTE_REPOSITORY_E621'),
        os.environ.get('REMOTE_REPOSITORY_E621_PUBLIC'),
        os.environ.get('REMOTE_REPOSITORY_E621_2024'),
        os.environ.get('REMOTE_REPOSITORY_E621_PUBLIC_4M'),
        os.environ.get('REMOTE_REPOSITORY_E621_4M_IDX'),
    ]:
        if remote_repo and len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= 500:
            logging.info(f'Squashing repository {remote_repo!r} ...')
            hf_client.super_squash_history(repo_id=remote_repo, repo_type='dataset',
                                           commit_message=f'Repository {remote_repo} squashed!')


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    repo_squash()
