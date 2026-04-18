from ditk import logging
import os
from functools import partial
from typing import Optional

from hfutils.utils import get_requests_session
from huggingface_hub import HfApi, configure_http_backend

from inf.utils.cli import env_default, run_callable_from_cli


def repo_squash(repository: Optional[str] = None, public_repository: Optional[str] = None,
                previous_repository: Optional[str] = None, public_repository_4m: Optional[str] = None,
                index_repository: Optional[str] = None, min_commits: int = 500,
                commit_message_template: str = 'Repository {repo_id} squashed!'):
    """Super-squash the configured e621 dataset repositories when history grows too large."""
    configure_http_backend(partial(get_requests_session, timeout=180))
    hf_client = HfApi(token=os.environ.get('HF_TOKEN'))

    for remote_repo in [
        repository,
        public_repository,
        previous_repository,
        public_repository_4m,
        index_repository,
    ]:
        if remote_repo and len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= min_commits:
            logging.info(f'Squashing repository {remote_repo!r} ...')
            hf_client.super_squash_history(
                repo_id=remote_repo,
                repo_type='dataset',
                commit_message=commit_message_template.format(repo_id=remote_repo),
            )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    run_callable_from_cli(repo_squash, defaults={
        'repository': env_default('REMOTE_REPOSITORY_E621', default=None),
        'public_repository': env_default('REMOTE_REPOSITORY_E621_PUBLIC', default=None),
        'previous_repository': env_default('REMOTE_REPOSITORY_E621_2024', default=None),
        'public_repository_4m': env_default('REMOTE_REPOSITORY_E621_PUBLIC_4M', default=None),
        'index_repository': env_default('REMOTE_REPOSITORY_E621_4M_IDX', default=None),
    })
