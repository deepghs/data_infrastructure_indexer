import os
from functools import partial
from typing import Optional

from hfutils.utils import get_requests_session
from huggingface_hub import HfApi, configure_http_backend

from inf.utils.cli import env_default, run_callable_from_cli


def repo_squash(repository: Optional[str] = None, public_repository: Optional[str] = None,
                public_repository_4m: Optional[str] = None, min_commits: int = 500,
                commit_message: str = 'Squashed!'):
    """Super-squash the configured Danbooru dataset repositories when history grows too large."""
    configure_http_backend(partial(get_requests_session, timeout=180))

    hf_client = HfApi(token=os.environ.get('HF_TOKEN'))

    for remote_repo in [repository, public_repository, public_repository_4m]:
        if remote_repo and len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= min_commits:
            hf_client.super_squash_history(
                repo_id=remote_repo,
                repo_type='dataset',
                commit_message=commit_message,
            )


if __name__ == '__main__':
    run_callable_from_cli(repo_squash, defaults={
        'repository': env_default('REMOTE_REPOSITORY_DB_N', default=None),
        'public_repository': env_default('REMOTE_REPOSITORY_DB_N_PUBLIC', default=None),
        'public_repository_4m': env_default('REMOTE_REPOSITORY_DB_N_PUBLIC_4M', default=None),
    })
