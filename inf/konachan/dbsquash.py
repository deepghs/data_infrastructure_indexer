from functools import partial
from typing import Optional

import click
from hfutils.operate import get_hf_client
from hfutils.utils import get_requests_session
from huggingface_hub import configure_http_backend


def repo_squash(repository: Optional[str] = None, public_repository: Optional[str] = None,
                public_repository_4m: Optional[str] = None, min_commits: int = 500,
                commit_message: str = 'Squashed!'):
    """Super-squash the configured Konachan dataset repositories when history grows too large."""
    configure_http_backend(partial(get_requests_session, timeout=180))

    hf_client = get_hf_client()

    for remote_repo in [repository, public_repository, public_repository_4m]:
        if remote_repo and len(hf_client.list_repo_commits(repo_id=remote_repo, repo_type='dataset')) >= min_commits:
            hf_client.super_squash_history(
                repo_id=remote_repo,
                repo_type='dataset',
                commit_message=commit_message,
            )


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Super-squash the configured Konachan dataset repositories when history grows too large. '
         'The command checks commit counts for the configured repositories '
         'and triggers Hugging Face history squashing once the threshold is reached.',
)
@click.option(
    '-r', '--repository',
    type=str,
    envvar='REMOTE_REPOSITORY_KN',
    default=None,
    show_envvar=True,
    help='Target Hugging Face dataset repository to process.',
)
@click.option(
    '-p', '--public-repository',
    type=str,
    envvar='REMOTE_REPOSITORY_KN_PUBLIC',
    default=None,
    show_envvar=True,
    help='Public Hugging Face dataset repository to process alongside the private repo.',
)
@click.option(
    '-P', '--public-repository-4m',
    type=str,
    envvar='REMOTE_REPOSITORY_KN_PUBLIC_4M',
    default=None,
    show_envvar=True,
    help='4M public Hugging Face dataset repository to process alongside the private repo.',
)
@click.option(
    '-m', '--min-commits',
    type=int,
    default=500,
    show_default=True,
    help='Only squash repositories when commit history reaches at least this count.',
)
@click.option(
    '-c', '--commit-message',
    type=str,
    default='Squashed!',
    show_default=True,
    help='Commit message to use for squash operations.',
)
def cli(repository: Optional[str], public_repository: Optional[str], public_repository_4m: Optional[str],
        min_commits: int, commit_message: str):
    return repo_squash(
        repository=repository,
        public_repository=public_repository,
        public_repository_4m=public_repository_4m,
        min_commits=min_commits,
        commit_message=commit_message,
    )


if __name__ == '__main__':
    cli()
