from ditk import logging
import os
from functools import partial
from typing import Optional

import click
from hfutils.utils import get_requests_session
from huggingface_hub import HfApi, configure_http_backend


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


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Super-squash the configured e621 dataset repositories when history grows too large. '
         'The command inspects commit counts across private, public and index repositories '
         'and squashes history once the configured threshold is reached.',
)
@click.option(
    '-r', '--repository',
    type=str,
    envvar='REMOTE_REPOSITORY_E621',
    default=None,
    show_envvar=True,
    help='Target Hugging Face dataset repository to process.',
)
@click.option(
    '-p', '--public-repository',
    type=str,
    envvar='REMOTE_REPOSITORY_E621_PUBLIC',
    default=None,
    show_envvar=True,
    help='Public Hugging Face dataset repository to process alongside the private repo.',
)
@click.option(
    '-v', '--previous-repository',
    type=str,
    envvar='REMOTE_REPOSITORY_E621_2024',
    default=None,
    show_envvar=True,
    help='Historical Hugging Face dataset repository to process.',
)
@click.option(
    '-P', '--public-repository-4m',
    type=str,
    envvar='REMOTE_REPOSITORY_E621_PUBLIC_4M',
    default=None,
    show_envvar=True,
    help='4M public Hugging Face dataset repository to process alongside the private repo.',
)
@click.option(
    '-i', '--index-repository',
    type=str,
    envvar='REMOTE_REPOSITORY_E621_4M_IDX',
    default=None,
    show_envvar=True,
    help='Index Hugging Face dataset repository to process.',
)
@click.option(
    '-m', '--min-commits',
    type=int,
    default=500,
    show_default=True,
    help='Only squash repositories when commit history reaches at least this count.',
)
@click.option(
    '-c', '--commit-message-template',
    type=str,
    default='Repository {repo_id} squashed!',
    show_default=True,
    help='Template used for squash commit messages; supports {repo_id}.',
)
def cli(repository: Optional[str], public_repository: Optional[str], previous_repository: Optional[str],
        public_repository_4m: Optional[str], index_repository: Optional[str], min_commits: int,
        commit_message_template: str):
    logging.try_init_root(logging.INFO)
    return repo_squash(
        repository=repository,
        public_repository=public_repository,
        previous_repository=previous_repository,
        public_repository_4m=public_repository_4m,
        index_repository=index_repository,
        min_commits=min_commits,
        commit_message_template=commit_message_template,
    )


if __name__ == '__main__':
    cli()
