import click
from ditk import logging

from .eidx import sync


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Build the 4M e621 previous-existence index from the historical dataset repository. '
         'The command reuses the standard existence-index pipeline with the 4M repository pair '
         'and uploads the generated ID snapshot to both destinations.',
)
@click.option(
    '-r', '--repository',
    type=str,
    envvar='REMOTE_REPOSITORY_E621_PUBLIC_4M',
    required=True,
    show_envvar=True,
    help='Target Hugging Face dataset repository to read from and write to.',
)
@click.option(
    '-e', '--exist-repo',
    type=str,
    envvar='REMOTE_REPOSITORY_E621_4M_IDX',
    required=True,
    show_envvar=True,
    help='Existing Hugging Face dataset repository used as the source index.',
)
def cli(repository: str, exist_repo: str):
    logging.try_init_root(logging.INFO)
    return sync(repository=repository, exist_repo=exist_repo)


if __name__ == '__main__':
    cli()
