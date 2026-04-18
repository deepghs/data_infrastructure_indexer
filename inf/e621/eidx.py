import json
import os

import click
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import hf_fs_path, parse_hf_fs_path

from inf.utils.safe import safe_hf_hub_download


def sync(repository: str, exist_repo: str):
    """Build the e621 previous-existence index from the historical dataset repository."""
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        hf_client.update_repo_visibility(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.db filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    exist_ids = set()
    for path in hf_fs.glob(hf_fs_path(
            repo_id=exist_repo,
            repo_type='dataset',
            filename='original/*.json',
    )):
        filename = parse_hf_fs_path(path).filename
        with open(safe_hf_hub_download(
                hf_client,
                repo_id=exist_repo,
                repo_type='dataset',
                filename=filename,
        ), 'r') as f:
            meta = json.load(f)
            for key in meta['files'].keys():
                fid = int(os.path.basename(key).split('.')[0])
                exist_ids.add(fid)

    with TemporaryDirectory() as td:
        with open(os.path.join(td, 'previous_exist_ids.json'), 'w') as f:
            json.dump(sorted(exist_ids), f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=td,
            path_in_repo='.',
            message=f'Sync {plural_word(len(exist_ids), "previous exist id")} from {exist_repo!r}',
        )
        upload_directory_as_directory(
            repo_id=exist_repo,
            repo_type='dataset',
            local_directory=td,
            path_in_repo='.',
            message=f'Sync {plural_word(len(exist_ids), "previous exist id")} from {exist_repo!r}',
        )


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Build the e621 previous-existence index from the historical dataset repository. '
         'The command scans archived original metadata, extracts historical file IDs, '
         'and uploads the generated existence index to the target repositories.',
)
@click.option(
    '-r', '--repository',
    type=str,
    envvar='REMOTE_REPOSITORY_E621',
    required=True,
    show_envvar=True,
    help='Target Hugging Face dataset repository to read from and write to.',
)
@click.option(
    '-e', '--exist-repo',
    type=str,
    envvar='REMOTE_REPOSITORY_E621_2024',
    required=True,
    show_envvar=True,
    help='Existing Hugging Face dataset repository used as the source index.',
)
def cli(repository: str, exist_repo: str):
    logging.try_init_root(logging.INFO)
    return sync(repository=repository, exist_repo=exist_repo)


if __name__ == '__main__':
    cli()
