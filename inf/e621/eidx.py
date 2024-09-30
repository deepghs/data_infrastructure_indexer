import json
import os

from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import hf_fs_path, parse_hf_fs_path


def sync(repository: str, exist_repo: str):
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
        with open(hf_client.hf_hub_download(
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


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_E621'],
        exist_repo=os.environ['REMOTE_REPOSITORY_E621_2024'],
    )
