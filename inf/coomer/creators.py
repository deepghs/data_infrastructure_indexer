import os
from typing import Optional

import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import get_requests_session

from .base import _ROOT


def _get_creators(session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    resp = session.get(f'{_ROOT}/api/v1/creators.txt')
    resp.raise_for_status()
    return resp.json()


def sync(repository: str):
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

    df = pd.DataFrame(_get_creators())
    df = df.sort_values(by=['indexed'], ascending=[False])
    with TemporaryDirectory() as td:
        df.to_parquet(os.path.join(td, 'creators.parquet'), engine='pyarrow', index=False)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=td,
            path_in_repo='.',
            message=f'Sync creators, {plural_word(len(df), "creator")} in total',
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_CMR']
    )
