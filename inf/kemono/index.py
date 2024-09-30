import os
import time
from pprint import pprint

import pandas as pd
from ditk import logging
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.utils import get_requests_session

from .base import _ROOT


def sync(repository: str):
    start_time = time.time()
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

    df_creators = pd.read_parquet(hf_client.hf_hub_download(
        repo_id=repository,
        repo_type='dataset',
        filename='creators.parquet',
    ))
    df_creators = df_creators.sort_values(by=['updated'], ascending=[False])
    df_creators = df_creators[df_creators['service'] != 'discord']
    print(df_creators)
    print(df_creators['service'].value_counts())

    # for s in sorted(set(df_creators['service'])):
    #     print(s, df_creators[df_creators['service'] == s])

    quit()

    session = get_requests_session()

    def _get_posts(service: str, uid: str):
        offset = 0
        while True:
            logging.info(f'Searching {service} #{uid}, offset: {offset} ...')
            resp = session.get(f'{_ROOT}/api/v1/{service}/user/{uid}', params={'o': str(offset)})
            if resp.status_code == 400:
                break
            resp.raise_for_status()
            yield from resp.json()
            if not resp.json():
                break
            offset += len(resp.json())

    pprint(list(_get_posts('fanbox', '3316400')))


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_KMN']
    )
