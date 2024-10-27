import os

import httpx
import pandas as pd
import requests.exceptions
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory, get_hf_client, get_hf_fs
from hfutils.utils import get_requests_session
from tqdm import tqdm
from waifuc.utils import srequest


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

    while True:
        session = get_requests_session()
        logging.info(f'Try new session, UA: {session.headers["User-Agent"]!r}.')
        try:
            _ = srequest(session, 'GET', 'https://konachan.com/tag.json')
        except (httpx.HTTPError, requests.exceptions.RequestException) as err:
            logging.info(f'Retrying session - {err!r}.')
            continue
        else:
            logging.info('Try success.')
            break

    logging.info('Loading all tags ...')
    resp = srequest(session, 'GET', 'https://konachan.com/tag.json', params={'limit': '0'})
    df_tags = pd.DataFrame(resp.json())
    df_tags = df_tags.sort_values(by=['id'], ascending=[False])
    d_tags = {item['id']: item for item in df_tags.to_dict('records')}
    logging.info(f'Tags:\n{df_tags}')

    page = 0
    exist_tag_alias_ids = set()
    alias_records = []
    pg_page = tqdm(desc='Tag Alias Pages')
    pg_alias = tqdm(desc='Tag Alias')
    while True:
        resp = srequest(session, 'GET', 'https://konachan.com/tag_alias.json', params={'page': str(page)})
        for item in resp.json():
            if item['id'] in exist_tag_alias_ids:
                continue

            item['alias_name'] = d_tags[item['alias_id']]['name']
            item['type'] = d_tags[item['alias_id']]['type']
            alias_records.append(item)
            exist_tag_alias_ids.add(item['id'])
            pg_alias.update()

        if not resp.json():
            break
        page += 1
        pg_page.update()

    df_tag_aliases = pd.DataFrame(alias_records)
    df_tag_aliases = df_tag_aliases.sort_values(by=['id'], ascending=[False])
    logging.info(f'Tag alias:\n{df_tag_aliases}')
    with TemporaryDirectory() as td:
        df_tags.to_parquet(os.path.join(td, 'index_tags.parquet'), engine='pyarrow', index=False)
        df_tag_aliases.to_parquet(os.path.join(td, 'index_tag_aliases.parquet'), engine='pyarrow', index=False)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=td,
            path_in_repo='.',
            message=f'Create tags index, with {plural_word(len(df_tags), "tag")} '
                    f'and {plural_word(len(df_tag_aliases), "tag alias")}',
            hf_token=os.environ['HF_TOKEN'],
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_KN'],
    )
