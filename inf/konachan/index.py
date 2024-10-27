import math
import mimetypes
import os
import re
import time

import httpx
import numpy as np
import pandas as pd
import requests.exceptions
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory, get_hf_client, get_hf_fs
from hfutils.utils import number_to_tag, get_requests_session
from pyrate_limiter import Rate, Duration, Limiter
from waifuc.utils import srequest

mimetypes.add_type('image/webp', '.webp')
_TAG_TYPES = {
    -1: 'unknown',
    0: 'general',
    1: 'artist',
    3: 'copyright',
    4: 'character',
    5: 'style',
    6: 'circle',
}


def sync(repository: str, max_time_limit: float = 50 * 60, upload_time_span: float = 30,
         deploy_span: float = 5 * 60, sync_mode: bool = False, no_recent: float = 60 * 60 * 24 * 15):
    start_time = time.time()
    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

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

    if hf_fs.exists(f'datasets/{repository}/konachan.parquet'):
        df_ = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='konachan.parquet',
        )).replace(np.NaN, None)
        exist_ids = set(df_['id'])
        pre_ids = set(df_['id'])
        records = df_.to_dict('records')
    else:
        exist_ids = set()
        pre_ids = set()
        records = []

    df_index_tags = pd.read_parquet(hf_client.hf_hub_download(
        repo_id=repository,
        repo_type='dataset',
        filename='index_tags.parquet'
    )).replace(np.NaN, None)
    d_index_tags = {item['name']: item for item in df_index_tags.to_dict('records')}

    if hf_fs.exists(f'datasets/{repository}/tags.parquet'):
        df_tags = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='tags.parquet'
        )).replace(np.NaN, None)
        d_tags = {item['name']: item for item in df_tags.to_dict('records')}
    else:
        d_tags = {}

    _last_update, has_update = None, False
    _total_count = len(records)

    def _deploy(force=False):
        nonlocal _last_update, has_update, _total_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            parquet_file = os.path.join(td, 'konachan.parquet')
            df_records = pd.DataFrame(records)
            df_records = df_records.sort_values(by=['id'], ascending=[False])
            df_records.to_parquet(parquet_file, engine='pyarrow', index=False)

            tags_file = os.path.join(td, 'tags.parquet')
            df_tags = pd.DataFrame(list(d_tags.values()))
            df_tags = df_tags.sort_values(['count', 'type'], ascending=[False, True])
            df_tags.to_parquet(tags_file, engine='pyarrow', index=False)

            with open(os.path.join(td, 'README.md'), 'w') as f:
                print('---', file=f)
                print('license: other', file=f)
                print('task_categories:', file=f)
                print('- image-classification', file=f)
                print('- zero-shot-image-classification', file=f)
                print('- text-to-image', file=f)
                print('language:', file=f)
                print('- en', file=f)
                print('- ja', file=f)
                print('tags:', file=f)
                print('- art', file=f)
                print('- anime', file=f)
                print('- not-for-all-audiences', file=f)
                print('size_categories:', file=f)
                print(f'- {number_to_tag(len(df_records))}', file=f)
                print('annotations_creators:', file=f)
                print('- no-annotation', file=f)
                print('source_datasets:', file=f)
                print('- konachan', file=f)
                print('---', file=f)
                print('', file=f)

                print('## Records', file=f)
                print(f'', file=f)
                df_records_shown = df_records[:50][
                    ['id', 'width', 'height', 'rating', 'file_size', 'mimetype', 'file_url']]
                print(f'{plural_word(len(df_records), "record")} in total. '
                      f'Only {plural_word(len(df_records_shown), "record")} shown.', file=f)
                print(f'', file=f)
                print(df_records_shown.to_markdown(index=False), file=f)
                print(f'', file=f)
                print('## Tags', file=f)
                print(f'', file=f)
                print(f'{plural_word(len(df_tags), "tag")} in total.', file=f)
                print(f'', file=f)
                for type_id in sorted(set(df_tags['type'])):
                    df_tags_type = df_tags[df_tags['type'] == type_id]
                    df_tags_type = df_tags_type[['id', 'name', 'type', 'ambiguous', 'count']]
                    df_tags_shown = df_tags_type[:30]
                    df_tags_shown = df_tags_shown.replace(np.NaN, '')
                    print(f'These are the top {plural_word(len(df_tags_shown), "tag")} '
                          f'({plural_word(len(df_tags_type), "tag")} in total) '
                          f'of type `{_TAG_TYPES.get(type_id, type_id)} ({type_id})`:', file=f)
                    print('', file=f)
                    print(df_tags_shown.to_markdown(index=False), file=f)
                    print('', file=f)

            limiter.try_acquire('hf upload limit')
            upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Add {plural_word(len(df_records) - _total_count, "new record")} into index',
            )
            has_update = False
            _last_update = time.time()
            _total_count = len(df_records)

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

    def _get_page(page_no):
        logging.info(f'Getting page {page_no!r} ...')
        resp = srequest(
            session, 'GET', 'https://konachan.com/post.json',
            params={'limit': '100', 'page': str(page_no)}
        )
        return resp.json()

    l, r = 1024, 2048
    while True:
        if _get_page(r):
            r <<= 1
            l <<= 1
        else:
            break

    while l < r:
        m = (l + r + 1) // 2
        if _get_page(m):
            l = m
        else:
            r = m - 1
    max_page = l
    page_size = len(_get_page(1))
    logging.info(f'The max page is {max_page!r}, page size is {page_size!r}.')

    if sync_mode:
        page_range = range(1, max_page + 1)
    else:
        start_page = max(min(max_page - len(records) // page_size + 5, max_page), 1)
        page_range = range(start_page, 0, -1)
    for page in page_range:
        if start_time + max_time_limit < time.time():
            break

        for item in _get_page(page):
            if item['created_at'] + no_recent > time.time():
                logging.info(f'Post {item["id"]} is too recent, skipped.')
                continue

            if item['id'] in exist_ids:
                logging.info(f'Post {item["id"]} already crawled, skipped.')
                continue

            logging.info(f'Post {item["id"]} confirmed.')
            mimetype, _ = mimetypes.guess_type(item['file_url'])
            tags = list(filter(bool, re.split(r'\s+', item['tags'])))
            item['mimetype'] = mimetype
            records.append({
                **item,
                'mimetype': mimetype,
                'tags': ' '.join(['', *tags, '']),
            })
            for tag in tags:
                if tag in d_index_tags:
                    pre_info = d_index_tags[tag]
                else:
                    pre_info = {
                        "id": -1,
                        "name": tag,
                        "count": 0,
                        "type": -1,
                        "ambiguous": False,
                    }
                if tag not in d_tags:
                    d_tags[tag] = {**pre_info, 'count': 0}
                count = d_tags[tag]['count']
                d_tags[tag].update(pre_info)
                d_tags[tag]['count'] = count + 1
            exist_ids.add(item['id'])
            has_update = True

        _deploy()

    _deploy(force=True)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_KN'],
        max_time_limit=5.5 * 60 * 60,
    )
