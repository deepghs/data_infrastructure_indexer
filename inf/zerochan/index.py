import json
import math
import mimetypes
import os
import re
import time
from functools import partial
from typing import Optional, List

import httpx
import numpy as np
import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import upload_directory_as_directory, get_hf_client, get_hf_fs
from hfutils.utils import number_to_tag
from pyrate_limiter import Duration, Limiter, Rate
from waifuc.utils import srequest

from .base import get_session
from .tag import _get_tag_info

mimetypes.add_type('image/webp', '.webp')


def get_record(zerochan_id: int, session: Optional[requests.Session] = None):
    session = session or get_session()
    resp = srequest(
        session, 'GET', f'https://www.zerochan.net/{zerochan_id}',
        params={'json': '1'}
    )
    return resp.json()


def sync(repository: str, max_time_limit: float = 50 * 60, upload_time_span: float = 30,
         tag_refresh_time: float = 15 * 24 * 60 * 60, deploy_span: float = 5 * 60, sync_mode: bool = False,
         try_failed_ids_first: bool = False, start_from_id: Optional[int] = None):
    start_time = time.time()
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    logging.info(f'Try failed ids first: {try_failed_ids_first!r}')
    session = get_session()
    delete_detached_cache()

    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    if hf_fs.exists(f'datasets/{repository}/zerochan.parquet'):
        df_ = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='zerochan.parquet',
        )).replace(np.NaN, None)
        exist_ids = set(df_['id'])
        pre_ids = set(df_['id'])
        records = df_.to_dict('records')
    else:
        exist_ids = set()
        pre_ids = set()
        records = []

    if hf_fs.exists(f'datasets/{repository}/meta.json'):
        meta_info = json.loads(hf_fs.read_text(f'datasets/{repository}/meta.json'))
        failed_ids = set(meta_info['failed_ids'])
    else:
        failed_ids = set()

    if hf_fs.exists(f'datasets/{repository}/tags.json'):
        tags_raw = json.loads(hf_fs.read_text(f'datasets/{repository}/tags.json'))
        d_tags = {item['name']: item for item in tags_raw}
    else:
        d_tags = {}

    ids_in_table = [*pre_ids, *failed_ids]
    if ids_in_table:
        min_id = min(ids_in_table)
    else:
        min_id = None

    def ping_tag(tag, primary: bool = False):
        if tag in d_tags and d_tags[tag]['created_at'] + tag_refresh_time > time.time():
            if primary:
                d_tags[tag]['strict'] += 1
            else:
                d_tags[tag]['count'] += 1
            return tag
        else:
            logging.info(f'Query for tag {tag!r}.')
            tag_info = _get_tag_info(tag)
            if not tag_info:
                logging.warning(f'Empty tag, dropped - {tag!r}.')
                return None

            tag = tag_info['name']
            if tag in d_tags:
                strict = d_tags[tag]['strict']
                count = d_tags[tag]['count']
            else:
                strict = 0
                count = 0

            d_tags[tag] = tag_info
            d_tags[tag]['created_at'] = time.time()
            if primary:
                d_tags[tag]['strict'] = strict + 1
                d_tags[tag]['count'] = count
            else:
                d_tags[tag]['strict'] = strict
                d_tags[tag]['count'] = count + 1
            return tag

    def _iter_image_ids(offset: Optional[int] = None, prefix_ids: Optional[List[int]] = None):
        nonlocal exist_ids
        prefix_ids = list(prefix_ids or [])
        for id_ in prefix_ids:
            if id_ not in exist_ids:
                yield id_
                exist_ids.add(id_)

        ptc = 0
        while True:
            if start_time + max_time_limit < time.time():
                return

            params = {'json': '1'}
            if offset is not None and offset > 0:
                params['o'] = str(offset)
            resp = srequest(
                session, 'GET', 'https://www.zerochan.net/',
                params=params
            )
            ids = list(map(int, re.findall(r'"id":\s*(\d+)\s*,', resp.text)))
            has_new = False
            new_count = 0
            for id_ in ids:
                if id_ not in exist_ids:
                    yield id_
                    has_new = True
                    new_count += 1

            if not has_new or new_count <= 1:
                ptc += 1
            else:
                ptc = 0
            logging.info(f'Current continuous empty pages: {ptc!r}, has new: {has_new!r}, new count: {new_count!r}')
            if sync_mode and ptc >= 10:
                break
            if not ids:
                break
            offset = min(ids)

    _last_update, has_update = None, False
    _total_count = len(records)

    def _deploy(force=False):
        nonlocal _last_update, has_update, _total_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            parquet_file = os.path.join(td, 'zerochan.parquet')
            df_records = pd.DataFrame(records)
            df_records = df_records.sort_values(by=['id'], ascending=[False])
            df_records.to_parquet(parquet_file, engine='pyarrow', index=False)

            df_tags = pd.DataFrame([
                {
                    'name': d_item['name'],
                    'category': d_item['category'],
                    'raw_category': d_item.get('raw_category'),
                    **{
                        f'lang_{k}': (v or '')
                        for k, v in d_item['langs'].items()
                    },
                    'count': d_item['count'],
                    'strict': d_item['strict'],
                }
                for d_item in d_tags.values()
            ])
            df_tags = df_tags.replace(np.NaN, '')
            df_tags = df_tags.sort_values(['count', 'category'], ascending=[False, True])
            pcolumns = ['name', 'category', 'raw_category', 'count', 'strict']
            columns = [*pcolumns, *filter(lambda x: x.startswith('lang_'), df_tags.columns)]
            df_tags = df_tags[columns]
            s_columns = [name for name in columns if name != 'raw_category']

            with open(os.path.join(td, 'tags.json'), 'w') as f:
                json.dump(list(d_tags.values()), f)

            with open(os.path.join(td, 'meta.json'), 'w') as f:
                json.dump({
                    'failed_ids': sorted(failed_ids),
                    'exist_ids': sorted(exist_ids),
                }, f)

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
                print('- zerochan', file=f)
                print('---', file=f)
                print('', file=f)

                print('## Records', file=f)
                print(f'', file=f)
                df_records_shown = df_records[:50][
                    ['id', 'width', 'height', 'file_size', 'mimetype', 'primary_tag', 'file_url', ]]
                print(f'{plural_word(len(df_records), "record")} in total. '
                      f'Only {plural_word(len(df_records_shown), "record")} shown.', file=f)
                print(f'', file=f)
                print(df_records_shown.to_markdown(index=False), file=f)
                print(f'', file=f)
                print('## Tags', file=f)
                print(f'', file=f)
                print(f'{plural_word(len(df_tags), "tag")} in total.', file=f)
                print(f'', file=f)
                for type_id in sorted(set(df_tags['category'])):
                    df_tags_type = df_tags[df_tags['category'] == type_id]
                    if type_id != 'unknown':
                        df_tags_type = df_tags_type[s_columns]
                    df_tags_shown = df_tags_type[:30]
                    print(f'These are the top {plural_word(len(df_tags_shown), "tag")} '
                          f'({plural_word(len(df_tags_type), "tag")} in total) '
                          f'of type `{type_id}`:', file=f)
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

    is_data_safe = True
    try:
        for post_id in _iter_image_ids(
                offset=min_id if not sync_mode else start_from_id,
                prefix_ids=sorted(failed_ids, reverse=True) if try_failed_ids_first else [],
        ):
            if start_time + max_time_limit < time.time():
                break
            # if post_id in pre_ids and sync_mode:
            #     break
            if post_id in exist_ids:
                continue

            logging.info(f'Post {post_id!r} confirmed.')
            try:
                item = get_record(post_id, session=session)
            except (requests.exceptions.RequestException, httpx.HTTPError, json.JSONDecodeError) as err:
                logging.info(f'Post {post_id!r} skipped due to error - {err!r}.')
                failed_ids.add(post_id)
                has_update = True
                continue

            mimetype, _ = mimetypes.guess_type(item['full'])
            tags = list(filter(bool, item['tags']))
            try:
                row = {
                    'id': item['id'],
                    'width': item['width'],
                    'height': item['height'],
                    'file_size': item['size'],
                    'mimetype': mimetype,
                    'file_url': item['full'],
                    'small_url': item['small'],
                    'medium_url': item['medium'],
                    'large_url': item['large'],
                    'hash': item['hash'],
                    'source': item.get('source'),
                    'primary_tag': ping_tag(item['primary'], primary=True) if item['primary'] else None,
                    'tags': json.dumps(list(filter(bool, map(partial(ping_tag, primary=False), tags)))),
                }
                records.append(row)
                exist_ids.add(item['id'])
                if item['id'] in failed_ids:
                    failed_ids.remove(item['id'])
            except:
                is_data_safe = False
                raise

            has_update = True
            _deploy()

    finally:
        if is_data_safe:
            _deploy(force=True)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_ZC'],
        max_time_limit=5.7 * 60 * 60,
        deploy_span=5 * 60,
        # try_failed_ids_first=random.random() < 0.0625,
        try_failed_ids_first=False,
        sync_mode=True,
        start_from_id=None,
    )
