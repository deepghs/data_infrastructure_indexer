import html
import json
import math
import mimetypes
import os
import re
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from ditk import logging
from hbutils.collection import unique
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import upload_directory_as_directory, get_hf_client, get_hf_fs
from hfutils.utils import number_to_tag
from natsort import natsorted
from pyrate_limiter import Rate, Duration, Limiter
from waifuc.utils import srequest

from .tags import _get_session, _LIMITER

mimetypes.add_type('image/webp', '.webp')
__site_url__ = 'https://rule34.xxx'
_TAG_TYPES = {
    -1: 'unknown',
    0: 'general',
    1: 'artist',
    3: 'copyright',
    4: 'character',
    5: 'metadata',
}


def sync(repository: str, max_time_limit: float = 50 * 60, upload_time_span: float = 30,
         deploy_span: float = 5 * 60, no_recent: float = 60 * 60 * 24 * 15,
         max_part_rows: int = 1500000, user_id: Optional[str] = None, api_key: Optional[str] = None):
    start_time = time.time()
    delete_detached_cache()
    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    hf_client = get_hf_client(hf_token=os.environ['HF_TOKEN'])
    hf_fs = get_hf_fs(hf_token=os.environ['HF_TOKEN'])

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

    if hf_client.file_exists(
            repo_id=repository,
            repo_type='dataset',
            filename='exist_ids.json',
    ):
        exist_ids = json.loads(hf_fs.read_text(f'datasets/{repository}/exist_ids.json'))
        exist_ids = set(exist_ids)
    else:
        exist_ids = set()

    if hf_fs.glob(f'datasets/{repository}/tables/rule34-*.parquet'):
        last_path = natsorted(hf_fs.glob(f'datasets/{repository}/tables/rule34-*.parquet'))[-1]
        last_file_name = os.path.basename(last_path)
        last_rel_file = os.path.relpath(last_path, f'datasets/{repository}')
        current_ptr = int(os.path.splitext(last_file_name)[0].split('-')[-1])
        df_record = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename=last_rel_file,
        )).replace(np.NaN, None)
        df_record = df_record[df_record['id'] < 10 ** 9]
        records = df_record.to_dict('records')
        if len(records) > max_part_rows:
            records = []
            current_ptr = current_ptr + 1
    else:
        records = []
        current_ptr = 1
    logging.info(f'Current table ptr: {current_ptr!r}, records: {len(records)}')

    df_origin_tags = pd.read_parquet(hf_client.hf_hub_download(
        repo_id=repository,
        repo_type='dataset',
        filename='index_tags.parquet',
    )).replace(np.NaN, None)
    d_origin_tags = {item['name'].strip(): item for item in df_origin_tags.to_dict('records')}

    if hf_client.file_exists(
            repo_id=repository,
            repo_type='dataset',
            filename='tags.parquet',
    ):
        df_tags = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='tags.parquet',
        )).replace(np.NaN, None)
        d_tags = {item['name']: item for item in df_tags.to_dict('records')}
    else:
        d_tags = {}

    _last_update, has_update = None, False
    last_image_count = len(exist_ids)

    def _deploy(force=False):
        nonlocal _last_update, has_update, last_image_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            parquet_file = os.path.join(td, 'tables', f'rule34-{current_ptr}.parquet')
            os.makedirs(os.path.dirname(parquet_file), exist_ok=True)
            df_records = pd.DataFrame(records)
            df_records = df_records.sort_values(by=['id'], ascending=[False])
            df_records.to_parquet(parquet_file, index=False)

            tags_file = os.path.join(td, 'tags.parquet')
            df_tags = pd.DataFrame(list(d_tags.values()))
            df_tags = df_tags.sort_values(['count', 'type'], ascending=[False, True])
            df_tags.to_parquet(tags_file, index=False)

            with open(os.path.join(td, 'exist_ids.json'), 'w') as f:
                json.dump(sorted(exist_ids), f)

            with open(os.path.join(td, 'README.md'), 'w') as f:
                print('---', file=f)
                print('license: other', file=f)
                print('task_categories:', file=f)
                print('- image-classification', file=f)
                print('- zero-shot-image-classification', file=f)
                print('- text-to-image', file=f)
                print('language:', file=f)
                print('- en', file=f)
                print('tags:', file=f)
                print('- art', file=f)
                print('- anime', file=f)
                print('- not-for-all-audiences', file=f)
                print('size_categories:', file=f)
                print(f'- {number_to_tag(len(exist_ids))}', file=f)
                print('annotations_creators:', file=f)
                print('- no-annotation', file=f)
                print('source_datasets:', file=f)
                print('- rule34', file=f)
                print('---', file=f)
                print('', file=f)

                print('## Records', file=f)
                print(f'', file=f)
                df_records_shown = df_records[:50][
                    ['id', 'width', 'height', 'rating', 'mimetype', 'file_url']]
                print(f'{plural_word(len(exist_ids), "record")} in total. '
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
                hf_token=os.environ['HF_TOKEN'],
                message=f'Add {plural_word(len(exist_ids) - last_image_count, "new record")}',
            )
            has_update = False
            _last_update = time.time()
            last_image_count = len(exist_ids)

    session = _get_session()

    def _get_posts(tags: List[str], page: int = 1):
        logging.info(f'Get posts for tags: {tags!r}, page: {page!r} ...')
        params = {
            'page': 'dapi',
            's': 'post',
            'q': 'index',
            'tags': ' '.join(tags),
            'json': '1',
            'limit': '1000',
            'pid': str(page),
        }
        if user_id and api_key:
            params['user_id'] = user_id
            params['api_key'] = api_key
        _LIMITER.try_acquire('api limit')
        resp = srequest(session, 'GET', f'{__site_url__}/index.php', params=params)
        return resp.json()

    def _yield_from_newest():
        no_new_cnt = 0
        pid, current_min_id, min_id = 0, None, None
        while True:
            if pid > 200:
                pid = 0
                min_id = current_min_id

            has_item = False
            has_new = False
            search_tags = [] if min_id is None else [f'id:<{min_id}']
            for item in _get_posts(search_tags, pid):
                if item['id'] not in exist_ids:
                    yield item
                    has_new = True
                has_item = True
                if current_min_id is None or item['id'] < current_min_id:
                    current_min_id = item['id']

            if not has_item:
                break
            if not has_new:
                no_new_cnt += 1
            else:
                no_new_cnt = 0
            if no_new_cnt >= 20:
                break

            pid += 1

    source = _yield_from_newest()
    for item in source:
        if start_time + max_time_limit < time.time():
            break
        if item['change'] and item['change'] + no_recent > time.time():
            logging.info(f'Post {item["id"]} too recent, skipped.')
            continue
        if item['id'] in exist_ids:
            logging.info(f'Post {item["id"]} already crawled, skipped.')
            continue

        logging.info(f'Post {item["id"]} confirmed.')
        tags = list(map(html.unescape, filter(bool, re.split(r'\s+', item['tags']))))
        tags = list(unique(tags))
        mimetype, _ = mimetypes.guess_type(item['file_url'])
        row = {
            'id': item['id'],
            'width': item['width'],
            'height': item['height'],
            'filename': item['image'],
            'mimetype': mimetype,
            'rating': item['rating'],
            'file_url': item['file_url'],
            **item,
            'tags': ' '.join(['', *tags, '']),
            'scraped_at': time.time(),
        }
        records.append(row)
        for tag in tags:
            if tag not in d_tags:
                if tag in d_origin_tags:
                    d_tags[tag] = {**d_origin_tags[tag], 'count': 0}
                else:
                    d_tags[tag] = {'id': -1, 'type': -1, 'name': tag, 'count': 0, 'ambiguous': False}
            count = d_tags[tag]['count']
            if tag in d_origin_tags:
                d_tags[tag].update(d_origin_tags[tag])
            d_tags[tag]['count'] = count + 1
        exist_ids.add(item['id'])
        has_update = True
        _deploy()

    _deploy(force=True)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_RX'],
        max_time_limit=5.5 * 60 * 60,
        no_recent=60 * 60 * 24 * 0,
        deploy_span=3 * 60,
        max_part_rows=2000000,
        user_id=os.environ['RULE34_USER_ID'],
        api_key=os.environ['RULE34_API_KEY'],
    )
