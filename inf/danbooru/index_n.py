import math
import mimetypes
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import upload_directory_as_directory, get_hf_client, get_hf_fs
from hfutils.utils import number_to_tag
from pyrate_limiter import Rate, Duration, Limiter
from waifuc.source import DanbooruSource
from waifuc.utils import srequest

mimetypes.add_type('image/webp', '.webp')


def sync(repository: str, upload_time_span: float = 30, deploy_span: float = 5 * 60,
         max_time_limit: float = 50 * 60, sync_mode: bool = False,
         site_username: Optional[str] = None, site_apikey: Optional[str] = None,
         site_golden: bool = False):
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

    if hf_fs.exists(f'datasets/{repository}/records.parquet'):
        df_records = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='records.parquet'
        )).replace(np.NaN, None)
        records = df_records.to_dict('records')
        exist_ids = set(df_records['id'])
    else:
        records = []
        exist_ids = set()

    _last_update, has_update = None, False
    _total_count = len(records)

    def _deploy(force=False):
        nonlocal _last_update, has_update, _total_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            table_parquet_file = os.path.join(td, 'records.parquet')
            os.makedirs(os.path.dirname(table_parquet_file), exist_ok=True)
            df_records = pd.DataFrame(records)
            df_records = df_records.sort_values(by=['id'], ascending=[False])
            df_records.to_parquet(table_parquet_file, engine='pyarrow', index=False)

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
                print('- danbooru', file=f)
                print('---', file=f)
                print('', file=f)

                print('## Records', file=f)
                print(f'', file=f)
                df_records_shown = df_records[:50][
                    ['id', 'image_width', 'image_height', 'rating', 'mimetype', 'file_size', 'file_url']]
                print(f'{plural_word(len(exist_ids), "record")} in total. '
                      f'Only {plural_word(len(df_records_shown), "record")} shown.', file=f)
                print(f'', file=f)
                print(df_records_shown.to_markdown(index=False), file=f)
                print(f'', file=f)

            limiter.try_acquire('hf upload limit')
            upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Adding {plural_word(len(df_records) - _total_count, "new record")} into index',
                hf_token=os.environ['HF_TOKEN'],
            )
            has_update = False
            _last_update = time.time()
            _total_count = len(df_records)

    if site_apikey and site_username:
        logging.info(f'Initializing session with username {site_username!r} and api_key {site_apikey!r} ...')
        source = DanbooruSource(['1girl'], api_key=site_apikey, username=site_username)
    else:
        logging.info('Initializing session without any API key ...')
        source = DanbooruSource(['1girl'])
    source._prune_session()
    session = source.session

    max_query_pages = 5000 if site_golden else 1000
    min_image_id: Optional[int] = None
    q_tags = []
    image_id_lower_bound: int = 7306660

    def _iter_items():
        nonlocal min_image_id, q_tags
        no_item_cnt = 0
        page = 1
        while True:
            if page > max_query_pages:
                page = 1
                assert min_image_id is not None
                q_tags = [f'id:<{min_image_id}']

            logging.info(f'Requesting for newest page {page!r} ...')
            resp = srequest(session, 'GET', 'https://danbooru.donmai.us/posts.json', params={
                'limit': '200',
                'page': str(page),
                "tags": ' '.join(q_tags),
            }, auth=source.auth)
            has_new_items = False
            for pitem in resp.json():
                if pitem['id'] not in exist_ids:
                    yield pitem
                    if pitem.get('file_url'):
                        has_new_items = True
                if not min_image_id or pitem['id'] < min_image_id:
                    min_image_id = pitem['id']

            if has_new_items:
                no_item_cnt = 0
            else:
                no_item_cnt = no_item_cnt + 1
            if sync_mode and no_item_cnt >= 10:
                return
            if min_image_id and min_image_id < image_id_lower_bound:
                return

            page += 1

    for item in _iter_items():
        if start_time + max_time_limit < time.time():
            break
        if item['id'] in exist_ids:
            logging.info(f'Post {item["id"]!r} already crawled, skipped.')
            continue
        if not item.get('file_url'):
            logging.info(f'Empty url post {item["id"]!r}, maybe you need a golden account to scrape that, skipped.')
            continue

        logging.info(f'Post {item["id"]!r} confirmed!')
        del item['media_asset']
        if item.get('file_url'):
            item['mimetype'], _ = mimetypes.guess_type(item['file_url'])
        else:
            item['mimetype'] = None
        records.append(item)
        exist_ids.add(item['id'])
        has_update = True
        _deploy()

    _deploy(force=True)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_DB_N'],
        max_time_limit=5.7 * 60 * 60,
        upload_time_span=30,
        deploy_span=5 * 60,
        sync_mode=True,
        site_username=os.environ.get('DANBOORU_USERNAME'),
        site_apikey=os.environ.get('DANBOORU_APITOKEN'),
        site_golden=True,
    )
