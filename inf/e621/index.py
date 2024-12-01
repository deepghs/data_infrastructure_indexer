import base64
import json
import math
import mimetypes
import os
import time
from itertools import chain
from typing import Optional, List

import httpx
import numpy as np
import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import get_requests_session, number_to_tag
from pyrate_limiter import Rate, Duration, Limiter

mimetypes.add_type('image/webp', '.webp')

_TAG_CATEGORIES = {
    0: 'general',
    1: 'artist',
    3: 'copyright',
    4: 'character',
    5: 'species',
    6: 'invalid',
    7: 'meta',
    8: 'lore',
}
_TAG_INV_CATEGORIES = {value: key for key, value in _TAG_CATEGORIES.items()}


def _parquet_safe(v):
    if isinstance(v, dict):
        if not v:
            return type(v)({'__dummy': None})  # dont save empty dict in parquet
        else:
            return type(v)({
                key: _parquet_safe(value)
                for key, value in v.items()
            })

    elif isinstance(v, (list, tuple)):
        return type(v)([_parquet_safe(item) for item in v])

    else:
        return v


def _get_posts(session: Optional[requests.Session] = None,
               before_id: Optional[int] = None, after_id: Optional[int] = None,
               limit: int = 1000) -> List[dict]:
    session = session or get_requests_session()
    params = {'limit': str(limit)}
    if before_id is not None:
        params['page'] = f'b{before_id}'
    elif after_id is not None:
        params['page'] = f'a{after_id}'
    logging.info(f'Query posts: {params!r} ...')
    resp = session.get('https://e621.net/posts.json', params=params)
    resp.raise_for_status()
    return sorted(resp.json()['posts'], key=lambda x: x['id'])


def sync(repository: str, deploy_span: float = 5 * 60, upload_time_span: float = 30.0,
         max_time_limit: float = 50 * 60, max_part_rows: int = 1000000,
         site_username: Optional[str] = None, site_apikey: Optional[str] = None,
         sync_mode: bool = True):
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

    while True:
        session = get_requests_session()
        if site_username and site_apikey:
            logging.info(f'Initializing session with username {site_username!r} and api_key {site_apikey!r} ...')
            auth_string = f"{site_username}:{site_apikey}"
            auth_bytes = auth_string.encode('utf-8')
            base64_auth = base64.b64encode(auth_bytes).decode('utf-8')
            session.headers.update({
                "Authorization": f"Basic {base64_auth}"
            })
        else:
            logging.info('Initializing session ...')
        logging.info(f'Try user agent: {session.headers["User-Agent"]!r} ...')
        try:
            _ = _get_posts(session)
        except (requests.RequestException, httpx.HTTPStatusError) as err:
            if err.response.status_code == 403:
                continue
            raise
        else:
            break

    if hf_fs.glob(f'datasets/{repository}/e621-*.parquet'):
        last_page_id = sorted([
            int(os.path.splitext(os.path.basename(path))[0].split('-')[-1])
            for path in hf_fs.glob(f'datasets/{repository}/e621-*.parquet')
        ])[-1]

        df = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename=f'e621-{last_page_id}.parquet',
        )).replace(np.nan, None)
        if len(df) >= max_part_rows:
            last_page_id += 1
            records = []
        else:
            records = df.to_dict('records')
    else:
        last_page_id = 1
        records = []

    if hf_client.file_exists(
            repo_id=repository,
            repo_type='dataset',
            filename='meta.json',
    ):
        meta_info = json.loads(hf_fs.read_text(f'datasets/{repository}/meta.json'))
        max_id = meta_info['max_id']
        exist_ids = set(meta_info['exist_ids'])
    else:
        max_id = 0
        exist_ids = set()

    df_origin_tags = pd.read_parquet(hf_client.hf_hub_download(
        repo_id=repository,
        repo_type='dataset',
        filename='index_tags.parquet',
    ))
    d_origin_tags = {(item['category'], item['name']): item for item in df_origin_tags.to_dict('records')}

    if hf_client.file_exists(
            repo_id=repository,
            repo_type='dataset',
            filename='tags.parquet',
    ):
        df_tags = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='tags.parquet',
        ))
        d_tags = {(item['category'], item['name']): item for item in df_tags.to_dict('records')}
    else:
        d_tags = {}

    _last_update, has_update = None, False
    _total_count = len(exist_ids)

    def _deploy(force=False):
        nonlocal _last_update, has_update, _total_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            parquet_file = os.path.join(td, f'e621-{last_page_id}.parquet')
            df_records = pd.DataFrame(records)
            df_records = df_records.sort_values(by=['id'], ascending=[False])
            df_records.to_parquet(parquet_file, engine='pyarrow', index=False)

            tags_file = os.path.join(td, 'tags.parquet')
            df_tags = pd.DataFrame(list(d_tags.values()))
            df_tags = df_tags.sort_values(['count', 'category'], ascending=[False, True])
            df_tags.to_parquet(tags_file, engine='pyarrow', index=False)

            with open(os.path.join(td, 'meta.json'), 'w') as f:
                json.dump({
                    'exist_ids': sorted(exist_ids),
                    'max_id': max_id,
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
                print(f'- {number_to_tag(len(exist_ids))}', file=f)
                print('annotations_creators:', file=f)
                print('- no-annotation', file=f)
                print('source_datasets:', file=f)
                print('- e621', file=f)
                print('---', file=f)
                print('', file=f)

                print('## Records', file=f)
                print(f'', file=f)
                df_records_shown = df_records[:50][
                    ['id', 'width', 'height', 'rating', 'tags', 'file_size', 'mimetype', 'file_url']]
                print(f'{plural_word(len(exist_ids), "record")} in total. '
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
                    df_tags_type = df_tags_type[['id', 'name', 'category', 'count', 'total_count']]
                    df_tags_shown = df_tags_type[:30]
                    df_tags_shown = df_tags_shown.replace(np.NaN, '')
                    print(f'These are the top {plural_word(len(df_tags_shown), "tag")} '
                          f'({plural_word(len(df_tags_type), "tag")} in total) '
                          f'of type `{_TAG_CATEGORIES.get(type_id, type_id)} ({type_id})`:', file=f)
                    print('', file=f)
                    print(df_tags_shown.to_markdown(index=False), file=f)
                    print('', file=f)

            limiter.try_acquire('hf upload limit')
            upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Add {plural_word(len(exist_ids) - _total_count, "new record")} into index',
                hf_token=os.environ['HF_TOKEN'],
            )
            has_update = False
            _last_update = time.time()
            _total_count = len(exist_ids)

    def _iter_up():
        nonlocal max_id, has_update
        if max_id is None:
            return

        while True:
            if start_time + max_time_limit < time.time():
                break
            has_new_items = False
            for item in _get_posts(session=session, after_id=max_id):
                yield item
                has_new_items = True
                if max_id is None or item['id'] > max_id:
                    max_id = item['id']
                    has_update = True

            if not has_new_items:
                break

    def _iter_down():
        min_id = None
        no_new_count = 0
        while True:
            if start_time + max_time_limit < time.time():
                break
            has_new_item, has_item = False, False
            for item in _get_posts(session=session, before_id=min_id):
                if item['id'] not in exist_ids:
                    yield item
                    has_new_item = True
                has_item = True
                if min_id is None or item['id'] < min_id:
                    min_id = item['id']

            if not has_item:
                break
            if not has_new_item:
                no_new_count += 1
            else:
                no_new_count = 0
            if sync_mode and no_new_count >= 10:
                break

    def _iter_posts():
        yield from _iter_down()
        # yield from _iter_up()

    for post in _iter_posts():
        if start_time + max_time_limit < time.time():
            break
        if post['id'] in exist_ids:
            logging.warning(f'Post {post["id"]!r} already exist, skipped.')
            continue
        if not post['file'].get('url'):
            logging.warning(f'Post {post["id"]!r} don\'t have an URL, maybe you need a higher privileged account.')
            continue

        logging.info(f'Post {post["id"]!r} confirmed.')
        file_info = post.pop('file')
        flags_info = post.pop('flags')
        preview_info = post.pop('preview')
        sample_info = post.pop('sample')
        score_info = post.pop('score')
        tags_info = post.pop('tags')
        relationships_info = post.pop('relationships')
        if file_info['url']:
            mimetype, _ = mimetypes.guess_type(file_info['url'])
        else:
            mimetype = None

        row = {
            'id': post['id'],

            'mimetype': mimetype,
            'file_ext': file_info['ext'],
            'width': file_info['width'],
            'height': file_info['height'],
            'md5': file_info['md5'],
            'file_url': file_info['url'],
            'file_size': file_info['size'],
            'rating': post['rating'],

            'tags': list(chain(*tags_info.values())),

            'uploader_id': post['uploader_id'],
            'approver_id': post['approver_id'],

            'score': score_info['total'],
            'up_score': score_info['up'],
            'down_score': score_info['down'],
            'fav_count': post['fav_count'],

            **{f'preview_{key}': value for key, value in preview_info.items()},
            **{f'sample_{key}': value for key, value in sample_info.items()},
            **{f'is_{key}': value for key, value in flags_info.items()},

            **relationships_info,
            **post,

            'created_at': post['created_at'],
            'updated_at': post['updated_at'],
        }
        row['sample_alternates'] = _parquet_safe(row['sample_alternates'])
        records.append(row)

        for key, values in tags_info.items():
            for value in values:
                category_id = _TAG_INV_CATEGORIES[key]
                tag_name = value
                token = (category_id, tag_name)
                if token not in d_tags:
                    if token in d_origin_tags:
                        d_tags[token] = {
                            'id': d_origin_tags[token]['id'],
                            'name': d_origin_tags[token]['name'],
                            'category': d_origin_tags[token]['category'],
                            'count': 0,
                            'total_count': d_origin_tags[token]['post_count']
                        }
                    else:
                        d_tags[token] = {
                            'id': -1,
                            'name': tag_name,
                            'category': category_id,
                            'count': 0,
                            'total_count': -1,
                        }
                if d_tags[token]['id'] == -1 and token in d_origin_tags:
                    d_tags[token]['id'] = d_origin_tags[token]['id']
                if token in d_origin_tags:
                    d_tags[token]['total_count'] = d_origin_tags[token]['post_count']
                d_tags[token]['count'] += 1

        exist_ids.add(row['id'])
        has_update = True
        _deploy(force=False)

    _deploy(force=True)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_E621'],
        deploy_span=5 * 60,
        max_time_limit=5.5 * 60 * 60,
        max_part_rows=1000000,
        site_username=os.environ.get('E621_USERNAME'),
        site_apikey=os.environ.get('E621_APITOKEN'),
        sync_mode=True,
    )
