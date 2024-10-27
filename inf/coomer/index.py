import json
import math
import mimetypes
import os
import time
from typing import Optional

import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.archive import get_archive_type
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import get_requests_session, number_to_tag
from pyrate_limiter import Rate, Limiter, Duration
from tqdm import tqdm
from waifuc.utils import srequest

from .base import _ROOT

mimetypes.add_type('image/webp', '.webp')


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


def _get_file_type(file_path) -> Optional[str]:
    filename = os.path.basename(file_path)
    mimetype, _ = mimetypes.guess_type(filename)
    if mimetype and 'photoshop' in mimetype:
        type_ = 'photoshop'
    elif mimetype and mimetype.startswith('image/'):
        type_ = 'image'
    elif mimetype and mimetype.startswith('audio/'):
        type_ = 'audio'
    elif mimetype and mimetype.startswith('video/'):
        type_ = 'video'
    elif mimetype and mimetype.startswith('text/'):
        type_ = 'text'
    else:
        try:
            get_archive_type(filename)
        except ValueError:
            type_ = None
        else:
            type_ = 'archive'

    return type_


def _get_posts(service: str, uid: str, session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    offset = 0
    while True:
        logging.info(f'Searching {service} #{uid}, offset: {offset} ...')
        resp = srequest(
            session, 'GET', f'{_ROOT}/api/v1/{service}/user/{uid}',
            params={'o': str(offset)},
            raise_for_status=False,
        )
        if resp.status_code in {400, 404}:
            break
        else:
            resp.raise_for_status()

        try:
            lst = resp.json()
        except requests.exceptions.JSONDecodeError as err:
            logging.warning(f'JSON Error - {err}, retry.')
            continue

        resp.raise_for_status()
        yield from lst
        if not lst:
            break
        offset += len(lst)


def sync(repository: str, deploy_span: float = 5 * 60, upload_time_span: float = 30.0,
         max_time_limit: float = 50 * 60, max_posts_count: int = 2000000, proxy_pool: Optional[str] = None):
    start_time = time.time()
    delete_detached_cache()
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

    if hf_fs.exists(f'datasets/{repository}/meta.json'):
        meta_info = json.loads(hf_fs.read_text(f'datasets/{repository}/meta.json'))
        exist_ids = set(meta_info['exist_ids'])
        d_last_updates = meta_info['d_last_updates']
        max_file_id = meta_info['max_file_id']
        d_types = meta_info['d_types']
    else:
        exist_ids = set()
        d_last_updates = {}
        max_file_id = 0
        d_types = {}

    if hf_fs.glob(f'datasets/{repository}/pages/*/posts.parquet'):
        current_page_id = sorted([
            int(os.path.basename(os.path.dirname(path)))
            for path in hf_fs.glob(f'datasets/{repository}/pages/*/posts.parquet')
        ])[-1]

        df_posts = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename=f'pages/{current_page_id}/posts.parquet',
        ))
        df_attachments = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename=f'pages/{current_page_id}/attachments.parquet',
        ))
        if len(df_posts) >= max_posts_count:
            current_page_id += 1
            records = []
            attachments = []
        else:
            records = df_posts.to_dict('records')
            attachments = df_attachments.to_dict('records')
    else:
        current_page_id = 1
        records = []
        attachments = []

    df_creators = pd.read_parquet(hf_client.hf_hub_download(
        repo_id=repository,
        repo_type='dataset',
        filename='creators.parquet',
    ))
    df_creators = df_creators.sort_values(by=['updated'], ascending=[False])
    df_creators = df_creators[df_creators['service'] != 'discord']

    session = get_requests_session()
    if proxy_pool:
        logging.info(f'Proxy pool enabled: {proxy_pool}')
        session.proxies.update({
            'http': proxy_pool,
            'https': proxy_pool,
        })

    _last_update, has_update = None, False
    _total_post_count, _total_file_count = len(exist_ids), max_file_id

    def _deploy(force=False):
        nonlocal _last_update, has_update, _total_post_count, _total_file_count

        if not has_update:
            return
        if len(attachments) < 1:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            parquet_file = os.path.join(td, 'pages', str(current_page_id), f'posts.parquet')
            os.makedirs(os.path.dirname(parquet_file), exist_ok=True)
            df_records = pd.DataFrame(records)
            df_records = df_records.sort_values(by=['added'], ascending=[False])
            df_records.to_parquet(parquet_file, engine='pyarrow', index=False)

            attachments_parquet_file = os.path.join(td, 'pages', str(current_page_id), 'attachments.parquet')
            os.makedirs(os.path.dirname(attachments_parquet_file), exist_ok=True)
            df_attachments = pd.DataFrame(attachments)
            df_attachments = df_attachments.sort_values(by=['id'], ascending=[False])
            df_attachments.to_parquet(attachments_parquet_file, engine='pyarrow', index=False)

            with open(os.path.join(td, 'meta.json'), 'w') as f:
                json.dump({
                    'exist_ids': sorted(exist_ids),
                    'd_last_updates': d_last_updates,
                    'max_file_id': max_file_id,
                    'd_types': d_types,
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
                print('- coomer', file=f)
                print('---', file=f)
                print('', file=f)

                print('## Records', file=f)
                print(f'', file=f)
                df_records_shown = df_records[:50][
                    ['id', 'service', 'user', "post_id", 'title', 'tags',
                     'page_url', 'file_count', 'published', 'added']]
                print(f'{plural_word(len(exist_ids), "record")} in total. '
                      f'Only {plural_word(len(df_records_shown), "record")} shown.', file=f)
                print(f'', file=f)
                print(df_records_shown.to_markdown(index=False), file=f)
                print(f'', file=f)

                print('## Files', file=f)
                print(f'', file=f)
                df_file_types = pd.DataFrame([
                    {'type': key, 'count': value}
                    for key, value in d_types.items()
                ])
                df_file_types = df_file_types.sort_values(by=['count', 'type'], ascending=[False, True])
                print(df_file_types.to_markdown(index=False), file=f)
                print(f'', file=f)
                df_attachments_shown = df_attachments[:50][
                    ['id', 'post_id', 'type', 'filename', 'mimetype', 'path']]
                print(f'{plural_word(max_file_id, "file")} in total. '
                      f'Only {plural_word(len(df_attachments_shown), "file")} shown.', file=f)
                print(f'', file=f)
                print(df_attachments_shown.to_markdown(index=False), file=f)
                print(f'', file=f)

            limiter.try_acquire('hf upload limit')
            upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Add {plural_word(len(exist_ids) - _total_post_count, "new record")} '
                        f'and {plural_word(max_file_id - _total_file_count, "new file")} into index',
            )
            has_update = False
            _last_update = time.time()
            _total_post_count = len(exist_ids)
            _total_file_count = max_file_id

    def _yield_posts(user_item: dict):
        nonlocal has_update
        user_token = f'{user_item["service"]}_{user_item["id"]}'
        if user_token not in d_last_updates or d_last_updates[user_token] < user_item['updated']:
            yield from _get_posts(service=user_item["service"], uid=user_item["id"], session=session)
            d_last_updates[user_token] = user_item['updated']
            has_update = True
        else:
            logging.info(f'No updates for user {user_token!r}.')

    def _yield_from_all_users():
        for item in tqdm(df_creators.to_dict('records'), desc='Scanning Users'):
            if start_time + max_time_limit < time.time():
                break
            yield from _yield_posts(item)

    for post_item in _yield_from_all_users():
        if start_time + max_time_limit < time.time():
            break
        id_ = f'{post_item["service"]}/{post_item["user"]}/{post_item["id"]}'
        if id_ in exist_ids:
            logging.info(f'Post {id_!r} already exist, skipped.')
            continue

        logging.info(f'Post {id_!r} confirmed.')
        file_info = post_item.pop('file')
        attachments_info = post_item.pop('attachments')
        if file_info:
            attachments_info = [file_info, *attachments_info]
        embed_info = post_item.pop('embed')
        post_id = post_item.pop('id')
        tags_info = post_item.pop('tags')

        attachment_ids = []
        exist_paths = set()
        for attachment_item in attachments_info:
            if attachment_item['path'] in exist_paths:
                continue
            filename = os.path.basename(attachment_item['name'])
            name_type = _get_file_type(filename)
            url_type = _get_file_type(attachment_item['path'])
            if not name_type and url_type:
                filename = filename + os.path.splitext(attachment_item['path'])[-1]
            mimetype, _ = mimetypes.guess_type(filename)
            type_ = _get_file_type(filename) or 'other'

            max_file_id += 1
            attachment_ids.append(max_file_id)
            attachments.append({
                'id': max_file_id,
                'post_id': id_,
                'type': type_,
                'filename': filename,
                'mimetype': mimetype,
                'name': attachment_item['name'],
                'path': attachment_item['path'],
            })
            d_types[type_] = d_types.get(type_, 0) + 1
            exist_paths.add(attachment_item['path'])

        row = {
            'id': id_,
            'post_id': post_id,
            'user_id': f'{post_item["service"]}_{post_item["user"]}',
            'service': post_item['service'],
            'user': post_item['user'],
            'page_url': f'https://coomer.su/{post_item["service"]}/user/{post_item["user"]}/post/{post_id}',

            'title': post_item['title'],
            'tags': tags_info if isinstance(tags_info, (str, type(None))) else json.dumps(tags_info),
            'content': post_item['content'],
            'captions': post_item['captions'],

            'embed': _parquet_safe(embed_info),
            'attachment_ids': attachment_ids,
            'file_count': len(attachments_info),
            'poll': post_item['poll'],

            **post_item,

            'added': post_item['added'],
            'edited': post_item['edited'],
            'published': post_item['published'],
        }
        records.append(row)
        exist_ids.add(id_)
        has_update = True
        _deploy(force=False)

    _deploy(force=True)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_CMR'],
        deploy_span=5 * 60,
        max_time_limit=0.45 * 60 * 60,
        proxy_pool=os.environ['PP_AO3'],
        # proxy_pool=None,
        max_posts_count=350000,
    )
