import glob
import html
import json
import math
import mimetypes
import os
import re
import time
from itertools import chain
from typing import Optional, List

import numpy as np
import pandas as pd
import requests
from ditk import logging
from hbutils.collection import unique, nested_map
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hbutils.testing import disable_output
from hfutils.cache import delete_detached_cache
from hfutils.operate import upload_directory_as_directory, download_archive_as_directory, get_hf_client, get_hf_fs
from hfutils.utils import number_to_tag
from natsort import natsorted
from pyrate_limiter import Rate, Duration, Limiter
from tqdm import tqdm
from waifuc.utils import srequest

from .tags import _get_session, _get_tags_by_page

mimetypes.add_type('image/webp', '.webp')
__site_url__ = 'https://gelbooru.com'
_TAG_TYPES = {
    -1: 'unknown',
    0: 'general',
    1: 'artist',
    3: 'copyright',
    4: 'character',
    5: 'metadata',
    6: 'deprecated',
}
_NAME_TO_TAG_TYPE = {
    value: key
    for key, value in _TAG_TYPES.items()
    if key >= 0
}
_NAME_TO_TAG_TYPE['tag'] = _NAME_TO_TAG_TYPE['general']

_ARCHIVE_REPO = 'NebulaeWis/gelbooru_images_fornarugo'
_INDEX_REPO = 'narugo/gelbooru_images_fornarugo_index'


def _get_posts_by_page(p: Optional[int] = None, id_: Optional[int] = None,
                       tags: List[str] = None, max_tries: int = 5, session=None,
                       user_id: Optional[str] = None, api_key: Optional[str] = None,
                       limiter: Optional[Limiter] = None):
    session = session or _get_session()
    if tags is not None:
        if p is not None:
            logging.info(f'Getting page {p} for posts with tags {tags!r} ...')
        else:
            logging.info(f'Getting posts with tags {tags!r} ...')
    elif p is not None:
        logging.info(f'Getting page {p} for posts ...')
    else:
        logging.info(f'Getting post {id_} ...')

    tries = 0
    resp = None
    while True:
        try:
            params = {
                'page': 'dapi',
                's': 'post',
                'q': 'index',
                'limit': '100',
                'json': '1',
            }
            if p is not None:
                params['pid'] = str(p)
            if id_ is not None:
                params['id'] = str(id_)
            if tags:
                params['tags'] = ' '.join(tags)
            if user_id:
                params['user_id'] = user_id
            if api_key:
                params['api_key'] = api_key
            if limiter is not None:
                limiter.try_acquire('API access')
            resp = srequest(session, 'GET', f'{__site_url__}/index.php', params=params)
        except requests.exceptions.RequestException as err:
            if err.response.status_code == 403:
                tries += 1
                if tries <= max_tries:
                    sleep_time = 2.0 ** tries
                    logging.warning(f'Request error for tag api, retry {tries}/{max_tries}, '
                                    f'sleep for {sleep_time:.1f}s - {err!r}')
                    time.sleep(sleep_time)
                    continue

            raise
        else:
            break

    if 'post' in resp.json():
        return resp.json()['post']
    else:
        return []


def sync(repository: str, max_time_limit: float = 50 * 60, upload_time_span: float = 30,
         deploy_span: float = 5 * 60, sync_mode: bool = False, no_recent: float = 60 * 60 * 24 * 15,
         max_part_rows: int = 2500000, sync_from_archives: bool = True,
         user_id: Optional[str] = None, api_key: Optional[str] = None, access_interval: Optional[float] = None):
    delete_detached_cache()
    start_time = time.time()
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    if access_interval is not None:
        r_rate = Rate(1, int(math.ceil(Duration.SECOND * access_interval)))
        r_limiter = Limiter(r_rate, max_delay=1 << 32)
    else:
        r_limiter = None

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

    if hf_fs.exists(f'datasets/{repository}/exist_ids.json'):
        exist_ids = json.loads(hf_fs.read_text(f'datasets/{repository}/exist_ids.json'))
        pre_ids = set(exist_ids)
        exist_ids = set(exist_ids)
    else:
        pre_ids = set()
        exist_ids = set()

    if hf_fs.exists(f'datasets/{repository}/scanned_archives.json'):
        scanned_archives = json.loads(hf_fs.read_text(f'datasets/{repository}/scanned_archives.json'))
        scanned_archives = set(scanned_archives)
    else:
        scanned_archives = set()

    if hf_fs.exists(f'datasets/{repository}/meta.json'):
        meta_info = json.loads(hf_fs.read_text(f'datasets/{repository}/meta.json'))
        max_blind_id = meta_info['max_blind_id']
        tag_mapping = dict(meta_info.get('tag_mapping') or {})
    else:
        max_blind_id = 9843600
        tag_mapping = {}

    if hf_fs.glob(f'datasets/{repository}/tables/gelbooru-*.parquet'):
        last_path = natsorted(hf_fs.glob(f'datasets/{repository}/tables/gelbooru-*.parquet'))[-1]
        last_file_name = os.path.basename(last_path)
        last_rel_file = os.path.relpath(last_path, f'datasets/{repository}')
        current_ptr = int(os.path.splitext(last_file_name)[0].split('-')[-1])
        df_record = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename=last_rel_file,
        ))
        df_record = df_record.replace(np.NaN, None)
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
    ))
    df_origin_tags = df_origin_tags.replace(np.NaN, None)
    d_origin_tags = {item['name']: item for item in df_origin_tags.to_dict('records')}

    if hf_fs.exists(f'datasets/{repository}/tags.parquet'):
        df_tags = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='tags.parquet',
        ))
        df_tags = df_tags.replace(np.NaN, None)
        df_tags['ambiguous'] = list(map(lambda x: bool(eval(x) if isinstance(x, str) else x), df_tags['ambiguous']))
        df_tags['name'] = list(map(html.unescape, df_tags['name']))
        d_tags = {item['name']: item for item in df_tags.to_dict('records')}
    else:
        d_tags = {}

    session = _get_session()
    _last_update, has_update = None, False
    _total_count = len(exist_ids)

    def _deploy(force=False):
        nonlocal _last_update, has_update, _total_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            parquet_file = os.path.join(td, 'tables', f'gelbooru-{current_ptr}.parquet')
            os.makedirs(os.path.dirname(parquet_file), exist_ok=True)
            df_records = pd.DataFrame(records)
            df_records = df_records.sort_values(by=['id'], ascending=[False])
            df_records.to_parquet(parquet_file, engine='pyarrow', index=False)

            tags_file = os.path.join(td, 'tags.parquet')
            df_tags = pd.DataFrame(list(d_tags.values()))
            df_tags = df_tags.sort_values(['count', 'type'], ascending=[False, True])
            df_tags.to_parquet(tags_file, engine='pyarrow', index=False)

            with open(os.path.join(td, 'exist_ids.json'), 'w') as f:
                json.dump(sorted(exist_ids), f)
            with open(os.path.join(td, 'scanned_archives.json'), 'w') as f:
                json.dump(natsorted(scanned_archives), f)
            with open(os.path.join(td, 'meta.json'), 'w') as f:
                json.dump({
                    'max_blind_id': max_blind_id,
                    'tag_mapping': tag_mapping,
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
                print('- gelbooru', file=f)
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
                message=f'Add {plural_word(len(exist_ids) - _total_count, "new record")} into index',
            )
            has_update = False
            _last_update = time.time()
            _total_count = len(exist_ids)

    def _yield_from_archives():
        if not sync_from_archives:
            return

        nonlocal has_update
        json_full_paths = natsorted(hf_fs.glob(f'datasets/{_ARCHIVE_REPO}/**/*.tar'))
        for json_path in tqdm(json_full_paths, desc='Archives'):
            archive_file = os.path.relpath(json_path, f'datasets/{_ARCHIVE_REPO}')
            if archive_file in scanned_archives:
                logging.info(f'Archive {archive_file!r} scanned, skipped.')
                continue

            logging.info(f'Get posts from archives {archive_file!r} ...')
            with TemporaryDirectory() as td:
                with disable_output():
                    download_archive_as_directory(
                        repo_id=_ARCHIVE_REPO,
                        repo_type='dataset',
                        file_in_repo=archive_file,
                        local_directory=td,
                    )

                files = natsorted(glob.glob(os.path.join(td, '*.json')))
                logging.info(f'{plural_word(len(files), "json file")} found in {archive_file!r} ...')
                for file in files:
                    with open(file, 'r') as f:
                        item = nested_map(
                            lambda x: x if not isinstance(x, float) or not math.isnan(x) else None,
                            json.load(f),
                        )

                    del item['Index']
                    if item['id'] not in exist_ids:
                        yield item

            scanned_archives.add(archive_file)
            has_update = True

    min_newest_id: Optional[int] = None

    def _yield_from_newest():
        nonlocal min_newest_id
        should_quit = False
        no_item_cnt = 0
        while True:
            if min_newest_id is not None:
                extra_tags = [f'id:<{min_newest_id}']
            else:
                extra_tags = []
            for pid in range(0, 201):
                has_item = False
                for item in _get_posts_by_page(p=pid, tags=extra_tags, session=session,
                                               limiter=r_limiter, user_id=user_id, api_key=api_key):
                    if item['id'] not in exist_ids:
                        yield item
                        has_item = True

                    if min_newest_id is None or item['id'] < min_newest_id:
                        min_newest_id = item['id']
                    if item['id'] < max_blind_id:
                        should_quit = True
                        break

                if has_item:
                    no_item_cnt = 0
                else:
                    no_item_cnt += 1
                    if no_item_cnt >= 10:
                        should_quit = True

                if should_quit:
                    break

            if should_quit:
                break

    def _yield_from_blind():
        # 9843600
        nonlocal max_blind_id, has_update
        i = max_blind_id
        while min_newest_id is None or i < min_newest_id:
            if i not in exist_ids:
                yield from _get_posts_by_page(id_=i, session=session,
                                              limiter=r_limiter, user_id=user_id, api_key=api_key)

            i += 1
            max_blind_id = i
            has_update = True

    if sync_mode:
        source = chain(
            _yield_from_newest(),
            # _yield_from_blind(),
        )
    else:
        source = chain(
            _yield_from_archives(),
            _yield_from_newest(),
            _yield_from_blind(),
        )
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
        if 'more_info_from_page' in item:
            tag_types = item.pop('more_info_from_page')
        else:
            tag_types = None

        tag_types_from_page = {}
        if tag_types:
            for type_, l_tags in tag_types.items():
                if l_tags and isinstance(l_tags, list):
                    for tag in l_tags:
                        tag_types_from_page[tag.replace(' ', '_')] = _NAME_TO_TAG_TYPE[type_]

        current_tags = []
        for tag in tags:
            if tag not in d_tags:
                if tag in d_origin_tags:
                    d_tags[tag] = {**d_origin_tags[tag], 'count': 0}
                else:
                    d_tags[tag] = {
                        'id': -1,
                        'type': tag_types_from_page.get(tag, -1),
                        'name': tag,
                        'count': 0,
                        'ambiguous': False
                    }

            count = d_tags[tag]['count']
            if tag in d_origin_tags:
                d_tags[tag].update(d_origin_tags[tag])
            if d_tags[tag]['id'] < 0:
                tags_info = _get_tags_by_page(name=tag, session=session,
                                              limiter=r_limiter, user_id=user_id, api_key=api_key)
                if tags_info:
                    d_tags[tag].update(tags_info[0])

            origin_tag = tag
            tag = d_tags[tag]['name']
            if origin_tag != tag:
                tag_mapping[origin_tag] = tag
                if tag in d_tags:
                    total_count = d_tags[tag]['count'] + d_tags[origin_tag]['count']
                    d_tags[tag] = d_tags[origin_tag]
                    d_tags[tag]['count'] = total_count
                    del d_tags[origin_tag]
                else:
                    d_tags[tag] = d_tags[origin_tag]
                    del d_tags[origin_tag]
            else:
                if origin_tag in tag_mapping:
                    del tag_mapping[origin_tag]
            if tag in current_tags:
                d_tags[tag]['count'] = count
            else:
                d_tags[tag]['count'] = count + 1
                current_tags.append(tag)

        row = {
            'id': item['id'],
            'width': item['width'],
            'height': item['height'],
            'filename': item['image'],
            'mimetype': mimetype,
            'rating': item['rating'],
            'file_url': item['file_url'],
            **item,
            'tags': ' '.join(['', *current_tags, '']),
            'has_notes': json.loads(item['has_notes']),
            'has_comments': json.loads(item['has_comments']),
            'has_children': json.loads(item['has_children']),
            'scraped_at': time.time(),
        }
        records.append(row)
        exist_ids.add(item['id'])
        has_update = True
        _deploy()

    _deploy(force=True)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_GB'],
        max_time_limit=(60 * 5 + 45) * 60,
        sync_mode=True,
        sync_from_archives=True,
        no_recent=60 * 60 * 24 * 0,
        deploy_span=5 * 60,
        max_part_rows=3000000,
        user_id=os.environ["GELBOORU_USER_ID"],
        api_key=os.environ["GELBOORU_API_KEY"],
    )
