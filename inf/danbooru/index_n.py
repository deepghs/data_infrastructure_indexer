import math
import mimetypes
import os
import time
from typing import Optional

import click
import numpy as np
import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.utils import number_to_tag
from pyrate_limiter import Rate, Duration, Limiter
from waifuc.source import DanbooruSource

from inf.utils.duration import duration_type
from inf.utils.safe import safe_hf_hub_download, safe_upload_directory_as_directory
from inf.utils.session import srequest

mimetypes.add_type('image/webp', '.webp')


def sync(repository: str, upload_time_span: float = 30, deploy_span: float = 5 * 60,
         max_time_limit: Optional[float] = 50 * 60, sync_mode: bool = False,
         site_username: Optional[str] = None, site_apikey: Optional[str] = None,
         site_golden: bool = False, start_from_id: Optional[int] = None):
    """Sync Danbooru post metadata into the target Hugging Face dataset repository."""
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
        df_records = pd.read_parquet(safe_hf_hub_download(
            hf_client,
            repo_id=repository,
            repo_type='dataset',
            filename='records.parquet'
        )).replace(np.NaN, None)
        d_records = {item['id']: item for item in df_records.to_dict('records')}
    else:
        d_records = {}

    _last_update, has_update = None, False
    _total_count = len(d_records)

    def _deploy(force=False):
        nonlocal _last_update, has_update, _total_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            table_parquet_file = os.path.join(td, 'records.parquet')
            os.makedirs(os.path.dirname(table_parquet_file), exist_ok=True)
            df_records = pd.DataFrame(list(d_records.values()))
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
                print(f'{plural_word(len(df_records), "record")} in total. '
                      f'Only {plural_word(len(df_records_shown), "record")} shown.', file=f)
                print(f'', file=f)
                print(df_records_shown.to_markdown(index=False), file=f)
                print(f'', file=f)

            limiter.try_acquire('hf upload limit')
            safe_upload_directory_as_directory(
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
    if start_from_id is not None:
        min_image_id = start_from_id
        q_tags = [f'id:<{min_image_id}']
        logging.info(f'Start from explicit image id boundary {start_from_id!r}.')

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
                if pitem['id'] not in d_records or d_records[pitem['id']]['file_url'] != pitem.get('file_url'):
                    yield pitem
                    if pitem.get('file_url'):
                        has_new_items = True
                if not min_image_id or pitem['id'] < min_image_id:
                    min_image_id = pitem['id']

            if has_new_items:
                no_item_cnt = 0
            else:
                no_item_cnt = no_item_cnt + 1
            if sync_mode and no_item_cnt >= 100:
                return
            if min_image_id and min_image_id < image_id_lower_bound:
                return

            page += 1

    for item in _iter_items():
        if max_time_limit is not None and start_time + max_time_limit < time.time():
            break
        # if item['id'] in exist_ids:
        #     logging.info(f'Post {item["id"]!r} already crawled, skipped.')
        #     continue
        if not item.get('file_url'):
            logging.info(f'Empty url post {item["id"]!r}, maybe you need a golden account to scrape that, skipped.')
            continue

        logging.info(f'Post {item["id"]!r} confirmed!')
        if item['id'] in d_records:
            logging.info(f'Post {item["id"]!r} already exist, but has file_url changed.')
        del item['media_asset']
        if item.get('file_url'):
            item['mimetype'], _ = mimetypes.guess_type(item['file_url'])
        else:
            item['mimetype'] = None
        d_records[item['id']] = item
        has_update = True
        _deploy()

    _deploy(force=True)


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Sync Danbooru posts into the target Hugging Face dataset repository. '
         'The command pulls upstream post metadata in batches, refreshes local parquet snapshots, '
         'and periodically deploys repository artifacts during the run.',
)
@click.option(
    '-r', '--repository',
    type=str,
    envvar='REMOTE_REPOSITORY_DB_N',
    required=True,
    show_envvar=True,
    help='Target Hugging Face dataset repository to read from and write to.',
)
@click.option(
    '-u', '--upload-time-span',
    type=duration_type(),
    default=30,
    show_default=True,
    help='Minimum interval between upload batches.',
)
@click.option(
    '-d', '--deploy-span',
    type=duration_type(),
    default=5 * 60,
    show_default=True,
    help='Minimum interval between deploy or upload commits.',
)
@click.option(
    '-m', '--max-time-limit',
    type=duration_type(allow_none=True),
    default=5.7 * 60 * 60,
    show_default=True,
    help='Stop the sync after this total runtime. Use none or unlimited to disable the limit.',
)
@click.option(
    '-s', '--sync-mode/--no-sync-mode',
    default=True,
    show_default=True,
    help='Continue incremental sync behavior instead of a fresh rebuild.',
)
@click.option(
    '-U', '--site-username',
    type=str,
    envvar='DANBOORU_USERNAME',
    default=None,
    show_envvar=True,
    help='Site username used for authenticated upstream requests.',
)
@click.option(
    '-A', '--site-apikey',
    type=str,
    envvar='DANBOORU_APITOKEN',
    default=None,
    show_envvar=True,
    help='Site API key used for authenticated upstream requests.',
)
@click.option(
    '-g', '--site-golden/--no-site-golden',
    default=True,
    show_default=True,
    help='Enable the Danbooru golden-account request mode.',
)
@click.option(
    '-i', '--start-from-id',
    type=int,
    default=None,
    help='Start scanning below this explicit Danbooru post ID boundary.',
)
def cli(repository: str, upload_time_span: float, deploy_span: float, max_time_limit: Optional[float],
        sync_mode: bool, site_username: Optional[str], site_apikey: Optional[str], site_golden: bool,
        start_from_id: Optional[int]):
    logging.try_init_root(logging.INFO)
    return sync(
        repository=repository,
        upload_time_span=upload_time_span,
        deploy_span=deploy_span,
        max_time_limit=max_time_limit,
        sync_mode=sync_mode,
        site_username=site_username,
        site_apikey=site_apikey,
        site_golden=site_golden,
        start_from_id=start_from_id,
    )


if __name__ == '__main__':
    cli()
