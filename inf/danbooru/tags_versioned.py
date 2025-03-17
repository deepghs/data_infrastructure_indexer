import calendar
import datetime
import json
import os
import time
from datetime import datetime
from datetime import timezone
from typing import Optional

import dateparser
import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import number_to_tag
from huggingface_hub import hf_hub_url
from tqdm import tqdm
from waifuc.source import DanbooruSource
from waifuc.utils import srequest


def generate_calendar_markdown(f, minx, url_function, last_month_count: int = 12):
    today = datetime.now()
    current_year = today.year
    current_month = today.month
    current_day = today.day

    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    min_year, min_month = minx

    for i in range(0, last_month_count):
        month = current_month - i
        year = current_year
        while month <= 0:
            month += 12
            year -= 1

        if (year, month) < (min_year, min_month):
            continue

        print(f"## {year}-{month:02d}", file=f)
        print(f'', file=f)
        cal = calendar.monthcalendar(year, month)

        df = pd.DataFrame(cal, columns=weekday_names)
        for row in range(len(df)):
            for col in range(7):
                day = df.iloc[row, col]
                if day == 0:
                    df.iloc[row, col] = ""
                else:
                    if year == current_year and month == current_month and day > current_day:
                        df.iloc[row, col] = ""
                    else:
                        url = url_function((year, month, day))
                        if url:
                            df.iloc[row, col] = f"[{day}]({url})"
                        else:
                            df.iloc[row, col] = str(day)

        print(df.to_markdown(index=False), file=f)
        print(f'', file=f)


def get_year_and_quarter(time=None):
    if time is None:
        time = datetime.now()
    utc_time = time.astimezone(timezone.utc)
    year = utc_time.year
    month = utc_time.month
    day = utc_time.day
    return f'{year}{month:02d}', f'{year}{month:02d}{day:02d}', (year, month, day)


def sync(repository: str, site_username: Optional[str] = None, site_apikey: Optional[str] = None, ):
    delete_detached_cache()
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

    if site_apikey and site_username:
        logging.info(f'Initializing session with username {site_username!r} and api_key {site_apikey!r} ...')
        source = DanbooruSource(['1girl'], api_key=site_apikey, username=site_username)
    else:
        logging.info('Initializing session without any API key ...')
        source = DanbooruSource(['1girl'])
    source._prune_session()
    session = source.session

    if hf_client.file_exists(
            repo_id=repository,
            repo_type='dataset',
            filename='meta.json',
    ):
        meta_info = json.loads(hf_fs.read_text(f'datasets/{repository}/meta.json'))
        exist_days = {tuple(x) for x in meta_info['exist_days']}
        exist_months = {(x[0], x[1]) for x in meta_info['exist_days']}
    else:
        exist_days = set()
        exist_months = set()

    ym, yd, (year, month, day) = get_year_and_quarter()
    logging.info(f'Current mark: {ym!r} - {yd!r} - {(year, month, day)!r}')

    def _get_ch_tags(pid: int):
        logging.info(f'Accessing tags on page #{pid!r} ...')
        resp = srequest(session, 'GET', 'https://danbooru.donmai.us/tags.json', params={
            'search[category]': '4',
            'search[post_count]': '>=0',
            'only': 'id,name,post_count,category,created_at,updated_at,is_deprecated,wiki_page[id,title,other_names],'
                    'antecedent_implications[consequent_name],consequent_implications[antecedent_name]',
            'limit': '1000',
            'page': str(pid),
        })
        return resp.json()

    l, r = 1, 2
    while _get_ch_tags(r):
        l <<= 1
        r <<= 1

    while l < r:
        m = (l + r + 1) // 2
        if _get_ch_tags(m):
            l = m
        else:
            r = m - 1

    max_page = l
    logging.info(f'Max page: {max_page!r}.')

    d_new_records = {}
    for page_id in tqdm(range(1, max_page + 1), desc='Iterating Pages'):
        call_time = time.time()
        for item in _get_ch_tags(page_id):
            wiki_item = item.get('wiki_page') or {}
            d_new_records[item['id']] = {
                'id': item['id'],
                'name': item['name'],
                'post_count': item['post_count'],
                'category': item['category'],
                'is_deprecated': item['is_deprecated'],
                'wiki_id': wiki_item.get('id'),
                'wiki_title': wiki_item.get('title'),
                'other_names': list(wiki_item.get('other_names') or []),
                'antecedents': [x['consequent_name'] for x in item['antecedent_implications']],
                'consequents': [x['antecedent_name'] for x in item['consequent_implications']],
                'created_at': dateparser.parse(item['created_at']).timestamp(),
                'updated_at': dateparser.parse(item['updated_at']).timestamp(),
                'scraped_at': call_time,
            }

    df_current = pd.DataFrame(list(d_new_records.values()))
    df_current = df_current.sort_values(by=['id'], ascending=[False])
    logging.info(f'Current scraping result:\n{df_current}')

    with TemporaryDirectory() as upload_dir:
        last_tags_file = os.path.join(upload_dir, 'current.parquet')
        df_current.to_parquet(last_tags_file, index=False)

        history_file = os.path.join(upload_dir, 'history', ym, f'{yd}.parquet')
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        df_current.to_parquet(history_file, index=False)

        exist_days.add((year, month, day))
        exist_months.add((year, month))
        min_year, min_month = min(exist_months)
        with open(os.path.join(upload_dir, 'meta.json'), 'w') as f:
            json.dump({
                'exist_days': sorted(exist_days),
                'last_date': {
                    'year': year,
                    'month': month,
                    'day': day,
                },
                'ym': ym,
                'yd': yd,
            }, f)

        with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
            print('---', file=f)
            print('license: other', file=f)
            print('language:', file=f)
            print('- en', file=f)
            print('- ja', file=f)
            print('tags:', file=f)
            print('- art', file=f)
            print('- anime', file=f)
            print('- text', file=f)
            print('- tag', file=f)
            print('- not-for-all-audiences', file=f)
            print('size_categories:', file=f)
            print(f'- {number_to_tag(len(df_current))}', file=f)
            print('annotations_creators:', file=f)
            print('- no-annotation', file=f)
            print('source_datasets:', file=f)
            print('- danbooru', file=f)
            print('---', file=f)
            print('', file=f)

            print(f'# Danbooru Character Tags History', file=f)
            print(f'', file=f)
            print(f'This is the historical repository of the danbooru character tags.', file=f)
            print(f'', file=f)
            print(f'You can analysis their post count history with these databases.', file=f)
            print(f'', file=f)

            def _get_url(d):
                (year, month, day) = d
                if (year, month, day) in exist_days:
                    return hf_hub_url(
                        repo_id=repository,
                        repo_type='dataset',
                        filename=f'history/{year}{month:02d}/{year}{month:02d}{day:02d}.parquet',
                    )
                else:
                    return None

            generate_calendar_markdown(
                f=f,
                minx=(min_year, min_month),
                url_function=_get_url,
                last_month_count=12,
            )

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=upload_dir,
            path_in_repo='.',
            message=f'Add #{yd}, with {plural_word(len(df_current), "tag")} in total, '
                    f'at month {ym!r}',
            hf_token=os.environ['HF_TOKEN'],
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_DB_CH_TAGS_VRAW'],
        site_username=os.environ.get('DANBOORU_USERNAME'),
        site_apikey=os.environ.get('DANBOORU_APITOKEN'),
    )
