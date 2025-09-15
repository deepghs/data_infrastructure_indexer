import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Optional

import cloudscraper
import pandas as pd
import xmltodict
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory, get_hf_fs, get_hf_client
from pyquery import PyQuery as pq
from pyrate_limiter import Rate, Limiter, Duration
from tqdm import tqdm
from waifuc.utils import srequest

__site_url__ = 'https://rule34.xxx'

_RATE = Rate(60, int(math.ceil(Duration.SECOND * 60)))
_LIMITER = Limiter(_RATE, max_delay=1 << 32)


def _get_session():
    return cloudscraper.create_scraper()
    # return get_requests_session()


def _get_tags_by_page(p: int, user_id: Optional[str] = None, api_key: Optional[str] = None, session=None):
    session = session or _get_session()
    logging.info(f'Getting page {p} for tags ...')
    params = {
        'page': 'dapi',
        's': 'tag',
        'q': 'index',
        'json': '1',
        'limit': '100',
        'pid': str(p),
    }
    if user_id and api_key:
        params['user_id'] = user_id
        params['api_key'] = api_key
    _LIMITER.try_acquire('api limit')
    resp = srequest(session, 'GET', f'{__site_url__}/index.php', params=params)
    resp.raise_for_status()

    print(resp.text)

    json_data = xmltodict.parse(resp.text)
    if 'tags' not in json_data or 'tag' not in json_data['tags']:
        return None

    data = []
    for item in json_data['tags']['tag']:
        item = {key.lstrip('@'): value for key, value in item.items()}
        item['id'] = int(item['id'])
        item['type'] = int(item['type'])
        item['count'] = int(item['count'])
        item['ambiguous'] = json.loads(item['ambiguous'])
        data.append(item)

    return data


def _get_all_tags(user_id: Optional[str] = None, api_key: Optional[str] = None, session=None):
    session = session or _get_session()
    l, r = 1, 2
    while _get_tags_by_page(r, user_id, api_key, session):
        l, r = l << 1, r << 1

    while l < r:
        m = (l + r + 1) // 2
        if _get_tags_by_page(m, user_id, api_key, session):
            l = m
        else:
            r = m - 1

    max_page_id = l + 5

    data = []
    exist_ids = set()
    lock = Lock()
    page_range = range(0, max_page_id + 1)
    pg_pages = tqdm(total=len(page_range), desc='Tag Pages')
    pg_tags = tqdm(desc='Tags')

    def _scrap_page(pid):
        res = _get_tags_by_page(pid, user_id, api_key, session)
        if not res:
            return
        for item in res:
            with lock:
                if item['id'] not in exist_ids:
                    data.append(item)
                    exist_ids.add(item['id'])
                    pg_tags.update()

        pg_pages.update()

    tp = ThreadPoolExecutor(max_workers=12)
    for i in page_range:
        tp.submit(_scrap_page, i)

    tp.shutdown(wait=True)

    df = pd.DataFrame(data)
    df = df[['id', 'name', 'type', 'count', 'ambiguous']].sort_values(by=['id'], ascending=[False])
    return df


def _get_tag_aliases_by_page(p, user_id: Optional[str] = None, api_key: Optional[str] = None, session=None):
    session = session or _get_session()
    logging.info(f'Getting page {p} for tag aliases ...')
    params = {
        'page': 'alias',
        's': 'list',
        'pid': str(p),
    }
    if user_id and api_key:
        params['user_id'] = user_id
        params['api_key'] = api_key
    _LIMITER.try_acquire('api limit')
    resp = srequest(session, 'GET', f'{__site_url__}/index.php', params=params)
    resp.raise_for_status()

    page = pq(resp.text)
    table = page('#aliases table')
    headers = [item.text().strip().lower() for item in table('th').parent('tr')('th').items()]

    data = []
    for row in table('tr').items():
        if len(list(row('td').items())) == 0:
            continue
        texts = [item('a').text().strip() for item in row('td').items()]
        v = dict(zip(headers, texts))
        data.append((v['alias'], v['to'], v['reason']))

    return data


def _get_all_tag_aliases(user_id: Optional[str] = None, api_key: Optional[str] = None, session=None):
    session = session or _get_session()
    l, r = 1, 2
    while _get_tag_aliases_by_page(r, user_id, api_key, session):
        l, r = l << 1, r << 1

    while l < r:
        m = (l + r + 1) // 2
        if _get_tag_aliases_by_page(m, user_id, api_key, session):
            l = m
        else:
            r = m - 1

    max_page_id = l // 45 + 5

    data = []
    exist_ids = set()
    lock = Lock()
    page_range = range(0, max_page_id + 1)
    pg_pages = tqdm(total=len(page_range), desc='Tag Alias Pages')
    pg_tags = tqdm(desc='Tag Aliases')

    def _scrap_page(pid):
        res = _get_tag_aliases_by_page(pid, user_id, api_key, session)
        if not res:
            return
        for item in res:
            with lock:
                idx = (item[0], item[1])
                row = {'alias': item[0], 'tag': item[1], 'reason': item[2]}
                if idx not in exist_ids:
                    data.append(row)
                    exist_ids.add(idx)
                    pg_tags.update()

        pg_pages.update()

    tp = ThreadPoolExecutor(max_workers=12)
    for i in page_range:
        tp.submit(_scrap_page, i * 45)

    tp.shutdown(wait=True)

    df = pd.DataFrame(data)
    df = df[['alias', 'tag', 'reason']].sort_values(by=['tag', 'alias'], ascending=[True, True])
    return df


def sync(repository: str, user_id: Optional[str] = None, api_key: Optional[str] = None):
    hf_fs = get_hf_fs(hf_token=os.environ['HF_TOKEN'])
    hf_client = get_hf_client(hf_token=os.environ['HF_TOKEN'])

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    session = _get_session()
    df_tags = _get_all_tags(user_id, api_key, session)
    df_tag_aliases = _get_all_tag_aliases(user_id, api_key, session)

    with TemporaryDirectory() as td:
        df_tags.to_parquet(os.path.join(td, 'index_tags.parquet'), index=False)
        df_tag_aliases.to_parquet(os.path.join(td, 'index_tag_aliases.parquet'), index=False)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=td,
            path_in_repo='.',
            message='Create tags index',
            hf_token=os.environ['HF_TOKEN'],
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_RX'],
        user_id=os.environ['RULE34_USER_ID'],
        api_key=os.environ['RULE34_API_KEY'],
    )
