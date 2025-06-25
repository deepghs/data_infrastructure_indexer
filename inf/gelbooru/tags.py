import html
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from threading import Lock
from typing import Optional

import pandas as pd
import requests.exceptions
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory, urlsplit
from hfutils.operate import upload_directory_as_directory, get_hf_client, get_hf_fs
from hfutils.utils import get_requests_session
from pyquery import PyQuery as pq
from pyrate_limiter import Rate, Limiter, Duration
from tqdm import tqdm
from waifuc.utils import srequest

__site_url__ = 'https://gelbooru.com'


def _get_session(proxy_pool: Optional[str] = None):
    session = get_requests_session()
    if proxy_pool:
        logging.info(f'Proxy pool enabled: {proxy_pool}')
        session.proxies.update({
            'http': proxy_pool,
            'https': proxy_pool,
        })
    return session


def _get_tags_by_page(p: Optional[int] = None, name: Optional[str] = None, max_tries: int = 5,
                      session=None, user_id: Optional[str] = None, api_key: Optional[str] = None,
                      limiter: Optional[Limiter] = None):
    session = session or _get_session()
    if p is not None:
        logging.info(f'Getting page {p} for tags ...')
    else:
        logging.info(f'Getting tag {name!r} ...')

    tries = 0
    resp = None
    while True:
        try:
            params = {
                'page': 'dapi',
                's': 'tag',
                'q': 'index',
                'limit': '100',
                'json': '1',
            }
            if p is not None:
                params['pid'] = str(p)
            if name is not None:
                params['name'] = name
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

    if 'tag' in resp.json():
        items = resp.json()['tag']
    else:
        items = []

    for item in items:
        item['name'] = html.unescape(item['name'])
        item['ambiguous'] = bool(item['ambiguous'])

    return items


def _get_all_tags(session=None, user_id: Optional[str] = None, api_key: Optional[str] = None,
                  limiter: Optional[Limiter] = None):
    session = session or _get_session()
    l, r = 1, 2
    while _get_tags_by_page(p=r, session=session, user_id=user_id, api_key=api_key, limiter=limiter):
        l, r = l << 1, r << 1

    while l < r:
        m = (l + r + 1) // 2
        if _get_tags_by_page(p=m, session=session, user_id=user_id, api_key=api_key, limiter=limiter):
            l = m
        else:
            r = m - 1

    max_page_id = l + 5

    data = {}
    lock = Lock()
    page_range = range(0, max_page_id + 1)
    pg_pages = tqdm(total=len(page_range), desc='Tag Pages')
    pg_tags = tqdm(desc='Tags')

    def _scrap_page(pid):
        try:
            res = _get_tags_by_page(p=pid, session=session, user_id=user_id, api_key=api_key, limiter=limiter)
            if not res:
                return
            for item in res:
                with lock:
                    data[item['id']] = item
                    pg_tags.update()
        except Exception as err:
            logging.error(f'Error when scrap tag page {pid!r} - {err!r}.')
            raise
        finally:
            pg_pages.update()

    tp = ThreadPoolExecutor(max_workers=12)
    for i in page_range:
        tp.submit(_scrap_page, i)

    tp.shutdown(wait=True)

    df = pd.DataFrame(list(data.values()))
    df = df[['id', 'name', 'type', 'count', 'ambiguous']].sort_values(by=['id'], ascending=[False])
    return df


def _get_tag_aliases_by_page(p, session=None):
    session = session or _get_session()
    logging.info(f'Getting page {p} for tag aliases ...')
    resp = srequest(session, 'GET', f'{__site_url__}/index.php', params={
        'page': 'alias',
        's': 'list',
        'pid': str((p - 1) * 50),
    })
    resp.raise_for_status()

    page = pq(resp.text)

    page_pg = page('#paginator .pagination')
    paginator_words = set([
        item.text().strip()
        for item in chain(page_pg('a').items(), page_pg('b').items())
    ])
    if str(p) not in paginator_words:
        return []

    table = page('#aliases table')

    data = []
    for row in table('tr').items():
        if len(list(row('td').items())) == 0:
            continue

        first_a, second_a = row('a').items()
        alias_tag = urlsplit(first_a.attr('href')).query_dict['tags']
        tag = urlsplit(second_a.attr('href')).query_dict['tags']
        data.append((alias_tag, tag))

    return data


def _get_all_tag_aliases(session=None):
    session = session or _get_session()
    l, r = 1, 2
    while _get_tag_aliases_by_page(r, session):
        l, r = l << 1, r << 1

    while l < r:
        m = (l + r + 1) // 2
        if _get_tag_aliases_by_page(m):
            l = m
        else:
            r = m - 1

    max_page_id = l + 5

    data = {}
    lock = Lock()
    page_range = range(0, max_page_id + 1)
    pg_pages = tqdm(total=len(page_range), desc='Tag Alias Pages')
    pg_tags = tqdm(desc='Tag Aliases')

    def _scrap_page(pid):
        try:
            res = _get_tag_aliases_by_page(p=pid, session=session)
            if not res:
                return
            for item in res:
                with lock:
                    idx = (item[0], item[1])
                    row = {'alias': item[0], 'tag': item[1]}
                    data[idx] = row
                    pg_tags.update()
        except Exception as err:
            logging.error(f'Error when scrap tag alias page {pid!r} - {err!r}.')
            raise
        finally:
            pg_pages.update()

    tp = ThreadPoolExecutor(max_workers=12)
    for i in page_range:
        tp.submit(_scrap_page, i)

    tp.shutdown(wait=True)

    df = pd.DataFrame(list(data.values()))
    df = df[['alias', 'tag']].sort_values(by=['tag', 'alias'], ascending=[True, True])
    return df


def sync(repository: str, proxy_pool: Optional[str] = None, access_interval: Optional[float] = None,
         user_id: Optional[str] = None, api_key: Optional[str] = None):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    if access_interval is not None:
        rate = Rate(1, int(math.ceil(Duration.SECOND * access_interval)))
        limiter = Limiter(rate, max_delay=1 << 32)
    else:
        limiter = None

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    session = _get_session(proxy_pool=proxy_pool)
    df_tags = _get_all_tags(session=session, limiter=limiter, api_key=api_key, user_id=user_id)
    df_tag_aliases = _get_all_tag_aliases(session=session)

    with TemporaryDirectory() as td:
        df_tags.to_parquet(os.path.join(td, 'index_tags.parquet'), engine='pyarrow', index=False)
        df_tag_aliases.to_parquet(os.path.join(td, 'index_tag_aliases.parquet'), engine='pyarrow', index=False)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=td,
            path_in_repo='.',
            message=f'Create tags index, with {plural_word(len(df_tags), "tag")} '
                    f'and {plural_word(len(df_tag_aliases), "tag alias")}',
            hf_token=os.environ['HF_TOKEN'],
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    # pprint(_get_tag_aliases_by_page(1))
    # pprint(_get_tags_by_page(
    #     p=1,
    #     user_id=os.environ["GELBOORU_USER_ID"],
    #     api_key=os.environ["GELBOORU_API_KEY"],
    # ))
    sync(
        repository=os.environ['REMOTE_REPOSITORY_GB'],
        user_id=os.environ["GELBOORU_USER_ID"],
        api_key=os.environ["GELBOORU_API_KEY"],
    )
