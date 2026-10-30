"""Index sync for asiaon.top.

Ported from the pyskeb prototype with the published layout left exactly as it was: the same
``posts.parquet`` / ``tags.parquet`` / ``categories.parquet`` / ``README.md`` in the same dataset
repository, with every existing row carried over untouched. A run reads the published tables, walks
the listing from the newest page, fetches the detail of only the posts it has not stored, and writes
the tables back whole.

Posts are keyed by the site's own database id (``dbid``). The listing is sixteen posts a page, newest
first. The walk stops once ``max_empty_pages`` consecutive pages carry nothing new, or when a page
comes back empty, which is the end of the listing; ``max_empty_pages`` of 0 disables the first rule and
walks the whole listing, which is how a full rebuild is done.

Download links are stored twice, as the ``download_urls`` struct and as one ``<host>_download_url``
column per host, so a host the site starts using appears as a new column on the next run.
"""
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
from requests import HTTPError

from inf.utils.duration import duration_type
from inf.utils.safe import configure_hf_http_backend, safe_hf_hub_download, safe_upload_directory_as_directory
from .client import AOClient


def _read_table(hf_client, hf_fs, repository: str, filename: str) -> Optional[pd.DataFrame]:
    if not hf_fs.exists(f'datasets/{repository}/{filename}'):
        return None
    df = pd.read_parquet(safe_hf_hub_download(
        hf_client,
        repo_id=repository,
        repo_type='dataset',
        filename=filename,
    ))
    # Parquet nulls come back as NaN; the rest of the pipeline expects None.
    return df.astype(object).where(df.notna(), None)


def _with_sorted_download_urls(posts: list) -> list:
    """
    Give every post's ``download_urls`` the same keys, in alphabetical order.

    The column is an Arrow struct whose field order comes from the order keys are first met, and the
    published table has them alphabetical. Rows added in a run carry only the hosts they use, so
    without this the new rows would put their own keys first and reorder the struct.
    """
    keys = sorted({key for post in posts for key in (post.get('download_urls') or {})})
    return [{**post, 'download_urls': {key: (post.get('download_urls') or {}).get(key) for key in keys}}
            for post in posts]


def _count_terms(terms: dict, items: list) -> list:
    """Count each term of a post into ``terms`` and return the post's term ids."""
    ids = []
    for item in items:
        if item['id'] not in terms:
            terms[item['id']] = {**item, 'count': 0}
        terms[item['id']]['count'] += 1
        terms[item['id']]['total_count'] = item['total_count']
        ids.append(item['id'])
    return ids


def _write_readme(file: str, df_posts: pd.DataFrame, df_tags: pd.DataFrame, df_categories: pd.DataFrame):
    with open(file, 'w') as f:
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
        print('- photo', file=f)
        print('- anime', file=f)
        print('- not-for-all-audiences', file=f)
        print('size_categories:', file=f)
        print(f'- {number_to_tag(len(df_posts))}', file=f)
        print('source_datasets:', file=f)
        print('- asiaon', file=f)
        print('---', file=f)
        print('', file=f)

        print('## Records', file=f)
        print('', file=f)
        df_posts_shown = df_posts[:50][[
            'dbid', 'id', 'title', 'page_url', 'tags', 'categories',
            'resolution', 'model', 'file_size_str', 'unzip_password', 'created_at'
        ]]
        df_posts_shown = df_posts_shown.replace(np.nan, '')
        print(f'{plural_word(len(df_posts), "record")} in total. '
              f'Only {plural_word(len(df_posts_shown), "record")} shown.', file=f)
        print('', file=f)
        print(df_posts_shown.to_markdown(index=False), file=f)
        print('', file=f)

        for title, noun, df in [('Tags', 'tag', df_tags), ('Categories', 'category', df_categories)]:
            print(f'## {title}', file=f)
            print('', file=f)
            df_shown = df[:50].replace(np.nan, '')
            print(f'{plural_word(len(df), noun)} in total, {plural_word(len(df_shown), noun)} shown.', file=f)
            print('', file=f)
            print('', file=f)
            print(df_shown.to_markdown(index=False), file=f)
            print('', file=f)


def sync(repository: str, max_time_limit: Optional[float] = 50 * 60, deploy_span: float = 5 * 60,
         max_empty_pages: int = 10, dry_run: bool = False, max_posts: Optional[int] = None,
         start_page: int = 1):
    start_time = time.time()
    delete_detached_cache()
    # Without a timeout a stalled hub request blocks for about sixteen minutes, which is how a run was lost once.
    configure_hf_http_backend(timeout=300)

    def _timeout():
        return max_time_limit is not None and start_time + max_time_limit < time.time()

    hf_client = get_hf_client(hf_token=os.environ['HF_TOKEN'])
    hf_fs = get_hf_fs(hf_token=os.environ['HF_TOKEN'])
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        raise RuntimeError(f'Repository {repository!r} does not exist, refusing to create a fresh index.')

    df_posts = _read_table(hf_client, hf_fs, repository, 'posts.parquet')
    d_posts = {item['dbid']: item for item in df_posts.to_dict('records')} if df_posts is not None else {}
    df_tags = _read_table(hf_client, hf_fs, repository, 'tags.parquet')
    d_tags = {item['id']: item for item in df_tags.to_dict('records')} if df_tags is not None else {}
    df_categories = _read_table(hf_client, hf_fs, repository, 'categories.parquet')
    d_categories = {item['id']: item for item in df_categories.to_dict('records')} \
        if df_categories is not None else {}
    logging.info(f'Loaded {plural_word(len(d_posts), "post")}, {plural_word(len(d_tags), "tag")} and '
                 f'{plural_word(len(d_categories), "category")} from {repository!r}.')

    client = AOClient()
    client.init()
    state = {'has_update': False, 'last_deploy': None, 'last_count': len(d_posts)}

    def _deploy(force=False):
        if not state['has_update']:
            return
        if not force and state['last_deploy'] is not None and state['last_deploy'] + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            df_posts = pd.DataFrame(_with_sorted_download_urls(list(d_posts.values())))
            df_posts = df_posts.sort_values(by=['dbid'], ascending=[False])
            df_posts.to_parquet(os.path.join(td, 'posts.parquet'), engine='pyarrow', index=False)

            df_tags = pd.DataFrame(list(d_tags.values()))
            if len(df_tags) > 0:
                df_tags = df_tags.sort_values(['count', 'id'], ascending=[False, True])
                df_tags.to_parquet(os.path.join(td, 'tags.parquet'), engine='pyarrow', index=False)

            df_categories = pd.DataFrame(list(d_categories.values()))
            if len(df_categories) > 0:
                df_categories = df_categories.sort_values(['count', 'id'], ascending=[False, True])
                df_categories.to_parquet(os.path.join(td, 'categories.parquet'), engine='pyarrow', index=False)

            _write_readme(os.path.join(td, 'README.md'), df_posts, df_tags, df_categories)

            added = len(d_posts) - state['last_count']
            if dry_run:
                logging.info(f'[dry-run] Would upload {plural_word(added, "new post")}, '
                             f'{plural_word(len(df_posts), "post")} in total.')
            else:
                safe_upload_directory_as_directory(
                    repo_id=repository,
                    repo_type='dataset',
                    local_directory=td,
                    path_in_repo='.',
                    # The upload helper appends ", on <time>" itself, matching the published history.
                    message=f'Add {plural_word(added, "post")}',
                    hf_token=os.environ['HF_TOKEN'],
                )
        state['has_update'] = False
        state['last_deploy'] = time.time()
        state['last_count'] = len(d_posts)

    progress = {'page': start_page}

    def _iter_new_posts():
        page_id, empty_pages = start_page, 0
        while not _timeout():
            progress['page'] = page_id
            items = client.get_lst_page(page_id)
            if not items:
                logging.info(f'Page {page_id} is past the end of the listing.')
                progress['page'] = None
                return
            has_new_item = False
            for item in items:
                if item['dbid'] not in d_posts:
                    has_new_item = True
                    yield item
            empty_pages = 0 if has_new_item else empty_pages + 1
            if max_empty_pages and empty_pages >= max_empty_pages:
                logging.info(f'{plural_word(empty_pages, "page")} in a row with nothing new, stopping.')
                progress['page'] = None
                return
            page_id += 1

    fetched = 0
    for item in _iter_new_posts():
        if _timeout() or (max_posts is not None and fetched >= max_posts):
            break
        if item['dbid'] in d_posts:
            continue

        logging.info(f'Item #{item["dbid"]!r}, ID: {item["id"]!r}, URL: {item["page_url"]!r} found.')
        try:
            record = client.get_info(item['id'])
        except HTTPError as err:
            if err.response.status_code == 404:
                logging.warning(f'404 on item #{item["dbid"]!r}, ID: {item["id"]!r}, URL: {item["page_url"]!r}.')
                continue
            raise
        fetched += 1

        record['categories'] = _count_terms(d_categories, record['categories'])
        record['tags'] = _count_terms(d_tags, record['tags'])
        for dname, durl in record['download_urls'].items():
            record[f'{dname}_download_url'] = durl

        d_posts[record['dbid']] = record
        state['has_update'] = True
        _deploy()

    _deploy(force=True)
    logging.info(f'Done, {plural_word(fetched, "post")} added, {plural_word(len(d_posts), "post")} in the index.')
    if progress['page'] is not None:
        # The walk was cut short, so older posts may still be missing. A new run starting from page 1
        # would find the pages above this one already stored, take them for the end of the new posts
        # and stop, never reaching the gap, so it has to be told where to pick up.
        logging.warning(f'Stopped before the walk finished, at listing page {progress["page"]}. '
                        f'To continue, run again with --start-page {progress["page"]}.')


@click.command()
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_AO',
              default='deepghs/asiaon_index', show_default=True,
              help='Target dataset repository. It must already exist.')
@click.option('-t', '--max-time-limit', type=duration_type(allow_none=True), default='5h',
              show_default=True, help='Stop fetching after this duration, leaving room for the '
                                      'final upload. Use "none" to disable.')
@click.option('-d', '--deploy-span', type=duration_type(), default=5 * 60, show_default=True,
              help='Minimum interval between commits.')
@click.option('-E', '--max-empty-pages', type=int, default=10, show_default=True,
              help='Stop after this many consecutive listing pages with nothing new. 0 walks the '
                   'whole listing.')
@click.option('-P', '--start-page', type=int, envvar='START_PAGE', default=1, show_default=True,
              help='First listing page to read. A run cut short by its time limit reports the page to '
                   'resume from; without this a later run would stop at the stored pages above the gap.')
@click.option('-n', '--max-posts', type=int, default=None,
              help='Fetch at most this many new posts, for trial runs.')
@click.option('--dry-run', is_flag=True, default=False,
              help='Fetch and build the tables but upload nothing.')
def cli(repository: str, max_time_limit: Optional[float], deploy_span: float,
        max_empty_pages: int, start_page: int, max_posts: Optional[int], dry_run: bool):
    """Sync the asiaon index."""
    logging.try_init_root(logging.INFO)
    sync(
        repository=repository,
        max_time_limit=max_time_limit,
        deploy_span=deploy_span,
        max_empty_pages=max_empty_pages,
        dry_run=dry_run,
        max_posts=max_posts,
        start_page=start_page,
    )


if __name__ == '__main__':
    cli()
