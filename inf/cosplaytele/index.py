"""Index sync for cosplaytele.com.

Ported from the pyskeb prototype with the published layout left exactly as it was: the same
``posts.parquet`` / ``tags.parquet`` / ``README.md`` in the same dataset repository, with every
existing row carried over untouched. A run reads the published tables, walks the listing from the
newest page, fetches only posts it has not stored, and writes the tables back whole.

Walking the site
================

The listing is ``/page/<n>/`` with twelve posts per page, newest first, and a page past the end
answers 404. Posts are keyed by their slug, so a post that slides to the next page while the walk
is running is simply seen twice and skipped the second time. The walk stops once
``max_empty_pages`` consecutive pages have nothing new; a value of 0 disables that and walks to the
last page, which is how a full rebuild is done.

Download links are stored one column per host (``<host>_download_url``), so a host the site starts
using appears as a new column on the next run instead of needing a schema change.
"""
import json
import os
import re
import time
from typing import Optional
from urllib.parse import urljoin

import click
import dateparser
import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import urlsplit, TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs
from pyquery import PyQuery as pq

from inf.utils.duration import duration_type
from inf.utils.safe import safe_hf_hub_download, safe_upload_directory_as_directory
from inf.utils.session import get_requests_session, srequest

__site_url__ = 'https://cosplaytele.com'

#: A datacentre address is sometimes answered with a 403 for one user agent and served normally
#: with the next, so a session is rebuilt until the site lets it through.
_SESSION_ATTEMPTS = 10


def get_cosplaytele_session():
    for _ in range(_SESSION_ATTEMPTS):
        session = get_requests_session(timeout=20)
        logging.info(f'Try getting session, with UA: {session.headers["User-Agent"]!r} ...')
        resp = session.head(f'{__site_url__}/')
        if resp.status_code != 403:
            resp.raise_for_status()
            return session
    raise RuntimeError(f'{__site_url__} kept answering 403 across {_SESSION_ATTEMPTS} sessions.')


def _category_check(url):
    split = urlsplit(url)
    if split.host == 'cosplaytele.com':
        segments = list(filter(bool, split.path_segments))
        if len(segments) >= 2 and segments[0] in {'tag', 'category'}:
            return segments[0], segments[1]

    return None, None


def _get_info_from_post_page(post_url, session):
    logging.info(f'Getting post information from {post_url!r} ...')
    resp = srequest(session, 'GET', post_url)

    page = pq(resp.text)
    title = page('.page-title .entry-title').text().strip()
    created_at = dateparser.parse(page('.page-title .posted-on time.published').attr('datetime')).timestamp()
    updated_at = dateparser.parse(page('.page-title .posted-on time.updated').attr('datetime')).timestamp()
    tags = {}
    for item in page('.page-title .entry-category a').items():
        tag_type, tag_id = _category_check(urljoin(resp.url, item.attr('href')))
        if tag_id:
            tags[(tag_type, tag_id)] = {
                'tag': tag_id,
                'name': item.text().strip(),
                'type': tag_type,
                'category': None,
            }

    body = page('article .entry-content')
    bx = body('blockquote')
    for item in bx('a').items():
        tag_type, tag_id = _category_check(urljoin(resp.url, item.attr('href')))
        vp_items = re.split(r'\s*:\s*', item.parent().text().lower(), maxsplit=1)
        prefix = vp_items[0].replace(' ', '') if len(vp_items) == 2 else None
        if tag_id:
            tag_token = (tag_type, tag_id)
            if tag_token in tags:
                tags[tag_token]['category'] = prefix
            else:
                tags[tag_token] = {
                    'tag': tag_id,
                    'name': item.text().strip(),
                    'type': tag_type,
                    'category': prefix,
                }

    unzip_password = bx('input').attr('value') or None

    body.remove('blockquote')
    body.remove('div.gallery')
    body.remove('.crp_related')
    body.remove('.blog-share')

    download_urls = []
    for item in body('a').items():
        link_url = urljoin(resp.url, item.attr('href'))
        split = urlsplit(link_url)
        if 'cosplaytele' not in split.host:
            download_urls.append({
                'site': split.host.split('.')[-2],
                'url': link_url,
            })

    return {
        'title': title,
        'tags': list(tags.values()),
        'download_urls': download_urls,
        'unzip_password': unzip_password,
        'created_at': created_at,
        'updated_at': updated_at,
    }


def _get_posts_from_listpage(page_no, session):
    logging.info(f'Listing posts page #{page_no!r} ...')
    resp = srequest(session, 'GET', f'{__site_url__}/page/{page_no}/', raise_for_status=False)
    if resp.status_code == 404:
        return
    resp.raise_for_status()

    page = pq(resp.text)
    for item in page('#post-list .post-item').items():
        link_url = urljoin(resp.url, item('a').attr('href'))
        thumbnail_url = urljoin(resp.url, item('a img').attr('src'))
        segments = list(filter(bool, urlsplit(link_url).path_segments))
        assert len(segments) == 1, f'Invalid link url: {link_url!r}'
        yield {
            'post_id': segments[0],
            'post_url': link_url,
            'thumbnail_url': thumbnail_url,
        }


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


def _write_readme(file, df_posts: pd.DataFrame, df_tags: pd.DataFrame, total_posts: int):
    with open(file, 'w') as f:
        print('---', file=f)
        print('license: other', file=f)
        print('---', file=f)
        print('', file=f)
        print('## Records', file=f)
        print('', file=f)
        df_posts_shown = df_posts[:50][['id', 'title', 'page_url', 'tags', 'unzip_password', 'created_at']]
        df_posts_shown = df_posts_shown.fillna('')
        print(f'{plural_word(total_posts, "record")} in total. '
              f'Only {plural_word(len(df_posts_shown), "record")} shown.', file=f)
        print('', file=f)
        print(df_posts_shown.to_markdown(index=False), file=f)
        print('', file=f)

        print('## Tags', file=f)
        print('', file=f)
        df_tags_shown = df_tags[:50].fillna('')
        print(f'{plural_word(len(df_tags), "tag")} in total, '
              f'{plural_word(len(df_tags_shown), "tag")} shown.', file=f)
        print('', file=f)
        print('', file=f)
        print(df_tags_shown.to_markdown(index=False), file=f)
        print('', file=f)


def sync(repository: str, max_time_limit: Optional[float] = 50 * 60, deploy_span: float = 5 * 60,
         max_empty_pages: int = 10, dry_run: bool = False):
    start_time = time.time()
    delete_detached_cache()

    def _timeout():
        return max_time_limit is not None and start_time + max_time_limit < time.time()

    hf_client = get_hf_client(hf_token=os.environ['HF_TOKEN'])
    hf_fs = get_hf_fs(hf_token=os.environ['HF_TOKEN'])
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        raise RuntimeError(f'Repository {repository!r} does not exist, refusing to create a fresh index.')

    df_posts = _read_table(hf_client, hf_fs, repository, 'posts.parquet')
    records = df_posts.to_dict('records') if df_posts is not None else []
    exist_post_ids = {r['id'] for r in records}
    df_tags = _read_table(hf_client, hf_fs, repository, 'tags.parquet')
    d_tags = {(r['type'], r['tag']): r for r in df_tags.to_dict('records')} if df_tags is not None else {}
    logging.info(f'Loaded {plural_word(len(records), "post")} and {plural_word(len(d_tags), "tag")} '
                 f'from {repository!r}.')

    session = get_cosplaytele_session()
    state = {'has_update': False, 'last_deploy': None, 'last_count': len(exist_post_ids)}

    def _deploy(force=False):
        if not state['has_update']:
            return
        if not force and state['last_deploy'] is not None and state['last_deploy'] + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            df_posts = pd.DataFrame(records)
            df_posts = df_posts.sort_values(by=['created_at', 'id'], ascending=[False, False])
            df_posts.to_parquet(os.path.join(td, 'posts.parquet'), engine='pyarrow', index=False)

            df_tags = pd.DataFrame(list(d_tags.values()))
            df_tags = df_tags.sort_values(['count', 'tag'], ascending=[False, True])
            df_tags.to_parquet(os.path.join(td, 'tags.parquet'), engine='pyarrow', index=False)
            _write_readme(os.path.join(td, 'README.md'), df_posts, df_tags, len(exist_post_ids))

            added = len(exist_post_ids) - state['last_count']
            if dry_run:
                logging.info(f'[dry-run] Would upload {plural_word(added, "new post")}, '
                             f'{plural_word(len(df_posts), "post")} in total.')
            else:
                safe_upload_directory_as_directory(
                    repo_id=repository,
                    repo_type='dataset',
                    local_directory=td,
                    path_in_repo='.',
                    message=f'Add {plural_word(added, "post")}, on '
                            f'{time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())}',
                    hf_token=os.environ['HF_TOKEN'],
                )
        state['has_update'] = False
        state['last_deploy'] = time.time()
        state['last_count'] = len(exist_post_ids)

    def _iter_new_posts():
        page_no, empty_pages = 1, 0
        while not _timeout():
            has_item, has_new_item = False, False
            for item in _get_posts_from_listpage(page_no, session=session):
                has_item = True
                if item['post_id'] not in exist_post_ids:
                    has_new_item = True
                    yield item
            if not has_item:
                logging.info(f'Page {page_no} is past the end of the listing.')
                return
            empty_pages = 0 if has_new_item else empty_pages + 1
            if max_empty_pages and empty_pages >= max_empty_pages:
                logging.info(f'{plural_word(empty_pages, "page")} in a row with nothing new, stopping.')
                return
            page_no += 1

    for p_item in _iter_new_posts():
        if _timeout():
            break
        if p_item['post_id'] in exist_post_ids:
            logging.warning(f'Post {p_item["post_id"]!r} already crawled, skipped.')
            continue

        logging.info(f'Post {p_item["post_id"]!r} confirmed.')
        post_info = _get_info_from_post_page(p_item['post_url'], session)
        records.append({
            'id': p_item['post_id'],
            'title': post_info['title'],
            'page_url': p_item['post_url'],
            'thumbnail_url': p_item['thumbnail_url'],
            'unzip_password': post_info['unzip_password'],
            'tags': json.dumps([f"{titem['type']}/{titem['tag']}" for titem in post_info['tags']]),
            **{f'{uitem["site"]}_download_url': uitem['url'] for uitem in post_info['download_urls']},
            'created_at': post_info['created_at'],
            'updated_at': post_info['updated_at'],
        })
        for pitem in post_info['tags']:
            tag_token = (pitem['type'], pitem['tag'])
            if tag_token not in d_tags:
                d_tags[tag_token] = {**pitem, 'count': 0}
            else:
                d_tags[tag_token]['category'] = d_tags[tag_token]['category'] or pitem['category']
                if len(pitem['name']) > len(d_tags[tag_token]['name']):
                    d_tags[tag_token]['name'] = pitem['name']
            d_tags[tag_token]['count'] += 1
        exist_post_ids.add(p_item['post_id'])
        state['has_update'] = True
        _deploy()

    _deploy(force=True)
    logging.info(f'Done, {plural_word(len(exist_post_ids), "post")} in the index.')


@click.command()
@click.option('-r', '--repository', type=str, envvar='REMOTE_REPOSITORY_CTEL',
              default='deepghs/cosplaytele_index', show_default=True,
              help='Target dataset repository. It must already exist.')
@click.option('-t', '--max-time-limit', type=duration_type(allow_none=True), default='5h',
              show_default=True, help='Stop fetching after this duration, leaving room for the '
                                      'final upload. Use "none" to disable.')
@click.option('-d', '--deploy-span', type=duration_type(), default=5 * 60, show_default=True,
              help='Minimum interval between commits.')
@click.option('-E', '--max-empty-pages', type=int, default=10, show_default=True,
              help='Stop after this many consecutive listing pages with nothing new. 0 walks the '
                   'whole listing.')
@click.option('--dry-run', is_flag=True, default=False,
              help='Fetch and build the tables but upload nothing.')
def cli(repository: str, max_time_limit: Optional[float], deploy_span: float,
        max_empty_pages: int, dry_run: bool):
    """Sync the cosplaytele index."""
    logging.try_init_root(logging.INFO)
    sync(
        repository=repository,
        max_time_limit=max_time_limit,
        deploy_span=deploy_span,
        max_empty_pages=max_empty_pages,
        dry_run=dry_run,
    )


if __name__ == '__main__':
    cli()
