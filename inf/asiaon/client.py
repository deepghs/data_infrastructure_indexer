"""Client for asiaon.top.

The site is a Next.js front end over WordPress, and its pages are served as JSON from
``/_next/data/<buildId>/...``. The build id changes whenever the site is redeployed, so it is read
from the home page at start-up instead of being hard-coded. Listing pages hold sixteen posts,
newest first, and a page past the end answers with an empty list.
"""
import json
import re
import time
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urljoin

import dateparser
from ditk import logging
from hbutils.system import urlsplit
from pyquery import PyQuery as pq
from requests import HTTPError

from inf.utils.session import get_cloudscraper_session, srequest

__site_url__ = 'https://asiaon.top'

#: A datacentre address is sometimes answered with a 403 for one user agent and served normally with
#: the next, so a session is rebuilt until the site lets it through.
_SESSION_ATTEMPTS = 10

#: The data API answers 500 now and then for a request that succeeds when repeated.
_API_ATTEMPTS = 8


#: ``51M``, ``412.93MB``, ``425Mb``, ``1.5GB``: a number and an optional decimal unit, nothing else.
_SIZE_PATTERN = re.compile(r'\s*(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>[KMGT]?)B{0,2}\s*', re.IGNORECASE)
_SIZE_UNITS = {'': 0, 'K': 1, 'M': 2, 'G': 3, 'T': 4}


def parse_size(text: str) -> Optional[float]:
    """
    A rough byte count from a size such as ``412.93MB``, or ``None`` when the text is not just a size.

    Units are decimal (1 MB is 1,000,000 bytes), which is what every size already published in the
    index was computed with. The library call this replaces changed its mind about that between
    versions, so the rule is spelled out here instead of inherited; text with anything else in it, a
    stray slash or a trailing plus sign, is left as ``None`` rather than guessed at.
    """
    match = _SIZE_PATTERN.fullmatch(text.replace(',', ''))
    if not match:
        return None
    return float(round(float(match.group('number')) * 1000 ** _SIZE_UNITS[match.group('unit').upper()]))


def get_asiaon_session():
    for _ in range(_SESSION_ATTEMPTS):
        session = get_cloudscraper_session()
        logging.info(f'Try getting session, with UA: {session.headers["User-Agent"]!r} ...')
        resp = session.head(f'{__site_url__}/')
        if resp.status_code != 403:
            resp.raise_for_status()
            return session
    raise RuntimeError(f'{__site_url__} kept answering 403 across {_SESSION_ATTEMPTS} sessions.')


class AOClient:
    def __init__(self, session=None):
        self.session = session or get_asiaon_session()
        self.session.headers.update({
            'Referer': f'{__site_url__}/',
            'Accept': '*/*',
        })
        self.build_id = None

    def init(self):
        resp = srequest(
            self.session, 'GET',
            f'{__site_url__}/page/1/',
            params={'sort': 'DATE/DESC'},
        )
        findings = re.findall(r'"buildId":(?P<string>"[^"]+")', resp.text)
        if not findings:
            raise RuntimeError('No buildId found on webpage.')

        self.build_id = json.loads(findings[0])
        logging.info(f'Build ID refreshed: {self.build_id!r}')

    def _get_json(self, url: str, params: Dict[str, str]) -> Dict:
        for attempt in range(1, _API_ATTEMPTS + 1):
            try:
                resp = srequest(self.session, 'GET', url, params=params)
            except HTTPError as err:
                if err.response.status_code == 500 and attempt < _API_ATTEMPTS:
                    logging.warning(f'Got {err!r}, retry {attempt}/{_API_ATTEMPTS} ...')
                    time.sleep(2)
                    continue
                raise
            resp.raise_for_status()
            return resp.json()
        raise AssertionError('Unreachable code reached in AOClient._get_json.')

    def _get_list_api_urls(self, page_id: int) -> List[str]:
        if page_id <= 1:
            return [
                f'{__site_url__}/_next/data/{self.build_id}/index.json',
                f'{__site_url__}/_next/data/{self.build_id}/page/1.json',
            ]
        return [f'{__site_url__}/_next/data/{self.build_id}/page/{page_id}.json']

    def get_lst_page(self, page_id: int = 1, raw: bool = False):
        logging.info(f'Accessing post list page #{page_id!r} ...')
        data = None
        for api_url in self._get_list_api_urls(page_id):
            page_props = self._get_json(api_url, {'sort': 'DATE/DESC', 'pageNumber': str(page_id)})['pageProps']
            if 'data' in page_props:
                data = page_props['data']['posts']['nodes']
                break
            if '__N_REDIRECT' in page_props:
                logging.warning(f'List api {api_url!r} redirected to {page_props["__N_REDIRECT"]!r}, '
                                f'trying next candidate.')

        if data is None:
            raise RuntimeError(f'No usable list payload found for page #{page_id!r}.')
        if raw:
            return data

        records = []
        for item in data:
            record = {}
            page_url = urljoin(f'{__site_url__}/page/{page_id}/?sort=DATE/DESC', item['uri'])
            record['id'] = urlsplit(page_url).path_segments[1]
            record['dbid'] = item['databaseId']
            record['title'] = item['title']
            record['page_url'] = page_url
            record['thumbnail_url'] = urljoin(page_url, item['featuredImage']['node']['sourceUrl']) \
                if item['featuredImage'] else None
            record['created_at'] = dateparser.parse(item['date']).timestamp()
            record['updated_at'] = dateparser.parse(item['modified']).timestamp()
            record['status'] = item['status']
            record['categories'] = [
                {
                    'id': '/'.join(filter(bool, urlsplit(urljoin(page_url, citem['uri'])).path_segments[2:])),
                    'name': citem['name'],
                    'dbid': citem['databaseId'],
                    'url': urljoin(page_url, citem['uri']),
                    'total_count': citem['count'],
                }
                for citem in item['categories']['nodes']
            ]
            records.append(record)

        return records

    @staticmethod
    def _terms(page_url: str, nodes: List[Dict]) -> List[Dict]:
        return [
            {
                'id': '/'.join(filter(bool, urlsplit(urljoin(page_url, node['uri'])).path_segments[2:])),
                'name': node['name'],
                'dbid': node['databaseId'],
                'url': urljoin(page_url, node['uri']),
                'total_count': node['count'],
            }
            for node in nodes
        ]

    @staticmethod
    def _to_int(text: str, what: str) -> Optional[int]:
        try:
            return int(text.strip())
        except ValueError:
            logging.warning(f'Error when processing {what}: {text.strip()!r}.')
            return None

    def get_info(self, id_: str, raw: bool = False):
        logging.info(f'Accessing post {id_!r} ...')
        resp_json = self._get_json(
            f'{__site_url__}/_next/data/{self.build_id}/{quote_plus(id_, safe="")}.json',
            {'wordpressNode': id_},
        )
        data = resp_json['pageProps']['__TEMPLATE_QUERY_DATA__']['post']
        if raw:
            return data

        record = {}
        page_url = f'{__site_url__}/{quote_plus(id_, safe="")}/'
        record['id'] = urlsplit(page_url).path_segments[1]
        record['dbid'] = data['databaseId']
        record['title'] = data['title']
        record['page_url'] = page_url
        record['thumbnail_url'] = urljoin(page_url, data['featuredImage']['node']['sourceUrl']) \
            if data['featuredImage'] else None
        record['created_at'] = dateparser.parse(data['date']).timestamp()
        record['updated_at'] = dateparser.parse(data['modified']).timestamp()
        record['status'] = data['status']
        record['categories'] = self._terms(page_url, data['categories']['nodes'])
        record['tags'] = self._terms(page_url, data['tags']['nodes'])

        record['resolution'] = data['detail']['resolution']
        record['width'], record['height'] = None, None
        if data['detail']['resolution']:
            segs = data['detail']['resolution'].split('x', maxsplit=1)
            if len(segs) == 2:
                record['width'] = self._to_int(segs[0], 'width')
                record['height'] = self._to_int(segs[1], 'height')
        record['model'] = data['detail']['model']

        record['file_size_str'] = data['detail']['fileSize']
        record['pages'], record['file_size_rough'] = None, None
        if data['detail']['fileSize']:
            segs = data['detail']['fileSize'].split('/', maxsplit=1)
            if len(segs) == 2:
                pages_str, file_size_str = segs
                record['pages'] = self._to_int(pages_str.strip().rstrip('P').rstrip('p'), 'pages')
                record['file_size_rough'] = parse_size(file_size_str)
                if record['file_size_rough'] is None:
                    logging.warning(f'Error when processing file size rough: {file_size_str.strip()!r}.')

        download_html = data['download']['down']
        password_text = None
        download_urls = {}
        if download_html:
            for pitem in pq(download_html)('p').items():
                if pitem('a'):
                    href_item = pitem('a')
                    pitem.remove('a')
                    prefix = pitem.text().strip().split(':')[0].strip()
                    prefix = re.sub(r'\([\s\S]*\)', '', prefix)
                    prefix = re.sub(r'[^\da-zA-Z]+', '', prefix)
                    prefix = prefix.lower()

                    href = href_item.attr('href')
                    if not href:
                        logging.warning(f'Download entry {prefix!r} has no link, skipped.')
                        continue
                    d_url = urljoin(__site_url__ + '/', href)
                    if not prefix:
                        prefix = urlsplit(d_url).host.split('.')[-2]
                    download_urls[prefix] = d_url

                elif 'password' in pitem.text().lower():
                    _, password_text = pitem.text().split(':', maxsplit=1)
                    password_text = password_text.strip()

        record['download_urls'] = download_urls
        record['unzip_password'] = password_text

        return record
