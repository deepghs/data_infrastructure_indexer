import os
import re
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word, humanize
from hbutils.system import urlsplit, TemporaryDirectory
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import get_requests_session, download_file
from natsort import natsorted
from pyquery import PyQuery as pq
from tqdm import tqdm
from waifuc.utils import srequest


def _get_latest_date_in_index(session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    resp = srequest(session, 'GET', 'https://e621.net/db_export/')
    resp.raise_for_status()
    detected_dates = []
    for aitem in pq(resp.text)('a[href]').items():
        url = urljoin(str(resp.url), aitem.attr('href'))
        filename = urlsplit(url).filename
        matching = re.fullmatch(r'^tags-(?P<date>\d{4}-\d{2}-\d{2})\.csv\.gz$', filename)
        if matching:
            date_str = matching.group('date')
            detected_dates.append(date_str)
    return natsorted(detected_dates)[-1]


def sync(repository: str):
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

    session = get_requests_session()
    last_date = _get_latest_date_in_index(session=session)
    logging.info(f'Last updated official index date is {last_date!r}.')
    mps = {}
    with TemporaryDirectory() as upload_dir:
        for tab_name in tqdm(['pools', 'posts', 'tag_aliases', 'tag_implications', 'tags', 'wiki_pages'],
                             desc='Sync Tables'):
            logging.info(f'Making for {tab_name!r} ...')
            source_url = urljoin('https://e621.net/db_export/', f'{tab_name}-{last_date}.csv.gz')
            with TemporaryDirectory() as td:
                csv_file = os.path.join(td, f'{tab_name}-{last_date}.csv.gz')
                download_file(
                    url=source_url,
                    filename=csv_file,
                    session=session,
                )

                df = pd.read_csv(csv_file, compression='gzip')
                mps[tab_name] = len(df)
                df.to_parquet(os.path.join(upload_dir, f'index_{tab_name}.parquet'), engine='pyarrow', index=False)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=upload_dir,
            path_in_repo='.',
            message=f'Sync index, {", ".join(map(lambda x: plural_word(x[1], humanize(x[0])), mps.items()))}',
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_E621'],
    )
