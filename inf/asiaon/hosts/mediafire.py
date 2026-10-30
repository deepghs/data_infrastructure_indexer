"""mediafire.com downloader.

A share page carries a download button whose link, on current pages, is base64-encoded in
``data-scrambled-url``. Cloudflare answers some requests with a "Just a moment..." challenge instead
of the page; that is not a verdict on the file, so each attempt uses a fresh session and a link that
still gets the challenge is reported as blocked rather than gone. Only the host's own "invalid or
deleted" answers mark a link as gone.
"""
import base64
import os
import re
from typing import Optional

from ditk import logging
from pyquery import PyQuery as pq
from urllib.parse import urljoin

from .base import (DownloadError, check_file_response, filename_from_response, get, get_download_session,
                   safe_local_name, stream_to_file)

#: Fresh sessions tried before a link is given up on as challenged.
_PAGE_ATTEMPTS = 3

_GONE = re.compile(r'Invalid or Deleted File|dangerous file|violat|has been (?:deleted|removed)', re.IGNORECASE)


def extract_download_link(html: str) -> Optional[str]:
    button = pq(html)('a#downloadButton')
    scrambled = button.attr('data-scrambled-url')
    if scrambled:
        return base64.b64decode(scrambled).decode()
    return button.attr('href')


def download_mediafire(url: str, output_dir: str) -> str:
    direct, session, last = None, None, None
    for attempt in range(1, _PAGE_ATTEMPTS + 1):
        session = get_download_session()
        resp = get(session, url)
        if 'downloadButton' in resp.text:
            direct = extract_download_link(resp.text)
            if direct:
                break
        elif resp.status_code == 404 or _GONE.search(resp.text):
            takedown = re.search(r'dangerous file|violat', resp.text, re.IGNORECASE)
            raise DownloadError(f'mediafire: {url!r} is deleted or invalid', gone=True,
                                kind='takedown' if takedown else 'deleted')
        last = f'HTTP {resp.status_code}'
        logging.warning(f'mediafire page gave no download link ({last}), attempt {attempt}/{_PAGE_ATTEMPTS}.')
    if not direct:
        raise DownloadError(f'mediafire: no download link after {_PAGE_ATTEMPTS} sessions ({last}), '
                            f'probably a Cloudflare challenge', blocked=True)

    resp = get(session, direct, stream=True, allow_redirects=True)
    if resp.status_code == 200 and not resp.headers.get('Content-Disposition') \
            and (resp.headers.get('Content-Type') or '').lower().startswith('text/html'):
        # An interstitial with a "continue" button in front of the file.
        target = pq(resp.text)('#continue-btn').attr('href')
        resp.close()
        if not target:
            raise DownloadError('mediafire: the download link answered an HTML page with no way on', retryable=True)
        resp = get(session, urljoin(resp.url, target), stream=True, allow_redirects=True)
    check_file_response(resp, 'mediafire', 'the download link')
    path = os.path.join(output_dir, safe_local_name(filename_from_response(resp)))
    stream_to_file(resp, path)
    return path
