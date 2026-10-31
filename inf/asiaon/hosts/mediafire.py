"""mediafire.com downloader.

A share page carries a download button whose link, on current pages, is base64-encoded in
``data-scrambled-url``. Cloudflare answers some requests with a "Just a moment..." challenge instead
of the page. Measured on runners, that is not a verdict on the host or on the file: about nine in ten
pages pass at once with a current Chrome fingerprint, and the rest pass on some tries and not on others,
whatever the client. So each try uses a fresh session with the next browser fingerprint (curl_cffi
reproduces their handshakes; cloudscraper and plain requests were turned away on the hard pages every
time), and only a page that still gets the challenge after all the tries is counted as a refusal. Only
the host's own "invalid or deleted" answers mark a link as gone. The file itself then downloads from
``download*.mediafire.com`` at over a hundred megabytes a second, so the page is the only obstacle.
"""
import base64
import os
import re
import time
from typing import Optional

from curl_cffi import requests as cffi_requests
from curl_cffi.requests import BrowserType
from ditk import logging
from pyquery import PyQuery as pq
from urllib.parse import urljoin

from .base import (DownloadError, check_file_response, filename_from_response, get, get_download_session,
                   safe_local_name, stream_to_file)

#: Fresh sessions tried before a link is given up on as challenged.
_PAGE_ATTEMPTS = 12
#: Seconds between two tries of one page.
_PAGE_PAUSE = 1.0
#: Browser fingerprints in the order they were found to pass the hard pages most often on runners; the ones
#: the installed curl_cffi does not know are left out.
_FINGERPRINTS = ['chrome152', 'safari15_5', 'chrome151', 'safari17_0', 'chrome120', 'safari155', 'chrome136',
                 'chrome119', 'safari170', 'chrome151_macos']

_GONE = re.compile(r'Invalid or Deleted File|dangerous file|violat|has been (?:deleted|removed)', re.IGNORECASE)


def clean_url(url: str) -> str:
    """The share link itself: some index rows hold a mangled prefix such as ``https://wwhttp://www.mediafire...``."""
    match = re.search(r'https?://(?:www\.)?mediafire\.com/\S+', url)
    return match.group(0) if match else url


def _fingerprints():
    known = {item.value for item in BrowserType}
    return [name for name in _FINGERPRINTS if name in known] or ['chrome']


def fetch_page(url: str, attempt: int):
    """The share page, through a fresh session with the fingerprint this try is due."""
    names = _fingerprints()
    return cffi_requests.Session(impersonate=names[(attempt - 1) % len(names)]).get(url, timeout=40)


def extract_download_link(html: str) -> Optional[str]:
    button = pq(html)('a#downloadButton')
    scrambled = button.attr('data-scrambled-url')
    if scrambled:
        return base64.b64decode(scrambled).decode()
    return button.attr('href')


def download_mediafire(url: str, output_dir: str) -> str:
    url = clean_url(url)
    direct, last = None, None
    for attempt in range(1, _PAGE_ATTEMPTS + 1):
        try:
            resp = fetch_page(url, attempt)
        except Exception as err:
            last = type(err).__name__
            logging.warning(f'mediafire page request failed ({last}), attempt {attempt}/{_PAGE_ATTEMPTS}.')
            time.sleep(_PAGE_PAUSE)
            continue
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
        time.sleep(_PAGE_PAUSE)
    if not direct:
        # Not blocked: one page can stay challenged while the next passes at once. A host that really turns
        # the job away shows as refusals in a row, which the caller counts.
        raise DownloadError(f'mediafire: no download link after {_PAGE_ATTEMPTS} tries ({last}), '
                            f'probably a Cloudflare challenge', refused=True)

    session = get_download_session()
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
