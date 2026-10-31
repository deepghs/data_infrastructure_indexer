"""pixeldrain.com downloader.

Single files are ``/u/<id>``; the file API gives the metadata and the bytes. Pixeldrain throttles an
address after about seven gigabytes: measured on a runner, a single stream held 300 to 600 MB/s for the
first 7.1 GB and then fell to about 0.5 MB/s. Opening more connections does not get round it - eight
parallel ranges together reach about 2 MB/s and sixteen no more, so the limit is on the address, not on a
connection - and it had not eased ten minutes later. A fresh runner has a fresh address, so each run can
take roughly seven gigabytes from this host. A throttled transfer is abandoned by the stall guard, and
two of those set the host aside for the rest of the run.
"""
import os
import re

from .base import (DownloadError, check_file_response, filename_from_response, get, get_plain_session,
                   safe_local_name, stream_to_file)

_API = 'https://pixeldrain.com/api/file'


def download_pixeldrain(url: str, output_dir: str) -> str:
    match = re.search(r'pixeldrain\.com/u/([\w-]+)', url)
    if not match:
        raise DownloadError(f'pixeldrain: unsupported link {url!r}')
    ident, session = match.group(1), get_plain_session()

    info = get(session, f'{_API}/{ident}/info')
    if info.status_code == 404:
        raise DownloadError(f'pixeldrain: file {ident} does not exist', gone=True, kind='deleted')
    if info.status_code != 200:
        raise DownloadError(f'pixeldrain: info answered HTTP {info.status_code}', retryable=info.status_code >= 500)
    availability = info.json().get('availability')
    if availability:
        # e.g. a captcha is required, or the file is rate limited: not something to wait out here
        raise DownloadError(f'pixeldrain: file {ident} is not available to us ({availability})', blocked=True)

    resp = get(session, f'{_API}/{ident}', params={'download': ''}, stream=True)
    check_file_response(resp, 'pixeldrain', 'the file')
    path = os.path.join(output_dir, safe_local_name(filename_from_response(resp)))
    stream_to_file(resp, path)
    return path
