"""pixeldrain.com downloader.

Single files are ``/u/<id>``; the file API gives the metadata and the bytes. Pixeldrain throttles an
address after several gigabytes - measured, a runner went from over a hundred MB/s to 1 MB/s at about
7 GB - and a throttled transfer is abandoned by the stall guard as ``blocked``, which stops the job
asking this host again until the next run.
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
