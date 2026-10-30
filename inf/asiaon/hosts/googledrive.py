"""Google Drive downloader, for files linked as ``drive.google.com/file/d/<id>``.

The files go through ``drive.usercontent.google.com/download?id=...&export=download&confirm=t``. The
gdown-style ``drive.google.com/uc?id=`` path is deliberately not used: it answers a "virus scan
warning" page for large files and is reported to stop after about forty requests per runner, while the
usercontent address served 166 consecutive requests here without a refusal. Requests are paced, and
an HTML answer is read to tell a quota refusal (the host is blocked for this run) from a file that is
gone or private (a verdict on this one link) and from anything else (try again).
"""
import os
import re

from .base import (DownloadError, Pacer, filename_from_response, get, get_plain_session, safe_local_name,
                   stream_to_file)

_PACER = Pacer(1.0)
_QUOTA = ('too many users', 'many accesses', 'quota exceeded', 'unusual traffic', 'download quota')
_GONE = ('request access', 'you need access', 'sign in', 'not found', 'permission', 'does not exist')
_PRIVATE = ('request access', 'you need access', 'sign in', 'permission')


def parse_file_id(url: str):
    if 'drive.google.com' not in url:
        return None
    match = re.search(r'/file/d/([\w-]{10,})', url) or re.search(r'[?&]id=([\w-]{10,})', url)
    return match.group(1) if match else None


def download_googledrive(url: str, output_dir: str) -> str:
    file_id = parse_file_id(url)
    if not file_id:
        raise DownloadError(f'googledrive: not a single-file Drive link: {url!r}')
    _PACER.wait()
    session = get_plain_session()
    resp = get(session, 'https://drive.usercontent.google.com/download',
               params={'id': file_id, 'export': 'download', 'confirm': 't'}, stream=True)
    if resp.status_code == 200 and not (resp.headers.get('Content-Type') or '').lower().startswith('text/html'):
        path = os.path.join(output_dir, safe_local_name(filename_from_response(resp)))
        stream_to_file(resp, path)
        return path

    text = next(resp.iter_content(6000), b'').decode('utf-8', 'replace').lower()
    resp.close()
    if any(word in text for word in _QUOTA):
        raise DownloadError('googledrive: download quota exceeded', blocked=True)
    if resp.status_code in (401, 403, 404) or any(word in text for word in _GONE):
        private = resp.status_code in (401, 403) or any(word in text for word in _PRIVATE)
        raise DownloadError(f'googledrive: file {file_id} is gone or private (HTTP {resp.status_code})', gone=True,
                            kind='private' if private else 'deleted')
    raise DownloadError(f'googledrive: unexpected answer HTTP {resp.status_code}', retryable=True)
