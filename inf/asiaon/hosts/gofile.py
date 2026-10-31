"""gofile.io downloader.

The content API wants a website token that the site's own script computes from a guest account
token. That script is third-party code fetched at run time, so it is run only under node's permission
model - no filesystem outside its own temporary directory, no child processes, an empty
environment - which keeps the job's credentials out of its reach.

A gofile folder can hold several files, and the same folder is sometimes linked from more than one post,
so a file is only taken when the folder holds exactly one candidate archive; anything else is reported
and skipped rather than guessed at, since guessing would file another post's package under this one.
"""
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Dict, Tuple

from ditk import logging

from .base import (DownloadError, MODERN_UA, check_file_response, get, get_plain_session, safe_local_name,
                   stream_to_file)

_ARCHIVE_EXT = {'.zip', '.rar', '.7z'}
_WT_TTL = 90 * 60.0
_LOCK = threading.Lock()
_STATE: Dict = {'token': None, 'wt': None, 'wt_at': 0.0}
_SCRIPT = os.path.join(os.path.dirname(__file__), 'gofile_wt.mjs')


def compute_wt(account_token: str, session) -> str:
    """Run the site's token generator for ``account_token`` under node's permission model."""
    node = shutil.which('node')
    if not node:
        raise DownloadError('gofile: node is not installed, cannot compute the website token', blocked=True)
    generator = get(session, 'https://gofile.io/js/wt.obf.js')
    generator.raise_for_status()
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'wt.obf.js'), 'w', encoding='utf-8') as f:
            f.write(generator.text)
        script = os.path.join(td, 'gofile_wt.mjs')
        shutil.copy(_SCRIPT, script)
        proc = subprocess.run(
            [node, '--permission', f'--allow-fs-read={td}', script, account_token, MODERN_UA],
            env={'PATH': os.environ.get('PATH', '/usr/bin:/bin')}, cwd=td,
            capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
        )
    wt = proc.stdout.strip()
    if proc.returncode != 0 or not re.fullmatch(r'[0-9a-f]{64}', wt):
        raise DownloadError(f'gofile: token generator failed (exit {proc.returncode}): {proc.stderr.strip()[-120:]!r}')
    return wt


def _credentials(session, refresh: bool = False) -> Tuple[str, str]:
    """One guest account per process, and a website token renewed before it goes stale."""
    with _LOCK:
        if _STATE['token'] is None:
            resp = session.post('https://api.gofile.io/accounts', timeout=(15, 60))
            if resp.status_code == 429:
                raise DownloadError('gofile: account creation is rate limited', blocked=True)
            resp.raise_for_status()
            _STATE['token'] = resp.json()['data']['token']
            refresh = True
        if refresh or _STATE['wt'] is None or time.time() - _STATE['wt_at'] > _WT_TTL:
            _STATE['wt'] = compute_wt(_STATE['token'], session)
            _STATE['wt_at'] = time.time()
        return _STATE['token'], _STATE['wt']


def _contents(session, ident: str):
    for attempt in (1, 2):
        token, wt = _credentials(session, refresh=attempt == 2)
        resp = get(session, f'https://api.gofile.io/contents/{ident}',
                   headers={'Authorization': f'Bearer {token}', 'X-Website-Token': wt, 'X-BL': 'en-US'})
        if resp.status_code == 429:
            raise DownloadError('gofile: rate limited', blocked=True)
        if resp.status_code == 404 or 'error-notFound' in resp.text:
            raise DownloadError(f'gofile: folder {ident} does not exist', gone=True, kind='deleted')
        if resp.status_code in (401, 403) and attempt == 1:
            logging.warning('gofile refused the website token, computing a new one.')
            continue
        if resp.status_code != 200:
            raise DownloadError(f'gofile: contents answered HTTP {resp.status_code}', retryable=resp.status_code >= 500)
        return token, resp.json()['data']
    raise DownloadError('gofile: the website token was refused twice', blocked=True)


def pick_file(data: Dict) -> Dict:
    files = [item for item in (data.get('children') or {}).values() if item.get('type') == 'file']
    if not files:
        # The folder is still there but nothing is left in it, which is how a purged share looks.
        raise DownloadError('gofile: the folder holds no files any more', gone=True, kind='emptied')
    if len(files) == 1:
        return files[0]
    archives = [item for item in files if os.path.splitext(item.get('name', ''))[1].lower() in _ARCHIVE_EXT]
    if len(archives) == 1:
        return archives[0]
    raise DownloadError(f'gofile: the folder holds {len(files)} files and {len(archives)} archives, '
                        f'which one belongs to this post cannot be told')


def download_gofile(url: str, output_dir: str) -> str:
    match = re.search(r'gofile\.io/d/([\w-]+)', url)
    if not match:
        raise DownloadError(f'gofile: unsupported link {url!r}')
    session = get_plain_session()
    token, data = _contents(session, match.group(1))
    item = pick_file(data)
    session.cookies.set('accountToken', token, domain='.gofile.io')
    resp = get(session, item['link'], headers={'Authorization': f'Bearer {token}'}, stream=True)
    check_file_response(resp, 'gofile', 'the file')
    path = os.path.join(output_dir, safe_local_name(item.get('name', 'package.zip')))
    stream_to_file(resp, path)
    return path
