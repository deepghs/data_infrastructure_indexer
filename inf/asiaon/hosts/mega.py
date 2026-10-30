"""mega.nz downloader, for single-file links.

A file link holds its handle and a 256-bit key: ``mega.nz/file/<handle>#<key>`` (older links are
``mega.nz/#!<handle>!<key>``). The API answers a handle with a download address and the size; what is
served from that address is encrypted with AES-CTR under a key folded out of the link's key, so it is
decrypted as it streams. The decrypted size must equal the size the API announced, which, with the
archive then having to open, is the integrity check; the MAC is not recomputed.

Folder links are not supported. Mega meters anonymous transfer, so a quota answer sets the host aside
for the run.
"""
import base64
import json
import os
import random
import re
import struct
import time
from typing import Optional, Tuple

import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .base import DownloadError, TIMEOUT, safe_local_name, stream_to_file

_API = 'https://g.api.mega.co.nz/cs'
#: Answers that say the link is permanently unusable: not found, blocked, no access, expired, bad handle.
_GONE_CODES = {-9, -16, -11, -8, -2}
#: How each of those answers reads: -16 is a takedown by Mega, -11 no access, -8 an expired link.
_GONE_KINDS = {-9: 'deleted', -2: 'deleted', -16: 'takedown', -11: 'private', -8: 'expired'}
_EOVERQUOTA = -17
_EAGAIN = -3
#: Answers that mean this client is asking too fast or too often, not that the link is bad: -3 try again,
#: -4 rate limit, -6 too many requests for the resource.
_BUSY = {-3, -4, -6}


def b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))


def parse_link(url: str) -> Optional[Tuple[str, str]]:
    """``(handle, key)`` of a single-file link, or ``None``."""
    if '/folder/' in url or '#F!' in url:
        return None
    match = re.search(r'mega\.nz/file/([\w-]{8})#([\w-]+)', url) or re.search(r'mega\.nz/#!([\w-]{8})!([\w-]+)', url)
    return (match.group(1), match.group(2)) if match else None


def fold_key(key: str) -> Tuple[bytes, bytes]:
    """The AES key and the CTR counter start a 256-bit file key stands for."""
    words = struct.unpack('>8I', b64url_decode(key))
    aes_key = struct.pack('>4I', words[0] ^ words[4], words[1] ^ words[5], words[2] ^ words[6], words[3] ^ words[7])
    nonce = struct.pack('>4I', words[4], words[5], 0, 0)
    return aes_key, nonce


def decrypt_name(attributes: str, aes_key: bytes) -> Optional[str]:
    """The file name from the encrypted attribute block (``MEGA`` followed by JSON), if it can be read."""
    try:
        raw = Cipher(algorithms.AES(aes_key), modes.CBC(b'\x00' * 16)).decryptor().update(b64url_decode(attributes))
        text = raw.rstrip(b'\x00').decode('utf-8')
        return json.loads(text[4:])['n'] if text.startswith('MEGA') else None
    except Exception:
        return None


class _DecryptingResponse:
    """A response whose body is decrypted chunk by chunk, with the interface the stream helper expects."""

    def __init__(self, response, decryptor, size: int):
        self._response, self._decryptor = response, decryptor
        self.headers = {'Content-Length': str(size)}

    def iter_content(self, chunk_size: int = 1 << 20):
        for chunk in self._response.iter_content(chunk_size=chunk_size):
            yield self._decryptor.update(chunk)

    def close(self):
        self._response.close()


def _ask_api(handle: str) -> dict:
    for attempt in range(1, 5):
        resp = requests.post(_API, params={'id': random.randint(1, 10 ** 9)},
                             json=[{'a': 'g', 'g': 1, 'p': handle}], timeout=TIMEOUT)
        resp.raise_for_status()
        answer = resp.json()
        answer = answer[0] if isinstance(answer, list) else answer
        if answer == _EAGAIN and attempt < 4:
            time.sleep(2 * attempt)
            continue
        if isinstance(answer, int):
            if answer in _GONE_CODES:
                raise DownloadError(f'mega: {handle} is gone (error {answer})', gone=True,
                                    kind=_GONE_KINDS.get(answer, 'deleted'))
            if answer == _EOVERQUOTA:
                raise DownloadError('mega: transfer quota exceeded', blocked=True)
            raise DownloadError(f'mega: the API answered error {answer}', retryable=answer in _BUSY,
                                blocked=answer in _BUSY)
        return answer
    raise AssertionError('Unreachable code reached in _ask_api.')


def download_mega(url: str, output_dir: str) -> str:
    parsed = parse_link(url)
    if not parsed:
        raise DownloadError(f'mega: not a single-file link: {url!r}')
    handle, key = parsed
    info = _ask_api(handle)
    if not info.get('g'):
        raise DownloadError(f'mega: {handle} has no download address', retryable=True)
    aes_key, nonce = fold_key(key)
    name = decrypt_name(info.get('at', ''), aes_key)

    resp = requests.get(info['g'], stream=True, timeout=TIMEOUT)
    if resp.status_code == 509:
        resp.close()
        raise DownloadError('mega: bandwidth limit reached', blocked=True)
    if resp.status_code != 200:
        resp.close()
        raise DownloadError(f'mega: the download address answered HTTP {resp.status_code}',
                            retryable=resp.status_code >= 500)
    decryptor = Cipher(algorithms.AES(aes_key), modes.CTR(nonce)).decryptor()
    path = os.path.join(output_dir, safe_local_name(name or 'package.zip'))
    stream_to_file(_DecryptingResponse(resp, decryptor, int(info['s'])), path, expected_size=int(info['s']))
    return path
