"""sorafolder.com downloader.

A share page embeds an encrypted key. Posting it to ``/file-down`` returns an AES-encrypted
link to a signed CDN URL, which is short-lived: resolve it and download straight away rather than
resolving a batch up front.

The key that decrypts the link is baked into the site's front end and has no business being
secret, but it is also what the site would rotate, so a failure to decode a plausible URL is
reported as such instead of as a generic error.
"""
import ast
import base64
import html
import json
import os
import re
from hashlib import md5
from typing import Dict

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7
from ditk import logging

from .base import DownloadError, safe_local_name, stream_to_file

_LINK_KEY = 'Ak7qrvvH4WKYxV2OgaeHAEg2a5eh16vE'.encode('utf-8')

#: A JavaScript string literal, single or double quoted, with backslash escapes.
_STRING = r'''('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")'''


def decrypt_link(encrypted_data: str) -> str:
    """Decrypt a CryptoJS-style AES-CBC payload (``Salted__`` header, EVP_BytesToKey)."""
    raw_data = base64.b64decode(encrypted_data)
    salt, ciphertext = raw_data[8:16], raw_data[16:]

    derived, prev = b'', b''
    while len(derived) < 48:
        prev = md5(prev + _LINK_KEY + salt).digest()
        derived += prev

    decryptor = Cipher(algorithms.AES(derived[:32]), modes.CBC(derived[32:48]),
                       backend=default_backend()).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = PKCS7(128).unpadder()
    return (unpadder.update(padded) + unpadder.finalize()).decode('utf-8')


def _page_constant(page: str, name: str) -> str:
    """
    Read ``const <name> = '<literal>';`` from the page.

    The value is parsed as a literal rather than evaluated, since it comes from a remote page, and
    HTML-unescaped because names are served escaped: ``Stepmom&#39;s Scent`` contains a ``;``
    that a naive "up to the next semicolon" match cuts in half.
    """
    match = re.search(rf'const\s+{name}\s*=\s*{_STRING}\s*;', page)
    if not match:
        raise DownloadError(f'sorafolder page has no {name!r}, the share was probably removed.')
    return html.unescape(ast.literal_eval(match.group(1)))


def get_download_info(page_url: str, session) -> Dict[str, str]:
    logging.info(f'Getting sorafolder page {page_url!r} ...')
    resp = session.get(page_url)
    resp.raise_for_status()
    key_encrypt = _page_constant(resp.text, 'keyEncrypte')
    file_name = _page_constant(resp.text, 'fileName')

    resp = session.post(
        'https://sorafolder.com/file-down',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'keyEncrypte': key_encrypt}),
    )
    resp.raise_for_status()
    data = resp.json()
    if 'error' in data:
        raise DownloadError(f'sorafolder refused the file: {data["error"]!r}')

    url = decrypt_link(data['url'])
    if 'https' not in url or 'sorafolder.com' not in url:
        raise DownloadError('Decoded sorafolder URL looks wrong, the link key was probably rotated.')
    return {'url': url, 'filename': file_name}


def download_sorafolder(url: str, output_dir: str, session) -> str:
    """Download the package behind a sorafolder share page and return its local path."""
    info = get_download_info(url, session)
    dst_file = os.path.join(output_dir, safe_local_name(info['filename']))
    logging.info(f'Downloading {info["filename"]!r} ...')
    resp = session.get(info['url'], stream=True)
    if resp.status_code != 200:
        raise DownloadError(f'sorafolder CDN answered {resp.status_code}.')
    stream_to_file(resp, dst_file)
    return dst_file
