"""Finding the password that really opens an asiaon archive.

The index keeps the site's own wording of the password, and that wording cannot be used as it stands:
since 2026-09-11 it is a sentence ("asiaontop.com or asiaon.top. Try one of the two; ..."), and only
the first of those two was ever right. Measured on 176 archives, the stored value opened 136 and the
first domain-like word in it opened all 176. The site also puts the real password into the comment of
the newer archives (``Password: <value>``).

So every archive is asked: the candidates below are tried in order and each one is checked by actually
reading an encrypted member of that archive. A password is only returned once the archive has accepted
it, and an archive nothing opens is an error, never a guess.
"""
import os
import re
import shutil
import subprocess
import zipfile
from typing import List, Optional

from ditk import logging

#: The domain-like words a password sentence is made of.
_DOMAIN_WORD = re.compile(r'[A-Za-z0-9][\w.\-]*\.(?:top|com|net|org)')
_COMMENT_PASSWORD = re.compile(r'Password:\s*(\S+)')
#: The two values the site has used, tried last in case the archive carries no comment and the index has none.
_KNOWN = ('asiaon.top', 'asiaontop.com')


class WrongPassword(Exception):
    """The archive is encrypted and none of the candidate passwords opened it."""


def password_candidates(stored: Optional[str], comment: Optional[str] = None) -> List[str]:
    """The passwords to try, best first and without repeats."""
    ordered = []
    match = _COMMENT_PASSWORD.search(comment or '')
    if match:
        ordered.append(match.group(1))
    words = _DOMAIN_WORD.findall(stored or '')
    ordered.extend(words[:1])
    ordered.extend(_KNOWN)
    ordered.extend(words[1:])
    return list(dict.fromkeys(ordered))


def _opens_with_7z(archive_file: str, password: str) -> bool:
    proc = subprocess.run(['7z', 't', '-y', f'-p{password}', archive_file], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=3600)
    return proc.returncode == 0


def find_password(archive_file: str, stored: Optional[str]) -> Optional[str]:
    """
    The password that opens ``archive_file``, or ``None`` when it is not encrypted.

    :raises WrongPassword: When the archive is encrypted and no candidate opens it.
    """
    if not archive_file.lower().endswith('.zip'):
        candidates = password_candidates(stored)
        if not shutil.which('7z'):
            raise WrongPassword('7z is not installed, so the password cannot be checked')
        for candidate in candidates:
            if _opens_with_7z(archive_file, candidate):
                return candidate
        raise WrongPassword(f'none of {candidates!r} opens {os.path.basename(archive_file)!r}')

    with zipfile.ZipFile(archive_file) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        encrypted = [info for info in members if info.flag_bits & 0x1]
        if not encrypted:
            return None
        comment = zf.comment.decode('utf-8', 'replace')
        candidates = password_candidates(stored, comment)
        smallest = min(encrypted, key=lambda info: info.file_size)
        # The standard library reads the traditional ZipCrypto scheme; WinZip AES (method 99) needs 7z.
        if smallest.compress_type != 99:
            for candidate in candidates:
                try:
                    zf.setpassword(candidate.encode('utf-8'))
                    zf.read(smallest)  # reading to the end checks the CRC, so a lucky header byte cannot pass
                except Exception:
                    continue
                return candidate
            raise WrongPassword(f'none of {candidates!r} opens {os.path.basename(archive_file)!r}')

    if not shutil.which('7z'):
        raise WrongPassword('the archive uses AES encryption and 7z is not installed')
    for candidate in candidates:
        if _opens_with_7z(archive_file, candidate):
            return candidate
    logging.warning(f'No candidate opened the AES archive {os.path.basename(archive_file)!r}.')
    raise WrongPassword(f'none of {candidates!r} opens {os.path.basename(archive_file)!r}')
