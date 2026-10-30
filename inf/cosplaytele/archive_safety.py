"""Checks that make unpacking a third-party archive safe to publish from.

The packages come from file hosts anyone can upload to, are unpacked on a runner that holds
credentials, and the result is republished to a dataset. So an archive is looked at twice: its
member list is checked *before* anything is extracted, because an archive can plant a symlinked
directory and then write "through" it before any after-the-fact check runs, and the extracted tree
is checked again *after*, because the member list is only what the tool reports.

Anything that is not a plain file or a plain directory is refused, as is any path that is absolute,
contains ``..``, or ends up outside the target directory. The checks fail closed: a listing that
cannot be read, or that yields no entries, rejects the archive.
"""
import os
import re
import shutil
import stat
import subprocess
import zipfile
from typing import Dict, List, Optional

from ditk import logging

#: Unix file-type characters of ``ls -l`` that are not a regular file or directory.
_SPECIAL_TYPES = 'lbcps'
_UNIX_MODE = re.compile(r'(?:^|\s)([-dlbcps])[rwxsStT-]{9}(?:\s|$)')


class ArchiveUnsafe(Exception):
    """The archive holds something that must not be unpacked or republished."""


def path_problem(path: str) -> Optional[str]:
    """Why a member path is unacceptable, or ``None``."""
    if not path or '\x00' in path:
        return 'empty or NUL path'
    norm = path.replace('\\', '/')
    if norm.startswith('/') or re.match(r'^[A-Za-z]:', norm):
        return 'absolute path'
    if '..' in norm.split('/'):
        return 'parent directory component'
    return None


def entry_problem(entry: Dict) -> Optional[str]:
    """Why a listed member is unacceptable, or ``None``."""
    problem = path_problem(entry['path'])
    if problem:
        return problem
    if entry.get('symlink') or entry.get('hardlink'):
        return 'link'
    match = _UNIX_MODE.search(entry.get('attributes') or '')
    if match and match.group(1) in _SPECIAL_TYPES:
        return f'special file type {match.group(1)!r}'
    return None


def parse_7z_listing(text: str) -> List[Dict]:
    """Parse ``7z l -slt`` output into one dict per archive member."""
    parts = re.split(r'^-{10,}\s*$', text, maxsplit=1, flags=re.M)
    if len(parts) < 2:
        return []
    entries = []
    for block in re.split(r'\n\s*\n', parts[1].strip('\n')):
        fields = {}
        for line in block.splitlines():
            key, sep, value = line.partition(' = ')
            if sep:
                fields[key.strip()] = value.strip()
        if 'Path' not in fields:
            continue
        entries.append({
            'path': fields['Path'],
            'attributes': fields.get('Attributes', ''),
            'symlink': bool(fields.get('Symbolic Link')),
            'hardlink': bool(fields.get('Hard Link')),
        })
    return entries


def _list_zip(archive_file: str) -> List[Dict]:
    entries = []
    with zipfile.ZipFile(archive_file) as zf:
        for info in zf.infolist():
            mode = info.external_attr >> 16 if info.create_system == 3 else 0
            attributes = stat.filemode(mode) if mode else ''
            entries.append({'path': info.filename, 'attributes': attributes, 'symlink': stat.S_ISLNK(mode)})
    return entries


def _list_with_7z(archive_file: str, password: Optional[str]) -> List[Dict]:
    cmd = ['7z', 'l', '-slt', '-y']
    if password:
        cmd.append(f'-p{password}')
    cmd.append(archive_file)
    proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600)
    if proc.returncode != 0:
        raise ArchiveUnsafe(f'Could not list archive (7z exit {proc.returncode}): {proc.stderr.strip()[:200]!r}')
    return parse_7z_listing(proc.stdout)


def check_archive_members(archive_file: str, password: Optional[str]) -> int:
    """
    Refuse the archive if any member is unsafe, before anything is extracted.

    :returns: The number of members listed.
    :raises ArchiveUnsafe: On an unsafe member, an unreadable listing or an empty one.
    """
    if archive_file.lower().endswith('.zip'):
        try:
            entries = _list_zip(archive_file)
        except zipfile.BadZipFile as err:
            raise ArchiveUnsafe(f'Not a readable zip: {err!r}') from err
    elif shutil.which('7z'):
        entries = _list_with_7z(archive_file, password)
    else:
        logging.warning('7z is not installed, so the member list could not be checked before extraction.')
        return -1

    if not entries:
        raise ArchiveUnsafe('Archive listing is empty or could not be parsed.')
    for entry in entries:
        problem = entry_problem(entry)
        if problem:
            raise ArchiveUnsafe(f'Unsafe archive member {entry["path"]!r}: {problem}.')
    return len(entries)


def validate_tree(directory: str) -> int:
    """
    Refuse an unpacked tree that holds anything but plain files and directories.

    Symlinked directories matter as much as symlinked files: ``os.walk`` lists them as directories
    and does not descend into them, so each entry is checked with ``lstat`` instead.

    :returns: The number of files.
    :raises ArchiveUnsafe: On a link, a special file, a hard link, an escaped path or no files at all.
    """
    root = os.path.realpath(directory)
    files = 0
    for dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
        for name in [*dirnames, *filenames]:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, directory)
            mode = os.lstat(path)
            if stat.S_ISLNK(mode.st_mode):
                raise ArchiveUnsafe(f'Symlink in unpacked archive: {rel!r}.')
            if not os.path.realpath(path).startswith(root + os.sep):
                raise ArchiveUnsafe(f'Path escapes the target directory: {rel!r}.')
            if stat.S_ISDIR(mode.st_mode):
                continue
            if not stat.S_ISREG(mode.st_mode):
                raise ArchiveUnsafe(f'Special file in unpacked archive: {rel!r}.')
            if mode.st_nlink > 1:
                raise ArchiveUnsafe(f'Hard link in unpacked archive: {rel!r}.')
            files += 1
    if files == 0:
        raise ArchiveUnsafe('Archive unpacked to no files.')
    return files
