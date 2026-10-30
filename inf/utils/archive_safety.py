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
import time
import zipfile
from typing import Dict, List, Optional, Tuple

from ditk import logging


#: Unix file-type characters of ``ls -l`` that are not a regular file or directory.
_SPECIAL_TYPES = 'lbcps'
_UNIX_MODE = re.compile(r'(?:^|\s)([-dlbcps])[rwxsStT-]{9}(?:\s|$)')


#: What each accepted extension must begin with, at offset 0. The listing tool and the extractor are
#: different programs that each decide what an archive is; requiring the signature up front stops a
#: polyglot, a self-extracting header or an embedded archive from being read one way by the check
#: and another way by the extractor.
_SIGNATURES = {
    'rar': (b'Rar!\x1a\x07\x00', b'Rar!\x1a\x07\x01\x00'),
    '7z': (b'7z\xbc\xaf\x27\x1c',),
    'zip': (b'PK\x03\x04', b'PK\x05\x06'),
}

#: The ``Type`` 7z reports for each extension.
_LISTED_TYPES = {'rar': {'Rar', 'Rar5'}, '7z': {'7z'}}


#: Characters that must not appear anywhere in a listing except as the ``\\n`` ending a line. Python's
#: ``str.splitlines`` also breaks on most of these, so a name holding one would be read as several
#: lines here while the extractor sees one name - the gap this list closes. The listing is split on
#: ``\\n`` alone, and anything else in this set is refused outright.
_LISTING_FORBIDDEN = re.compile('[\\x00-\\x09\\x0b-\\x1f\\x7f\\x85\\u2028\\u2029]')

#: The ``7z l -slt`` member fields this check understands. An unknown field is refused, so a name
#: cannot pass as a harmless extra field and a new 7z field is noticed instead of ignored.
_KNOWN_FIELDS = {
    'Path', 'Folder', 'Size', 'Packed Size', 'Modified', 'Created', 'Accessed', 'Attributes',
    'Encrypted', 'Comment', 'CRC', 'Method', 'Characteristics', 'Host OS', 'Version',
    'Volume Index', 'Offset', 'Symbolic Link', 'Hard Link', 'Block', 'Solid', 'Position',
    'Link', 'Checksum', 'Code Page', 'Big Endian', 'Reparse', 'Archive', 'Warnings', 'Errors',
    'Alternate Stream', 'Copy Link', 'NT Security', 'Split Before', 'Split After',
    'Aux', 'Deleted', 'Tree', 'SHA-1', 'SHA-256', 'Anti', 'Short Name', 'User', 'Group',
    'User ID', 'Group ID', 'Error Type', 'Warning Flags', 'Unconfirmed', 'Volume',
}
# Device and inode fields are left out on purpose: an archive holding a device node should be refused.


class ArchiveUnsafe(Exception):
    """The archive holds something that must not be unpacked or republished."""


def path_problem(path: str) -> Optional[str]:
    """Why a member path is unacceptable, or ``None``."""
    if not path or any(ord(c) < 32 or ord(c) == 127 for c in path):
        return 'empty path or control character in path'
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
    # A RAR5 "file copy" duplicates another member of the same archive. It is how RAR deduplicates,
    # not a filesystem link, so it is accepted as long as what it points at is a safe path too.
    if entry.get('copy_link') and path_problem(entry['copy_link']):
        return f'copy link target: {path_problem(entry["copy_link"])}'
    match = _UNIX_MODE.search(entry.get('attributes') or '')
    if match and match.group(1) in _SPECIAL_TYPES:
        return f'special file type {match.group(1)!r}'
    return None


def _parse_fields(block: str, unknown: set) -> Dict[str, str]:
    """
    Read ``Key = Value`` lines, refusing anything else.

    A member name may contain a newline, which would fake extra lines in the listing. Every line
    must therefore be a well-formed field and no key may repeat, so a name cannot smuggle in a
    second ``Path`` or hide a ``Symbolic Link`` marker. Keys this check does not know are collected
    into ``unknown`` rather than raised one at a time, so a listing that uses several new fields is
    reported whole.
    """
    fields = {}
    for line in block.split('\n'):
        if not line.strip():
            continue
        key, sep, value = line.partition(' = ')
        key = key.strip()
        if not sep or not key:
            raise ArchiveUnsafe(f'Malformed line in archive listing: {line[:80]!r}.')
        if key not in _KNOWN_FIELDS:
            unknown.add(key)
        if key in fields:
            raise ArchiveUnsafe(f'Repeated field {key!r} in archive listing.')
        fields[key] = value.strip()
    return fields


def parse_7z_listing(text: str) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Parse ``7z l -slt`` output into the archive's header fields and one dict per member.

    :raises ArchiveUnsafe: On any block that is not a well-formed member, rather than skipping it.
    """
    forbidden = _LISTING_FORBIDDEN.search(text)
    if forbidden:
        raise ArchiveUnsafe(f'Control character {forbidden.group()!r} in archive listing.')
    parts = re.split(r'^-{10,}[ \t]*$', text, maxsplit=1, flags=re.M)
    if len(parts) < 2:
        return {}, []
    header = {}
    for line in parts[0].split('\n'):
        key, sep, value = line.partition(' = ')
        if sep:
            header[key.strip()] = value.strip()

    entries, unknown = [], set()
    for block in re.split(r'\n[ \t]*\n', parts[1].strip('\n')):
        if not block.strip():
            continue
        fields = _parse_fields(block, unknown)
        if 'Path' not in fields:
            raise ArchiveUnsafe('Archive listing has a member block without a path.')
        entries.append({
            'path': fields['Path'],
            'attributes': fields.get('Attributes', ''),
            'symlink': bool(fields.get('Symbolic Link')),
            'hardlink': bool(fields.get('Hard Link')),
            'copy_link': fields.get('Copy Link') or None,
        })
    if unknown:
        raise ArchiveUnsafe(f'Unknown fields in archive listing: {sorted(unknown)!r}.')
    return header, entries


def _list_zip(archive_file: str) -> List[Dict]:
    entries = []
    with zipfile.ZipFile(archive_file) as zf:
        for info in zf.infolist():
            mode = info.external_attr >> 16 if info.create_system == 3 else 0
            attributes = stat.filemode(mode) if mode else ''
            entries.append({'path': info.filename, 'attributes': attributes, 'symlink': stat.S_ISLNK(mode)})
    return entries


#: ``7z l`` has been seen to die with a segmentation fault (exit -11) on packages that list fine on
#: the next attempt, so a failed listing is repeated before the archive is refused.
_LIST_ATTEMPTS = 3


def _list_with_7z(archive_file: str, password: Optional[str]) -> Tuple[Dict[str, str], List[Dict]]:
    cmd = ['7z', 'l', '-slt', '-y']
    if password:
        cmd.append(f'-p{password}')
    cmd.append(archive_file)
    for attempt in range(1, _LIST_ATTEMPTS + 1):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='strict',
                                  stdin=subprocess.DEVNULL, timeout=600)
        except UnicodeDecodeError as err:
            raise ArchiveUnsafe('Archive listing is not valid UTF-8.') from err
        if proc.returncode == 0:
            return parse_7z_listing(proc.stdout)
        logging.warning(f'7z listing failed (exit {proc.returncode}), attempt {attempt}/{_LIST_ATTEMPTS}.')
        if attempt < _LIST_ATTEMPTS:
            time.sleep(2)
    raise ArchiveUnsafe(f'Could not list archive (7z exit {proc.returncode}): {proc.stderr.strip()[:200]!r}')


#: RAR5 redirect record types: 1 and 2 are symbolic links, 3 a junction, 4 a hard link, 5 a copy of another member.
_REDIR_SYMLINKS = (1, 2, 3)
_REDIR_HARDLINK = 4
_REDIR_COPY = 5


def _list_with_rarfile(archive_file: str, password: Optional[str]) -> List[Dict]:
    """
    List a RAR through ``rarfile``, which drives the ``unrar`` tool and answers with structured entries.

    Used when ``7z`` cannot list an archive: 7-Zip 23.01 was seen segfaulting on RAR5 packages that 26.00
    lists without trouble, and the version on a runner is not something a job controls.
    """
    import rarfile

    try:
        rf = rarfile.RarFile(archive_file)
        if password:
            rf.setpassword(password)
        infos = rf.infolist()
    except Exception as err:
        raise ArchiveUnsafe(f'rarfile could not list the archive either: {err!r}') from err

    entries = []
    for info in infos:
        redirect = getattr(info, 'file_redir', None)
        kind = redirect[0] if redirect else None
        entries.append({
            'path': info.filename,
            'attributes': '',
            'symlink': bool(info.is_symlink()) or kind in _REDIR_SYMLINKS,
            'hardlink': kind == _REDIR_HARDLINK,
            'copy_link': redirect[2] if kind == _REDIR_COPY else None,
        })
    return entries


def check_signature(archive_file: str, extension: str) -> None:
    """Require the file to start with the signature its extension promises."""
    signatures = _SIGNATURES.get(extension)
    if not signatures:
        raise ArchiveUnsafe(f'Unsupported archive extension {extension!r}.')
    with open(archive_file, 'rb') as f:
        head = f.read(16)
    if not any(head.startswith(sig) for sig in signatures):
        raise ArchiveUnsafe(f'File does not start with a {extension} signature.')


def check_archive_members(archive_file: str, password: Optional[str]) -> int:
    """
    Refuse the archive if any member is unsafe, before anything is extracted.

    :returns: The number of members listed.
    :raises ArchiveUnsafe: On an unsafe member, a mismatched format, an unreadable or empty listing,
        or a missing listing tool - the check never passes by not running.
    """
    extension = os.path.splitext(archive_file)[1].lstrip('.').lower()
    check_signature(archive_file, extension)

    if extension == 'zip':
        try:
            entries = _list_zip(archive_file)
        except zipfile.BadZipFile as err:
            raise ArchiveUnsafe(f'Not a readable zip: {err!r}') from err
    else:
        if not shutil.which('7z'):
            raise ArchiveUnsafe('7z is not installed, so the member list cannot be checked.')
        try:
            header, entries = _list_with_7z(archive_file, password)
        except ArchiveUnsafe as err:
            if extension != 'rar' or 'Could not list archive' not in str(err):
                raise
            logging.warning(f'{err} Listing with rarfile instead.')
            header, entries = None, _list_with_rarfile(archive_file, password)
        if header is not None and header.get('Type') not in _LISTED_TYPES[extension]:
            raise ArchiveUnsafe(f'Archive is a {header.get("Type")!r}, not the {extension} its name claims.')

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
            if any(ord(c) < 32 or ord(c) == 127 for c in name):
                raise ArchiveUnsafe(f'Control character in unpacked name: {rel!r}.')
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
