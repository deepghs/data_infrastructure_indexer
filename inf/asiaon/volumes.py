"""Posts whose package is a split archive: every volume has to be fetched and joined.

Fifty-odd posts link, in the index, only the first mediafire volume (``x.zip.001``), because the index keeps one
link per host. Their pages list all of them, as ``MediaFire: Part 1 | Part 2 | Part 3``. The volumes are the raw
byte splits 7-Zip makes of a zip (or a 7z), so writing them one after the other into a single file gives back
the archive, which is then unpacked like any other. Split RAR volumes are not raw splits and are left alone.
"""
import os
import re
import shutil
import threading
from typing import Dict, List, Optional, Tuple

from ditk import logging
from pyquery import PyQuery as pq

#: ``x.zip.001`` and ``x.7z.002``: the raw splits of an archive whose own extension is the one before the number.
_VOLUME = re.compile(r'\.(zip|7z)\.(\d{2,4})$', re.IGNORECASE)
_PART_TEXT = re.compile(r'part\s*(\d+)', re.IGNORECASE)

#: One split post at a time: each needs room for its volumes, the unpacked files and the repacked tar.
LOCK = threading.Lock()
#: Space needed, as a multiple of the volumes' total size: the joined archive, its unpacked files and the tar.
_SPACE_FACTOR = 3.2
_SPACE_MARGIN = 2 * 1024 ** 3
#: Posts with more volume bytes than this are left for a bigger disk.
MAX_TOTAL_BYTES = 20 * 1024 ** 3


def parse_volume_name(name: Optional[str]) -> Optional[Tuple[str, int]]:
    """``(archive extension, volume number)`` of a split volume's file name, or ``None``."""
    match = _VOLUME.search(name or '')
    return (match.group(1).lower(), int(match.group(2))) if match else None


def mediafire_volume_urls(raw_post: Dict) -> List[str]:
    """The links of the ``MediaFire:`` line of a post's download section, in volume order."""
    html = (((raw_post or {}).get('download') or {}).get('down')) or ''
    for paragraph in pq(f'<div>{html}</div>')('p').items():
        text = paragraph.text().strip()
        if not text.lower().startswith('mediafire'):
            continue
        found = []
        for position, anchor in enumerate(paragraph('a').items()):
            href = anchor.attr('href') or ''
            if 'mediafire.com/file/' not in href:
                continue
            number = _PART_TEXT.search(anchor.text() or '')
            found.append((int(number.group(1)) if number else position + 1, href))
        found.sort(key=lambda item: item[0])
        return [href for _, href in found]
    return []


def check_volume_set(names: List[Optional[str]]) -> str:
    """The archive extension of a complete run of volumes ``.001``, ``.002``, ... ; raises ``ValueError`` otherwise."""
    parsed = [parse_volume_name(name) for name in names]
    if not parsed or any(item is None for item in parsed):
        raise ValueError(f'not all of the files are volumes of a split archive: {names!r}')
    if len({ext for ext, _ in parsed}) != 1:
        raise ValueError(f'the volumes are of different archive types: {names!r}')
    if [number for _, number in parsed] != list(range(1, len(parsed) + 1)):
        raise ValueError(f'the volumes are not numbered from 1 without gaps: {names!r}')
    return parsed[0][0]


def check_space(directory: str, total_bytes: int) -> None:
    free = shutil.disk_usage(directory).free
    need = int(total_bytes * _SPACE_FACTOR) + _SPACE_MARGIN
    if free < need:
        raise OSError(f'only {free / 1024 ** 3:.1f} GiB free, {need / 1024 ** 3:.1f} GiB needed for a '
                      f'{total_bytes / 1024 ** 3:.1f} GiB split archive')


def join_volumes(paths: List[str], target: str) -> int:
    """Write the volumes one after the other into ``target`` and delete them; returns the bytes written."""
    written = 0
    with open(target, 'wb') as out:
        for path in paths:
            with open(path, 'rb') as part:
                while True:
                    chunk = part.read(1 << 24)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
            os.remove(path)
    logging.info(f'Joined {len(paths)} volumes into {os.path.basename(target)!r}, {written} bytes.')
    return written
