"""Download hosts, in the order a post's links are tried.

Each entry maps the host to the ``<host>_download_url`` column of the index and to a function
``(url, output_dir, session) -> local archive path`` that raises :class:`DownloadError` on failure.
A post is stored from the first host whose package downloads *and* unpacks; a post that no host
can deliver stays out of storage and is simply tried again on the next run.

Hosts the index records but nothing here can fetch (gofile, mediafire, yandex, terabox, ...) are
deliberately absent, so those posts stay unstored for a later attempt.
"""
from typing import Callable, List, NamedTuple

from .base import DownloadError, get_download_session
from .sorafolder import download_sorafolder


class Host(NamedTuple):
    name: str
    download: Callable[[str, str, object], str]

    @property
    def column(self) -> str:
        return f'{self.name}_download_url'


HOSTS: List[Host] = [
    Host('sorafolder', download_sorafolder),
]

__all__ = ['DownloadError', 'get_download_session', 'Host', 'HOSTS']
