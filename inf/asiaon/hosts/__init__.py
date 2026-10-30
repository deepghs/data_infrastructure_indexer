"""Download hosts, in the order a post's links are tried.

Each entry maps a host to the ``<host>_download_url`` column of the index and to a function
``(url, output_dir) -> local archive path`` that raises :class:`DownloadError` on failure. A post is
stored from the first host whose package downloads *and* unpacks.

The order follows what was measured on a runner without a proxy: mediafire is by far the fastest and has
no byte limit, though Cloudflare turns away about a fifth of requests; pixeldrain is dependable but
throttled to a crawl after roughly seven gigabytes; gofile works but only when its folder holds a
single archive; Google Drive is where most of the surviving old posts live. Hosts nothing here can fetch
(mega, terabox, vikingfile, buzzheavier, rapidgator, workupload and the dead file hosts) are left out, so
posts linked only from those stay unstored for a later attempt.
"""
from typing import Callable, List, NamedTuple

from .base import DownloadError
from .gofile import download_gofile
from .googledrive import download_googledrive
from .mediafire import download_mediafire
from .pixeldrain import download_pixeldrain


class Host(NamedTuple):
    name: str
    download: Callable[[str, str], str]

    @property
    def column(self) -> str:
        return f'{self.name}_download_url'


HOSTS: List[Host] = [
    Host('mediafire', download_mediafire),
    Host('pixeldrain', download_pixeldrain),
    Host('gofile', download_gofile),
    Host('googledrive', download_googledrive),
]

__all__ = ['DownloadError', 'Host', 'HOSTS']
