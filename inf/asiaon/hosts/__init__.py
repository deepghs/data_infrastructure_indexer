"""Download hosts, in the order a post's links are tried.

Each entry maps a host to the ``<host>_download_url`` column of the index and to a function
``(url, output_dir) -> local archive path`` that raises :class:`DownloadError` on failure. A post is
stored from the first host whose package downloads *and* unpacks.

The order follows what was measured on a runner without a proxy: mediafire is by far the fastest and has
no byte limit, though Cloudflare turns away about a fifth of requests; pixeldrain is dependable but
throttled to a crawl after roughly seven gigabytes; gofile works but only when its folder holds a
single archive; Google Drive is where most of the surviving old posts live; mega works but only a
handful of old links are still alive and it is slow. Hosts nothing here can fetch are left out, each for
a measured reason: buzzheavier answers every runner request with 403 whatever the browser profile,
terabox wants a human verification, vikingfile a Turnstile challenge, rapidgator a premium account,
workupload a human check, and the old file hosts are shut down. Posts linked only from those stay
unstored for a later attempt.
"""
import re
from typing import Callable, List, NamedTuple

from .base import DownloadError
from .gofile import download_gofile
from .googledrive import download_googledrive, parse_file_id as parse_drive_id
from .mediafire import download_mediafire
from .mega import download_mega, parse_link as parse_mega_link
from .pixeldrain import download_pixeldrain


class Host(NamedTuple):
    name: str
    download: Callable[[str, str], str]
    #: Whether a link in this host's column is one the downloader can take. Index columns also hold
    #: link shorteners and folder links, which are not a verdict on the post and are left alone.
    accepts: Callable[[str], bool] = lambda url: True

    @property
    def column(self) -> str:
        return f'{self.name}_download_url'


HOSTS: List[Host] = [
    Host('mediafire', download_mediafire, lambda url: 'mediafire.com' in url),
    Host('pixeldrain', download_pixeldrain, lambda url: re.search(r'pixeldrain\.com/u/[\w-]+', url) is not None),
    Host('gofile', download_gofile, lambda url: re.search(r'gofile\.io/d/[\w-]+', url) is not None),
    Host('googledrive', download_googledrive, lambda url: parse_drive_id(url) is not None),
    Host('mega', download_mega, lambda url: parse_mega_link(url) is not None),
]

__all__ = ['DownloadError', 'Host', 'HOSTS']
