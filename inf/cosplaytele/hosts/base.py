"""Shared pieces for the download hosts that cosplaytele posts link to, now in :mod:`inf.utils.transfer`."""
from inf.utils.transfer import DownloadError, get_download_session, safe_local_name, stream_to_file

__all__ = ['DownloadError', 'get_download_session', 'safe_local_name', 'stream_to_file']
