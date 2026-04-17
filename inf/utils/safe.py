import os
import shutil
import time
from typing import Dict, Optional, Union, Literal

import httpx
import requests
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.archive import archive_unpack
from hfutils.operate import get_hf_client
from hfutils.operate.download import is_local_file_ready
from huggingface_hub import HfApi, constants
from huggingface_hub.utils import HfHubHTTPError, reset_sessions

_RETRYABLE_STATUS_CODES = {
    408, 409, 425, 429,
    500, 501, 502, 503, 504,
    521, 522, 523, 524,
}


def _join_repo_path(filename: str, subfolder: Optional[str] = None) -> str:
    parts = []
    if subfolder:
        parts.append(subfolder.strip('/'))
    if filename:
        parts.append(filename.strip('/'))
    return '/'.join(filter(bool, parts))


def _get_local_download_target(local_dir: Union[str, os.PathLike], filename: str,
                               subfolder: Optional[str] = None) -> str:
    relative_filename = _join_repo_path(filename=filename, subfolder=subfolder)
    return os.path.join(os.fspath(local_dir), *relative_filename.split('/'))


def _cleanup_target(path: str):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)


def _is_retryable_download_error(err: Exception) -> bool:
    if isinstance(err, HfHubHTTPError):
        status_code = getattr(getattr(err, 'response', None), 'status_code', None)
        return status_code in _RETRYABLE_STATUS_CODES

    if isinstance(err, (
            requests.ConnectionError,
            requests.Timeout,
            requests.RequestException,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ProtocolError,
            httpx.RemoteProtocolError,
            httpx.RequestError,
    )):
        return True

    if isinstance(err, OSError):
        return 'Consistency check failed' in str(err)

    return False


def safe_hf_hub_download(
        hf_client: HfApi,
        repo_id: str,
        filename: str,
        *,
        subfolder: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Union[str, os.PathLike, None] = None,
        local_dir: Union[str, os.PathLike, None] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
        token: Union[bool, str, None] = None,
        local_files_only: bool = False,
        resume_download: Optional[bool] = None,
        force_filename: Optional[str] = None,
        local_dir_use_symlinks: Union[bool, Literal['auto']] = 'auto',
        max_retries: int = 3,
        retry_wait_time: float = 5.0,
) -> str:
    for attempt in range(1, max_retries + 1):
        call_force_download = force_download or attempt > 1
        if local_dir is not None and attempt > 1:
            _cleanup_target(_get_local_download_target(local_dir=local_dir, filename=filename, subfolder=subfolder))

        try:
            return hf_client.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                repo_type=repo_type,
                revision=revision,
                cache_dir=cache_dir,
                local_dir=local_dir,
                force_download=call_force_download,
                proxies=proxies,
                etag_timeout=etag_timeout,
                token=token,
                local_files_only=local_files_only,
                resume_download=resume_download,
                force_filename=force_filename,
                local_dir_use_symlinks=local_dir_use_symlinks,
            )
        except Exception as err:
            if attempt >= max_retries or not _is_retryable_download_error(err):
                raise

            logging.warning(
                f'HF download {repo_id!r}/{_join_repo_path(filename=filename, subfolder=subfolder)!r} '
                f'failed on attempt {attempt}/{max_retries} - {err!r}, retry later.'
            )
            reset_sessions()
            time.sleep(retry_wait_time)

    raise AssertionError('Unreachable code reached in safe_hf_hub_download.')


def safe_download_file_to_file(local_file: str, repo_id: str, file_in_repo: str,
                               repo_type: Literal['dataset', 'model', 'space'] = 'dataset',
                               revision: str = 'main', soft_mode_when_check: bool = False,
                               hf_token: Optional[str] = None, hf_client: Optional[HfApi] = None,
                               max_retries: int = 3, retry_wait_time: float = 5.0) -> str:
    hf_client = hf_client or get_hf_client(hf_token=hf_token)
    if hf_token is None:
        hf_token = getattr(hf_client, 'token', None)

    if os.path.exists(local_file) and is_local_file_ready(
            repo_id=repo_id,
            repo_type=repo_type,
            local_file=local_file,
            file_in_repo=file_in_repo,
            revision=revision,
            hf_token=hf_token,
            soft_mode=soft_mode_when_check,
    ):
        logging.info(f'Local file {local_file!r} is ready, download skipped.')
        return local_file

    with TemporaryDirectory() as td:
        downloaded_file = safe_hf_hub_download(
            hf_client,
            repo_id=repo_id,
            repo_type=repo_type,
            filename=file_in_repo,
            revision=revision,
            local_dir=td,
            max_retries=max_retries,
            retry_wait_time=retry_wait_time,
        )

        if os.path.dirname(local_file):
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
        if os.path.exists(local_file):
            _cleanup_target(local_file)
        shutil.move(downloaded_file, local_file)
        return local_file


def safe_download_archive_as_directory(local_directory: str, repo_id: str, file_in_repo: str,
                                       repo_type: Literal['dataset', 'model', 'space'] = 'dataset',
                                       revision: str = 'main', password: Optional[str] = None,
                                       hf_token: Optional[str] = None, hf_client: Optional[HfApi] = None,
                                       max_retries: int = 3, retry_wait_time: float = 5.0) -> str:
    hf_client = hf_client or get_hf_client(hf_token=hf_token)

    for attempt in range(1, max_retries + 1):
        if os.path.exists(local_directory):
            _cleanup_target(local_directory)
        os.makedirs(local_directory, exist_ok=True)

        try:
            with TemporaryDirectory() as td:
                archive_file = os.path.join(td, os.path.basename(file_in_repo))
                safe_download_file_to_file(
                    local_file=archive_file,
                    repo_id=repo_id,
                    file_in_repo=file_in_repo,
                    repo_type=repo_type,
                    revision=revision,
                    hf_token=hf_token,
                    hf_client=hf_client,
                    max_retries=1,
                    retry_wait_time=retry_wait_time,
                )
                archive_unpack(archive_file, local_directory, password=password)
                return local_directory
        except Exception as err:
            if attempt >= max_retries:
                raise

            logging.warning(
                f'HF archive download {repo_id!r}/{file_in_repo!r} failed on attempt '
                f'{attempt}/{max_retries} - {err!r}, retry later.'
            )
            reset_sessions()
            time.sleep(retry_wait_time)

    raise AssertionError('Unreachable code reached in safe_download_archive_as_directory.')


__all__ = [
    'safe_hf_hub_download',
    'safe_download_file_to_file',
    'safe_download_archive_as_directory',
]
