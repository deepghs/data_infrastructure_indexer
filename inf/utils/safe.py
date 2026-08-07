import os
import re
import shutil
import time
from typing import Dict, Optional, Union, Literal

import httpx
import requests
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.archive import archive_unpack
from hfutils.operate import get_hf_client, upload_directory_as_directory
from hfutils.operate.download import is_local_file_ready
from huggingface_hub import HfApi, constants
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError, FileMetadataError, \
    configure_http_backend, reset_sessions

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
    if isinstance(err, (LocalEntryNotFoundError, FileMetadataError)):
        return True

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

    if isinstance(err, ValueError):
        return 'Force download failed due to the above error.' in str(err)

    return False


def _get_error_status_code(err: Exception) -> Optional[int]:
    seen = set()
    current = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        response = getattr(current, 'response', None)
        if response is not None:
            status_code = getattr(response, 'status_code', None)
            if status_code is not None:
                return status_code

        if isinstance(current, httpx.HTTPStatusError):
            return current.response.status_code

        current = getattr(current, '__cause__', None) or getattr(current, '__context__', None)

    matched = re.search(r'(^|\D)(?P<status>504)(\D|$)', str(err))
    if matched and 'gateway timeout' in str(err).lower():
        return int(matched.group('status'))

    return None


def _is_retryable_upload_error(err: Exception) -> bool:
    status_code = _get_error_status_code(err)
    if status_code is not None:
        return status_code in _RETRYABLE_STATUS_CODES

    # A dropped or timed-out connection says nothing about whether the commit was valid, so it
    # deserves the same retry a gateway timeout gets. Without this a single stalled request
    # ends a run that had already done all the work.
    return isinstance(err, (
        requests.ConnectionError,
        requests.Timeout,
        requests.RequestException,
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.ProtocolError,
        httpx.RemoteProtocolError,
        httpx.RequestError,
    ))


def configure_hf_http_backend(timeout: float = 120.0, max_retries: int = 3):
    """
    Give every Hugging Face Hub HTTP call a timeout.

    ``huggingface_hub`` issues its requests without one, so a hub endpoint that accepts a
    connection and then stops responding leaves the caller blocked until the kernel gives up,
    which takes roughly a quarter of an hour. That is long enough to consume a scheduled run.
    One observed case: ``create_commit`` posts ``README.md`` to ``/api/validate-yaml`` before
    uploading anything, and a stall there wasted sixteen minutes and then failed the job.

    With a timeout in place the same stall surfaces in seconds as a retryable error, which
    :func:`safe_upload_directory_as_directory` and friends already know how to handle.

    :param timeout: Connect and read timeout in seconds. Generous enough for a slow but
        progressing transfer, since the read timeout applies between bytes rather than to the
        whole request.
    :type timeout: float
    :param max_retries: Transport-level retries for the underlying adapter.
    :type max_retries: int
    """
    from .session import TimeoutHTTPAdapter

    def _backend_factory() -> requests.Session:
        session = requests.Session()
        adapter = TimeoutHTTPAdapter(timeout=timeout, max_retries=max_retries,
                                     pool_connections=32, pool_maxsize=32)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    configure_http_backend(_backend_factory)
    logging.info(f'Hugging Face HTTP backend configured with a {timeout:.0f}s timeout.')


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


def safe_upload_directory_as_directory(
        local_directory: str,
        repo_id: str,
        path_in_repo: str,
        pattern: Optional[str] = None,
        repo_type: Literal['dataset', 'model', 'space'] = 'dataset',
        revision: str = 'main',
        message: Optional[str] = None,
        time_suffix: bool = True,
        clear: bool = False,
        hf_token: Optional[str] = None,
        operation_chunk_size: Optional[int] = None,
        upload_timespan: float = 5.0,
        max_retries: int = 3,
        retry_wait_time: float = 5.0,
):
    for attempt in range(1, max_retries + 1):
        try:
            return upload_directory_as_directory(
                local_directory=local_directory,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                pattern=pattern,
                repo_type=repo_type,
                revision=revision,
                message=message,
                time_suffix=time_suffix,
                clear=clear,
                hf_token=hf_token,
                operation_chunk_size=operation_chunk_size,
                upload_timespan=upload_timespan,
            )
        except Exception as err:
            if attempt >= max_retries or not _is_retryable_upload_error(err):
                raise

            logging.warning(
                f'HF upload {repo_id!r}/{path_in_repo!r} failed with gateway timeout '
                f'on attempt {attempt}/{max_retries} - {err!r}, retry later.'
            )
            reset_sessions()
            time.sleep(retry_wait_time)

    raise AssertionError('Unreachable code reached in safe_upload_directory_as_directory.')


__all__ = [
    'configure_hf_http_backend',
    'safe_hf_hub_download',
    'safe_download_file_to_file',
    'safe_download_archive_as_directory',
    'safe_upload_directory_as_directory',
]
