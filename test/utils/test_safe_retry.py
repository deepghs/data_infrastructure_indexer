import requests
import pytest

from inf.utils.safe import _is_retryable_upload_error


def _reraise_as_runtime(err: Exception) -> RuntimeError:
    try:
        raise RuntimeError("Error while uploading 'meta.json' to the Hub.") from err
    except RuntimeError as wrapped:
        return wrapped


@pytest.mark.unittest
class TestRetryableUploadError:
    def test_bare_connection_error_is_retryable(self):
        assert _is_retryable_upload_error(requests.ConnectionError('Connection aborted.'))

    def test_wrapped_connection_error_is_retryable(self):
        # The shape that ended a run: huggingface_hub raises RuntimeError from the real cause,
        # so a check on the outermost type alone sees only an unfamiliar RuntimeError.
        err = _reraise_as_runtime(requests.ConnectionError('Connection aborted.'))
        assert isinstance(err, RuntimeError)
        assert _is_retryable_upload_error(err)

    def test_wrapped_status_error_is_retryable(self):
        response = requests.Response()
        response.status_code = 504
        assert _is_retryable_upload_error(
            _reraise_as_runtime(requests.HTTPError('gateway timeout', response=response)))

    def test_wrapped_client_error_is_not_retryable(self):
        response = requests.Response()
        response.status_code = 401
        assert not _is_retryable_upload_error(
            _reraise_as_runtime(requests.HTTPError('unauthorized', response=response)))

    def test_unrelated_error_is_not_retryable(self):
        assert not _is_retryable_upload_error(ValueError('bad value'))

    def test_self_referential_chain_terminates(self):
        err = RuntimeError('loop')
        err.__cause__ = err
        assert _is_retryable_upload_error(err) is False
