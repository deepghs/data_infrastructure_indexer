import pytest
import requests

from inf.danbooru.quick_dl import UndecodableImage, _classify_error
from inf.utils.download import DownloadDigestMismatch, DownloadSizeMismatch


def _http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f'HTTP Error {status}', response=response)


@pytest.mark.unittest
class TestClassifyError:
    @pytest.mark.parametrize('status', [404, 410])
    def test_missing_post_is_permanent(self, status):
        assert _classify_error(_http_error(status)) == 'permanent'

    def test_refused_fingerprint_is_blocked(self):
        # 403 must stay distinct from 429: the cure is a new session, not a slower fleet.
        assert _classify_error(_http_error(403)) == 'blocked'

    @pytest.mark.parametrize('status', [429, 500, 502, 503])
    def test_metering_and_server_faults_are_rate_limited(self, status):
        assert _classify_error(_http_error(status)) == 'rate_limit'

    def test_undecodable_image_is_permanent(self):
        # The bytes already matched the index's length and md5, so every retry would fetch the
        # same unusable file. Blacklisting is the only thing that terminates.
        err = UndecodableImage(11831758, 'https://example.invalid/a.png', ValueError('bad chunk'))
        assert _classify_error(err) == 'permanent'

    @pytest.mark.parametrize('err', [
        DownloadDigestMismatch('https://example.invalid/a.png', 'a' * 32, 'b' * 32),
        DownloadSizeMismatch('https://example.invalid/a.png', 100, 90),
    ])
    def test_corrupt_transfer_is_retried(self, err):
        # The mirror image of the case above: here the bytes are wrong, so the transfer is at
        # fault and retrying is exactly right.
        assert _classify_error(err) == 'transient'

    def test_unknown_failure_is_retried(self):
        assert _classify_error(ConnectionError('reset by peer')) == 'transient'


@pytest.mark.unittest
class TestUndecodableImage:
    def test_carries_the_context_needed_to_report_it(self):
        cause = ValueError('broken PNG file (bad header checksum in b\'eXIf\')')
        err = UndecodableImage(11831758, 'https://example.invalid/a.png', cause)
        assert err.post_id == 11831758
        assert err.url == 'https://example.invalid/a.png'
        assert err.reason is cause
        assert '11831758' in str(err)
