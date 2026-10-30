import base64

import pytest

from inf.asiaon.hosts import HOSTS, base, gofile, googledrive, mediafire


class Resp:
    def __init__(self, headers=None, status_code=200, text=''):
        self.headers, self.status_code, self.text = headers or {}, status_code, text


@pytest.mark.unittest
class TestRegistry:
    def test_order_follows_what_was_measured(self):
        assert [h.name for h in HOSTS] == ['mediafire', 'pixeldrain', 'gofile', 'googledrive']
        assert [h.column for h in HOSTS][0] == 'mediafire_download_url'

    def test_hosts_that_cannot_be_fetched_are_absent(self):
        names = {h.name for h in HOSTS}
        assert not names & {'mega', 'terabox', 'vikingfile', 'buzzheavier', 'rapidgator', 'workupload'}


@pytest.mark.unittest
class TestMediafire:
    def test_scrambled_link_is_decoded(self):
        scrambled = base64.b64encode(b'https://download123.mediafire.com/x/file.zip').decode()
        html = f'<a id="downloadButton" data-scrambled-url="{scrambled}" href="#">Download</a>'
        assert mediafire.extract_download_link(html) == 'https://download123.mediafire.com/x/file.zip'

    def test_plain_href_is_used_when_nothing_is_scrambled(self):
        html = '<a id="downloadButton" href="https://download1.mediafire.com/y/f.zip">Download</a>'
        assert mediafire.extract_download_link(html) == 'https://download1.mediafire.com/y/f.zip'

    def test_a_page_without_the_button_gives_nothing(self):
        assert mediafire.extract_download_link('<html>Just a moment...</html>') is None

    @pytest.mark.parametrize('text', ['Invalid or Deleted File', 'dangerous file', 'This file has been deleted'])
    def test_gone_wording(self, text):
        assert mediafire._GONE.search(text)

    def test_the_challenge_page_is_not_gone(self):
        assert not mediafire._GONE.search('<title>Just a moment...</title> Enable JavaScript and cookies to continue')


@pytest.mark.unittest
class TestGoogleDrive:
    @pytest.mark.parametrize('url, expected', [
        ('https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUv/view?usp=sharing', '1AbCdEfGhIjKlMnOpQrStUv'),
        ('https://drive.google.com/a/my.smccd.edu/file/d/1AbCdEfGhIjKlMnOpQrStUv/view', '1AbCdEfGhIjKlMnOpQrStUv'),
        ('https://drive.google.com/open?id=1AbCdEfGhIjKlMnOpQrStUv', '1AbCdEfGhIjKlMnOpQrStUv'),
        ('https://terabox.com/s/1pPGnalyDiKt3Ml8mHeXexA', None),
        ('https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUv', None),
    ])
    def test_file_id(self, url, expected):
        assert googledrive.parse_file_id(url) == expected


@pytest.mark.unittest
class TestGofile:
    def _folder(self, *names):
        files = {str(i): {'type': 'file', 'name': n, 'link': f'https://x/{n}'} for i, n in enumerate(names)}
        return {'children': files}

    def test_a_single_file_is_taken(self):
        assert gofile.pick_file(self._folder('pack.zip'))['name'] == 'pack.zip'

    def test_a_single_archive_among_other_files_is_taken(self):
        assert gofile.pick_file(self._folder('readme.txt', 'pack.zip'))['name'] == 'pack.zip'

    def test_several_archives_are_not_guessed_between(self):
        with pytest.raises(base.DownloadError, match='cannot be told'):
            gofile.pick_file(self._folder('a.zip', 'b.zip'))

    def test_a_folder_without_an_archive_among_several_files_is_refused(self):
        with pytest.raises(base.DownloadError):
            gofile.pick_file(self._folder('a.jpg', 'b.jpg'))


@pytest.mark.unittest
class TestBase:
    def test_filename_from_the_header(self):
        resp = Resp({'Content-Disposition': 'attachment; filename="Pack Vol.1.zip"'})
        assert base.filename_from_response(resp) == 'Pack Vol.1.zip'

    def test_default_without_a_header(self):
        assert base.filename_from_response(Resp()) == 'package.zip'

    def test_html_answer_is_not_a_file(self):
        with pytest.raises(base.DownloadError, match='HTML'):
            base.check_file_response(Resp({'Content-Type': 'text/html; charset=utf-8'}), 'host', 'x')

    def test_error_status_is_not_a_file(self):
        with pytest.raises(base.DownloadError) as exc:
            base.check_file_response(Resp(status_code=503), 'host', 'x')
        assert exc.value.retryable
        with pytest.raises(base.DownloadError) as exc:
            base.check_file_response(Resp(status_code=403), 'host', 'x')
        assert not exc.value.retryable

    def test_a_good_answer_passes(self):
        base.check_file_response(Resp({'Content-Type': 'application/zip'}), 'host', 'x')
