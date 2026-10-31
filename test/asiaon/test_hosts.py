import base64

import pytest

from inf.asiaon.hosts import HOSTS, base, gofile, googledrive, mediafire, mega


class Resp:
    def __init__(self, headers=None, status_code=200, text=''):
        self.headers, self.status_code, self.text = headers or {}, status_code, text


@pytest.mark.unittest
class TestRegistry:
    def test_order_follows_what_was_measured(self):
        assert [h.name for h in HOSTS] == ['mediafire', 'pixeldrain', 'gofile', 'googledrive', 'mega']
        assert [h.column for h in HOSTS][0] == 'mediafire_download_url'

    def test_hosts_that_cannot_be_fetched_are_absent(self):
        names = {h.name for h in HOSTS}
        assert not names & {'terabox', 'vikingfile', 'buzzheavier', 'rapidgator', 'workupload'}


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

    def test_an_emptied_folder_is_gone_not_a_failure(self):
        with pytest.raises(base.DownloadError) as info:
            gofile.pick_file(self._folder())
        assert info.value.gone and info.value.kind == 'emptied'


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


K = 'T4bK7JREgkxuIaAF0Vj0qIejocbcZ_rw5BBH5tzIhPY'


@pytest.mark.unittest
class TestMega:
    KEY_WORDS = [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10, 0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20]

    def _key(self):
        import struct
        raw = struct.pack('>8I', *self.KEY_WORDS)
        return base64.urlsafe_b64encode(raw).decode().rstrip('=')

    @pytest.mark.parametrize('url, expected', [
        (f'https://mega.nz/file/UZRACZqb#{K}', ('UZRACZqb', K)),
        (f'https://mega.nz/#!UZRACZqb!{K}', ('UZRACZqb', K)),
        (f'https://mega.nz/folder/UZRACZqb#{K}', None),
        (f'https://mega.nz/#F!UZRACZqb!{K}', None),
        ('https://example.com/file/x', None),
    ])
    def test_link_parsing(self, url, expected):
        assert mega.parse_link(url) == expected

    def test_key_folding(self):
        import struct
        aes_key, nonce = mega.fold_key(self._key())
        w = self.KEY_WORDS
        assert aes_key == struct.pack('>4I', w[0] ^ w[4], w[1] ^ w[5], w[2] ^ w[6], w[3] ^ w[7])
        assert nonce == struct.pack('>4I', w[4], w[5], 0, 0)

    def test_streamed_decryption_roundtrip_and_attribute_name(self):
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        aes_key, nonce = mega.fold_key(self._key())
        plain = bytes(range(256)) * 400  # not a multiple of the AES block size on purpose
        plain += b'tail'
        cipher = Cipher(algorithms.AES(aes_key), modes.CTR(nonce)).encryptor().update(plain)

        class Raw:
            def iter_content(self, chunk_size):
                for i in range(0, len(cipher), 777):  # chunks that cut across AES blocks
                    yield cipher[i:i + 777]

            def close(self):
                pass

        dec = Cipher(algorithms.AES(aes_key), modes.CTR(nonce)).decryptor()
        out = b''.join(mega._DecryptingResponse(Raw(), dec, len(plain)).iter_content())
        assert out == plain

        meta = b'MEGA{"n":"Pack Vol.1.zip"}'
        meta += b'\x00' * (-len(meta) % 16)
        enc = Cipher(algorithms.AES(aes_key), modes.CBC(b'\x00' * 16)).encryptor().update(meta)
        assert mega.decrypt_name(base64.urlsafe_b64encode(enc).decode().rstrip('='), aes_key) == 'Pack Vol.1.zip'
        assert mega.decrypt_name('not base64 at all!', aes_key) is None

    @pytest.mark.parametrize('code, gone, blocked', [
        (-9, True, False), (-16, True, False), (-17, False, True), (-6, False, True), (-4, False, True),
        (-7, False, False),
    ])
    def test_api_error_codes(self, monkeypatch, code, gone, blocked):
        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return [code]

        monkeypatch.setattr(mega.requests, 'post', lambda *a, **k: R())
        monkeypatch.setattr(mega.time, 'sleep', lambda s: None)
        with pytest.raises(base.DownloadError) as exc:
            mega._ask_api('UZRACZqb')
        assert (exc.value.gone, exc.value.blocked) == (gone, blocked)

    def test_eagain_is_retried(self, monkeypatch):
        answers = iter([[-3], [-3], [{'g': 'https://x', 's': 5, 'at': ''}]])

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return next(answers)

        monkeypatch.setattr(mega.requests, 'post', lambda *a, **k: R())
        monkeypatch.setattr(mega.time, 'sleep', lambda s: None)
        assert mega._ask_api('UZRACZqb')['s'] == 5


@pytest.mark.unittest
class TestMediafireRetries:
    class Page:
        def __init__(self, text, status=200):
            self.text, self.status_code = text, status

    BUTTON = '<a id="downloadButton" href="https://download1.mediafire.com/y/f.zip">Download</a>'
    CHALLENGE = '<title>Just a moment...</title>'

    def _patch(self, monkeypatch, pages):
        used = []

        def fetch(url, attempt):
            used.append((url, attempt))
            return pages[min(len(used), len(pages)) - 1]

        monkeypatch.setattr(mediafire, 'fetch_page', fetch)
        monkeypatch.setattr(mediafire.time, 'sleep', lambda s: None)
        return used

    def test_a_challenged_page_is_asked_again_until_it_passes(self, monkeypatch, tmp_path):
        used = self._patch(monkeypatch, [self.Page(self.CHALLENGE, 403)] * 4 + [self.Page(self.BUTTON)])

        def stop(*args, **kwargs):
            raise RuntimeError('reached the transfer')

        monkeypatch.setattr(mediafire, 'get', stop)
        with pytest.raises(RuntimeError, match='reached the transfer'):
            mediafire.download_mediafire('https://www.mediafire.com/file/abc/x.zip/file', str(tmp_path))
        assert [attempt for _, attempt in used] == [1, 2, 3, 4, 5]

    def test_a_page_that_stays_challenged_counts_as_a_refusal_not_a_block(self, monkeypatch, tmp_path):
        used = self._patch(monkeypatch, [self.Page(self.CHALLENGE, 403)])
        with pytest.raises(base.DownloadError) as info:
            mediafire.download_mediafire('https://www.mediafire.com/file/abc/x.zip/file', str(tmp_path))
        assert len(used) == mediafire._PAGE_ATTEMPTS
        assert info.value.refused and not info.value.blocked and not info.value.gone

    def test_a_deleted_page_is_gone_at_once(self, monkeypatch, tmp_path):
        used = self._patch(monkeypatch, [self.Page('Invalid or Deleted File', 200)])
        with pytest.raises(base.DownloadError) as info:
            mediafire.download_mediafire('https://www.mediafire.com/file/abc/x.zip/file', str(tmp_path))
        assert info.value.gone and len(used) == 1

    def test_a_mangled_prefix_is_cleaned(self):
        assert mediafire.clean_url('https://wwhttp://www.mediafire.com/file/abc/x.zip/file') == \
            'http://www.mediafire.com/file/abc/x.zip/file'
        assert mediafire.clean_url('https://www.mediafire.com/file/abc/x.zip/file') == \
            'https://www.mediafire.com/file/abc/x.zip/file'

    def test_fingerprints_unknown_to_the_installed_client_are_left_out(self, monkeypatch):
        monkeypatch.setattr(mediafire, '_FINGERPRINTS', ['no-such-browser', 'chrome120'])
        assert mediafire._fingerprints() in (['chrome120'], ['chrome'])


@pytest.mark.unittest
class TestGofileColdStorage:
    class Resp:
        status_code = 403
        text = "This content has been moved to cold storage due to inactivity. To access it, you'll need to import it."

        def close(self):
            pass

    def test_a_cold_storage_answer_is_gone_not_a_fault(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gofile, 'get_plain_session', lambda: type('S', (), {'cookies': type('C', (), {
            'set': lambda *a, **k: None})()})())
        monkeypatch.setattr(gofile, '_contents', lambda session, ident: (
            'tok', {'children': {'1': {'type': 'file', 'name': 'a.zip', 'link': 'https://store/a.zip'}}}))
        monkeypatch.setattr(gofile, 'get', lambda *a, **k: self.Resp())
        with pytest.raises(base.DownloadError) as info:
            gofile.download_gofile('https://gofile.io/d/abc', str(tmp_path))
        assert info.value.gone and info.value.kind == 'cold_storage'


@pytest.mark.unittest
class TestSplitVolumes:
    class Resp:
        status_code = 200

        def __init__(self, disposition):
            self.headers = {'Content-Disposition': disposition, 'Content-Type': 'application/zip'}
            self.closed = False

        def close(self):
            self.closed = True

    @pytest.mark.parametrize('name', ['x.zip.001', 'x.7z.002', 'x.RAR.01', 'x.part2.rar', 'x.z01'])
    def test_volumes_of_a_split_archive_are_recognised(self, name):
        assert base.is_split_volume(name)

    @pytest.mark.parametrize('name', ['x.zip', 'x.rar', 'x.7z', 'photos.2019.05.06.zip', '', None])
    def test_whole_archives_are_not(self, name):
        assert not base.is_split_volume(name)

    def test_a_volume_is_turned_away_before_any_bytes_are_read(self):
        resp = self.Resp('attachment; filename="pack.zip.001"')
        with pytest.raises(base.DownloadError) as info:
            base.check_file_response(resp, 'mediafire', 'the file')
        assert info.value.unusable and not info.value.gone and not info.value.retryable and resp.closed

    def test_a_whole_archive_passes(self):
        base.check_file_response(self.Resp('attachment; filename="pack.zip"'), 'mediafire', 'the file')
