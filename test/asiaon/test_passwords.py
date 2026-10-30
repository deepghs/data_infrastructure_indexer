import shutil
import subprocess

import pytest

from inf.asiaon import passwords as pw

SENTENCE = ('asiaontop.com or asiaon.top. Try one of the two; '
            'new archives also include their password in the ZIP comment.')
needs_zip = pytest.mark.skipif(shutil.which('zip') is None, reason='the zip tool is needed to make encrypted fixtures')


def make_zip(path, password=None, comment=None):
    src = path.parent / 'payload'
    src.mkdir(exist_ok=True)
    (src / 'a.txt').write_text('hello')
    (src / 'b.txt').write_text('a much longer member that is not the smallest one')
    cmd = ['zip', '-q', '-j', str(path), str(src / 'a.txt'), str(src / 'b.txt')]
    if password:
        cmd[1:1] = ['-P', password]
    if comment:
        cmd[1:1] = ['-z']
    subprocess.run(cmd, input=comment, text=True, check=True, cwd=str(path.parent))
    return str(path)


@pytest.mark.unittest
class TestCandidates:
    def test_the_sentence_gives_the_first_domain_first(self):
        assert pw.password_candidates(SENTENCE) == ['asiaontop.com', 'asiaon.top']

    def test_a_plain_value_is_tried_first(self):
        assert pw.password_candidates('asiaon.top') == ['asiaon.top', 'asiaontop.com']

    def test_the_zip_comment_comes_before_everything(self):
        got = pw.password_candidates('asiaon.top', 'Website: https://x\r\nPassword: secret.example.com')
        assert got[0] == 'secret.example.com' and got[1:] == ['asiaon.top', 'asiaontop.com']

    def test_junk_around_the_value_is_ignored(self):
        assert pw.password_candidates('asiaon.top\n+')[0] == 'asiaon.top'
        assert pw.password_candidates('asiaon.top\ne')[0] == 'asiaon.top'

    def test_no_stored_value_still_offers_the_known_ones(self):
        assert pw.password_candidates(None) == ['asiaon.top', 'asiaontop.com']
        assert pw.password_candidates('') == ['asiaon.top', 'asiaontop.com']


@pytest.mark.unittest
@needs_zip
class TestFindPassword:
    def test_old_archive_with_the_old_password(self, tmp_path):
        assert pw.find_password(make_zip(tmp_path / 'a.zip', 'asiaon.top'), 'asiaon.top') == 'asiaon.top'

    def test_new_archive_is_opened_by_the_first_domain_of_the_sentence(self, tmp_path):
        assert pw.find_password(make_zip(tmp_path / 'a.zip', 'asiaontop.com'), SENTENCE) == 'asiaontop.com'

    def test_the_comment_password_is_found_even_when_the_index_is_no_help(self, tmp_path):
        comment = 'Website: https://asiaontop.com\r\nPassword: custom-secret'
        path = make_zip(tmp_path / 'a.zip', 'custom-secret', comment)
        assert pw.find_password(path, SENTENCE) == 'custom-secret'

    def test_a_wrong_stored_value_falls_back_to_the_known_passwords(self, tmp_path):
        # the index says asiaon.top but the archive was made with the new password
        assert pw.find_password(make_zip(tmp_path / 'a.zip', 'asiaontop.com'), 'asiaon.top') == 'asiaontop.com'

    def test_junk_in_the_stored_value_does_not_matter(self, tmp_path):
        assert pw.find_password(make_zip(tmp_path / 'a.zip', 'asiaon.top'), 'asiaon.top\n+') == 'asiaon.top'

    def test_an_archive_nothing_opens_is_an_error_not_a_guess(self, tmp_path):
        with pytest.raises(pw.WrongPassword):
            pw.find_password(make_zip(tmp_path / 'a.zip', 'somethingelse'), SENTENCE)

    def test_an_unencrypted_archive_needs_no_password(self, tmp_path):
        assert pw.find_password(make_zip(tmp_path / 'a.zip'), SENTENCE) is None

    def test_the_verdict_comes_from_reading_the_archive_not_from_the_text(self, tmp_path):
        # the sentence itself is not a password, and the archive says so
        path = make_zip(tmp_path / 'a.zip', 'asiaontop.com')
        assert pw.find_password(path, SENTENCE) != SENTENCE
