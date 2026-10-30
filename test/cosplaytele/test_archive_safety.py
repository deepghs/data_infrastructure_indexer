import os
import stat
import zipfile

import pytest

from inf.cosplaytele import archive_safety as sf
from inf.cosplaytele import storage as st

LISTING = """
7-Zip 23.01 : Copyright (c) 1999-2023 Igor Pavlov

Path = /tmp/p.rar
Type = Rar5
Physical Size = 100

----------
Path = Post/Pic/1.jpg
Folder = -
Size = 3
Attributes = A -rw-r--r--

Path = Post/Pic
Folder = +
Size = 0
Attributes = D drwxr-xr-x

Path = Post/link
Folder = -
Size = 4
Attributes = A lrwxrwxrwx
Symbolic Link = /etc
"""


@pytest.mark.unittest
class TestMemberChecks:
    @pytest.mark.parametrize('path', ['/etc/passwd', 'C:/x', '..\\evil', 'a/../../b', 'a/..', ''])
    def test_bad_paths(self, path):
        assert sf.path_problem(path)

    @pytest.mark.parametrize('path', ['Post/Pic/1.jpg', 'a b/c (1).jpg', 'dir/..hidden/x', 'ビョル/1.jpg'])
    def test_good_paths(self, path):
        assert sf.path_problem(path) is None

    def test_parse_7z_listing(self):
        header, entries = sf.parse_7z_listing(LISTING)
        assert header['Type'] == 'Rar5'
        assert [e['path'] for e in entries] == ['Post/Pic/1.jpg', 'Post/Pic', 'Post/link']
        assert sf.entry_problem(entries[0]) is None and sf.entry_problem(entries[1]) is None
        assert sf.entry_problem(entries[2]) == 'link'

    def test_special_unix_types_are_refused_even_without_a_link_field(self):
        assert 'special' in sf.entry_problem({'path': 'a', 'attributes': 'A prw-r--r--'})
        assert sf.entry_problem({'path': 'a', 'attributes': 'A'}) is None  # DOS-style attributes only

    def test_unparseable_listing_yields_nothing(self):
        assert sf.parse_7z_listing('7z: Cannot open file') == ({}, [])

    def test_a_block_without_a_path_is_refused_not_skipped(self):
        text = LISTING + "\nFolder = -\nSize = 9\n"
        with pytest.raises(sf.ArchiveUnsafe, match='without a path'):
            sf.parse_7z_listing(text)

    def test_newline_in_a_name_cannot_fake_fields(self):
        # a member named "a\nPath = ok.jpg" would otherwise read as two Path lines, or as a clean entry
        injected = LISTING.replace('Path = Post/Pic/1.jpg', 'Path = x\nPath = ok.jpg')
        with pytest.raises(sf.ArchiveUnsafe, match='Repeated field'):
            sf.parse_7z_listing(injected)
        stray = LISTING.replace('Path = Post/Pic/1.jpg', 'Path = x\nSymbolic Link')
        with pytest.raises(sf.ArchiveUnsafe, match='Malformed line'):
            sf.parse_7z_listing(stray)

    def test_control_characters_in_paths_are_refused(self):
        assert sf.path_problem('a\nb.jpg') and sf.path_problem('a\x00b') and sf.path_problem('a\x1bb')


def _zip(path, members):
    with zipfile.ZipFile(path, 'w') as zf:
        for item in members:
            if isinstance(item, zipfile.ZipInfo):
                zf.writestr(item, '/etc')
            else:
                zf.writestr(item, b'x')


@pytest.mark.unittest
class TestZipPrecheck:
    def test_clean_archive_passes(self, tmp_path):
        _zip(tmp_path / 'a.zip', ['P/1.jpg', 'P/2.jpg'])
        assert sf.check_archive_members(str(tmp_path / 'a.zip'), None) == 2

    def test_traversal_member_is_refused_before_extraction(self, tmp_path):
        _zip(tmp_path / 'a.zip', ['P/1.jpg', '../escape.txt'])
        with pytest.raises(sf.ArchiveUnsafe, match='parent directory'):
            sf.check_archive_members(str(tmp_path / 'a.zip'), None)

    def test_symlink_member_is_refused(self, tmp_path):
        info = zipfile.ZipInfo('evil')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        _zip(tmp_path / 'a.zip', [info])
        with pytest.raises(sf.ArchiveUnsafe, match='link'):
            sf.check_archive_members(str(tmp_path / 'a.zip'), None)

    def test_empty_and_corrupt_archives_fail_closed(self, tmp_path):
        _zip(tmp_path / 'e.zip', [])
        with pytest.raises(sf.ArchiveUnsafe, match='empty'):
            sf.check_archive_members(str(tmp_path / 'e.zip'), None)
        (tmp_path / 'bad.zip').write_bytes(b'nope')
        with pytest.raises(sf.ArchiveUnsafe):
            sf.check_archive_members(str(tmp_path / 'bad.zip'), None)


@pytest.mark.unittest
class TestTreeValidation:
    def test_plain_tree_passes(self, tmp_path):
        (tmp_path / 'a' / 'b').mkdir(parents=True)
        (tmp_path / 'a' / 'b' / '1.jpg').write_bytes(b'x')
        (tmp_path / 'a' / '2.jpg').write_bytes(b'x')
        assert sf.validate_tree(str(tmp_path)) == 2

    def test_symlinked_directory_is_refused(self, tmp_path):
        outside = tmp_path / 'outside'
        outside.mkdir()
        (outside / 'secret.txt').write_bytes(b'secret')
        tree = tmp_path / 'tree'
        tree.mkdir()
        (tree / 'ok.jpg').write_bytes(b'x')
        os.symlink(outside, tree / 'evil')
        with pytest.raises(sf.ArchiveUnsafe, match='Symlink'):
            sf.validate_tree(str(tree))

    def test_symlinked_file_is_refused(self, tmp_path):
        tree = tmp_path / 'tree'
        tree.mkdir()
        (tmp_path / 'secret.txt').write_bytes(b'secret')
        os.symlink(tmp_path / 'secret.txt', tree / 'a.jpg')
        with pytest.raises(sf.ArchiveUnsafe, match='Symlink'):
            sf.validate_tree(str(tree))

    def test_hard_link_is_refused(self, tmp_path):
        tree = tmp_path / 'tree'
        tree.mkdir()
        (tmp_path / 'secret.txt').write_bytes(b'secret')
        os.link(tmp_path / 'secret.txt', tree / 'a.jpg')
        with pytest.raises(sf.ArchiveUnsafe, match='Hard link'):
            sf.validate_tree(str(tree))

    def test_no_files_is_refused(self, tmp_path):
        (tmp_path / 'empty').mkdir()
        with pytest.raises(sf.ArchiveUnsafe, match='no files'):
            sf.validate_tree(str(tmp_path))


@pytest.mark.unittest
class TestUnpackPackageIsSafe:
    def test_hostile_zip_is_refused_and_nothing_is_written_outside(self, tmp_path):
        _zip(tmp_path / 'a.zip', ['P/1.jpg', '../escape.txt'])
        out = tmp_path / 'sub' / 'out'
        out.mkdir(parents=True)
        with pytest.raises(st.UnpackError, match='unsafe'):
            st.unpack_package(str(tmp_path / 'a.zip'), str(out), None)
        assert not (tmp_path / 'sub' / 'escape.txt').exists() and not any(out.iterdir())


@pytest.mark.unittest
class TestSignatureAndTool:
    RAR5 = b'Rar!\x1a\x07\x01\x00' + b'\x00' * 32

    def test_signature_must_be_at_offset_zero(self, tmp_path):
        good, bad = tmp_path / 'g.rar', tmp_path / 'b.rar'
        good.write_bytes(self.RAR5)
        bad.write_bytes(b'MZ stub' + self.RAR5)  # an SFX-style header in front of a real archive
        sf.check_signature(str(good), 'rar')
        with pytest.raises(sf.ArchiveUnsafe, match='signature'):
            sf.check_signature(str(bad), 'rar')

    def test_extension_and_content_must_agree(self, tmp_path):
        z = tmp_path / 'a.rar'
        _zip(tmp_path / 'real.zip', ['P/1.jpg'])
        z.write_bytes((tmp_path / 'real.zip').read_bytes())  # a zip wearing a .rar name
        with pytest.raises(sf.ArchiveUnsafe, match='signature'):
            sf.check_archive_members(str(z), None)

    def test_unknown_extension_is_refused(self, tmp_path):
        f = tmp_path / 'a.sh'
        f.write_bytes(self.RAR5)
        with pytest.raises(sf.ArchiveUnsafe, match='Unsupported'):
            sf.check_signature(str(f), 'sh')

    def test_missing_7z_fails_closed(self, tmp_path, monkeypatch):
        f = tmp_path / 'a.rar'
        f.write_bytes(self.RAR5)
        monkeypatch.setattr(sf.shutil, 'which', lambda name: None)
        with pytest.raises(sf.ArchiveUnsafe, match='7z is not installed'):
            sf.check_archive_members(str(f), None)

    def test_reported_type_must_match_the_extension(self, tmp_path, monkeypatch):
        f = tmp_path / 'a.rar'
        f.write_bytes(self.RAR5)
        monkeypatch.setattr(sf.shutil, 'which', lambda name: '/usr/bin/7z')
        listing = LISTING.replace('Type = Rar5', 'Type = zip')
        monkeypatch.setattr(sf, '_list_with_7z', lambda *a: sf.parse_7z_listing(listing))
        with pytest.raises(sf.ArchiveUnsafe, match="not the rar"):
            sf.check_archive_members(str(f), None)

    def test_clean_rar_listing_is_accepted(self, tmp_path, monkeypatch):
        f = tmp_path / 'a.rar'
        f.write_bytes(self.RAR5)
        clean = LISTING.split('Path = Post/link')[0]
        monkeypatch.setattr(sf.shutil, 'which', lambda name: '/usr/bin/7z')
        monkeypatch.setattr(sf, '_list_with_7z', lambda *a: sf.parse_7z_listing(clean))
        assert sf.check_archive_members(str(f), None) == 2

    def test_control_character_in_an_unpacked_name_is_refused(self, tmp_path):
        (tmp_path / 'tree').mkdir()
        (tmp_path / 'tree' / 'a\nb.jpg').write_bytes(b'x')
        with pytest.raises(sf.ArchiveUnsafe, match='Control character'):
            sf.validate_tree(str(tmp_path / 'tree'))
