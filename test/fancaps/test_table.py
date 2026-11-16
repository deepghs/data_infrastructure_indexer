import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from inf.fancaps import table as t
from inf.fancaps.client import Episode


def rows_of(ids, ep=1, bangumi=1, name='Bangumi', thumb='https://ant.fancaps.net'):
    episode = Episode(ep, f'Episode {ep}', f'https://fancaps.net/anime/episodeimages.php?{ep}-x/y')
    return t.episode_rows(bangumi, name, 'https://fancaps.net/anime/showimages.php?1-x', episode,
                          [(i, f'{thumb}/{i}.jpg') for i in ids])


def reference(*tables):
    """What the earlier job wrote: the whole frame sorted by id descending, ``to_parquet(index=False)``."""
    df = pd.concat([x.to_pandas() for x in tables], ignore_index=True)
    return df.sort_values(by=['id'], ascending=[False])


@pytest.fixture()
def small_groups(monkeypatch):
    monkeypatch.setattr(t, 'ROW_GROUP_SIZE', 50)


@pytest.mark.unittest
class TestLayout:
    def test_schema_is_the_published_one(self):
        assert [f.name for f in t.SCHEMA] == t.COLUMNS
        assert [str(f.type) for f in t.SCHEMA] == [
            'int64', 'string', 'string', 'int64', 'string', 'string', 'int64', 'string', 'string']

    def test_pandas_metadata_matches_what_pandas_writes(self, tmp_path):
        path = tmp_path / 'ref.parquet'
        reference(rows_of([3, 2, 1])).to_parquet(path, index=False)
        written = pq.ParquetFile(path).schema_arrow
        assert written.equals(t.SCHEMA, check_metadata=False)
        assert pq.ParquetFile(path).schema_arrow.metadata[b'pandas'].decode().count('"name"') == 9

    def test_url_is_derived_from_the_id_and_thumbnail_is_kept_as_given(self):
        row = rows_of([31449533], thumb='https://animethumbs.fancaps.net').to_pylist()[0]
        assert row['url'] == 'https://cdni.fancaps.net/file/fancaps-animeimages/31449533.jpg'
        assert row['thumbnail_url'] == 'https://animethumbs.fancaps.net/31449533.jpg'


@pytest.mark.unittest
class TestMerge:
    def test_result_equals_the_whole_frame_sort_and_has_the_published_physical_shape(self, tmp_path, small_groups):
        base = tmp_path / 'base.parquet'
        reference(rows_of(range(400, 0, -2), ep=1), rows_of(range(399, 1, -2), ep=2)).to_parquet(base, index=False)
        new = t.fresh_rows(pa.concat_tables([rows_of([1000, 999, 998], ep=3), rows_of([5, 201, 202, 7], ep=4)]),
                           t.scan_table(str(base)).ids)
        out = tmp_path / 'out.parquet'
        written = t.merge_table(str(base), new, str(out))
        expected = reference(pq.read_table(base), new).reset_index(drop=True)
        got = pq.read_table(out).to_pandas()
        pd.testing.assert_frame_equal(got, expected)
        assert written == len(expected)
        pf = pq.ParquetFile(out)
        assert [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)][:-1] == \
               [50] * (pf.metadata.num_row_groups - 1)
        col = pf.metadata.row_group(0).column(0)
        assert str(col.compression) == 'SNAPPY' and 'RLE_DICTIONARY' in col.encodings
        t.check_table(str(out), len(expected))

    def test_new_rows_between_existing_batches_land_in_the_right_place(self, tmp_path, small_groups):
        base = tmp_path / 'base.parquet'
        reference(rows_of(list(range(1000, 900, -1)))).to_parquet(base, index=False)
        # one id above everything, one in the middle of the base, one below everything; batches of 25 rows
        new = t.fresh_rows(rows_of([1001, 949, 3]), t.scan_table(str(base)).ids)
        out = tmp_path / 'out.parquet'
        t.merge_table(str(base), new, str(out), batch_rows=25)
        got = pq.read_table(out)['id'].to_pylist()
        assert got == sorted(set(list(range(1000, 900, -1)) + [1001, 949, 3]), reverse=True)

    def test_empty_base_and_empty_new(self, tmp_path, small_groups):
        out = tmp_path / 'a.parquet'
        assert t.merge_table(None, rows_of([3, 1, 2]), str(out)) == 3
        assert pq.read_table(out)['id'].to_pylist() == [3, 2, 1]
        out2 = tmp_path / 'b.parquet'
        assert t.merge_table(str(out), t.empty_table(), str(out2)) == 3
        assert pq.read_table(out2).equals(pq.read_table(out))

    def test_every_new_id_above_or_below_everything(self, tmp_path, small_groups):
        base = tmp_path / 'base.parquet'
        reference(rows_of(range(500, 300, -1))).to_parquet(base, index=False)
        for ids in ([900, 800], [10, 5], [900, 400, 5]):
            out = tmp_path / f'o{ids[0]}.parquet'
            new = t.fresh_rows(rows_of(ids), t.scan_table(str(base)).ids)
            t.merge_table(str(base), new, str(out), batch_rows=64)
            got = pq.read_table(out)['id'].to_pylist()
            assert got == sorted(set(range(301, 501)) | set(ids), reverse=True)

    def test_fresh_rows_drop_known_and_repeated_ids(self):
        new = pa.concat_tables([rows_of([5, 6, 7], ep=1), rows_of([7, 8], ep=2)])
        out = t.fresh_rows(new, np.array([6]))
        assert out['id'].to_pylist() == [8, 7, 5]
        assert out['episode_id'].to_pylist() == [2, 1, 1]               # the first seen copy of 7 is kept

    def test_check_table_rejects_a_wrong_count_and_a_bad_order(self, tmp_path):
        good = tmp_path / 'good.parquet'
        reference(rows_of([3, 2, 1])).to_parquet(good, index=False)
        t.check_table(str(good), 3)
        with pytest.raises(ValueError, match='expected 4'):
            t.check_table(str(good), 4)
        bad = tmp_path / 'bad.parquet'
        rows_of([1, 2, 3]).to_pandas().to_parquet(bad, index=False)
        with pytest.raises(ValueError, match='descending'):
            t.check_table(str(bad), 3)


@pytest.mark.unittest
class TestScan:
    def test_scan_reports_ids_counts_and_episode_descriptions(self, tmp_path, small_groups):
        path = tmp_path / 'x.parquet'
        reference(rows_of(range(300, 200, -1), ep=1, bangumi=7, name='Alpha'),
                  rows_of(range(100, 70, -1), ep=2, bangumi=7, name='Alpha')).to_parquet(path, index=False)
        scan = t.scan_table(str(path))
        assert scan.rows == 130 and len(scan.ids) == 130 and bool((scan.ids[1:] > scan.ids[:-1]).all())
        assert scan.episode_counts == {1: 100, 2: 30}
        assert scan.episodes[2] == t.EpisodeInfo(7, 'Alpha', 'https://fancaps.net/anime/showimages.php?1-x',
                                                 'Episode 2', 'https://fancaps.net/anime/episodeimages.php?2-x/y')

    def test_scan_of_a_missing_table_is_empty(self):
        scan = t.scan_table(None)
        assert scan.rows == 0 and len(scan.ids) == 0 and scan.episode_counts == {} and scan.episodes == {}

    def test_duplicate_image_ids_are_an_error(self, tmp_path):
        path = tmp_path / 'dup.parquet'
        pd.concat([rows_of([5, 4]).to_pandas(), rows_of([4, 3], ep=2).to_pandas()]).sort_values(
            'id', ascending=False).to_parquet(path, index=False)
        with pytest.raises(ValueError, match='distinct'):
            t.scan_table(str(path))
