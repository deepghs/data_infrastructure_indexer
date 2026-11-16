import random

import pytest

from inf.fancaps import discover as d
from inf.fancaps.client import SiteError
from .fakes import FakeSite


def site_with(shows):
    return FakeSite({sid: {'name': f'S{sid}', 'episodes': list(eps)} for sid, eps in shows.items()})


def stored(site):
    return {str(sid): {'name': s['name'], 'slug': s['name'], 'url': '', 'episodes': list(s['episodes'])}
            for sid, s in site.shows.items()}


@pytest.mark.unittest
class TestWalkAndProbe:
    def test_walk_reads_pages_until_the_first_empty_one(self):
        site = site_with({1: range(100, 125)})
        assert [e.id for e in d.walk_show(site, 1, 'S1')] == list(range(100, 125))
        assert site.requests['list'] == 4                       # three pages of ten, five and the empty one

    def test_probe_is_one_request_and_two_when_the_last_page_is_full(self):
        site = site_with({1: range(15), 2: range(20)})
        assert d.tail_changed(site, 1, 'S1', list(range(15))) is False
        assert site.requests['list'] == 1
        site.requests['list'] = 0
        assert d.tail_changed(site, 2, 'S2', list(range(20))) is False
        assert site.requests['list'] == 2

    @pytest.mark.parametrize('how', ['head', 'middle', 'tail', 'scattered', 'adjacent', 'fills_the_page'])
    def test_any_pure_addition_is_seen_by_the_probe(self, how):
        order = list(range(1000, 1043))
        positions = {'head': [0], 'middle': [21], 'tail': [42], 'scattered': [3, 17, 30], 'adjacent': [20, 21],
                     'fills_the_page': [40, 41, 42]}[how]
        known = [e for i, e in enumerate(order) if i not in positions]
        assert d.tail_changed(site_with({1: order}), 1, 'S1', known) is True

    def test_a_new_page_opened_by_a_full_last_page_is_seen(self):
        known = list(range(20))
        assert d.tail_changed(site_with({1: range(21)}), 1, 'S1', known) is True

    def test_exhaustive_single_additions_are_never_missed(self):
        rng = random.Random(3)
        for n in (2, 9, 10, 11, 19, 20, 21, 55):
            order = list(range(n))
            for i in range(n):
                known = order[:i] + order[i + 1:]
                if known:
                    assert d.tail_changed(site_with({1: order}), 1, 'S1', known), (n, i)
        assert rng

    def test_one_addition_and_one_removal_in_the_same_show_is_invisible_to_the_probe(self):
        """The documented blind spot: the tail does not move, so only a complete walk can see it."""
        order = list(range(30))
        site_state = [e for e in order if e != 12]               # the site delisted 12 ...
        site_state.insert(5, 99)                                  # ... and added 99 near the front
        known = order[:5] + order[5:]                             # what was stored
        site = site_with({1: site_state})
        # the site list is now [0..4, 99, 5..11, 13..29]: shifted by zero after the removal point
        assert d.tail_changed(site, 1, 'S1', known) is False
        assert [e.id for e in d.walk_show(site, 1, 'S1')] == site_state


@pytest.mark.unittest
class TestRefresh:
    def test_first_run_walks_every_show(self):
        site = site_with({1: range(12), 2: range(100, 103)})
        r = d.refresh_shows(site, site.show_list(), {}, workers=2)
        assert r.stats['new'] == 2 and set(r.shows) == {'1', '2'}
        assert r.shows['1']['episodes'] == list(range(12))
        assert set(r.episodes) == set(range(12)) | {100, 101, 102}

    def test_unchanged_shows_cost_one_request_each(self):
        site = site_with({1: range(12), 2: range(100, 103), 3: range(200, 220)})
        before = stored(site)
        r = d.refresh_shows(site, site.show_list(), before, workers=2)
        assert r.stats['same'] == 3 and r.episodes == {}
        assert site.requests['list'] == 1 + 1 + 2
        assert {k: v['episodes'] for k, v in r.shows.items()} == {k: v['episodes'] for k, v in before.items()}

    def test_a_changed_show_is_walked_and_its_order_replaced_exactly(self):
        site = site_with({1: range(25)})
        before = stored(site)
        site.shows[1]['episodes'].insert(7, 500)                  # a late upload lands in the middle
        r = d.refresh_shows(site, site.show_list(), before, workers=1)
        assert r.stats['changed'] == 1
        assert r.shows['1']['episodes'] == site.shows[1]['episodes']
        assert 500 in r.episodes

    def test_a_complete_walk_finds_what_the_probe_cannot_and_notices_removals(self):
        order = list(range(30))
        site = site_with({1: order})
        before = stored(site)
        site.shows[1]['episodes'] = [e for e in order if e != 12]
        site.shows[1]['episodes'].insert(5, 99)
        assert d.refresh_shows(site, site.show_list(), before, workers=1).stats['same'] == 1
        r = d.refresh_shows(site, site.show_list(), before, full=True, workers=1)
        assert r.stats['full'] == 1 and r.stats['delisted_episodes'] == 1
        assert 99 in r.episodes and 12 not in r.shows['1']['episodes']

    def test_a_failing_show_keeps_its_previous_entry(self):
        site = site_with({1: range(5), 2: range(100, 105), 3: range(200, 205), 4: range(300, 305), 5: range(400, 405),
                          6: range(500, 505)})
        before = stored(site)
        site.shows[1]['episodes'].append(77)
        real = site.list_page

        def flaky(show_id, slug, number):
            if show_id == 1:
                raise SiteError('boom')
            return real(show_id, slug, number)

        site.list_page = flaky
        r = d.refresh_shows(site, site.show_list(), before, workers=2)
        assert r.stats['failed'] == 1 and r.shows['1']['episodes'] == list(range(5))

    def test_when_too_many_shows_fail_the_run_is_aborted(self):
        site = site_with({i: range(5) for i in range(1, 6)})
        site.list_page = lambda *a: (_ for _ in ()).throw(SiteError('down'))
        with pytest.raises(SiteError, match='not usable'):
            d.refresh_shows(site, site.show_list(), stored(site), workers=1)

    def test_shows_that_left_the_listing_keep_their_stored_entry(self):
        site = site_with({1: range(5), 2: range(100, 105)})
        before = stored(site)
        del site.shows[2]
        r = d.refresh_shows(site, site.show_list(), before, workers=1)
        assert r.shows['2'] == before['2']
