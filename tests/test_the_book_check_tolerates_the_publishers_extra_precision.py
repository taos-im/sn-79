# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The miner library rebuilds each book from the events it received and compares the result with the snapshot the
validator published. The rebuild rounds every level to the book's volume decimals; the published level keeps the
engine's full precision (RECON 557.0339 against TARGET 557.03394573 on the 6 October testnet log), so an exact
comparison flagged a mismatch on every partially filled level. A level matches within half a volume increment, a
level within that of zero counts as absent, and each book is rounded to its own grid, not the first class's."""
import copy
from types import SimpleNamespace

from taos.im.protocol.book import L2Snapshot, LevelInfo
from taos.im.utils import history as H


def _snap(bids, asks, ts=0):
    def lv(p, q):
        return LevelInfo.model_construct(price=p, quantity=q, orders=None)
    return L2Snapshot(timestamp=ts, bids={p: lv(p, q) for p, q in bids.items()}, asks={p: lv(p, q) for p, q in asks.items()})


CFG4 = SimpleNamespace(volumeDecimals=4, publish_interval=1_000_000_000)


def test_the_publishers_extra_precision_is_not_a_discrepancy():
    recon = _snap({100.0: 557.0339}, {101.0: 12.5})
    target = _snap({100.0: 557.03394573}, {101.0: 12.49996})
    matched, discrepancies, existing = recon.compare(target, CFG4)
    assert matched and discrepancies == [], discrepancies
    assert existing == {'bid': {}, 'ask': {}}, "nothing to reconcile inside the tolerance"


def test_one_increment_apart_is_still_a_discrepancy_with_the_reconciliation_volume():
    recon = _snap({100.0: 557.0339}, {})
    target = _snap({100.0: 557.0340}, {})
    matched, discrepancies, existing = recon.compare(target, CFG4)
    assert not matched and len(discrepancies) == 1 and "BID" in discrepancies[0]
    assert existing['bid'] == {100.0: 0.0001}


def test_a_level_within_half_an_increment_of_zero_is_absent():
    recon = _snap({100.0: 0.00004}, {})
    target = _snap({}, {101.0: 0.00003})
    matched, discrepancies, _ = recon.compare(target, CFG4)
    assert matched, discrepancies
    recon2 = _snap({100.0: 0.0002}, {})
    matched, discrepancies, _ = recon2.compare(_snap({}, {}), CFG4)
    assert not matched and len(discrepancies) == 1


def test_the_dict_replay_rounds_each_book_to_its_own_grid():
    snaps = {0: {'timestamp': 0, 'bids': {100.0: {'p': 100.0, 'q': 1.0, 'o': None}}, 'asks': {}},
             96: {'timestamp': 0, 'bids': {30.0: {'p': 30.0, 'q': 1.0, 'o': None}}, 'asks': {}}}
    events = {0: [{'y': 'o', 'i': 1, 't': 1, 'p': 100.0, 'q': 0.123456, 's': 0}],
              96: [{'y': 'o', 'i': 2, 't': 1, 'p': 30.0, 'q': 0.123456, 's': 0}]}
    out = H.history_batch(copy.deepcopy(snaps), events, {0: 4, 96: 2})
    assert out[0][0][1]['bids'][100.0]['q'] == round(1.123456, 4)
    assert out[96][0][1]['bids'][30.0]['q'] == round(1.123456, 2)
    # the replay advances its starting snapshot in place, so the second run starts from a fresh copy
    same = H.history_batch(copy.deepcopy(snaps), events, 4)
    assert same[96][0][1]['bids'][30.0]['q'] == round(1.123456, 4), "a single integer still applies to every book"


def test_the_history_manager_resolves_a_books_own_configuration():
    from taos.im.agents import StateHistoryManager
    single = SimpleNamespace(config=SimpleNamespace(volumeDecimals=4))
    assert StateHistoryManager._config_for(single, 5).volumeDecimals == 4
    multi = SimpleNamespace(config=SimpleNamespace(volumeDecimals=4, config_for_book=lambda b: SimpleNamespace(volumeDecimals=2 if b >= 96 else 4)))
    assert StateHistoryManager._config_for(multi, 0).volumeDecimals == 4
    assert StateHistoryManager._config_for(multi, 100).volumeDecimals == 2
