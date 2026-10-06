# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""An exchange book id is a netuid, so the config must admit the ids the engine actually serves.

The exchange engine takes its books from the chain: `_book_ids = sorted(netuid for netuid in
pools if netuid != 0)`, and publishes the agent-visible state keyed by that netuid
(`books[netuid] = book`). `bookId == netuid == index` is the engine's own statement of the id space.

`book_count` is deliberately the TRADED count, not the width of that space -- reward.py scores
against it and already notes it reads "the actual traded book-id set (netuids), NOT
range(book_count)". So deriving the admissible id set from the count is wrong in both directions:
it rejects a traded id above the count, and admits an untraded id below it.

With pools on netuids 2, 5 and 10, a config deriving the admissible set from the count admits only
{0,1,2}, so an agent asking about its own book 5 or 10 raises inside
`respond -> round_price -> config_for` and stops answering on that mechanism entirely.
"""
import pytest

from taos.im.protocol.exchange_config import ExchangeConfig


def _cfg(**kw):
    return ExchangeConfig(book_count=kw.pop("book_count", 3), **kw)


def test_the_traded_ids_are_admitted_even_when_they_exceed_the_count():
    cfg = _cfg(book_count=3, traded_book_ids=[2, 5, 10])
    assert cfg.book_ids == [2, 5, 10]
    for bid in (2, 5, 10):
        assert cfg.config_for_book(bid) is cfg


def test_an_id_the_exchange_does_not_serve_is_still_refused():
    cfg = _cfg(book_count=3, traded_book_ids=[2, 5, 10])
    with pytest.raises(KeyError):
        cfg.config_for_book(7)
    # and 0/1 are NOT books here, even though they are below the count
    with pytest.raises(KeyError):
        cfg.config_for_book(1)


def test_without_an_explicit_set_it_falls_back_to_the_dense_range():
    """Older state carries only the count; the previous behaviour must still hold."""
    cfg = _cfg(book_count=3)
    assert cfg.book_ids == [0, 1, 2]
    assert cfg.config_for_book(0) is cfg
    with pytest.raises(KeyError):
        cfg.config_for_book(3)


def test_the_asset_class_surface_names_the_same_books():
    cfg = _cfg(book_count=3, traded_book_ids=[2, 5, 10])
    classes = cfg.asset_classes()
    assert len(classes) == 1
    assert list(classes[0].books) == [2, 5, 10]


def test_price_and_volume_grids_resolve_for_a_traded_id():
    """The path the agents actually take: config_for -> priceDecimals."""
    cfg = _cfg(book_count=3, traded_book_ids=[2, 5, 10], priceDecimals=4, volumeDecimals=4)
    assert cfg.config_for_book(10).priceDecimals == 4
    assert cfg.config_for_book(10).volumeDecimals == 4
