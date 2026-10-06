# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""One agent serves both mechanisms, and the asset-class surface (asset_classes, class_of, config_for, the
per-book grids) is the way it reads a book's grid, so the exchange configuration exposes the same surface as
the simulation one: one class over every book, on the exchange's own grid. Found on 30 September 2026 when an
exchange state update reached a 0.6.3 example agent and `config_for_book` was missing on ExchangeConfig."""
import types

import pytest


def _agent(cfg):
    from taos.im.agents import FinanceAgentBase

    a = types.SimpleNamespace(simulation_config=cfg)
    for name in ("asset_classes", "class_of", "config_for", "price_decimals", "volume_decimals", "round_price",
                 "round_volume"):
        setattr(a, name, getattr(FinanceAgentBase, name).__get__(a))
    return a


def test_the_exchange_is_one_asset_class_over_every_book_on_its_own_grid():
    from taos.im.protocol.exchange_config import ExchangeConfig

    cfg = ExchangeConfig(book_count=3, priceDecimals=5, volumeDecimals=3)
    classes = cfg.asset_classes()
    assert [(c.name, c.books, c.priceDecimals, c.volumeDecimals) for c in classes] == [("exchange", [0, 1, 2], 5, 3)]
    assert classes[0].config is cfg
    assert cfg.config_for_book(2) is cfg
    for bad in (3, -1):
        with pytest.raises(KeyError):
            cfg.config_for_book(bad)


def test_an_agent_reads_the_exchange_grid_through_the_same_calls():
    from taos.im.protocol.exchange_config import ExchangeConfig

    a = _agent(ExchangeConfig(book_count=2, priceDecimals=5, volumeDecimals=3))
    assert a.class_of(1) == "exchange"
    assert (a.price_decimals(0), a.volume_decimals(0)) == (5, 3)
    assert a.round_price(1, 1.2345678) == 1.23457
    assert a.round_volume(1, 0.12345) == 0.123
