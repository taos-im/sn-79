# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""ExchangeConfig exposes the same book_ids surface as the simulation configs, dense over book_count,
so a consumer can iterate one attribute whichever config class the state carries."""
from taos.im.protocol.exchange_config import ExchangeConfig


def test_book_ids_are_dense_over_book_count():
    cfg = ExchangeConfig.model_construct(book_count=4)
    assert cfg.book_ids == [0, 1, 2, 3]


def test_no_books_means_no_ids():
    cfg = ExchangeConfig.model_construct(book_count=0)
    assert cfg.book_ids == []
