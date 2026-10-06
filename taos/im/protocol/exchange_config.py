# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Exchange-mode validator config Pydantic model.

Kept in its own module so the public testnet release (sim-only) can omit it
via apply_testnet.sh without disturbing the rest of taos.im.protocol.
"""
import xml.etree.ElementTree as ET

from taos.common.protocol import BaseModel


class ExchangeConfig(BaseModel):
    """
    Config object for exchange mode, mirroring the fields from MarketSimulationConfig
    that downstream scoring logic (trade.py, reward.py, persistence.py) and miner
    agents read from ``self.simulation``.  Populated from the exchange engine's XML
    (``from_xml``) so the values match the running engine; the field defaults are
    only a fallback for when the XML is unavailable.
    """

    book_count:           int
    # THE IDS, NOT JUST HOW MANY. An exchange book id IS a netuid: the engine takes its books from
    # the chain's pools and publishes agent-visible state keyed by that netuid. book_count is
    # deliberately the TRADED count (reward.py scores against it), so it is not the width of the id
    # space and the ids cannot be derived from it. Empty means "not told", and the properties below
    # fall back to the dense range so state that carries only a count behaves as it always did.
    traded_book_ids:      list[int] = []
    # ── engine precision + limits (read from MultiBookExchangeAgent in the XML) ──
    priceDecimals:        int   = 4
    volumeDecimals:       int   = 4
    baseDecimals:         int   = 8
    quoteDecimals:        int   = 10
    max_loan:             float = 10_000.0
    max_open_orders:      int   = 100
    init_price:           float = 300.0
    grace_period:         int   = 0
    duration:             int   = 86_400_000_000_000
    # ── exchange-mode runtime (not in the engine XML) ───────────────────────────
    miner_wealth:         float = 0.0
    publish_interval:     int   = 12_000_000_000  # BLOCK_TIME_NS
    time_unit:            str   = "ns"
    logDir:               str | None = None
    simulation_id:        str   = "exchange"
    block_time_ns:        int   = 12_000_000_000  # BLOCK_TIME_NS
    response_timeout:     float = 60.0
    max_response_retries: int   = 3

    @property
    def book_ids(self) -> list[int]:
        """The book ids of this exchange run: the same surface MarketSimulationConfig and
        MultiAssetSimulationConfig expose, so FinanceAgentBase.update iterates one attribute
        whichever config class the state carries.

        The engine's ids when it has told us them, because they are netuids and need not begin at
        zero; the dense range otherwise, which is what a state carrying only a count means."""
        return list(self.traded_book_ids) if self.traded_book_ids else list(range(self.book_count))

    def asset_classes(self) -> list:
        """The exchange as one asset class over every book, on its own grid: the surface an agent reads a
        book's grid from, the same calls in either mechanism."""
        from taos.im.protocol.config import AssetClass

        return [AssetClass(name="exchange", books=self.book_ids, priceDecimals=self.priceDecimals,
                           volumeDecimals=self.volumeDecimals, config=self)]

    def config_for_book(self, book_id: int) -> "ExchangeConfig":
        """The configuration governing a book: this one, for every book the exchange has."""
        if int(book_id) not in self.book_ids:
            raise KeyError(f"book id {book_id} names no book of this {self.book_count}-book exchange")
        return self

    def label(self) -> str:
        """Human-readable label for this book's parameters."""
        return "exchange"

    @classmethod
    def from_xml(cls, path: str, **overrides):
        """Build from the exchange engine's XML (same ``MultiBookExchangeAgent``
        schema as the sim config). Reads only the elements exchange_*.xml carries
        — the root element (``<Exchange>``; ``<Simulation>`` in older configs and
        in sim, and the tag name is never checked), ``MultiBookExchangeAgent``,
        ``Books`` — with
        ``.get`` + per-field defaults, so a missing attrib or a slightly different
        layout never raises (unlike the sim's ``_req``-based parser, which needs
        sim-only agent elements absent from exchange_*.xml). ``overrides`` win
        (e.g. book_count from the traded-netuid set, response_timeout from CLI)."""
        vals: dict = {}
        try:
            root = ET.parse(path).getroot()
            agents = root.find("Agents")
            mbe = agents.find("MultiBookExchangeAgent") if agents is not None else None
            books = mbe.find("Books") if mbe is not None else None

            def _put(key, src, attr, cast):
                if src is not None and attr in src.attrib:
                    vals[key] = cast(src.attrib[attr])

            _put("priceDecimals", mbe, "priceDecimals", int)
            _put("volumeDecimals", mbe, "volumeDecimals", int)
            _put("baseDecimals", mbe, "baseDecimals", int)
            _put("quoteDecimals", mbe, "quoteDecimals", int)
            _put("max_loan", mbe, "maxLoan", float)
            _put("max_open_orders", mbe, "maxOpenOrders", int)
            _put("init_price", mbe, "initialPrice", float)
            _put("grace_period", mbe, "gracePeriod", int)
            _put("duration", root, "duration", int)
            _put("time_unit", root, "timescale", str)

            # book_count = blockCount × Books.instanceCount, mirroring the sim parse.
            if books is not None and "instanceCount" in books.attrib and "blockCount" in root.attrib:
                vals["book_count"] = int(root.attrib["blockCount"]) * int(books.attrib["instanceCount"])
        except (ET.ParseError, OSError, ValueError, TypeError):
            # Unreadable/malformed XML → fall back entirely to defaults + overrides.
            pass

        vals.update(overrides)
        vals.setdefault("book_count", 0)
        return cls(**vals)
