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
