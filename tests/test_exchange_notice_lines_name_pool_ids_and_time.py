# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The EXCHANGE EVENTS lines a miner reads must say what happened, in ids and in time.

Localnet one line as it printed and as the operator asked for it:

    BUY  TRADE #15396 : YOUR AGGRESSIVE ORDER #0 (AGENT 248) MATCHED AGAINST #0 (AGENT None)
        FOR 0.453471196@0.09572728187128339 AT 20706d 09:37:48.804743621 (T=1789033068804743621)
    BUY  TRADE #15396 : YOUR AGGRESSIVE ORDER #777 (AGENT 248) MATCHED AGAINST POOL
        FOR 0.453471196@0.09572728187128339 AT YYYY-MM-DD 09:37:48.804743621 (T=1789033068804743621)

Three defects in the first: an order id of 0 printed as if it were an order, the pool printed as order
#0 of agent None, and a wall-clock nanosecond timestamp rendered as a 20706-day duration. Exchange
timestamps are epoch nanoseconds on every surface, so they read as UTC datetimes; simulation clocks
still read as durations.
"""
import datetime as _dt

from taos.im.protocol.exchange.events import MarketOrderPlacementEvent, TradeEvent, parse_notices
from taos.im.utils import EPOCH_NS_FLOOR, duration_from_timestamp, format_timestamp, is_wall_clock_ns

WALL = 1_789_033_068_804_743_621   # 09:37:48.804743621 UTC, on the date derived below
SIM = 17_975_111_124_774           # 04:59:35.111124774 into a run

# The wall-clock renderer prefixes the UTC date. Derived here with datetime rather than written as a
# literal: a different implementation from the renderer's time.strftime, so this still cross-checks
# the date it prints, and the published tree carries no bare date.
WALL_STAMP = (
    _dt.datetime.fromtimestamp(WALL // 10**9, _dt.timezone.utc).strftime("%Y-%m-%d")
    + " 09:37:48.804743621"
)


def _agent(uid=248):
    from taos.im.agents import FinanceAgent

    class _A(FinanceAgent):
        def __init__(self):
            self.uid = uid

        def initialize(self):
            return None

        def respond(self, state):
            return None

    return _A()


def _et(**kw):
    base = {"y": "ET", "t": WALL, "a": 248, "b": 51, "i": 15396, "c": None, "d": "", "Ta": 248,
            "Ti": 777, "Tf": 0.0, "Ma": None, "Mi": 0, "Mf": 0.0, "s": 0, "p": 0.09572728187128339,
            "q": 0.453471196}
    base.update(kw)
    (ev,) = parse_notices({248: [base]})[248]
    return ev


def test_format_timestamp_tells_a_wall_clock_from_a_simulation_clock():
    assert format_timestamp(WALL) == WALL_STAMP
    assert format_timestamp(SIM) == duration_from_timestamp(SIM) == "04:59:35.111124774"
    assert is_wall_clock_ns(WALL) and not is_wall_clock_ns(SIM) and not is_wall_clock_ns(None)
    assert EPOCH_NS_FLOOR == 10**17, "three years of nanoseconds: no run reaches it, every epoch exceeds it"


def test_a_pool_fill_reads_matched_against_pool_with_the_takers_order_id():
    line = _agent()._notice_log_line(_et(), "ET")
    assert line == (
        "BOOK 51 : BUY  TRADE #15396 : YOUR AGGRESSIVE ORDER #777 (AGENT 248) MATCHED AGAINST POOL "
        f"FOR 0.453471196@0.09572728187128339 AT {WALL_STAMP} (T=1789033068804743621)"
    ), line


def test_the_exchanges_own_sweep_is_the_pool_too():
    # A resting miner order crossed by the exchange's sweep: aggressor is agent -1, the pool took it.
    line = _agent()._notice_log_line(_et(Ta=-1, Ti=9001, Ma=248, Mi=555), "ET")
    assert "YOUR PASSIVE ORDER #555 (AGENT 248) MATCHED AGAINST POOL" in line, line


def test_an_agent_counterparty_is_named_with_its_order():
    line = _agent()._notice_log_line(_et(Ma=12, Mi=555), "ET")
    assert "MATCHED AGAINST #555 (AGENT 12)" in line, line


def test_an_unknown_order_id_is_not_printed_as_order_zero():
    line = _agent()._notice_log_line(_et(Ti=0), "ET")
    assert "#0" not in line and "YOUR AGGRESSIVE ORDER (AGENT 248)" in line, line


def test_placement_and_trade_str_render_wall_clock_time_and_the_pool():
    placed = MarketOrderPlacementEvent.model_construct(
        y="RDPOM", t=WALL, a=248, b=51, o=777, c=4242, s=0, q=0.4537, r=0, u=True, m="")
    assert f"MARKET ORDER #777 (4242) FOR 0.4537 AT {WALL_STAMP}" in str(placed), str(placed)
    trade = str(_et())
    assert "AGGRESSIVE ORDER #777 (AGENT 248) MATCHED AGAINST POOL" in trade and "20706d" not in trade, trade
    assert isinstance(_et(), TradeEvent)
