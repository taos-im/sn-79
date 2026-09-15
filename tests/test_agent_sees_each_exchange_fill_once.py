# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner's handlers and log see each exchange fill once, however often the validator redelivers it.

Exchange-mode settled-fill notices are re-sent on every state update for fill_notice_window_seconds
(900 s by default) so a miner that was unreachable still learns of its fill. There is no
acknowledgement, so a reachable miner receives the same fill about 75 times, and until now the agent base
fired onTrade and printed the EXCHANGE EVENTS line for every one of them: localnet the same
five trades printed on nine consecutive updates. The validator's own consumers already key on the trade
id (test_et_notice_redelivery_double_counts); the agent base now does the same.
"""
import pytest

from taos.im.protocol.exchange import ExchangeStateUpdate


class _Recorder:
    def _seen(self):
        if not hasattr(self, "_calls"):
            self._calls = []
        return self._calls

    def onTrade(self, event, hotkey=None):
        self._seen().append(("onTrade", getattr(event, "tradeId", None)))

    def onOrderAccepted(self, event):
        self._seen().append(("onOrderAccepted", getattr(event, "orderId", None)))


def _agent(uid=248):
    from taos.im.agents import FinanceAgent

    class _A(_Recorder, FinanceAgent):
        def __init__(self):
            self.uid = uid
            self.accounts = {}
            self._pools = {}
            self._calls = []

        def initialize(self):
            return None

        def respond(self, state):
            return None

    return _A()


def _state(notices):
    st = ExchangeStateUpdate.model_construct()
    st.notices = notices
    st.accounts = {}
    st.config = {}
    st.pools = {51: {"price": 0.09}}
    st.dendrite = {"hotkey": "5HotKey"}
    return st


def _et(trade_id, book=51, uid=248):
    return {"y": "ET", "t": 1_789_033_068_804_743_621, "a": uid, "b": book, "i": trade_id, "c": None,
            "d": "", "Ta": uid, "Ti": 777, "Tf": 0.0, "Ma": None, "Mi": 0, "Mf": 0.0, "s": 0,
            "p": 0.09572728187128339, "q": 0.453471196}


def _rdpom(order_id, uid=248):
    return {"type": "RDPOM", "timestamp": 1_789_033_062_036_468_852, "agentId": uid, "bookId": 51,
            "orderId": order_id, "clientOrderId": 4242, "side": 0, "quantity": 0.4537, "success": True,
            "message": "", "r": 0}


def _parsed(uid, wires):
    from taos.im.protocol.exchange.events import parse_notices

    return parse_notices({uid: list(wires)})


@pytest.fixture
def logged(monkeypatch):
    import taos.im.agents as agents_mod

    lines = []
    monkeypatch.setattr(agents_mod.bt.logging, "info", lambda msg, *a, **k: lines.append(str(msg)))
    return lines


def test_a_fill_redelivered_on_later_updates_fires_and_prints_once(logged):
    agent = _agent()
    for _ in range(9):
        agent.update(_state(_parsed(248, [_et(15396)])))
    assert [c for c in agent._calls if c[0] == "onTrade"] == [("onTrade", 15396)], (
        f"onTrade must fire once per fill, got {agent._calls}"
    )
    blob = "\n".join(logged)
    assert blob.count("TRADE #15396") == 1, f"the fill was printed {blob.count('TRADE #15396')} times"


def test_a_fill_repeated_inside_one_update_is_also_collapsed():
    agent = _agent()
    agent.update(_state(_parsed(248, [_et(15396)] * 4)))
    assert [c for c in agent._calls if c[0] == "onTrade"] == [("onTrade", 15396)]


def test_distinct_fills_all_reach_the_handler():
    agent = _agent()
    agent.update(_state(_parsed(248, [_et(15396), _et(15397), _et(15398)])))
    agent.update(_state(_parsed(248, [_et(15396), _et(15397), _et(15398), _et(15399)])))
    assert [t for n, t in agent._calls if n == "onTrade"] == [15396, 15397, 15398, 15399]


def test_the_same_trade_id_on_another_book_is_another_fill():
    agent = _agent()
    agent.update(_state(_parsed(248, [_et(7, book=51), _et(7, book=52)])))
    assert len([c for c in agent._calls if c[0] == "onTrade"]) == 2


def test_acknowledgements_are_never_treated_as_repeats():
    # Two identical placements in one update are two orders; only fills are deduplicated.
    agent = _agent()
    agent.update(_state(_parsed(248, [_rdpom(777), _rdpom(777)])))
    assert len([c for c in agent._calls if c[0] == "onOrderAccepted"]) == 2


def test_the_ledger_is_bounded():
    agent = _agent()
    agent._REDELIVERY_LEDGER_CAP = 3
    agent.update(_state(_parsed(248, [_et(1), _et(2), _et(3), _et(4)])))
    assert len(agent._seen_fill_keys) == 3
    assert (51, 1) not in agent._seen_fill_keys and (51, 4) in agent._seen_fill_keys


def test_no_per_update_diagnostic_line_is_logged(logged, monkeypatch):
    """The NOTICEWIRE and NOTICEKEYS probes are gone.

    They compared what the validator packed with what arrived while notices were being lost, and
    once every redelivered fill was dropped they printed `received {} redelivered_dropped=N` on every
    update for the whole redelivery window, about 75 lines per fill saying nothing the BOOK lines had
    not already said. The per-notice log line is the record of what arrived.
    """
    import taos.im.agents as agents_mod

    warned = []
    monkeypatch.setattr(agents_mod.bt.logging, "warning", lambda msg, *a, **k: warned.append(str(msg)))
    agent = _agent()
    agent.update(_state(_parsed(248, [_et(15396)])))
    agent.update(_state(_parsed(248, [_et(15396)])))
    agent.update(_state(_parsed(249, [_et(15397)])))  # notices for another uid, none for this one
    assert not [ln for ln in logged + warned if "NOTICEWIRE" in ln or "NOTICEKEYS" in ln], (logged, warned)
