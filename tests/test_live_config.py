from threading import Lock
from types import SimpleNamespace

from taos.im.agents import FinanceSimulationAgent as FA


class _StubAgent(FA):
    # Concrete so it can be __new__'d without ABC blocking; we never call __init__.
    def initialize(self):
        pass


def test_coerce_live_value():
    assert FA._coerce_live_value("0.2") == 0.2
    assert FA._coerce_live_value("256") == 256.0
    assert FA._coerce_live_value("false") == "false"
    assert FA._coerce_live_value(7) == 7


def test_parse_kv_csv():
    assert FA._parse_kv_csv("a=1,b=2.5,c=hi") == {"a": 1.0, "b": 2.5, "c": "hi"}
    assert FA._parse_kv_csv("") == {}
    assert FA._parse_kv_csv("  x = 3 ") == {"x": 3.0}
    assert FA._parse_kv_csv("junk,noeq,ok=1") == {"ok": 1.0}


def test_parse_live_config_body_csv_and_json():
    inst = _StubAgent.__new__(_StubAgent)
    assert inst._parse_live_config_body("a=1,b=2") == {"a": 1.0, "b": 2.0}
    assert inst._parse_live_config_body('{"agent_params":"x=3"}') == {"x": 3.0}
    assert inst._parse_live_config_body('{"y":4,"z":"hi"}') == {"y": 4, "z": "hi"}
    assert inst._parse_live_config_body("") == {}
    assert inst._parse_live_config_body("   ") == {}


def test_apply_pending_live_config_mutates_and_reports_changes():
    inst = _StubAgent.__new__(_StubAgent)
    inst.config = SimpleNamespace(agent_range=256.0, foo="a")
    inst._live_cfg_lock = Lock()
    inst._live_cfg_pending = {"agent_range": 128.0, "foo": "a", "new_key": 5.0}
    reloaded = {}
    inst.on_config_reload = lambda changed: reloaded.update(changed)

    inst._apply_pending_live_config()

    assert inst.config.agent_range == 128.0
    assert inst.config.new_key == 5.0
    assert inst._live_cfg_pending is None
    # foo was identical -> not reported as changed; only real changes fire the hook
    assert set(reloaded) == {"agent_range", "new_key"}


def test_apply_pending_noop_when_poller_never_started():
    inst = _StubAgent.__new__(_StubAgent)
    inst.config = SimpleNamespace(agent_range=256.0)
    # No _live_cfg_pending attribute at all -> cheap fast-path returns, no error.
    inst._apply_pending_live_config()
    assert inst.config.agent_range == 256.0
