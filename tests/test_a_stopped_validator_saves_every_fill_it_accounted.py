# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A validator stopped cleanly saves the state it holds, so its restart loses no fill.

The validator saves its state every second scoring interval. A validator that tore down without a save of its own
would restart from the last of those, up to two intervals behind its clock. The engine, stopped after it, checkpoints
at the last state it published and resumes one step after that, so the fills between the save and the checkpoint
would never be accounted, and the open positions of every miner that traded in them would stay wrong for the rest of
the run.

So a stop waits for the listener to hand over the state the engine checkpoints at, applies every reward queued before
it, and writes both state files at the last state it accounted, under the reward lock, after the scoring service
(which writes the validator file too) has stopped. A stop with nothing accounted since a save main wrote itself
leaves the files alone. A periodic save still writing when the stop save writes does not put its older files in
place afterwards. A second signal during the stop does not cut the save short, and a save cut off by a kill leaves
the last complete files in place.
"""
import asyncio
import concurrent.futures
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

import test_scoring_shadow as tss
import test_shadow_snapshot_completeness as tsc
import taos.im.utils.save as save_mod
import taos.im.validator.cleanup as C
from taos.im.validator import persistence as P
from taos.im.validator.scoring_shadow import ScoringShadow, child_save_validator_state, compute_parity_components
from taos.im.validator.trade import update_trade_volumes

_S = tss._S
# Rebuilt at runtime rather than saved: the per-uid pnl total is summed again from the books, the prune clock restarts.
_NOT_SAVED = ("pnl_total", "prune")


class _Engine:
    mode = "simulation"

    def __init__(self, v):
        self._v = v
        self.book_ids = list(range(tss._BOOKS))

    def build_simulation_state(self):
        return {"simulation_timestamp": self._v.simulation_timestamp}

    def stop(self):
        pass


def _shell(tmp_path):
    v = tss._fresh_main()
    n = v.effective_max_uids
    v.scores = torch.zeros(n)
    v.gentrx_scores = torch.zeros(n)
    v.hotkeys = [f"hk{u}" for u in range(n)]
    v.unnormalized_scores = {}
    v.deregistered_uids = []
    v.miner_stats = {}
    v.miner_presence = {}
    v.simulation_timestamp = 0
    v.device = "cpu"
    v.config.neuron = SimpleNamespace(reset=False)
    v.simulation.simulation_id = "run"
    v.engine = _Engine(v)
    v.validator_state_file = str(tmp_path / "validator.mp")
    v.simulation_state_file = str(tmp_path / "simulation.mp")
    return v


def _validator(tmp_path):
    v = _shell(tmp_path)
    v.save_state_executor = ThreadPoolExecutor(max_workers=1)
    v._scoring_shadow = None
    v._scoring_proc_cutover = False
    v._last_defrag_hour = 0
    for flag in ("shared_state_rewarding", "shared_state_saving", "shared_state_reporting",
                 "maintaining", "compressing", "querying", "_cleanup_done"):
        setattr(v, flag, False)
    v._reward_lock = asyncio.Lock()
    v.main_loop = asyncio.new_event_loop()
    threading.Thread(target=v.main_loop.run_forever, daemon=True).start()
    return v


async def _reward(v, st, compute_s=0.0, score=None):
    """The reward's shape: under the reward lock it accounts the state, then awaits the score compute and applies it."""
    async with v._reward_lock:
        v.shared_state_rewarding = True
        try:
            update_trade_volumes(v, st)
            if compute_s:
                await asyncio.sleep(compute_s)
            if score is not None:
                v.scores[score[0]] = score[1]
        finally:
            v.shared_state_rewarding = False


def _handle(v, st, **kw):
    """What handling a state does before it is answered: the clock and the step move and the reward is queued."""
    v.simulation_timestamp = st.timestamp
    v.step += 1
    return asyncio.run_coroutine_threadsafe(_reward(v, st, **kw), v.main_loop)


def _periodic_save(v):
    assert asyncio.run_coroutine_threadsafe(P.save_state_async(v), v.main_loop).result(timeout=30)


def _reloaded(v, tmp_path):
    """What a restart reads: the validator file through the validator's own loader."""
    r = _shell(tmp_path)
    r.validator_state_file = v.validator_state_file
    P._load_validator_state(r)
    return r


def _accounted(*rounds):
    ref = tss._fresh_main()
    for st in rounds:
        update_trade_volumes(ref, st)
    return ref


def _fills(x):
    parts = {k: d for k, d in compute_parity_components(x).items() if k not in _NOT_SAVED}
    return parts, tss._mp_norm(P.snapshot_open_positions(x))


def _stat(path):
    st = os.stat(path)
    return st.st_ino, st.st_mtime_ns


def test_a_validator_stopped_between_two_saves_reloads_every_fill_it_accounted(tmp_path):
    rounds = tsc._debeta_rounds(1, 6)
    v = _validator(tmp_path)
    for st in rounds[:3]:
        _handle(v, st).result(timeout=5)
    _periodic_save(v)
    for st in rounds[3:]:
        _handle(v, st).result(timeout=5)

    C.cleanup(v)

    r = _reloaded(v, tmp_path)
    assert r.simulation_timestamp == v._last_accounted_ts == rounds[-1].timestamp, (
        "the restart starts from the last periodic save instead of the last state the validator accounted")
    assert _fills(r) == _fills(_accounted(*rounds)), "a fill accounted after the last periodic save is lost"
    sim = P._try_load_state_file(v.simulation_state_file, "simulation")
    assert sim["simulation_timestamp"] == rounds[-1].timestamp, "the simulation file was not saved with it"
    assert v._cleanup_finished


def test_rewards_queued_at_the_stop_are_applied_before_the_save(tmp_path):
    rounds = tsc._debeta_rounds(1, 5)
    v = _validator(tmp_path)
    for st in rounds[:3]:
        _handle(v, st).result(timeout=5)
    _periodic_save(v)
    # One reward is computing its scores when the stop arrives and the next waits for the lock behind it.
    _handle(v, rounds[3], compute_s=0.5, score=(1, 0.5))
    _handle(v, rounds[4])

    C.cleanup(v)

    r = _reloaded(v, tmp_path)
    assert r.simulation_timestamp == rounds[4].timestamp, "the save did not wait for the rewards queued before it"
    assert _fills(r) == _fills(_accounted(*rounds))
    assert float(r.scores[1]) == pytest.approx(0.5), "the save was taken while a reward was still computing"


def test_the_state_the_engine_checkpoints_at_is_accounted_before_the_save(tmp_path):
    """The engine publishes one more state after the validator's last answer and checkpoints at it. The listener
    handles that state without answering it and then leaves its loop; the save waits for that."""
    rounds = tsc._debeta_rounds(1, 4)
    v = _validator(tmp_path)
    for st in rounds[:3]:
        _handle(v, st).result(timeout=5)
    _periodic_save(v)
    v._listener_stopped = threading.Event()

    def _listener():
        while not v._cleanup_done:
            time.sleep(0.01)
        time.sleep(0.3)
        _handle(v, rounds[3])
        v._listener_stopped.set()

    t = threading.Thread(target=_listener, daemon=True)
    v._listener_thread = t
    t.start()

    C.cleanup(v)

    r = _reloaded(v, tmp_path)
    assert r.simulation_timestamp == rounds[3].timestamp, "the save was taken before the engine's last state"
    assert _fills(r) == _fills(_accounted(*rounds))


def test_a_listener_that_hands_over_no_state_is_closed_before_the_save(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "STOP_LISTEN_WAIT_S", 0.2)
    v = _validator(tmp_path)
    _handle(v, tsc._debeta_rounds(1, 1)[0]).result(timeout=5)
    v._listener_stopped = threading.Event()
    v._listener_thread = threading.Thread(target=lambda: None)
    t0 = time.monotonic()
    C.cleanup(v)
    assert time.monotonic() - t0 < 5
    assert v._listener_closed, "a state arriving after the save began would be handled and never saved"


def test_a_listener_on_the_stopping_thread_is_not_waited_for(tmp_path, monkeypatch):
    """On the exchange the listener runs on the main thread, the one the signal interrupts, so it cannot hand over
    anything while the stop runs."""
    monkeypatch.setattr(C, "STOP_LISTEN_WAIT_S", 30.0)
    v = _validator(tmp_path)
    v._listener_stopped = threading.Event()
    v._listener_thread = threading.current_thread()
    t0 = time.monotonic()
    C.cleanup(v)
    assert time.monotonic() - t0 < 5


def test_a_stop_with_nothing_accounted_since_the_last_save_leaves_the_files_alone(tmp_path):
    v = _validator(tmp_path)
    for st in tsc._debeta_rounds(1, 3):
        _handle(v, st).result(timeout=5)
    _periodic_save(v)
    before = (_stat(v.validator_state_file), _stat(v.simulation_state_file))

    C.cleanup(v)

    assert (_stat(v.validator_state_file), _stat(v.simulation_state_file)) == before, (
        "the stop rewrote files that already hold everything it accounted")


class _ScoringService:
    """The scoring service's save offload: it writes the validator file from its replica of main's structures."""

    initialized = True
    cutover_suspended = False

    def __init__(self, v):
        self._v = v
        self.written = None
        self.untouched_at_stop = None

    def request_save(self, path, light):
        n = child_save_validator_state(tss._snapshot_to_shadow(self._v), light, path)
        self.written = os.stat(path).st_ino
        fut = concurrent.futures.Future()
        fut.set_result((n, 0.0))
        return self._v._last_accounted_ts, fut

    def stop(self):
        self.untouched_at_stop = os.stat(self._v.validator_state_file).st_ino == self.written


def test_a_stop_after_the_scoring_service_wrote_the_file_writes_it_again_from_main(tmp_path):
    """Main cannot vouch for a file the scoring service wrote: its replica can run ahead of main and the main-only
    fields it carries are from the moment main asked. So the stop writes the file from main's own state, and stops
    the service first, so that no write of the service's lands after it."""
    rounds = tsc._debeta_rounds(1, 3)
    v = _validator(tmp_path)
    service = v._scoring_shadow = _ScoringService(v)
    v._scoring_proc_cutover = True
    for st in rounds:
        _handle(v, st).result(timeout=5)
    _periodic_save(v)
    assert os.stat(v.validator_state_file).st_ino == service.written

    C.cleanup(v)

    assert service.untouched_at_stop, "the scoring service was still running when main wrote the file"
    assert os.stat(v.validator_state_file).st_ino != service.written, "the stop kept the scoring service's file"
    r = _reloaded(v, tmp_path)
    assert r.simulation_timestamp == rounds[-1].timestamp
    assert _fills(r) == _fills(_accounted(*rounds))


class _SlowExecutor(ThreadPoolExecutor):
    def __init__(self, delay):
        super().__init__(max_workers=1)
        self._delay = delay

    def submit(self, fn, *args, **kwargs):
        def _late():
            time.sleep(self._delay)
            return fn(*args, **kwargs)
        return super().submit(_late)


def test_a_periodic_save_still_writing_does_not_put_older_files_over_the_stop_save(tmp_path):
    rounds = tsc._debeta_rounds(1, 5)
    v = _validator(tmp_path)
    v.save_state_executor = _SlowExecutor(0.3)
    for st in rounds[:3]:
        _handle(v, st).result(timeout=5)
    periodic = asyncio.run_coroutine_threadsafe(P.save_state_async(v), v.main_loop)
    deadline = time.monotonic() + 5
    while not (v.shared_state_saving and v._reward_lock.locked()) and time.monotonic() < deadline:
        time.sleep(0.005)
    # Accounted after the periodic save took its snapshot, while it is still building and writing.
    _handle(v, rounds[3])
    _handle(v, rounds[4])

    C.cleanup(v)

    assert periodic.result(timeout=30) is False, "the periodic save put its files in place after the stop save"
    r = _reloaded(v, tmp_path)
    assert r.simulation_timestamp == rounds[4].timestamp
    assert _fills(r) == _fills(_accounted(*rounds))
    assert P._try_load_state_file(v.simulation_state_file, "simulation")["simulation_timestamp"] == rounds[4].timestamp


class _UnansweredScoringService(_ScoringService):
    """Asked for a save it has not written yet when the stop arrives. Stopping it answers the save with nothing,
    as the service does."""

    def request_save(self, path, light):
        self.pending = concurrent.futures.Future()
        return self._v._last_accounted_ts, self.pending

    def stop(self):
        self.pending.set_result(None)


def test_a_periodic_save_waiting_on_the_scoring_service_gives_way_to_the_stop_save(tmp_path, monkeypatch):
    """Without the service's file, a periodic save falls back to building the validator state itself. Once the stop
    save has begun that work is wasted inside the kill timeout and its result could never be put in place."""
    rebuilt = []
    _build = P.construct_validator_state_async

    async def _counted(v):
        rebuilt.append(v.step)
        return await _build(v)

    monkeypatch.setattr(P, "construct_validator_state_async", _counted)
    rounds = tsc._debeta_rounds(1, 4)
    v = _validator(tmp_path)
    v._scoring_shadow = _UnansweredScoringService(v)
    v._scoring_proc_cutover = True
    for st in rounds[:3]:
        _handle(v, st).result(timeout=5)
    periodic = asyncio.run_coroutine_threadsafe(P.save_state_async(v), v.main_loop)
    # The periodic save has put its simulation file in place and waits for the service's validator file.
    deadline = time.monotonic() + 5
    while not os.path.exists(v.simulation_state_file) and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.05)
    _handle(v, rounds[3])

    C.cleanup(v)

    assert periodic.result(timeout=30) is False
    assert rebuilt == [], "the periodic save rebuilt the validator state after the stop save had begun"
    r = _reloaded(v, tmp_path)
    assert r.simulation_timestamp == rounds[3].timestamp
    assert _fills(r) == _fills(_accounted(*rounds))


def test_no_periodic_save_starts_once_the_stop_has_begun(tmp_path):
    v = _shell(tmp_path)
    v.last_state = SimpleNamespace(timestamp=4 * _S)
    v.shared_state_saving = False
    v.querying = False
    v._cleanup_done = True
    scheduled = []
    v.main_loop = SimpleNamespace(call_soon_threadsafe=lambda fn: scheduled.append(fn))
    P.schedule_save(v)
    assert scheduled == [], "a periodic save started during the stop competes with the stop save"


def test_stopping_the_scoring_service_answers_a_save_it_will_never_write():
    """A periodic save waits up to minutes for the service's answer; a stopped service never sends one."""
    service = ScoringShadow()
    pending = concurrent.futures.Future()
    service._save_futures[7] = pending
    service.stop()
    assert pending.done() and pending.result() is None


@pytest.fixture
def _signal_handlers():
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    yield
    for s, h in saved.items():
        signal.signal(s, h)


def test_a_second_signal_during_the_stop_lets_the_save_finish(_signal_handlers):
    import taos.im.neurons.validator as V

    calls = []
    stopping = SimpleNamespace(_cleanup_done=True, cleanup=lambda: calls.append("cleanup"))
    V.Validator._setup_signal_handlers(stopping)
    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    assert calls == [], "a second signal started another cleanup"

    stopped = SimpleNamespace(_cleanup_done=True, _cleanup_finished=True, cleanup=lambda: calls.append("cleanup"))
    V.Validator._setup_signal_handlers(stopped)
    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    running = SimpleNamespace(_cleanup_done=False, cleanup=lambda: calls.append("cleanup"))
    V.Validator._setup_signal_handlers(running)
    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    assert calls == ["cleanup", "cleanup"]


class _Killed(BaseException):
    """What a kill does to a write in progress: it ends there, and no handler runs."""


def test_a_stop_save_cut_off_by_a_kill_leaves_the_last_complete_files(tmp_path, monkeypatch):
    rounds = tsc._debeta_rounds(1, 5)
    v = _validator(tmp_path)
    for st in rounds[:3]:
        _handle(v, st).result(timeout=5)
    _periodic_save(v)
    before = {p: open(p, "rb").read() for p in (v.validator_state_file, v.simulation_state_file)}
    for st in rounds[3:]:
        _handle(v, st).result(timeout=5)

    class _KilledPacker:
        def __init__(self, **kw):
            self._calls = 0

        def pack(self, obj):
            self._calls += 1
            if self._calls > 1:
                raise _Killed()
            return b"\x00"

        def pack_map_header(self, n):
            return b"\x80"

    monkeypatch.setattr(save_mod, "msgpack", SimpleNamespace(Packer=_KilledPacker))
    with pytest.raises(_Killed):
        asyncio.run_coroutine_threadsafe(P.save_state_on_stop(v), v.main_loop).result(timeout=30)

    for path, data in before.items():
        assert open(path, "rb").read() == data, f"a cut-off save replaced {os.path.basename(path)}"
    assert P._try_load_state_file(v.validator_state_file, "validator")["simulation_timestamp"] == rounds[2].timestamp
    v.main_loop.call_soon_threadsafe(v.main_loop.stop)


def test_the_stop_fits_inside_the_kill_timeout():
    """pm2 kills a validator 60 s after the stop signal. The API server's shutdown, which comes first in simulation
    mode, the wait for the engine's last state and the save together leave the teardown at least five seconds of it."""
    from taos.im.validator.http_surface import API_GRACEFUL_SHUTDOWN_S

    assert API_GRACEFUL_SHUTDOWN_S + C.STOP_LISTEN_WAIT_S + C.STOP_SAVE_TIMEOUT_S <= 55


def _exchange_validator(tmp_path):
    v = _validator(tmp_path)
    v.engine.mode = "exchange"
    v.simulation.simulation_id = "exchange"
    return v


def test_an_exchange_stop_writes_no_file_and_waits_for_no_listener(tmp_path, monkeypatch):
    """The save on stop is simulation mode's. The exchange validator stops as it did: no wait for a last state and no
    save of its own, so its files stay those of its last periodic save."""
    monkeypatch.setattr(C, "STOP_LISTEN_WAIT_S", 30.0)
    rounds = tsc._debeta_rounds(1, 5)
    v = _exchange_validator(tmp_path)
    for st in rounds[:3]:
        _handle(v, st).result(timeout=5)
    _periodic_save(v)
    before = (_stat(v.validator_state_file), _stat(v.simulation_state_file))
    for st in rounds[3:]:
        _handle(v, st).result(timeout=5)
    v._listener_stopped = threading.Event()
    v._listener_thread = threading.Thread(target=lambda: None)

    t0 = time.monotonic()
    C.cleanup(v)

    assert time.monotonic() - t0 < 5, "the exchange stop waited for a listener"
    assert not getattr(v, "_listener_closed", False)
    assert (_stat(v.validator_state_file), _stat(v.simulation_state_file)) == before, "the exchange stop saved"
    assert not getattr(v, "_stop_save_begun", False)


def test_an_exchange_stop_keeps_its_periodic_save_and_its_exit_on_a_second_signal(tmp_path, _signal_handlers):
    import taos.im.neurons.validator as V

    v = _shell(tmp_path)
    v.engine.mode = "exchange"
    v.last_state = SimpleNamespace(timestamp=4 * _S)
    v.shared_state_saving = False
    v.querying = False
    v._cleanup_done = True
    scheduled = []
    v.main_loop = SimpleNamespace(call_soon_threadsafe=lambda fn: scheduled.append(fn))
    # schedule_save runs on the listener's thread, which has a loop; it names the current one in a log line.
    caller_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(caller_loop)
    try:
        P.schedule_save(v)
    finally:
        asyncio.set_event_loop(None)
        caller_loop.close()
    assert len(scheduled) == 1, "a periodic save due during an exchange stop no longer starts"

    calls = []
    stopping = SimpleNamespace(_cleanup_done=True, engine=SimpleNamespace(mode="exchange"),
                               cleanup=lambda: calls.append("cleanup"))
    V.Validator._setup_signal_handlers(stopping)
    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    assert calls == ["cleanup"]
