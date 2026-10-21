# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A captured state (about 12 MB on mainnet) is compressed, written and trimmed off the listener loop: the loop only
hands it to a bounded queue and returns at once, a writer thread does the rest, the directory is trimmed on a timer,
and when the writer falls behind the oldest pending state is dropped and counted rather than the loop waiting."""
import time

from taos.im.validator import state_capture as sc


def test_submit_is_a_no_op_without_the_directory():
    w = sc.CaptureWriter(env={})
    assert w.submit(b"abc", 1) is False
    assert w._thread is None, "no writer thread starts while capture is off"


def test_submit_returns_at_once_and_drops_the_oldest_when_the_writer_lags(tmp_path, monkeypatch):
    real_write = sc._write

    def slow_write(raw, timestamp, env, trim=True):
        time.sleep(0.3)
        return real_write(raw, timestamp, env, trim=trim)

    monkeypatch.setattr(sc, "_write", slow_write)
    w = sc.CaptureWriter(env={sc.ENV_DIR: str(tmp_path)}, maxsize=2, trim_every_s=3600)
    t0 = time.monotonic()
    for ts in range(1, 11):
        assert w.submit(b"x" * 1000, ts * 1_000_000_000) is True
    assert time.monotonic() - t0 < 0.2, "ten submits against a 0.3 s writer must not wait for it"
    w.flush(timeout=10)
    kept = [ts for ts, _ in sc.captured_files(str(tmp_path))]
    assert w.dropped >= 6 and w.written == len(kept)
    assert kept[-1] == 10_000_000_000, "the newest state is never the one dropped"
    assert w.dropped + w.written == 10


def test_the_writer_writes_every_state_it_keeps_and_trims_on_its_timer(tmp_path):
    env = {sc.ENV_DIR: str(tmp_path), sc.ENV_MAX_BYTES: "1"}
    w = sc.CaptureWriter(env=env, maxsize=8, trim_every_s=0.0)
    for ts in (1, 2, 3):
        w.submit(b"y" * 3000, ts * 1_000_000_000)
    w.flush(timeout=10)
    assert [ts for ts, _ in sc.captured_files(str(tmp_path))] == [3_000_000_000], "a one-byte budget keeps the newest"
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())


def test_a_failing_write_is_counted_and_never_raises(tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")
    w = sc.CaptureWriter(env={sc.ENV_DIR: str(blocker)})
    assert w.submit(b"abc", 5) is True
    w.flush(timeout=10)
    assert w.written == 0 and w.failed == 1
