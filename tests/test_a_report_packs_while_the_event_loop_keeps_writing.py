# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The report is packed in a worker thread while the event loop keeps mutating the dicts the report refers to.
On mainnet (30 September 2026, twice) that raised `RuntimeError: dictionary changed size during iteration` out
of persistence._stream_pack and the cycle's report was lost. The streamed packer must therefore read each
container in one atomic step and pack what it read, whatever another thread does to the live object meanwhile,
and the bytes must still decode to a well-formed document."""
import io
import threading

import msgpack


def _pack(data, depth):
    from taos.im.validator.persistence import _stream_pack

    buf = io.BytesIO()
    _stream_pack(msgpack.Packer(use_bin_type=True), buf.write, data, depth)
    return buf.getvalue()


def _mutate_until(stop, live):
    i = 0
    while not stop.is_set():
        live[f"k{i}"] = {"n": i, "row": [i, i + 1]}
        live.pop(f"k{i - 300}", None)
        live["gauges"][f"g{i % 50}"] = i
        live["gauges"].pop(f"g{(i + 7) % 50}", None)
        live["series"].append(i)
        if len(live["series"]) > 400:
            del live["series"][:100]
        i += 1


def test_the_streamed_pack_survives_a_writer_on_the_live_dicts_at_every_depth():
    live = {"gauges": {f"g{i}": i for i in range(50)}, "series": list(range(100)), "step": 1}
    live.update({f"k{i}": {"n": i, "row": [i, i + 1]} for i in range(300)})
    report = {"state": live, "nested": {"inner": live}, "step": 7}
    stop = threading.Event()
    writer = threading.Thread(target=_mutate_until, args=(stop, live), daemon=True)
    writer.start()
    try:
        for depth in (0, 1, 2, 3, 4):
            for _ in range(12):
                out = msgpack.unpackb(_pack(report, depth), raw=False, strict_map_key=False)
                assert out["step"] == 7
                assert isinstance(out["state"]["gauges"], dict)
                assert isinstance(out["nested"]["inner"]["series"], list)
    finally:
        stop.set()
        writer.join(timeout=5)


def test_the_streamed_bytes_stay_identical_to_a_whole_pack_of_a_still_document():
    still = {"a": {"b": [1, 2, {"c": "d"}], "e": 3.5}, "f": None, "g": [True, False]}
    for depth in (0, 1, 2, 3):
        assert _pack(still, depth) == msgpack.packb(still, use_bin_type=True)
