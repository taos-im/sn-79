# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator's scoring state at each engine checkpoint, so that a resume restarts both from the same instant.

The engine writes a checkpoint after it has published the state at T and taken the validator's answer, as
<run dir>/ckpt/<T>.ckptd, before it publishes the next state. An engine that is killed or crashes resumes from its
newest periodic checkpoint, behind the clock the validator holds. Re-opening the run on the resumed clock would keep
every entry of the abandoned stretch, shifted back, under the replay's own (the stretch counts twice in every window),
and would reset the inventory, the marks and the open positions as for a new run.

So the validator keeps its own state at each checkpoint and reloads it on a resume to T. The boundary is read on disk
at the next state: on its tick, before that state's query, the query-side windows are kept; at the start of its
reward, under the reward lock and before that state is accounted, every scoring structure (the scoring service's INIT
set, complete by construction) and the scores and EMAs. A snapshot is bound to its checkpoint directory's identity
and kept while the engine keeps that checkpoint.
"""
import asyncio
import logging
import os
import pickle
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from taos.im.validator.scoring_shadow import _STRUCT_NAMES, _rebuild_structs

logger = logging.getLogger(__name__)

CKPT_STORE = "ckpt"
CKPT_EXT = ".ckptd"
SNAPSHOT_DIR = "checkpoint_snapshots"
SNAPSHOT_EXT = ".snapshot"
# A bound of its own beside the engine's retention, for a store whose checkpoints are never removed.
SNAPSHOT_CAP = max(1, int(os.environ.get("CHECKPOINT_SNAPSHOTS_MAX", "8")))
_FORMAT = 1
_FINISHER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt-snapshot")


def engine_checkpoint(log_dir, ts):
    """(inode, mtime_ns) of the engine's checkpoint directory for `ts` under the run directory, None when absent.
    The engine restores the directory's write time when it marks a checkpoint to keep, so the pair names one
    checkpoint for as long as it exists."""
    if not log_dir or ts is None:
        return None
    try:
        st = os.stat(os.path.join(str(log_dir), CKPT_STORE, f"{int(ts)}{CKPT_EXT}"))
    except (OSError, TypeError, ValueError):
        return None
    return (st.st_ino, st.st_mtime_ns)


def snapshot_dir(v) -> str:
    return os.path.join(v.config.neuron.full_path, SNAPSHOT_DIR)


def snapshot_path(v, run_id, ts) -> str:
    return os.path.join(snapshot_dir(v), f"{run_id}.{int(ts)}{SNAPSHOT_EXT}")


def _parse(name):
    """(run id, ts) from a snapshot file name, None for anything else."""
    if not name.endswith(SNAPSHOT_EXT):
        return None
    run_id, _, ts = name[: -len(SNAPSHOT_EXT)].rpartition(".")
    try:
        return run_id, int(ts)
    except ValueError:
        return None


def query_fields(v) -> dict:
    """The per-query windows, read on the tick before the next state's query."""
    return {
        "miner_presence": {u: deque(d, maxlen=getattr(d, "maxlen", None))
                           for u, d in dict(getattr(v, "miner_presence", None) or {}).items()},
        "miner_stats": {u: {**s, "call_time": list(s.get("call_time", []))}
                        for u, s in dict(getattr(v, "miner_stats", None) or {}).items() if isinstance(s, dict)},
    }


def _tensor(t):
    return t.detach().clone().cpu() if hasattr(t, "detach") else None


def scoring_fields(v) -> dict:
    """The scores, the EMAs and the registration view, read under the reward lock with every earlier reward done.
    Each moves once per scoring boundary, the GenTRX pair included (its EMA and moving average step on the service's
    current scores at every boundary), so a replayed boundary would move them twice. The registration view rolls back
    with them: a deregistration the engine lost in the abandoned stretch is seen again at the next metagraph sync, and
    its reset is sent again."""
    return {
        "scores": _tensor(getattr(v, "scores", None)),
        "gentrx_scores": _tensor(getattr(v, "gentrx_scores", None)),
        "_gentrx_ema": dict(getattr(v, "_gentrx_ema", None) or {}),
        "unnormalized_scores": dict(getattr(v, "unnormalized_scores", None) or {}),
        "hotkeys": list(getattr(v, "hotkeys", None) or []),
        "deregistered_uids": list(getattr(v, "deregistered_uids", None) or []),
        "_trading_score_ema": dict(getattr(v, "_trading_score_ema", None) or {}),
        "_trading_score_ema_n": dict(getattr(v, "_trading_score_ema_n", None) or {}),
        "_trading_score_ema_ts": getattr(v, "_trading_score_ema_ts", None),
        "_debeta_pool_ema": {k: dict(d or {}) for k, d in (getattr(v, "_debeta_pool_ema", None) or {}).items()},
    }


def snapshot_meta(v, run_id, ts, ckpt) -> dict:
    return {
        "format": _FORMAT,
        "run_id": run_id,
        "ts": int(ts),
        "checkpoint": tuple(ckpt) if ckpt else None,
        "last_accounted_ts": getattr(v, "_last_accounted_ts", None),
        "last_prune_timestamp": getattr(v, "_last_prune_timestamp", None),
        "effective_max_uids": getattr(v, "effective_max_uids", None),
    }


class _PlainPickler(pickle.Pickler):
    """Pickles a defaultdict as the plain dict it holds, as the scoring service's INIT plainifies it, without a
    copy: the reload rebuilds the same shells the service does."""

    def reducer_override(self, obj):
        if isinstance(obj, defaultdict):
            return dict, (), None, None, iter(obj.items())
        return NotImplemented


def _write_records(f, v, meta, fields) -> None:
    """One record per top-level entry, so no single pickle holds the GIL for a whole structure."""
    p = _PlainPickler(f, protocol=5)

    def dump(rec):
        p.dump(rec)
        p.clear_memo()

    dump(("meta", meta))
    for name in _STRUCT_NAMES:
        obj = getattr(v, name, {})
        if isinstance(obj, dict):
            dump(("struct", name))
            for key, value in list(dict(obj).items()):
                dump(("item", key, value))
        else:
            dump(("whole", name, obj))
    dump(("fields", fields))
    dump(("end", len(_STRUCT_NAMES)))


def _finish(tmp, path) -> None:
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_snapshot(v, path, meta, fields) -> int:
    """Write the snapshot atomically (temp, fsync, rename). Returns its size in bytes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb", buffering=1 << 20) as f:
        _write_records(f, v, meta, fields)
    nbytes = os.path.getsize(tmp)
    _finish(tmp, path)
    return nbytes


def load_snapshot(path) -> dict:
    """{'meta', 'parts', 'fields'}; raises on a truncated or foreign file."""
    parts, current, meta, fields, ended = {}, None, None, None, False
    with open(path, "rb") as f:
        while True:
            try:
                rec = pickle.load(f)
            except EOFError:
                break
            kind = rec[0]
            if kind == "meta":
                meta = rec[1]
            elif kind == "struct":
                current = parts[rec[1]] = {}
            elif kind == "item":
                current[rec[1]] = rec[2]
            elif kind == "whole":
                parts[rec[1]] = rec[2]
            elif kind == "fields":
                fields = rec[1]
            elif kind == "end":
                ended = rec[1] == len(parts)
    if not ended or meta is None or meta.get("format") != _FORMAT or fields is None:
        raise ValueError(f"incomplete checkpoint snapshot {path}")
    return {"meta": meta, "parts": parts, "fields": fields}


def read_meta(path) -> dict:
    """The snapshot's first record, its meta, without reading further; raises on a file that does not start with one."""
    with open(path, "rb") as f:
        rec = pickle.load(f)
    if not (isinstance(rec, tuple) and len(rec) == 2 and rec[0] == "meta" and isinstance(rec[1], dict)):
        raise ValueError(f"{path} does not start with a snapshot meta record")
    return rec[1]


def check_snapshot(v, run_id, ts, log_dir):
    """The snapshot kept at the engine's checkpoint `ts` of `run_id`, checked from its meta record alone against the
    run, the checkpoint's clock, the clock it was taken at, the uid count and the checkpoint directory's identity:
    {'path', 'meta'}, or None with the reason logged. Nothing of the validator is touched."""
    path = snapshot_path(v, run_id, ts)
    if not os.path.exists(path):
        logger.warning(f"No scoring snapshot at the engine's checkpoint {ts} of run {run_id!r} ({path})")
        return None
    try:
        meta = read_meta(path)
    except Exception:
        logger.exception(f"The scoring snapshot {path} could not be read")
        return None
    ckpt = engine_checkpoint(log_dir, ts)
    why = None
    if meta.get("format") != _FORMAT:
        why = f"it is format {meta.get('format')}, this validator reads {_FORMAT}"
    elif meta.get("run_id") != run_id or meta.get("ts") != int(ts) or meta.get("last_accounted_ts") != int(ts):
        why = f"it was written for run {meta.get('run_id')!r} at {meta.get('last_accounted_ts')}"
    elif meta.get("effective_max_uids") != getattr(v, "effective_max_uids", None):
        why = f"it holds {meta.get('effective_max_uids')} uids, the validator {getattr(v, 'effective_max_uids', None)}"
    elif ckpt is None or tuple(meta.get("checkpoint") or ()) != tuple(ckpt):
        why = "the engine's checkpoint directory is not the one it was taken at"
    if why:
        logger.warning(f"The scoring snapshot at {ts} of run {run_id!r} is not used: {why}")
        return None
    return {"path": path, "meta": meta}


def find_snapshot(v, run_id, ts, log_dir):
    """The checked snapshot read whole ({'meta', 'parts', 'fields'}), or None. The validator's own reload streams it
    (restore_snapshot_from); this is for a reader that wants every structure at once."""
    checked = check_snapshot(v, run_id, ts, log_dir)
    if checked is None:
        return None
    try:
        return load_snapshot(checked["path"])
    except Exception:
        logger.exception(f"The scoring snapshot {checked['path']} could not be read")
        return None


def restore_snapshot(v, snap) -> None:
    """Put the validator's scoring state back to the snapshot's instant. The caller holds the reward lock."""
    for name, value in _rebuild_structs(snap["parts"]).items():
        setattr(v, name, value)
    _apply_fields(v, snap["meta"], snap["fields"])


def _apply_fields(v, meta, fields) -> None:
    v._last_accounted_ts = meta.get("last_accounted_ts")
    v._last_prune_timestamp = meta.get("last_prune_timestamp")
    for name, value in fields.items():
        if name in ("scores", "gentrx_scores"):
            if value is not None:
                setattr(v, name, value.to(getattr(v, "device", "cpu")))
        else:
            setattr(v, name, value)


class SnapshotReadError(RuntimeError):
    """A streamed reload stopped partway: `swapped` structures had been replaced by then."""

    def __init__(self, swapped, cause):
        super().__init__(f"after {swapped} of {len(_STRUCT_NAMES)} structures: {type(cause).__name__}: {cause}")
        self.swapped = swapped


def restore_snapshot_from(v, path) -> int:
    """Reload the validator's scoring state from a checked snapshot one structure at a time: a structure's records are
    read, its shell rebuilt and swapped onto the validator, and the one it replaces dropped before the next is read, so
    the peak is the state plus the largest structure. The scores, EMAs and clocks follow once the end record confirms
    every structure arrived. Raises on a record it cannot read or a file that ends early; some structures may have
    been swapped by then, so the caller recovers the whole state. Returns the structures swapped."""
    known = set(_STRUCT_NAMES)
    swapped, fields, name, plain = 0, None, None, None

    def swap():
        nonlocal swapped, name, plain
        if name is not None:
            setattr(v, name, _rebuild_structs({name: plain})[name])
            swapped += 1
            name = plain = None

    try:
        with open(path, "rb") as f:
            rec = pickle.load(f)
            if not (isinstance(rec, tuple) and rec[0] == "meta"):
                raise ValueError(f"{path} does not start with a snapshot meta record")
            meta = rec[1]
            while True:
                rec = pickle.load(f)
                kind = rec[0]
                if kind == "item" and plain is not None:
                    plain[rec[1]] = rec[2]
                elif kind in ("struct", "whole") and rec[1] in known:
                    swap()
                    if kind == "struct":
                        name, plain = rec[1], {}
                    else:
                        setattr(v, rec[1], _rebuild_structs({rec[1]: rec[2]})[rec[1]])
                        swapped += 1
                elif kind == "fields":
                    swap()
                    fields = rec[1]
                elif kind == "end" and fields is not None and rec[1] == swapped:
                    break
                else:
                    raise ValueError(f"unexpected record {kind!r}")
        _apply_fields(v, meta, fields)
    except Exception as exc:
        raise SnapshotReadError(swapped, exc) from exc
    return swapped


def prune_snapshots(v, run_id, log_dir) -> int:
    """Keep a snapshot while the engine keeps its checkpoint, and never more than SNAPSHOT_CAP; a snapshot of
    another run, or a temp left by a write that did not finish, goes. Returns the files removed."""
    root = snapshot_dir(v)
    try:
        names = os.listdir(root)
    except OSError:
        return 0
    ours, drop = [], []
    for name in names:
        parsed = _parse(name)
        if name.endswith(".tmp") or parsed is None:
            if name.endswith(".tmp"):
                drop.append(name)
            continue
        run, ts = parsed
        if run != run_id or engine_checkpoint(log_dir, ts) is None:
            drop.append(name)
        else:
            ours.append((ts, name))
    ours.sort()
    drop.extend(name for _, name in ours[:-SNAPSHOT_CAP])
    for name in drop:
        try:
            os.remove(os.path.join(root, name))
        except OSError:
            logger.debug(f"Could not remove {name}", exc_info=True)
    return len(drop)


def _kept_at(path):
    """The checkpoint identity a snapshot file was taken at (its first record), None when there is none."""
    try:
        return tuple(read_meta(path).get("checkpoint") or ())
    except Exception:
        return None


def mark_if_checkpointed(v, state, prev_ts, run_id) -> bool:
    """On the tick of a state after the one the engine checkpointed: mark the state, with the query-side windows as
    they stood after that checkpoint's query, for its reward to take the snapshot."""
    log_dir = getattr(state, "logDir", None) or getattr(getattr(v, "simulation", None), "logDir", None)
    if not run_id or not prev_ts or state.timestamp <= prev_ts:
        return False
    try:
        ckpt = engine_checkpoint(log_dir, prev_ts)
        if ckpt is None or _kept_at(snapshot_path(v, run_id, prev_ts)) == tuple(ckpt):
            return False
        state._checkpoint_snapshot = (int(prev_ts), run_id, ckpt, log_dir, query_fields(v))
    except Exception:
        logger.exception(f"Could not mark the engine's checkpoint {prev_ts} for a scoring snapshot")
        return False
    return True


async def take_due_snapshot(v, state):
    """At the start of a marked state's reward, under the reward lock: write the snapshot of the state the engine
    checkpointed, which every structure is still at. The structures are pickled under the lock; the fsync and the
    rename finish after it. Returns that finishing future, or None when nothing was due or the write failed."""
    req = getattr(state, "_checkpoint_snapshot", None)
    if not req:
        return None
    state._checkpoint_snapshot = None
    ts, run_id, ckpt, log_dir, query = req
    if getattr(v, "_last_accounted_ts", None) != ts:
        logger.warning(f"No scoring snapshot at the engine's checkpoint {ts}: the scoring state is at "
                       f"{getattr(v, '_last_accounted_ts', None)}")
        return None
    path = snapshot_path(v, run_id, ts)
    meta = snapshot_meta(v, run_id, ts, ckpt)
    fields = {**scoring_fields(v), **query}
    tmp = path + ".tmp"
    loop = asyncio.get_running_loop()

    def _write():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "wb", buffering=1 << 20) as f:
            _write_records(f, v, meta, fields)
        return os.path.getsize(tmp)

    t0 = time.time()
    try:
        nbytes = await loop.run_in_executor(None, _write)
    except Exception:
        logger.exception(f"The scoring snapshot at the engine's checkpoint {ts} could not be written")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None
    held = time.time() - t0

    def _complete():
        try:
            _finish(tmp, path)
        except OSError:
            logger.exception(f"The scoring snapshot at the engine's checkpoint {ts} could not be completed")
            return
        removed = prune_snapshots(v, run_id, log_dir)
        logger.info(f"Scoring snapshot kept at the engine's checkpoint {ts} ({nbytes / 1e6:.1f}MB, {held:.2f}s "
                    f"under the reward lock, {time.time() - t0:.2f}s in all); {removed} older removed")

    return _FINISHER.submit(_complete)
