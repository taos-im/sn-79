#!/usr/bin/env bash
# THE SIMULATOR'S LAUNCH DECISION, SO THAT A RESTART CONTINUES THE RUN.
#
# pm2 re-runs an app's registered command verbatim, so registering `taosim -f config/...` meant every
# `pm2 restart simulator` silently started a BRAND NEW simulation: fresh per-book trade id counters
# over a tape that already holds those ids, a new price path, and every scenario span computed against
# a watermark belonging to the run that just ended.
#
# The engine has both halves of the alternative already: `-c latest` resumes the newest run directory
# under logs/, and a SIGTERM is held until the next barrier so the shutdown writes a checkpoint first
# (SimulationManager::saveCheckpointOnShutdown). A resumed run reads the run directory's own saved
# config.xml, which carries the stamped id, so it keeps the same log directory and the same sim id and
# the tape remains one continuous series.
#
# exec, not a plain call: pm2 signals the process it started, and the engine must receive that SIGTERM
# itself or it never reaches the barrier that writes the shutdown checkpoint.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

# OVERRIDABLE SO THE LAUNCH DECISION CAN BE TESTED WITHOUT AN ENGINE. The decision this script makes
# -- resume, or open a new simulation -- is the whole of its behaviour, and it ends in an exec, so the
# only way to observe it is to exec something that reports its arguments.
BIN=${TAOSIM_BIN:-../build/src/cpp/taosim}

# WHERE THE ENGINE'S NARRATION GOES.
#
# At debug="1" the engine writes on the order of a megabyte a second. Routed through a process
# manager's stdout capture that is a problem in its own right: the manager buffers it, its log
# rotation runs continuously, and a rotation racing a write can take the manager down and with it
# every process it supervises. Pointing the stream at a file takes the manager out of the path.
#
# Opt-in, so a plain launch still prints to the console as before. Set SIM_NARRATION_FILE to a path
# and everything from here is appended to it instead.
if [ -n "${SIM_NARRATION_FILE:-}" ]; then
    mkdir -p "$(dirname "$SIM_NARRATION_FILE")" 2>/dev/null || true
    exec >>"$SIM_NARRATION_FILE" 2>&1
fi
# THE CONFIG IS AN ARGUMENT, NOT AN INHERITED VARIABLE. The launchers set SIMULATION_CONFIG without
# exporting it, so pm2 captures no such variable and the fallback below silently decided the config
# for them. It matched only because the default is the same name; launched with -g on another config
# the engine would have come up on the wrong one.
CONFIG=${1:-${SIMULATION_CONFIG:-multiasset_simulation_0}}

# A directory is resumable only with a SIMULATION checkpoint in it. The exchange service writes
# state.ckpt into logs/ckpt and must never be mistaken for one; the engine's own runDirLatest()
# applies the same common.ckpt test.
# RESUMABLE MEANS THE NEWEST RUN DIRECTORY, NOT ANY OF THEM.
#
# `-c latest` resumes the newest directory that HAS a checkpoint, which is not the same as the newest
# directory. A simulation writes its first checkpoint about twelve minutes in, so during that window
# the newest directory has none and "latest" silently resolves to the one before it -- resuming a
# simulation that was superseded, rewinding by however long the current one ran, while the tape, the
# data service and the validator's ledgers all keep the interim minutes. The engine then re-mints
# trade ids the downstream record already holds, and the two disagree with nothing to say why.
#
# So: ask only about the newest directory. If it cannot be resumed, opening a new simulation is the
# honest answer -- it is visible everywhere as a new sim id, where a silent rewind is visible nowhere.
# THE NEWEST RUN DIRECTORY THAT IS A RUN, NOT AN ARTEFACT.
#
# Signalling the taosim child lets pm2 autorestart the wrapper, and that restart can open a run
# directory a second before a deliberate relaunch does, holding config.xml and nothing else. The
# newest directory is then that empty one, resumable() finds no checkpoint in it, and a resumable run
# cold-starts instead: the rewind the resume logic exists to prevent, caused by a directory no
# simulation ever ticked in.
#
# A directory holding only config.xml is such an artefact. Skipping it does NOT reopen the hazard the
# newest-only rule guards against -- that rule exists so a SUPERSEDED simulation is never resumed, and
# a directory with no ticks supersedes nothing. Anything with a ckpt/ directory or any log output is
# treated as a real run and still answered on.
_newest_real_rundir() {
    # SKIP ONLY THE UNAMBIGUOUS ARTEFACT: a directory holding config.xml AND NOTHING ELSE. A pm2
    # autorestart that dies immediately leaves exactly that, and treating it as the newest run makes
    # resumable() answer about an empty directory, so a live run cold-starts and loses its tape.
    #
    # AN EMPTY DIRECTORY IS NOT THAT CASE, and must count as real. A simulation that has just been
    # opened does not write its first checkpoint for about twelve minutes, and for part of that it has
    # no files at all. Skipping it makes `-c latest` resolve to the run BEFORE it, so a restart inside
    # that window silently rewinds the engine while the tape and every ledger keep the interim
    # minutes. Counting entries excluding config.xml is equally true of an EMPTY directory, and so
    # reintroduces exactly that rewind.
    #
    # The tie-break, when the two cases cannot be told apart by content, is the one that test states:
    # opening a new simulation is visible everywhere, a silent rewind is visible nowhere. So anything
    # that is not exactly-config.xml-only counts as real.
    local d _entries
    for d in $(ls -d logs/*/ 2>/dev/null | grep -E 'logs/[0-9]{8}_[0-9]{6}/$' | sort -r); do
        _entries=$(ls -A "$d" 2>/dev/null)
        [ "$_entries" = "config.xml" ] && continue
        printf '%s' "$d"
        return 0
    done
    return 1
}

resumable() {
    local newest
    newest=$(_newest_real_rundir)
    [ -n "$newest" ] || return 1
    local d
    for d in "$newest"ckpt/*.ckptd; do
        [ -f "$d/common.ckpt" ] && return 0
    done
    return 1
}

# A COLD START IS A ONE-SHOT INTENT AND MUST NOT LIVE IN THE PM2 ENTRY.
#
# pm2 re-runs the registered command verbatim, so registering
# `bash -c 'SIM_COLD_START=1 exec bash start_simulator.sh ...'` makes EVERY later restart open a new
# simulation -- a maintenance restart, a redeploy, an automated recovery -- for a flag that was meant
# to apply once. The env var is invisible in `pm2 list` and survives `pm2 save`, so nothing about the
# host says the next restart will discard the run.
#
# The failure this produces is quiet and expensive. An engine is stopped deliberately, writes its
# shutdown checkpoint, restarts, and comes back as a brand new simulation instead of the one that was
# saved. Anything checking continuity across that restart -- that a resting order survived, that
# balances carried -- then reports on a simulation that never held what it is asking about.
#
# The marker below matches the intent it is asked to express: drop the file, restart once, and the
# wrapper removes it BEFORE exec so the restart after that resumes.
#
#     touch simulate/trading/run/.cold_start_once && pm2 restart simulator
#
# SIM_COLD_START is still honoured for a direct manual launch, where the process is the only thing it
# can affect. It does not belong in a registered pm2 entry.
COLD_ONCE=.cold_start_once
COLD_START=${SIM_COLD_START:-0}
if [ -f "$COLD_ONCE" ]; then
    rm -f "$COLD_ONCE"
    COLD_START=1
    echo "simulator: one-shot cold-start marker consumed; the restart after this one will resume"
fi

# A RESUME MUST BE THE SAME LAYOUT. `-c latest` resumes the newest run directory under its OWN saved
# config.xml, whatever config/$CONFIG.xml now names. When the configured layout changed since that run
# was opened (0.6.3: the default moved from the single-market simulation_0 to the multi-asset
# multiasset_simulation_0, and an acceptance layout's second class was replaced), resuming would bring
# back the old layout under a validator configured for the new one, the mechanism mismatch the
# acceptance gate catches after the fact. Compared here before the decision: the root element and the
# ordered list of <Background> declarations (name, path, instanceCount) of the saved config against the
# requested one. A difference opens a new simulation from the requested config and says so.
layout_of() {
    local f="$1"
    [ -f "$f" ] || { echo "missing"; return; }
    local root
    root=$(grep -oE '<(MultiAssetSimulation|Simulation)\b' "$f" | head -1 | tr -d '<')
    # The engine saves a run's config with ABSOLUTE background paths (and its own spacing); the requested config
    # names them relative to config/. Compare what identifies a layout: each background's name, the basename of
    # its path and its instanceCount (1 when absent), in document order.
    local bgs
    bgs=$(grep -oE '<Background [^>]*>' "$f" | sed -E '
        s/.*name="([^"]*)".*path="([^"]*)".*instanceCount="([^"]*)".*/\1 \2 \3/;
        t done;
        s/.*name="([^"]*)".*path="([^"]*)".*/\1 \2 1/;
        :done' | sed -E 's#([^ ]+) ([^ ]*/)?([^ /]+) ([^ ]+)#\1 \3 \4#' | tr '\n' ';')
    echo "${root}|${bgs}"
}
same_layout_as_latest() {
    local newest
    # The SAME directory resumable() answered on, or the two can disagree about which run is latest.
    newest=$(_newest_real_rundir)
    [ -n "$newest" ] || return 1
    # a run directory the engine opened always carries its config.xml; without one there is nothing to
    # compare and the decision stays what it was (resume), as the acceptance's cold-start test expects
    [ -f "${newest}config.xml" ] || return 0
    local have want
    have=$(layout_of "${newest}config.xml")
    want=$(layout_of "config/$CONFIG.xml")
    [ "$have" = "$want" ] && return 0
    echo "simulator: the latest run (${newest}) is a different layout [${have}] from config/$CONFIG.xml [${want}]; not resuming it"
    return 1
}
if [ "$COLD_START" = "1" ]; then
    echo "simulator: cold start requested, starting a new simulation from config/$CONFIG.xml"
elif resumable && same_layout_as_latest; then
    echo "simulator: resuming the latest checkpoint (-c latest); the run and its trade ids continue"
    exec "$BIN" -c latest
else
    echo "simulator: no simulation checkpoint under logs/, so there is nothing to resume"
fi
echo "simulator: starting a new simulation from config/$CONFIG.xml"
exec "$BIN" -f "config/$CONFIG.xml"
