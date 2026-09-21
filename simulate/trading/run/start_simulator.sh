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
# THE CONFIG IS AN ARGUMENT, NOT AN INHERITED VARIABLE. The launchers set SIMULATION_CONFIG without
# exporting it, so pm2 captures no such variable and the fallback below silently decided the config
# for them. It matched only because the default is the same name; launched with -g on another config
# the engine would have come up on the wrong one.
CONFIG=${1:-${SIMULATION_CONFIG:-simulation_0}}

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
resumable() {
    local newest
    newest=$(ls -d logs/*/ 2>/dev/null | grep -E 'logs/[0-9]{8}_[0-9]{6}/$' | sort | tail -1)
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

if [ "$COLD_START" = "1" ]; then
    echo "simulator: cold start requested, starting a new simulation from config/$CONFIG.xml"
elif resumable; then
    echo "simulator: resuming the latest checkpoint (-c latest); the run and its trade ids continue"
    exec "$BIN" -c latest
else
    echo "simulator: no simulation checkpoint under logs/, so there is nothing to resume"
fi
echo "simulator: starting a new simulation from config/$CONFIG.xml"
exec "$BIN" -f "config/$CONFIG.xml"
