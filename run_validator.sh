#!/bin/bash
# Validator launcher: starts the C++ simulator and the validator process under
# pm2, with an optional tmux dashboard.
#
# GenTRX distributed training (optional):
#   -G              enable as sibling validator
#   -G aggregator   enable as uid-0 aggregator (subnet operator only)
#   -Q <url>        gradient server URL — skip auto-start and use a remote GPU
REPO_ROOT=$(pwd)
ENDPOINT="wss://entrypoint-finney.opentensor.ai:443"
WALLET_PATH=~/.bittensor/wallets/
WALLET_NAME=taos
HOTKEY_NAME=validator
NETUID=""
CHECKPOINT=0
LOG_LEVEL=info
PD_KEY="\"\""
PROM_PORT=9001
VALIDATOR_PORT=8000
TIMEOUT=3.0
SIMULATION_CONFIG=simulation_0
PRESERVE_SIMULATOR=0
USE_TMUX=1
MECHID=0

# ── GenTRX distributed training ───────────────────────────────────────────────
GENTRX_MODE=""
GRAD_URL=""
GRAD_PORT="${GRAD_PORT:-8100}"
GRAD_CORES_COUNT="${GRAD_CORES_COUNT:-0}"
export GRAD_CORES_COUNT
_EXPLICIT_GRAD_URL=0

# Pre-parse -E before getopts (env file sourced after defaults, before CLI flags).
_ENV_FILE=""
_prev=""
for _a in "$@"; do
    [ "$_prev" = "-E" ] && _ENV_FILE="$_a"
    _prev="$_a"
done
if [ -n "$_ENV_FILE" ]; then
    [ -f "$_ENV_FILE" ] || { echo "ERROR: env file '$_ENV_FILE' not found" >&2; exit 1; }
    # shellcheck source=/dev/null
    . "$_ENV_FILE"
fi

# Load machine-local secrets (~/.taos.env wins over -E env file).
# shellcheck source=/dev/null
[ -f "$HOME/.taos.env" ] && . "$HOME/.taos.env"

# Restore saved GenTRX mode from $REPO_ROOT/.env.
# shellcheck source=/dev/null
[ -f "$REPO_ROOT/.env" ] && . "$REPO_ROOT/.env"

# Normalize -G: allow `-G` with no argument to mean sibling mode.
_norm_args=()
_ia=1
while [ "$_ia" -le "$#" ]; do
    _cur="${!_ia}"
    if [ "$_cur" = "-G" ]; then
        _next_idx=$((_ia + 1))
        if [ "$_next_idx" -le "$#" ]; then
            _next="${!_next_idx}"
            case "$_next" in
                aggregator|agg|a) _norm_args+=("-G" "aggregator"); _ia=$((_ia + 2));;
                sibling|sib|s)    _norm_args+=("-G" "sibling");    _ia=$((_ia + 2));;
                -*|"")            _norm_args+=("-G" "sibling");    _ia=$((_ia + 1));;
                *)                _norm_args+=("-G" "sibling");    _ia=$((_ia + 1));;
            esac
        else
            _norm_args+=("-G" "sibling"); _ia=$((_ia + 1))
        fi
    else
        _norm_args+=("$_cur"); _ia=$((_ia + 1))
    fi
done
set -- "${_norm_args[@]}"

while getopts ":e:p:w:h:u:l:d:o:t:g:s:x:c:G:Q:E:P:M:" opt; do
    case "$opt" in
        e) ENDPOINT=${OPTARG};;
        p) WALLET_PATH=${OPTARG};;
        w) WALLET_NAME=${OPTARG};;
        h) HOTKEY_NAME=${OPTARG};;
        u) NETUID=${OPTARG};;
        l) LOG_LEVEL=${OPTARG};;
        d) PD_KEY=${OPTARG};;
        o) PROM_PORT=${OPTARG};;
        t) TIMEOUT=${OPTARG};;
        g) SIMULATION_CONFIG=${OPTARG};;
        s) PRESERVE_SIMULATOR=${OPTARG};;
        x) USE_TMUX=${OPTARG};;
        c) CHECKPOINT=${OPTARG};;
        G) GENTRX_MODE=${OPTARG};;
        Q) GRAD_URL=${OPTARG}; _EXPLICIT_GRAD_URL=1;;
        E) ;;  # already pre-parsed
        P) VALIDATOR_PORT=${OPTARG};;
        M) MECHID=${OPTARG};;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1;;
    esac
done

# GenTRX restore
if [ -z "$GENTRX_MODE" ] && [ -n "${GENTRX_SAVED_MODE:-}" ]; then
    GENTRX_MODE="$GENTRX_SAVED_MODE"
fi
if [ -n "$GRAD_URL" ] && [ -z "$GENTRX_MODE" ]; then
    GENTRX_MODE="sibling"
fi

# Default netuid from endpoint
if [ -z "$NETUID" ]; then
    case "$ENDPOINT" in
        *entrypoint-finney*) NETUID=79 ;;
        *test.finney*)       NETUID=366 ;;
        *)                   NETUID=79 ;;
    esac
fi

VALIDATOR_NAME="validator"
TMUX_SESSION="mvtrx"

echo "ENDPOINT: $ENDPOINT"
echo "WALLET_PATH: $WALLET_PATH"
echo "WALLET_NAME: $WALLET_NAME"
echo "HOTKEY_NAME: $HOTKEY_NAME"
echo "NETUID: $NETUID"
echo "VALIDATOR_PORT: $VALIDATOR_PORT"
echo "PROMETHEUS PORT: $PROM_PORT"
echo "SIMULATION_CONFIG: $SIMULATION_CONFIG"
echo "CHECKPOINT: $CHECKPOINT"
echo "TMUX_SESSION: $TMUX_SESSION"

_pm2_clean() {
    _ids=$(pm2 jlist 2>/dev/null | python3 -c "
import sys, json
try:
    procs = json.load(sys.stdin)
    names = set(sys.argv[1:])
    print(' '.join(str(p['pm_id']) for p in procs if p['name'] in names))
except Exception:
    pass
" "$@" 2>/dev/null)
    if [ -n "$_ids" ]; then
        pm2 stop    $_ids 2>/dev/null || true
        pm2 delete  $_ids 2>/dev/null || true
    fi
}

if [ "$PRESERVE_SIMULATOR" = "0" ]; then
    _pm2_clean simulator validator "$VALIDATOR_NAME"
    # Free the memory-heavy gradient server BEFORE the C++ rebuild so it doesn't
    # contend for RAM/CPU during cmake (BUILD_JOBS is memory-bounded, so a 6GB
    # resident aggregator throttles the build). The GenTRX block below restarts
    # it fresh; its aggregation state persists on disk and reloads on restart.
    if [ -n "$GENTRX_MODE" ]; then
        pm2 stop gradient-server 2>/dev/null || true
    fi
    [ "$USE_TMUX" = "1" ] && tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
fi

echo "Updating Validator"
git pull
git submodule sync
git submodule update --init
pip install -e .

# Strip orphaned packages bittensor ≥8 doesn't need.
pip uninstall substrate-interface scalecodec py-scale-codec cyscale -y 2>/dev/null || true
pip install cyscale --force-reinstall --quiet

if [ "$PRESERVE_SIMULATOR" = "0" ]; then
    echo "Updating C++ Engine"
    export LD_LIBRARY_PATH="/usr/local/gcc-14.1.0/lib/../lib64:$LD_LIBRARY_PATH"
    cd simulate/trading
    cd vcpkg
    CURRENT_VCPKG_HASH=$(git rev-parse HEAD)
    EXPECTED_VCPKG_HASH="e140b1fde236eb682b0d47f905e65008a191800f"
    if [ "$CURRENT_VCPKG_HASH" != "$EXPECTED_VCPKG_HASH" ]; then
        git fetch --all --quiet
        git reset --hard "$EXPECTED_VCPKG_HASH"
        rm -r ../build
        mkdir ../build
        ./bootstrap-vcpkg.sh -disableMetrics
    fi
    cd ..
    MEM_MB=$(free -m | awk '/^Mem:/{print $7}')
    MEM_JOBS=$(( MEM_MB / 2048 ))
    MEM_JOBS=$(( MEM_JOBS < 1 ? 1 : MEM_JOBS ))
    NPROC=$(nproc)
    BUILD_JOBS=$(( MEM_JOBS < NPROC ? MEM_JOBS : NPROC ))
    _TRADING_DIR="$REPO_ROOT/simulate/trading"
    _build_ok=0
    for _try in 1 2; do
        cd "$_TRADING_DIR/build" || exit 1
        if ! g++ -dumpversion | grep -q "14"; then
            cmake -DENABLE_TRACES=1 -DCMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++-14 .. \
                && cmake --build . -j "$BUILD_JOBS" \
                && _build_ok=1
        else
            cmake -DENABLE_TRACES=1 -DCMAKE_BUILD_TYPE=Release .. \
                && cmake --build . -j "$BUILD_JOBS" \
                && _build_ok=1
        fi
        [ "$_build_ok" = "1" ] && break
        echo "Simulator build attempt $_try failed — wiping vcpkg buildtrees/packages and retrying..."
        cd "$_TRADING_DIR"
        rm -rf vcpkg/buildtrees vcpkg/packages vcpkg/downloads/temp build
        mkdir build
    done
    if [ "$_build_ok" = "0" ]; then
        echo "ERROR: Simulator build failed after 2 attempts." >&2
        exit 1
    fi
    cd "$REPO_ROOT/taos/im/neurons"
else
    cd taos/im/neurons
fi

# ── GenTRX gradient server (optional sibling) ─────────────────────────────────
GENTRX_VAL_ARGS=""
if [ -n "$GENTRX_MODE" ]; then
    _GRAD_URL="${GRAD_URL:-http://127.0.0.1:${GRAD_PORT}/gentrx}"
    if [ "$_EXPLICIT_GRAD_URL" = "0" ]; then
        _GRAD_SKIP_UPDATE=""
        [ "$PRESERVE_SIMULATOR" = "1" ] && _GRAD_SKIP_UPDATE="-S 1"
        "$REPO_ROOT/run_gradients.sh" -m "$GENTRX_MODE" -e "$ENDPOINT" -u "$NETUID" -V 1 -p "$GRAD_PORT" -x 0 $_GRAD_SKIP_UPDATE \
            || { echo "ERROR: gradient server start failed" >&2; exit 1; }
        echo " ✓  Gradient server started locally (mode=$GENTRX_MODE, port=$GRAD_PORT)"
    fi
    GENTRX_VAL_ARGS="--gentrx.enabled --gentrx.gradient_server_url $_GRAD_URL"
    if grep -q '^GENTRX_SAVED_MODE=' "$REPO_ROOT/.env" 2>/dev/null; then
        sed -i "s|^GENTRX_SAVED_MODE=.*|GENTRX_SAVED_MODE=$GENTRX_MODE|" "$REPO_ROOT/.env"
    else
        echo "GENTRX_SAVED_MODE=$GENTRX_MODE" >> "$REPO_ROOT/.env"
    fi
fi

MECHID_ARG="--neuron.mechid $MECHID"
ENGINE_ARGS="--engine simulation --simulation.xml_config ../../../simulate/trading/run/config/$SIMULATION_CONFIG.xml"
# Benchmark agents config file. Dev-local env files set BENCHMARK_AGENTS to the
# test config (../config/benchmark_agents_test.json); when unset the validator
# falls back to its default (../config/benchmark_agents.json). Mirrors the
# --benchmark.agents wiring in run_validator.sh / run_mvtrx.sh.
BENCHMARK_ARG=""
[ -n "${BENCHMARK_AGENTS:-}" ] && BENCHMARK_ARG="--benchmark.agents ${BENCHMARK_AGENTS}"

VALIDATOR_CMD="python $(pwd)/validator.py \
  --netuid $NETUID \
  --subtensor.chain_endpoint $ENDPOINT \
  --wallet.path $WALLET_PATH \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $HOTKEY_NAME \
  --logging.$LOG_LEVEL \
  --alerting.pagerduty.integration_key $PD_KEY \
  --port $VALIDATOR_PORT \
  --prometheus.port $PROM_PORT \
  --neuron.timeout $TIMEOUT \
  $MECHID_ARG \
  $ENGINE_ARGS \
  $BENCHMARK_ARG \
  $GENTRX_VAL_ARGS"
GENTRX_API_KEY="${GENTRX_API_KEY:-}" \
GENTRX_VALIDATOR_S3_ENDPOINT_URL="${GENTRX_VALIDATOR_S3_ENDPOINT_URL:-}" \
GENTRX_VALIDATOR_S3_BUCKET="${GENTRX_VALIDATOR_S3_BUCKET:-}" \
GENTRX_VALIDATOR_S3_WRITE_ACCESS_KEY="${GENTRX_VALIDATOR_S3_WRITE_ACCESS_KEY:-}" \
GENTRX_VALIDATOR_S3_WRITE_SECRET_KEY="${GENTRX_VALIDATOR_S3_WRITE_SECRET_KEY:-}" \
GENTRX_VALIDATOR_S3_READ_ACCESS_KEY="${GENTRX_VALIDATOR_S3_READ_ACCESS_KEY:-${GENTRX_VALIDATOR_S3_WRITE_ACCESS_KEY:-}}" \
GENTRX_VALIDATOR_S3_READ_SECRET_KEY="${GENTRX_VALIDATOR_S3_READ_SECRET_KEY:-${GENTRX_VALIDATOR_S3_WRITE_SECRET_KEY:-}}" \
pm2 start --name=$VALIDATOR_NAME bash -- -c "$VALIDATOR_CMD"

# ── C++ Simulator ─────────────────────────────────────────────────────────────
if [ "$PRESERVE_SIMULATOR" = "0" ]; then
    cd ../../../simulate/trading/run
    echo "Starting Simulator"
    if [ "$CHECKPOINT" = "0" ]; then
        # --kill-timeout: pm2 SIGKILLs 1600ms after the signal by default, long before the engine can
        # reach a barrier and write its shutdown checkpoint, so an intended stop behaves like a crash
        # and the next resume rewinds to the last periodic checkpoint.
        pm2 start --no-autorestart --kill-timeout 60000 --name=simulator "../build/src/cpp/taosim -f config/$SIMULATION_CONFIG.xml"
    else
        pm2 start --no-autorestart --kill-timeout 60000 --name=simulator "../build/src/cpp/taosim -c $CHECKPOINT"
    fi
fi
pm2 save
pm2 startup

if [ "$USE_TMUX" = "1" ]; then
    echo "Setting Up Tmux Session"
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    _HTOP='P=$(pgrep -d, -f "validator.py|taosim" 2>/dev/null); [ -n "$P" ] && exec htop -p "$P" || exec htop'
    P0=$(tmux new-session -d -s "$TMUX_SESSION" -n 'monitor' -P -F '#{pane_id}' "$_HTOP")
    tmux set-option -t "$TMUX_SESSION" mouse on
    P1=$(tmux split-window -v -p 75 -t "$P0" -P -F '#{pane_id}' "pm2 logs $VALIDATOR_NAME")
    tmux split-window -h -p 50 -t "$P1" 'pm2 logs simulator'
    if [ -n "$GENTRX_MODE" ]; then
        tmux new-window -t "$TMUX_SESSION:" -n 'gentrx'
        _GTX_HTOP='P=$(pgrep -d, -f "gradient_server" 2>/dev/null); [ -n "$P" ] && exec htop -p "$P" || exec htop'
        tmux send-keys -t "$TMUX_SESSION:gentrx" "$_GTX_HTOP" Enter
        PG=$(tmux split-window -v -p 67 -t "$TMUX_SESSION:gentrx" -P -F '#{pane_id}')
        tmux send-keys -t "$PG" 'pm2 logs gradient-server' Enter
    fi
    tmux select-window -t "$TMUX_SESSION:monitor"
    tmux select-pane -t "$P0"
fi

# Network sysctl tuning (idempotent).
setting_exists() { grep -q "^${1}=" /etc/sysctl.conf; }
SETTINGS="net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.core.rmem_default=8388608
net.core.wmem_default=8388608
net.ipv4.tcp_rmem=4096 87380 67108864
net.ipv4.tcp_wmem=4096 87380 67108864
net.ipv4.tcp_tw_reuse=1
net.ipv4.tcp_fin_timeout=30"
echo "$SETTINGS" | while IFS= read -r line; do
    sudo sysctl -w "${line}" >/dev/null 2>&1 || true
done
needs_update=false
echo "$SETTINGS" | while IFS= read -r line; do
    setting="${line%%=*}"
    setting_exists "$setting" || { echo "needs_update"; break; }
done | grep -q "needs_update" && needs_update=true
if [ "$needs_update" = true ]; then
    echo "$SETTINGS" | while IFS= read -r line; do
        setting="${line%%=*}"
        if ! setting_exists "$setting"; then
            echo "$line" | sudo tee -a /etc/sysctl.conf >/dev/null
        fi
    done
fi

if [ "$USE_TMUX" = "1" ]; then
    tmux attach-session -t "$TMUX_SESSION"
fi
