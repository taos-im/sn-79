#!/bin/bash
# See run_validator.sh: set before any python launch so bittensor 10.3.2's
# import-time logging singleton isn't built with an empty config (which silently
# disables logging). pm2 captures it for the gradient-server process.
export BT_NO_PARSE_CLI_ARGS=false
# run_gradients.sh — start (or restart) the GenTRX gradient server
#
# All S3 credentials and the API key can be passed as flags so this script
# can run on a remote GPU machine without a separate .env file.  Falls back
# to the same-named environment variables (or .env if present) when flags
# are omitted.
#
# Same-machine use (called automatically by run_validator.sh -G):
#   ./run_gradients.sh -m sibling -e wss://... -u 79 -V 1
#
# Remote GPU machine — copy this file and paste the printed command, e.g.:
#   ./run_gradients.sh -m sibling -e wss://... -u 79 -V 1 -b 0.0.0.0 \
#       -E https://<acct>.r2.cloudflarestorage.com -B <bucket> \
#       -w <write-key> -W <write-secret> -k <api-key>
#
# Flags:
#   -m|-G sibling          Run mode (default: sibling; -G accepted as alias). Use
#                          'sibling' unless you are the subnet operator running
#                          the canonical uid-0 aggregator that publishes checkpoints
#                          on-chain for the whole subnet.
#   -e <endpoint>          Subtensor network endpoint
#   -u <netuid>            Subnet UID (default: 79)
#   -p <port>              Listen port (default: 8100)
#   -b <bind>              Bind address (default: 127.0.0.1; use 0.0.0.0 for remote)
#   -V <uid>               Validator UID for Prometheus labels
#   -E <s3-endpoint-url>   S3 endpoint URL (overrides GENTRX_VALIDATOR_S3_ENDPOINT_URL)
#   -B <bucket>            S3 bucket name  (overrides GENTRX_VALIDATOR_S3_BUCKET)
#   -w <write-access-key>  S3 write access key
#   -W <write-secret>      S3 write secret key
#   -k <api-key>           Shared API key (required when -b is not 127.0.0.1)
#   -c <path>              Checkpoint path (default: checkpoints/GenTRX/best.pt)
#   -d <path>              Val-data path   (default: data/gentrx_val)
#   -o <path>              Output path     (default: checkpoints/GenTRX/latest.pt)
#   -s 1                   Skip pm2 delete (keep existing process, just regenerate launcher)
#   -x 0                   Skip tmux window setup
#   -N <name>              pm2 process name (default: gradient-server). The launchers pass a
#                          suffixed name so two stacks on one host don't replace each other.
#   -U                     Opt in to updating on launch (git pull + pip install -e .). Off by
#                          default so a running node never jumps to an unpinned upstream HEAD.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# set -a auto-exports every var the sourced file sets, so any child process
# inherits them. The generated launcher (.gradient_server.sh) also uses
# set -a; matching that here keeps pre-pm2 sanity checks consistent.
[ -f "$REPO_ROOT/.env" ] && { set -a; . "$REPO_ROOT/.env"; set +a; }

_ok()   { printf ' \033[32m✓\033[0m  %s\n' "$*"; }
_warn() { printf ' \033[33m⚠\033[0m  %s\n' "$*"; }
_step() { printf '\n \033[1m─ %s\033[0m\n' "$*"; }
_info() { printf '   %s\n' "$*"; }

_env_write() {
    local key="$1" val="$2"
    [ -f "$REPO_ROOT/.env" ] || touch "$REPO_ROOT/.env"
    python3 -c "
import sys, os, shlex
path, key, val = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(path).readlines() if os.path.exists(path) else []
new = []; found = False
quoted = shlex.quote(val) if val else \"''\"
for l in lines:
    if l.strip().startswith(key + '=') and not l.strip().startswith('#'):
        new.append(key + '=' + quoted + '\n'); found = True
    else:
        new.append(l)
if not found:
    new.append(key + '=' + quoted + '\n')
open(path, 'w').writelines(new)
" "$REPO_ROOT/.env" "$key" "$val"
}

_prompt() {
    local varname="$1" msg="$2" default="${3:-}"
    local current="${!varname:-}"
    if [ -n "$current" ]; then
        _info "$varname: [already set — skipping]"; return
    fi
    if [ -n "$default" ]; then
        read -r -p "    $msg [$default]: " _in
        printf -v "$varname" '%s' "${_in:-$default}"
    else
        read -r -p "    $msg: " _in
        printf -v "$varname" '%s' "$_in"
    fi
    [ -z "${!varname:-}" ] && { echo "ERROR: $varname is required."; exit 1; }
    _env_write "$varname" "${!varname}"
    export "$varname"
}

_prompt_secret() {
    local varname="$1" msg="$2"
    local current="${!varname:-}"
    if [ -n "$current" ]; then
        _info "$varname: [already set — skipping]"; return
    fi
    read -r -s -p "    $msg: " _in; echo
    printf -v "$varname" '%s' "$_in"
    [ -z "${!varname:-}" ] && { echo "ERROR: $varname is required."; exit 1; }
    _env_write "$varname" "${!varname}"
    export "$varname"
}

# Defaults — overridden by flags below; fall back to env vars if flags not given
GENTRX_ROLE=sibling
ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
NETUID=79
GRAD_PORT=8100
GRAD_BIND=127.0.0.1
VALIDATOR_UID=""
# Training-data shard for the gradient server's --mode (simulation|exchange).
# Distinct from -m/-G, which is the ROLE (aggregator|sibling). The launcher
# scripts (run_validator.sh, run_miner.sh) pass -M "$MODE" so an exchange stack's
# aggregator uses the gentrx/<net>/exchange/ prefix instead of simulation.
GRAD_TRAIN_MODE="simulation"
CHECKPOINT_PATH=checkpoints/GenTRX/best.pt
VAL_DATA_PATH=data/gentrx_val
OUTPUT_PATH=checkpoints/GenTRX/latest.pt
S3_ENDPOINT_URL="${GENTRX_VALIDATOR_S3_ENDPOINT_URL:-}"
S3_BUCKET="${GENTRX_VALIDATOR_S3_BUCKET:-}"
S3_WRITE_KEY="${GENTRX_VALIDATOR_S3_WRITE_ACCESS_KEY:-}"
S3_WRITE_SECRET="${GENTRX_VALIDATOR_S3_WRITE_SECRET_KEY:-}"
API_KEY="${GENTRX_API_KEY:-}"
PRESERVE=0
LOG_LEVEL=info
USE_TMUX=1
# Auto-update is OPT-IN: by default the launcher does NOT `git pull` /
# `pip install -e .`, so a running node never silently jumps to an unpinned
# upstream HEAD. Pass -U to update on launch; -n is kept as a no-op alias.
SKIP_UPDATE=1
# pm2 process name. The launcher scripts (run_validator.sh, run_miner.sh) pass
# a suffixed name (e.g. gradient-server-sim-local) so multiple stacks on one
# host don't clobber each other's instance: pm2 start with an existing name
# REPLACES it.
GRADIENT_SERVER_NAME="${GRADIENT_SERVER_NAME:-gradient-server}"

while getopts m:G:e:u:p:b:V:c:d:o:E:B:w:W:k:s:l:x:nUN:M: flag; do
    case "${flag}" in
        m|G) GENTRX_ROLE=${OPTARG};;
        M) GRAD_TRAIN_MODE=${OPTARG};;
        N) GRADIENT_SERVER_NAME=${OPTARG};;
        e) ENDPOINT=${OPTARG};;
        u) NETUID=${OPTARG};;
        p) GRAD_PORT=${OPTARG};;
        b) GRAD_BIND=${OPTARG};;
        V) VALIDATOR_UID=${OPTARG};;
        c) CHECKPOINT_PATH=${OPTARG};;
        d) VAL_DATA_PATH=${OPTARG};;
        o) OUTPUT_PATH=${OPTARG};;
        E) S3_ENDPOINT_URL=${OPTARG};;
        B) S3_BUCKET=${OPTARG};;
        w) S3_WRITE_KEY=${OPTARG};;
        W) S3_WRITE_SECRET=${OPTARG};;
        k) API_KEY=${OPTARG};;
        s) PRESERVE=${OPTARG};;
        l) LOG_LEVEL=${OPTARG};;
        x) USE_TMUX=${OPTARG};;
        U) SKIP_UPDATE=0;;
        n) SKIP_UPDATE=1;;
    esac
done

# Normalize role (aggregator mode is reserved for subnet operators)
case "$GENTRX_ROLE" in
    a|agg|aggregator) GENTRX_ROLE=aggregator; AGG_FLAG="--is-aggregator" ;;
    *)                GENTRX_ROLE=sibling;    AGG_FLAG="--no-is-aggregator" ;;
esac

# Normalize training-data shard: only 'exchange' is special; anything else
# (obs, blank, typo) falls back to 'simulation' so the gradient server's
# --mode choices=[simulation,exchange] never rejects it.
case "$GRAD_TRAIN_MODE" in
    exchange) GRAD_TRAIN_MODE=exchange ;;
    *)        GRAD_TRAIN_MODE=simulation ;;
esac

# ── CPU core pinning ──────────────────────────────────────────────────────────
# GRAD_CORES_COUNT: reserve the last N CPU cores for the gradient server.
# Affinity is set internally by the server via os.sched_setaffinity() (same
# pattern as the validator).  OMP/MKL thread counts must be set in the
# environment before the process starts so BLAS libraries read them on load.
# Set in .env: GRAD_CORES_COUNT=8  (or pass GRAD_CORES_COUNT=8 before running)
# run_validator.sh exports GRAD_CORES_COUNT automatically when launching locally.
GRAD_CORES_COUNT="${GRAD_CORES_COUNT:-0}"

_OMP_EXPORT=""
if [ "${GRAD_CORES_COUNT:-0}" -gt 0 ] 2>/dev/null; then
    _OMP_EXPORT="export OMP_NUM_THREADS=$GRAD_CORES_COUNT
export MKL_NUM_THREADS=$GRAD_CORES_COUNT"
    _ok "BLAS threads limited to $GRAD_CORES_COUNT (OMP_NUM_THREADS, MKL_NUM_THREADS)"
fi

# ── Network shard ─────────────────────────────────────────────────────────────
# GENTRX_NETWORK: override the automatic network→bucket-shard mapping.
# Auto-detection handles finney / test / local / standard public endpoints and
# custom wss:// addresses (assumed mainnet).  Only needed for:
#   - a private testnet node:  GENTRX_NETWORK=testnet
#   - localnet with mainnet bucket layout (unusual): GENTRX_NETWORK=mainnet
GENTRX_NETWORK="${GENTRX_NETWORK:-}"
_NETWORK_ARG=""
[ -n "$GENTRX_NETWORK" ] && _NETWORK_ARG="--network $GENTRX_NETWORK"

# ── Credentials setup wizard ──────────────────────────────────────────────────
# Runs interactively when credentials are missing; skips if already in .env or flags.
if [ -z "$S3_BUCKET" ] || [ -z "$S3_WRITE_KEY" ] || [ -z "$S3_WRITE_SECRET" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " GenTRX Gradient Server — Credentials Setup"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    _step "Validator S3 bucket"
    _info "The gradient server needs WRITE credentials to upload checkpoints,"
    _info "training data, and proposals to your validator S3 bucket."
    _info "You will enter the write token first, then the read token."
    _info "These match GENTRX_VALIDATOR_S3_WRITE_* / GENTRX_VALIDATOR_S3_READ_* in .env"
    _info "on the validator host — copy them here, or create new ones for this host."
    _info "From each R2 token, copy the Access Key ID and Secret Access Key"
    _info "(NOT the Token Value — that bearer string is for a different API)."
    echo

    # Provider selection
    _provider=""
    if [ -n "${GENTRX_VALIDATOR_S3_ACCOUNT_ID:-}" ]; then
        _provider="configured-r2"
    elif [ -n "${GENTRX_VALIDATOR_S3_ENDPOINT_URL:-}" ]; then
        _provider="2"
    else
        printf '    Provider: (1) Cloudflare R2  (2) Hippius  [1]: '
        read -r _provider; _provider="${_provider:-1}"
    fi

    case "$_provider" in
        2|hippius|Hippius)
            _prompt GENTRX_VALIDATOR_S3_ENDPOINT_URL \
                "Hippius endpoint URL" "https://s3.hippius.com"
            S3_ENDPOINT_URL="${GENTRX_VALIDATOR_S3_ENDPOINT_URL}"
            _prompt GENTRX_VALIDATOR_S3_BUCKET "Bucket name"
            S3_BUCKET="${GENTRX_VALIDATOR_S3_BUCKET}"
            ;;
        *)
            _prompt GENTRX_VALIDATOR_S3_ACCOUNT_ID \
                "Cloudflare R2 account ID (32-char hex, from R2 dashboard)"
            _default_bucket="${GENTRX_VALIDATOR_S3_ACCOUNT_ID:-}"
            _prompt GENTRX_VALIDATOR_S3_BUCKET "Bucket name" "$_default_bucket"
            S3_BUCKET="${GENTRX_VALIDATOR_S3_BUCKET}"
            if [ -z "${GENTRX_VALIDATOR_S3_ENDPOINT_URL:-}" ]; then
                GENTRX_VALIDATOR_S3_ENDPOINT_URL="https://${GENTRX_VALIDATOR_S3_ACCOUNT_ID}.r2.cloudflarestorage.com"
                _env_write "GENTRX_VALIDATOR_S3_ENDPOINT_URL" "$GENTRX_VALIDATOR_S3_ENDPOINT_URL"
                export GENTRX_VALIDATOR_S3_ENDPOINT_URL
                _info "Endpoint URL: $GENTRX_VALIDATOR_S3_ENDPOINT_URL"
            fi
            S3_ENDPOINT_URL="${GENTRX_VALIDATOR_S3_ENDPOINT_URL}"
            ;;
    esac

    echo
    _step "Write token (1 of 2)"
    _prompt_secret GENTRX_VALIDATOR_S3_WRITE_ACCESS_KEY "Write access key ID"
    _prompt_secret GENTRX_VALIDATOR_S3_WRITE_SECRET_KEY "Write secret access key"
    S3_WRITE_KEY="${GENTRX_VALIDATOR_S3_WRITE_ACCESS_KEY}"
    S3_WRITE_SECRET="${GENTRX_VALIDATOR_S3_WRITE_SECRET_KEY}"
    echo
    _ok "Write credentials saved to .env"
fi

_step "Read-only token (2 of 2) — committed on-chain so miners can find your bucket"
_info "A separate read-only API token on the same bucket."
_prompt_secret GENTRX_VALIDATOR_S3_READ_ACCESS_KEY "Read-only access key ID"
_prompt_secret GENTRX_VALIDATOR_S3_READ_SECRET_KEY "Read-only secret access key"
echo
_ok "S3 credentials ready."

# Persist all S3 config to .env regardless of whether values came from the wizard
# or from the shell environment (e.g. localnet runner exports).  run_validator.sh
# sources .env in a fresh shell, so everything must be written there.
[ -n "$S3_BUCKET" ]      && _env_write "GENTRX_VALIDATOR_S3_BUCKET"               "$S3_BUCKET"
[ -n "$S3_ENDPOINT_URL" ] && _env_write "GENTRX_VALIDATOR_S3_ENDPOINT_URL"          "$S3_ENDPOINT_URL"
[ -n "$S3_WRITE_KEY" ]   && _env_write "GENTRX_VALIDATOR_S3_WRITE_ACCESS_KEY"      "$S3_WRITE_KEY"
[ -n "$S3_WRITE_SECRET" ] && _env_write "GENTRX_VALIDATOR_S3_WRITE_SECRET_KEY"      "$S3_WRITE_SECRET"
[ -n "${GENTRX_VALIDATOR_S3_READ_ACCESS_KEY:-}" ] && \
    _env_write "GENTRX_VALIDATOR_S3_READ_ACCESS_KEY" "$GENTRX_VALIDATOR_S3_READ_ACCESS_KEY"
[ -n "${GENTRX_VALIDATOR_S3_READ_SECRET_KEY:-}" ] && \
    _env_write "GENTRX_VALIDATOR_S3_READ_SECRET_KEY" "$GENTRX_VALIDATOR_S3_READ_SECRET_KEY"

# Derive or prompt for S3 endpoint URL if not already set
if [ -z "$S3_ENDPOINT_URL" ]; then
    _acct="${GENTRX_VALIDATOR_S3_ACCOUNT_ID:-}"
    if [ -n "$_acct" ]; then
        S3_ENDPOINT_URL="https://${_acct}.r2.cloudflarestorage.com"
        _env_write "GENTRX_VALIDATOR_S3_ENDPOINT_URL" "$S3_ENDPOINT_URL"
        export GENTRX_VALIDATOR_S3_ENDPOINT_URL="$S3_ENDPOINT_URL"
    else
        _step "S3 endpoint URL"
        _info "For Cloudflare R2: https://<account-id>.r2.cloudflarestorage.com"
        _info "For Hippius: https://s3.hippius.com"
        read -r -p "    S3 endpoint URL: " S3_ENDPOINT_URL
        [ -z "$S3_ENDPOINT_URL" ] && { echo "ERROR: S3 endpoint URL is required."; exit 1; }
        GENTRX_VALIDATOR_S3_ENDPOINT_URL="$S3_ENDPOINT_URL"
        _env_write "GENTRX_VALIDATOR_S3_ENDPOINT_URL" "$S3_ENDPOINT_URL"
        export GENTRX_VALIDATOR_S3_ENDPOINT_URL
    fi
fi

# API key wizard — required when binding to a non-loopback address
if [ "$GRAD_BIND" != "127.0.0.1" ] && [ -z "$API_KEY" ]; then
    _step "API key (required for non-loopback binding)"
    _info "The gradient server is bound to $GRAD_BIND."
    _info "An API key is required so only your validator can push state."
    echo
    printf '    (1) Generate a new key automatically\n'
    printf '    (2) Enter an existing key\n'
    printf '    Choice [1]: '
    read -r _key_choice; _key_choice="${_key_choice:-1}"
    if [ "$_key_choice" = "2" ]; then
        _prompt_secret GENTRX_API_KEY "API key"
    else
        GENTRX_API_KEY=$(openssl rand -hex 32)
        _env_write "GENTRX_API_KEY" "$GENTRX_API_KEY"
        export GENTRX_API_KEY
        _ok "Generated API key (saved to .env): $GENTRX_API_KEY"
        _info "Set the same value as GENTRX_API_KEY in .env on the validator host."
    fi
    API_KEY="${GENTRX_API_KEY}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " GenTRX Gradient Server: $GENTRX_ROLE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  netuid=$NETUID  endpoint=$ENDPOINT"
echo "  port=$GRAD_PORT  bind=$GRAD_BIND"
echo "  validator_uid=${VALIDATOR_UID:-(unset)}"
echo "  bucket=$S3_BUCKET  endpoint_url=$S3_ENDPOINT_URL"
echo ""

# ── Install / update ──────────────────────────────────────────────────────────
cd "$REPO_ROOT"
if [ "$SKIP_UPDATE" = "0" ]; then
    echo "Updating gradient server (opt-in via -U)"
    git pull || { echo "WARNING: git pull failed (no tracking branch?). Continue without updating? [y/N]"; read -r _yn; [ "$_yn" = "y" ] || exit 1; }
    # Honour the committed lockfile when present so the update resolves the
    # pinned, reproducible dependency set rather than floating to newer releases. The gradient
    # server needs the [gentrx] extra (boto3/transformers/polars), not just the core set.
    if [ -f "$REPO_ROOT/constraints.txt" ]; then
        pip install -e ".[gentrx]" -c "$REPO_ROOT/constraints.txt"
    else
        pip install -e ".[gentrx]"
    fi
fi

# scalecodec ↔ cyscale conflict: bittensor's transitive deps drag py-scale-codec
# (PyPI package `scalecodec`) back in on every `pip install -e .`, even though
# we want cyscale to provide the `scalecodec` namespace. Both installed at once
# → async_substrate_interface raises a RuntimeError at `import bittensor` time
# and the gradient server never starts. Detect by trying to import bittensor;
# on failure, uninstall both and force-reinstall cyscale. Idempotent. Mirrors
# the repair in run_validator.sh.
# Runs even when -n (SKIP_UPDATE=1) is passed: the conflict can be present on
# a host that skipped the update step.
if ! python3 -c "import bittensor" >/dev/null 2>&1; then
    echo "Repairing scalecodec/cyscale conflict (bittensor import failed)"
    pip uninstall -y scalecodec cyscale 2>/dev/null || true
    pip install --force-reinstall --no-cache-dir cyscale
fi

# CUDA check — warn with install command if torch has no CUDA
_cuda_ok=$(python3 -c "
try:
    import torch
except ImportError:
    print('notinstalled')
    raise SystemExit(0)
if torch.cuda.is_available():
    print('ok:' + torch.cuda.get_device_name(0))
else:
    try:
        import subprocess, re
        out = subprocess.check_output(['nvcc','--version'], stderr=subprocess.STDOUT).decode()
        m = re.search(r'release ([\d.]+)', out)
        ver = m.group(1).replace('.','') if m else '128'
        major, minor = ver[:2], ver[2:] if len(ver)>2 else '1'
        print('cpu:cu' + major + minor)
    except Exception:
        print('cpu:cu128')
" 2>/dev/null || echo "cpu:cu128")
if echo "$_cuda_ok" | grep -q "^ok:"; then
    echo " ✓  CUDA: ${_cuda_ok#ok:}"
elif echo "$_cuda_ok" | grep -q "^notinstalled"; then
    echo ""
    echo " ⚠  torch is not installed. Install with CUDA support before running:"
    echo "      pip install torch --index-url https://download.pytorch.org/whl/cu128"
    echo "    Then re-run this script."
    echo ""
else
    _cu_tag="${_cuda_ok#cpu:}"
    echo ""
    echo " ⚠  torch has no CUDA support. Gradient scoring will be ~5× slower on CPU."
    echo "    To install CUDA-enabled torch:"
    echo "      pip install torch --index-url https://download.pytorch.org/whl/${_cu_tag}"
    echo "    Then re-run this script."
    echo ""
fi

# nvitop — GPU monitor used in tmux pane; install silently if missing
command -v nvitop > /dev/null 2>&1 || pip install nvitop -q || true

# pm2 — install via npm if missing
if ! command -v pm2 > /dev/null 2>&1; then
    if command -v npm > /dev/null 2>&1; then
        echo " ─  pm2 not found — installing via npm..."
        npm install -g pm2 -q
    else
        echo "ERROR: pm2 is required. Install node.js + npm, then: npm install -g pm2"
        exit 1
    fi
fi

# Generate the launcher script.  pm2 restarts this on crash; it re-sources
# credentials at each start so .env rotation takes effect without re-running
# run_gradients.sh.
_PYTHON=$(command -v python3)
LAUNCHER="$REPO_ROOT/.gradient_server.sh"
cat > "$LAUNCHER" << LAUNCHER_EOF
#!/bin/bash
# Auto-generated by run_gradients.sh — re-run run_gradients.sh to update.
_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
cd "\$_DIR"
set -a; [ -f "\$_DIR/.env" ] && . "\$_DIR/.env"; set +a
$_OMP_EXPORT
exec "$_PYTHON" -m GenTRX.src.gradient_server \
    --checkpoint "$CHECKPOINT_PATH" \
    --val-data "$VAL_DATA_PATH" \
    --output "$OUTPUT_PATH" \
    --port $GRAD_PORT \
    --bind $GRAD_BIND \
    $AGG_FLAG \
    --netuid $NETUID \
    --subtensor-network $ENDPOINT \
    --validator-uid "${VALIDATOR_UID:-}" \
    --mode "$GRAD_TRAIN_MODE" \
    --log-level $LOG_LEVEL \
    \${GENTRX_API_KEY:+--api-key "\$GENTRX_API_KEY"} \
    $_NETWORK_ARG \
    \${GENTRX_GRAD_EXTRA_ARGS:-}
LAUNCHER_EOF
chmod +x "$LAUNCHER"

if [ "$PRESERVE" = "0" ]; then
    pm2 delete "$GRADIENT_SERVER_NAME" 2>/dev/null || true
fi
# pm2 log rotation: cap each .log at 100 MB, keep 10 rotated copies, gzip
# old ones. Idempotent — `pm2 install` is a no-op when already present, and
# `pm2 set` overwrites silently. Applies daemon-wide (also covers validator,
# miner, etc. if they're running under the same pm2).
if ! pm2 describe pm2-logrotate >/dev/null 2>&1; then
    pm2 install pm2-logrotate >/dev/null 2>&1 || true
fi
pm2 set pm2-logrotate:max_size 100M >/dev/null 2>&1 || true
pm2 set pm2-logrotate:retain 10 >/dev/null 2>&1 || true
pm2 set pm2-logrotate:compress true >/dev/null 2>&1 || true

pm2 start --name="$GRADIENT_SERVER_NAME" "$LAUNCHER"
pm2 save
echo ""
echo " ✓  Gradient server started (pm2: $GRADIENT_SERVER_NAME)"
echo "    Logs:    pm2 logs $GRADIENT_SERVER_NAME"
echo "    Metrics: http://${GRAD_BIND}:${GRAD_PORT}/gentrx/metrics"
echo "    Version: curl http://${GRAD_BIND}:${GRAD_PORT}/gentrx/version"
echo ""

if [ "$USE_TMUX" = "1" ]; then
    tmux kill-session -t gentrx 2>/dev/null || true
    tmux new-session -d -s gentrx -n 'gentrx' 'htop'
    tmux set-option -t gentrx mouse on
    tmux split-window -h -t gentrx:gentrx 'nvitop 2>/dev/null || watch -n2 nvidia-smi'
    tmux select-pane -t 0
    tmux split-window -v -t gentrx:gentrx "pm2 logs $GRADIENT_SERVER_NAME"
    tmux attach-session -t gentrx
fi
