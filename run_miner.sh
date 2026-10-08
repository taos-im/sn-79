#!/bin/bash
set -e
# See run_validator.sh: set before any python launch so bittensor 10.3.2's
# import-time logging singleton isn't built with an empty config (which silently
# disables logging). pm2 captures it; spawned subprocesses inherit it.
export BT_NO_PARSE_CLI_ARGS=false
# run_miner.sh — launch MVTRX miner
#
# Pass -G to enable GenTRX distributed training:
#   ./run_miner.sh -G -w mywallet -h myhotkey
#
# First run: prompts interactively for bucket credentials and commits them
# on-chain; saves all config (agent name/params, GenTRX params) to .env.
#
# Subsequent update runs: run without any flags — saved GenTRX mode, agent,
# and params are restored automatically, setup prompts are skipped.
# Pass -n/-m/-t explicitly to override saved agent/params on a specific run.
#
# Examples:
#   First setup:    ./run_miner.sh -G -w mywallet -h myhotkey
#   Update/restart: ./run_miner.sh
#   Override steps: ./run_miner.sh -t "gtx_train_steps=100 gtx_train_batch_size=8"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
WALLET_PATH=~/.bittensor/wallets/
WALLET_NAME=taos
HOTKEY_NAME=miner
NETUID=79
AXON_PORT=8091
AGENT_PATH=~/.taos/agents
AGENT_NAME=SimpleRegressorAgent
AGENT_PARAMS="min_quantity=0.1 max_quantity=1.0 expiry_period=200 model=PassiveAggressiveRegressor signal_threshold=0.0025"
LOG_LEVEL=info
GENTRX=0          # 1 = enable GenTRX training mode
GENTRX_PARAMS=""  # override GenTRX-specific params (gtx_* keys)

# Track whether agent/params were explicitly provided on the command line
_EXPLICIT_AGENT=0
_EXPLICIT_PARAMS=0
_EXPLICIT_GENTRX_PARAMS=0

# set -a auto-exports every var the sourced file sets, so pm2 (and any other
# child process) inherits them. Plain `. .env` would only populate this
# script's shell vars.
[ -f "$REPO_ROOT/.env" ] && { set -a; . "$REPO_ROOT/.env"; set +a; }

while getopts e:p:w:h:u:a:g:n:m:t:l:G flag; do
    case "${flag}" in
        e) ENDPOINT=${OPTARG};;
        p) WALLET_PATH=${OPTARG};;
        w) WALLET_NAME=${OPTARG};;
        h) HOTKEY_NAME=${OPTARG};;
        u) NETUID=${OPTARG};;
        a) AXON_PORT=${OPTARG};;
        g) AGENT_PATH=${OPTARG};;
        n) AGENT_NAME=${OPTARG};       _EXPLICIT_AGENT=1;;
        m) AGENT_PARAMS=${OPTARG};     _EXPLICIT_PARAMS=1;;
        t) GENTRX_PARAMS=${OPTARG};    _EXPLICIT_GENTRX_PARAMS=1;;
        l) LOG_LEVEL=${OPTARG};;
        G) GENTRX=1;;
    esac
done

# If -G not passed but GenTRX was previously enabled, restore saved mode
if [ "$GENTRX" = "0" ] && [ "${GENTRX_ENABLED:-0}" = "1" ]; then
    GENTRX=1
fi

# On GenTRX update runs: restore saved agent/params unless overridden on CLI.
# If -n is explicitly set to a different agent, don't restore the old params
# (they belong to the previous agent and are likely incompatible).
if [ "$GENTRX" = "1" ]; then
    [ "$_EXPLICIT_AGENT" = "0" ] && [ -n "${GENTRX_SAVED_AGENT_NAME:-}" ] && \
        AGENT_NAME="$GENTRX_SAVED_AGENT_NAME"
    # Only restore params if agent wasn't explicitly changed to something different
    if [ "$_EXPLICIT_PARAMS" = "0" ] && [ "$_EXPLICIT_AGENT" = "0" ] && \
       [ -n "${GENTRX_SAVED_AGENT_PARAMS:-}" ]; then
        AGENT_PARAMS="$GENTRX_SAVED_AGENT_PARAMS"
    fi
    [ "$_EXPLICIT_GENTRX_PARAMS" = "0" ] && [ -n "${GENTRX_SAVED_GENTRX_PARAMS:-}" ] && \
        GENTRX_PARAMS="$GENTRX_SAVED_GENTRX_PARAMS"
    [ -z "$AGENT_PATH" ] || [ "$AGENT_PATH" = "$HOME/.taos/agents" ] && \
        [ -n "${GENTRX_SAVED_AGENT_PATH:-}" ] && AGENT_PATH="$GENTRX_SAVED_AGENT_PATH"
fi

# Apply GenTRX training agent defaults only on first setup (no saved config, no
# explicit agent flag, and agent is still the non-training default).
if [ "$GENTRX" = "1" ] && [ "$_EXPLICIT_AGENT" = "0" ] && \
   [ -z "${GENTRX_SAVED_AGENT_NAME:-}" ] && [ "$AGENT_NAME" = "SimpleRegressorAgent" ]; then
    AGENT_NAME=HybridTrainingAgent
    AGENT_PARAMS="imbalance_depth=5 history_retention_mins=1 \
entry_threshold=0.35 cancel_threshold=0.20 \
stop_loss_bps=40 base_quote_size=0.3 enter_size_mult=3.0 \
max_flat_inventory=2.0 expiry_period=500000000 max_fee_rate=0.005"
fi

echo "ENDPOINT:        $ENDPOINT"
echo "WALLET:          $WALLET_PATH $WALLET_NAME / $HOTKEY_NAME"
echo "NETUID:          $NETUID"
echo "AXON_PORT:       $AXON_PORT"
echo "AGENT_PATH:      $AGENT_PATH"
echo "AGENT_NAME:      $AGENT_NAME"
echo "AGENT_PARAMS:    $AGENT_PARAMS"
echo "GENTRX:          $GENTRX"
echo "GENTRX_PARAMS:   ${GENTRX_PARAMS:-(defaults)}"

cd "$REPO_ROOT"
git pull || { echo "WARNING: git pull failed (no tracking branch?). Continue without updating? [y/N]"; read -r _yn; [ "$_yn" = "y" ] || exit 1; }
# THE LOCKFILE ON EVERY START, not only at install. A host that already had an older package kept it as
# long as it satisfied the unpinned requirement; 0.6.3 testnet validators then served HTTP 500 on every
# metrics family with a prometheus_client older than the pin.
_CONSTRAINTS=""; [ -f constraints.txt ] && _CONSTRAINTS="-c constraints.txt"
pip install -e . $_CONSTRAINTS
cd "$REPO_ROOT/taos/im/neurons"

# ══════════════════════════════════════════════════════════════════════════════
# GenTRX setup (runs when -G is passed)
# ══════════════════════════════════════════════════════════════════════════════

_hr()         { echo; printf '%s\n' "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
_ok()         { printf ' \033[32m✓\033[0m  %s\n' "$*"; }
_warn()       { printf ' \033[33m⚠\033[0m  %s\n' "$*"; }
_step()       { printf '\n \033[1m─ %s\033[0m\n' "$*"; }
_info()       { printf '   %s\n' "$*"; }

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


_gentrx_miner_setup() {
    _hr
    printf ' \033[1mGenTRX Miner Setup\033[0m\n'
    _hr
    echo
    _info "Docs: doc/gentrx/miner_setup.md"
    echo

    # ── Data directory ───────────────────────────────────────────────────────
    _step "Miner data directory"
    _info "Where the miner writes parquets, gradient cache, downloaded"
    _info "checkpoints, pending-upload buffer, and rotating logs."
    _info ""
    _info "Default: <repo>/agents/data/<uid>/ resolved from the GenTRX"
    _info "package path (Path(__file__).resolve()), independent of the"
    _info "directory you launch from."
    _info ""
    _info "Override with an absolute path if your repo lives on a small"
    _info "disk and you want data on a larger volume."
    echo
    local _default_data_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/agents/data"
    _prompt GENTRX_AGENT_OUTPUT_DIR "Miner data directory" "$_default_data_dir"

    # ── Miner bucket ─────────────────────────────────────────────────────────
    _step "Miner gradient bucket"
    _info "Each miner needs one R2, Storj, or Hippius bucket for gradient uploads."
    _info "In your provider's dashboard, create a bucket and generate TWO keys:"
    _info "  • Write key  (Object Read & Write) — stays on this host"
    _info "  • Read key   (Object Read only)    — committed on-chain"
    _info "Copy the Access Key ID and Secret Access Key from each."
    echo

    local _provider
    if [ "${GENTRX_AGENT_S3_PROVIDER:-}" = "storj" ]; then
        _provider="3"               # Storj: provider hint already in env
    elif [ -n "${GENTRX_AGENT_S3_ACCOUNT_ID:-}" ]; then
        _provider="configured-r2"   # R2: account_id present
    elif [ -n "${GENTRX_AGENT_S3_ENDPOINT_URL:-}" ] || [ -n "${GENTRX_AGENT_S3_BUCKET:-}" ]; then
        _provider="2"               # custom endpoint (MinIO/Hippius): no account_id
    else
        printf '    Provider: (1) Cloudflare R2  (2) Hippius  (3) Storj  [1]: '
        read -r _provider; _provider="${_provider:-1}"
    fi

    case "$_provider" in
        2|hippius|Hippius)
            _prompt GENTRX_AGENT_S3_ENDPOINT_URL \
                "Hippius endpoint URL" "https://s3.hippius.com"
            _prompt GENTRX_AGENT_S3_BUCKET "Bucket name"
            ;;
        3|storj|Storj)
            GENTRX_AGENT_S3_PROVIDER="storj"
            _env_write "GENTRX_AGENT_S3_PROVIDER" "storj"
            export GENTRX_AGENT_S3_PROVIDER
            _prompt GENTRX_AGENT_S3_ENDPOINT_URL \
                "Storj gateway URL" "https://gateway.storjshare.io"
            _prompt GENTRX_AGENT_S3_BUCKET "Bucket name"
            _info "Storj read grant needs only GetObject scope; ListBucket is not required."
            ;;
        *)
            _prompt GENTRX_AGENT_S3_ACCOUNT_ID \
                "Cloudflare R2 account ID (32-char hex)"
            local _default_bucket="${GENTRX_AGENT_S3_ACCOUNT_ID:-}"
            _prompt GENTRX_AGENT_S3_BUCKET "Bucket name" "$_default_bucket"
            if [ -z "${GENTRX_AGENT_S3_ENDPOINT_URL:-}" ]; then
                GENTRX_AGENT_S3_ENDPOINT_URL="https://${GENTRX_AGENT_S3_ACCOUNT_ID}.r2.cloudflarestorage.com"
                _env_write "GENTRX_AGENT_S3_ENDPOINT_URL" "$GENTRX_AGENT_S3_ENDPOINT_URL"
                export GENTRX_AGENT_S3_ENDPOINT_URL
                _info "Endpoint URL: $GENTRX_AGENT_S3_ENDPOINT_URL"
            fi
            ;;
    esac
    echo
    _step "Write token (1 of 2) — for gradient uploads, stays on this host"
    _prompt_secret GENTRX_AGENT_S3_ACCESS_KEY "Write access key ID"
    _prompt_secret GENTRX_AGENT_S3_SECRET_KEY "Write secret access key"

    _step "Read-only token (2 of 2) — committed on-chain so validators can score your gradients"
    _prompt_secret GENTRX_AGENT_S3_READ_ACCESS_KEY "Read-only access key ID"
    _prompt_secret GENTRX_AGENT_S3_READ_SECRET_KEY "Read-only secret access key"
    _ok "Miner S3 credentials saved to .env"

    # ── Chain commitment ──────────────────────────────────────────────────────
    _step "On-chain bucket commitment"
    _info "Checking if your bucket is already committed to chain..."

    local _committed _chain_out
    _chain_out=$(python3 -c "
import bittensor as bt, sys
try:
    sub = bt.Subtensor(network='$ENDPOINT')
    meta = sub.metagraph($NETUID)
    w = bt.Wallet(name='$WALLET_NAME', hotkey='$HOTKEY_NAME', path='$WALLET_PATH')
    uid = list(meta.hotkeys).index(w.hotkey.ss58_address)
    data = sub.get_commitment($NETUID, uid)
    print(uid, 'yes' if len(data) == 128 else 'no')
except Exception:
    print('', 'no')
" 2>/dev/null)
    # _MINER_UID intentionally not declared local — used outside _gentrx_miner_setup
    _MINER_UID=$(echo "$_chain_out" | awk '{print $1}')
    _committed=$(echo "$_chain_out" | awk '{print $2}')

    if [ "$_committed" = "yes" ]; then
        _ok "Bucket already committed on-chain — skipping chain transaction."
        echo
        _info "To re-commit (e.g. after rotating credentials), run:"
        _info "  python bin/setup_miner_bucket.py \\"
        _info "    --account-id \$GENTRX_AGENT_S3_ACCOUNT_ID \\"
        _info "    --wallet-name $WALLET_NAME --wallet-hotkey $HOTKEY_NAME \\"
        _info "    --wallet-path $WALLET_PATH \\"
        _info "    --netuid $NETUID --subtensor-network $ENDPOINT"
        _info "  (credentials are read from GENTRX_AGENT_S3_* env vars in .env)"
    else
        _info "Committing bucket credentials to chain..."
        echo
        # Export credentials so setup_miner_bucket.py picks them up from env
        # (.env is sourced without set -a, so vars are set but not yet exported)
        export GENTRX_AGENT_S3_ACCESS_KEY GENTRX_AGENT_S3_SECRET_KEY \
               GENTRX_AGENT_S3_READ_ACCESS_KEY GENTRX_AGENT_S3_READ_SECRET_KEY
        # Determine endpoint/account-id args for setup_miner_bucket.py
        local _bucket_args
        if [ -n "${GENTRX_AGENT_S3_ACCOUNT_ID:-}" ]; then
            _bucket_args="--account-id $GENTRX_AGENT_S3_ACCOUNT_ID"
        else
            _bucket_args="--endpoint $GENTRX_AGENT_S3_ENDPOINT_URL --bucket $GENTRX_AGENT_S3_BUCKET"
        fi
        python3 "$REPO_ROOT/bin/setup_miner_bucket.py" \
            $_bucket_args \
            --wallet-name "$WALLET_NAME" \
            --wallet-hotkey "$HOTKEY_NAME" \
            --wallet-path "$WALLET_PATH" \
            --netuid "$NETUID" \
            --subtensor-network "$ENDPOINT"
        _ok "Bucket committed on-chain"
    fi

    # Export S3 vars so pm2-managed miner.py inherits them.
    # `. .env` sets variables in the shell but doesn't export them; pm2 only
    # sees exported vars, so without this the miner's aggregator store is None.
    for _v in GENTRX_AGGREGATOR_S3_BUCKET GENTRX_AGGREGATOR_S3_ENDPOINT_URL \
              GENTRX_AGGREGATOR_S3_READ_ACCESS_KEY GENTRX_AGGREGATOR_S3_READ_SECRET_KEY \
              GENTRX_AGENT_S3_ENDPOINT_URL GENTRX_AGENT_S3_BUCKET \
              GENTRX_AGENT_S3_ACCESS_KEY GENTRX_AGENT_S3_SECRET_KEY \
              GENTRX_AGENT_S3_READ_ACCESS_KEY GENTRX_AGENT_S3_READ_SECRET_KEY \
              GENTRX_CHAIN_ENDPOINT_OVERRIDE; do
        [ -n "${!_v:-}" ] && export "$_v"
    done

    # ── Default GenTRX params ─────────────────────────────────────────────────
    # Use UID for the output directory so it matches existing data layout
    # (orders, trades, cancellations are all keyed by UID). Fall back to
    # hotkey name if chain lookup didn't return a UID.
    local _uid_dir="${_MINER_UID:-$HOTKEY_NAME}"
    if [ -z "$GENTRX_PARAMS" ]; then
        GENTRX_PARAMS="gtx_training_enabled=true gtx_train_steps=50 gtx_train_batch_size=16 gtx_train_seq_len=256 gtx_output_dir=$REPO_ROOT/agents/data/$_uid_dir"
    fi
    # Ensure gtx_output_dir is present even when params were set externally.
    if [[ "$GENTRX_PARAMS" != *gtx_output_dir=* ]]; then
        GENTRX_PARAMS="${GENTRX_PARAMS} gtx_output_dir=$REPO_ROOT/agents/data/$_uid_dir"
    fi

    _ok "GenTRX miner setup complete."
    _info "Agent:         $AGENT_NAME"
    _info "GENTRX_PARAMS: $GENTRX_PARAMS"
    _info "Monitor:  pm2 logs miner | grep '\\[GTX\\]'"
}

_print_miner_cmd() {
    local _qparams _qgtx
    _qparams=$(python3 -c "import sys,shlex; print(shlex.quote(sys.argv[1]))" "$AGENT_PARAMS")
    _qgtx=$(python3 -c "import sys,shlex; print(shlex.quote(sys.argv[1]))" "$GENTRX_PARAMS")
    _hr
    printf ' \033[1mSave this command — run it for updates or as a template for additional UIDs:\033[0m\n'
    printf ' \033[1m(Adjust -w/-h/-a for each UID)\033[0m\n'
    _hr
    echo
    printf '    \033[1m./run_miner.sh\033[0m \\\n'
    printf '        -G \\\n'
    printf '        -e %s \\\n'   "$ENDPOINT"
    printf '        -w %s \\\n'   "$WALLET_NAME"
    printf '        -h %s \\\n'   "$HOTKEY_NAME"
    printf '        -u %s \\\n'   "$NETUID"
    printf '        -a %s \\\n'   "$AXON_PORT"
    printf '        -n %s \\\n'   "$AGENT_NAME"
    printf '        -m %s \\\n'   "$_qparams"
    printf '        -t %s\n'      "$_qgtx"
    echo
    printf '  Note: bucket credentials are in .env.\n'
    printf '  Copy .env to each new deployment (or re-run without -G to use the same .env).\n'
    _hr
    echo
    printf ' \033[33m⚠\033[0m  Copy the command above, then press Enter to continue... '
    read -r
    echo
}

if [ "$GENTRX" = "1" ]; then
    _gentrx_miner_setup
    # Persist config so subsequent runs (no flags) restart with same settings
    _env_write "GENTRX_ENABLED"              "1"
    _env_write "GENTRX_SAVED_AGENT_PATH"     "$AGENT_PATH"
    _env_write "GENTRX_SAVED_AGENT_NAME"     "$AGENT_NAME"
    _env_write "GENTRX_SAVED_AGENT_PARAMS"   "$AGENT_PARAMS"
    _env_write "GENTRX_SAVED_GENTRX_PARAMS"  "$GENTRX_PARAMS"
    _print_miner_cmd
fi

# ── Launch miner ───────────────────────────────────────────────────────────────
# Split AGENT_PARAMS and GENTRX_PARAMS strings into arrays for safe pm2 passing
read -ra _agent_params <<< "$AGENT_PARAMS"
read -ra _gentrx_params <<< "$GENTRX_PARAMS"

if [ "$GENTRX" = "1" ]; then
    pm2 delete miner 2>/dev/null || true
    pm2 start miner.py \
        --name=miner \
        --interpreter python \
        --cwd "$REPO_ROOT/taos/im/neurons" \
        -- \
        --netuid "$NETUID" \
        --subtensor.chain_endpoint "$ENDPOINT" \
        --wallet.path "$WALLET_PATH" \
        --wallet.name "$WALLET_NAME" \
        --wallet.hotkey "$HOTKEY_NAME" \
        --axon.port "$AXON_PORT" \
        --logging."$LOG_LEVEL" \
        --agent.path "$AGENT_PATH" \
        --agent.name "$AGENT_NAME" \
        --agent.params "${_agent_params[@]}" "${_gentrx_params[@]}"
    pm2 save || true
    pm2 startup || true

    # Derive training log path from gtx_output_dir param
    _gtx_out=$(python3 -c "
import re, sys
m = re.search(r'gtx_output_dir=(\S+)', sys.argv[1])
print(m.group(1) if m else '')
" "$AGENT_PARAMS $GENTRX_PARAMS")
    if [ -z "$_gtx_out" ]; then
        TRAIN_LOG="$REPO_ROOT/agents/data/$HOTKEY_NAME/gradients/train.log"
    elif [[ "$_gtx_out" = /* ]]; then
        TRAIN_LOG="$_gtx_out/gradients/train.log"
    else
        TRAIN_LOG="$REPO_ROOT/taos/im/neurons/$_gtx_out/gradients/train.log"
    fi

    tmux kill-session -t miner 2>/dev/null || true
    tmux new-session -d -s miner -n 'miner' 'htop -F miner.py'
    tmux set-option -t miner mouse on
    tmux split-window -v -p 50 -t miner:miner.0 "pm2 logs miner"
    tmux split-window -h -p 50 -t miner:miner.1 "echo 'Waiting for training log...'; tail -F '$TRAIN_LOG' 2>/dev/null"
    tmux select-pane -t miner:miner.0
    tmux attach-session -t miner
else
    pm2 delete miner 2>/dev/null || true
    pm2 start miner.py \
        --name=miner \
        --interpreter python \
        --cwd "$REPO_ROOT/taos/im/neurons" \
        -- \
        --netuid "$NETUID" \
        --subtensor.chain_endpoint "$ENDPOINT" \
        --wallet.path "$WALLET_PATH" \
        --wallet.name "$WALLET_NAME" \
        --wallet.hotkey "$HOTKEY_NAME" \
        --axon.port "$AXON_PORT" \
        --logging."$LOG_LEVEL" \
        --agent.path "$AGENT_PATH" \
        --agent.name "$AGENT_NAME" \
        --agent.params "${_agent_params[@]}"
    pm2 save || true
    pm2 startup || true
    pm2 logs miner
fi
