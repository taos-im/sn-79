#!/bin/sh
set -e
echo $PATH
echo 'apt update'
apt-get update

if ! command -v pm2
then
	if ! command -v node
	then
		echo 'Installing latest Node.js via NodeSource...'
		curl -fsSL https://deb.nodesource.com/setup_current.x | bash -
		apt-get install -y nodejs
	fi
	echo 'Installing pm2...'
	npm install --location=global pm2
	pm2 install pm2-logrotate
	pm2 set pm2-logrotate:max_size 100M
	pm2 set pm2-logrotate:compress true
fi

if ! command -v htop
then
	echo 'Installing htop...'
	apt-get install htop -y
fi

if ! command -v tmux
then
	echo 'Installing tmux...'
	apt-get install -y tmux
fi

if ! command -v pyenv && [ ! -d "$HOME/.pyenv" ]
then
    echo 'Installing pyenv'
    apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    # Fetch the installer to a file first (fail on HTTP error, no partial-pipe
    # execution) rather than piping the network stream straight into bash.
    _pyenv_installer="$(mktemp)"
    curl -fsSL https://pyenv.run -o "$_pyenv_installer"
    bash "$_pyenv_installer"
    rm -f "$_pyenv_installer"
    echo 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
fi
# Always activate pyenv in this shell, even on re-runs where the dir already
# existed and the bashrc PATH addition hasn't propagated to the current shell yet.
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
if ! pyenv versions | grep -Fq '3.10.9';
then
    echo 'Installing and activating Python 3.10.9'
    pyenv install 3.10.9
    pyenv global 3.10.9
fi

python -m pip install -U pyopenssl cryptography

echo "Installing taos"
# GenTRX distributed-training deps (torch stack + polars/pyarrow) are an OPT-IN
# extra. A plain miner runs basic trading on core deps; set INSTALL_GENTRX=1 to
# also install the [gentrx] extra for training/inference agents.
_TAOS_EXTRAS=""
if [ "${INSTALL_GENTRX:-0}" = "1" ]; then
    _TAOS_EXTRAS="[gentrx]"
    echo "Including GenTRX training extra ([gentrx])"
fi
# Use the committed lockfile when present for a reproducible dependency set.
if [ -f constraints.txt ]; then
    python -m pip install -e ".${_TAOS_EXTRAS}" -c constraints.txt
else
    python -m pip install -e ".${_TAOS_EXTRAS}"
fi
mkdir -p ~/.taos/agents
# Copy CONTENTS, not the directory: `cp -r agents ~/.taos/agents` nests
# agents/agents on any re-run once the destination exists.
cp -r agents/. ~/.taos/agents/
