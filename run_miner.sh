#!/bin/sh
ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
WALLET_PATH=~/.bittensor/wallets/
WALLET_NAME=huc_q
HOTKEY_NAME=miner
NETUID=79
AXON_PORT=31313
AGENT_PATH=~/.taos/agents
AGENT_NAME=MyNewAgent
AGENT_PARAMS="min_quantity=0.1 max_quantity=1.0 expiry_period=200 model=PassiveAggressiveRegressor signal_threshold=0.0025"
LOG_LEVEL=info
while getopts e:p:w:h:u:a:g:n:m:l: flag
do
    case "${flag}" in
        e) ENDPOINT=${OPTARG};;
        p) WALLET_PATH=${OPTARG};;
        w) WALLET_NAME=${OPTARG};;
        h) HOTKEY_NAME=${OPTARG};;
        u) NETUID=${OPTARG};;
        a) AXON_PORT=${OPTARG};;
        g) AGENT_PATH=${OPTARG};;
        n) AGENT_NAME=${OPTARG};;
        m) AGENT_PARAMS=${OPTARG};;
        l) LOG_LEVEL=${OPTARG};;
    esac
done
echo "ENDPOINT: $ENDPOINT"
echo "WALLET_PATH: $WALLET_PATH"
echo "WALLET_NAME: $WALLET_NAME"
echo "HOTKEY_NAME: $HOTKEY_NAME"
echo "NETUID: $NETUID"
echo "AXON_PORT: $AXON_PORT"
echo "AGENT_PATH: $AGENT_PATH"
echo "AGENT_NAME: $AGENT_NAME"
echo "AGENT_PARAMS: $AGENT_PARAMS"

git pull
pip install -e .
cd taos/im/neurons
# pm2 delete sn79-1
pm2 start --name=sn79-1 "python miner.py  --netuid $NETUID --subtensor.chain_endpoint $ENDPOINT --wallet.path $WALLET_PATH --wallet.name $WALLET_NAME --wallet.hotkey $HOTKEY_NAME --axon.port $AXON_PORT --logging.debug --agent.path $AGENT_PATH --agent.name $AGENT_NAME  --agent.params $AGENT_PARAMS --logging.$LOG_LEVEL"
pm2 save
pm2 startup
pm2 logs sn79-1