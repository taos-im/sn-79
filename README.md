<div align="center">

# **τaos** ☯ **‪ي‬n 79**<!-- omit in toc -->

### **Decentralized Simulation of Automated Trading in Intelligent Markets:** <!-- omit in toc -->

### **Risk-Averse Agent Optimization** <!-- omit in toc -->

## [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**τaos** operates as a [Bittensor](https://bittensor.com) subnet at netuid 79, where participants are incentivized to produce meaningful contributions using carefully risk-managed algorithms in a large-scale agent-based **s**imulation **o**f **a**utomated **t**rading strategies.

[![Website](https://img.shields.io/badge/website-black?logo=googlechrome)](https://taos.im)
[![Whitepaper](https://img.shields.io/badge/whitepaper-white?logo=proton)](https://simulate.trading/taos-im-paper)
[![Dashboard](https://img.shields.io/badge/dashboard-white?logo=grafana)](https://taos.simulate.trading)
[![Discord](https://img.shields.io/badge/discord-black?logo=discord)](https://discord.com/channels/799672011265015819/1353733356470276096)

---

**_taos_ (/ˈtɑos/)** : To make things out of metal by heating it until it is soft and then bending and hitting it with a hammer to create the right shape.

---

### Table of Contents

</div>

1. [Incentive Mechanism](#mechanism)
   - [Owner Role](#mechanism-owner)
   - [Validator Role](#mechanism-validator)
   - [Miner Role](#mechanism-miner)
2. [Technical Operation](#technical)
   - [Simulator](#technical-simulator)
   - [Validator](#technical-validator)
   - [Miner](#technical-miner)
3. [Requirements](#requirements)
   - [Validator](#requirements-validator)
   - [Miner](#requirements-miner)
4. [Install](#install)
   - [Validator](#install-validator)
   - [Miner](#install-miner)
   - [Docker](#install-docker)
5. [Agents](#agents)
6. [Run](#run)
   - [Validator](#run-validator)
   - [Miner](#run-miner)

---

<div style="page-break-after: always;"></div>

## Incentive Mechanism <span id="mechanism"><span>

The incentive mechanism of subnet 79 is designed to promote intelligent, risk-managed trading logic to be applied by agents, in order that we are able to produce valid and valuable datasets mimicing the properties of a variety of different real-world asset classes and market conditions.

See the [whitepaper](https://simulate.trading/taos-im-paper) for a more detailed exploration of the background, goals and scope of the τaos project.

### Owner Role <span id="mechanism-owner"><span>

The subnet owners are tasked with ensuring fair, equitable and correct operation of the subnet mechanisms (as in all other subnets), while also being responsible for the design, refinement, tuning and publishing of the simulation parameters and logic. This involves consistent monitoring, testing and development to expand the capabilities of the simulator and determine parameters which result in the most useful possible outputs being generated through the subnet's operation. The owner must also ensure that the metrics utilized in determining miner rewards are chosen such that miners are incentivized to act fairly and in such a way that outputs are of optimal value in research, trading strategy development, market surveillance and other applications.

### Validator Role <span id="mechanism-validator"><span>

Validators in the subnet are responsible primarily for maintaining the state of the simulation, and rewarding agents (miners) which achieve the best results over all realizations of the simulated market. They deploy two components:

- The C++ simulator, which handles all the computation necessary to simulate asset markets
- The Python validator, which receives state updates from the simulator, forwards these to miners, submits instructions received in response back to the simulator, and calculates miner scores based on their performance throughout the simulation.

### Miner Role <span id="mechanism-miner"><span>

Miners in the subnet function as trading agents in the distributed simulation; their role is to develop and host trading strategies which maximize their average risk-adjusted performance measures over all simulated market realizations, while also maintaining a sufficient level of trading activity. There are no strict limitations on what strategies are able to be applied, but the simulation parameters and performance evaluation metrics will be continually reviewed, selected and adjusted with the intention of maximizing the utility of the output data, and promoting the use of intelligent, risk-averse and budget-constrained trading logic.

---

<div style="page-break-after: always;"></div>

## Technical Operation <span id="technical"><span>

The subnet operates at technical level in the first implementation in quite familiar manner for the Bittensor ecosystem. Validators construct requests containing the simulation state, which results from a series of computations by the simulator, and publishes these requests to miners at a pre-defined interval. Miners must respond to validator requests within a reasonable timeframe in order for their instructions to be submitted to the simulation for execution. Scores are calculated in general as a weighted sum of several risk-adjusted performance metrics; although, at least until others are required, only an intraday Sharpe ratio is evaluated. Miners are also required to maintain a certain level of trading volume in order for their risk-adjusted performance score to be allocated in full - this prevents inactive miners from gaining incentives, and aligns with the objective of the project to encourage active automated trading rather than simple buy & hold or other very low-frequency strategies.

In the current approach, a new simulation configuration is intended to be deployed on approximately weekly basis, with each simulation being executed as an independent run where all miner agents begin with the same initial capital allocation. Multiple runs of a particular configuration may be executed by validators before a new configuration is published, due to varying rate of progression resulting from differing resources deployed by validators. Miner scores are however calculated using a rolling window which is not cleared at the start of a new simulation, so that performance in previous races does still contribute to the miner's overall weighting. Deregistrations are handled by resetting the account balance and positions of the agent associated with the UID which was newly registered to the configured starting values.

### Simulator <span id="technical-simulator"><span>

The bulk of the computation involved in running the simulations is handled by a C++ agent-based simulation engine, which is built on top of [MAXE](https://github.com/maxe-team/maxe). This engine is deployed as a companion application to the validator logic, and handles the orderbook and account state maintenance necessary to simulate the high-frequency level microstructure of intelligent markets.

The construction of a fully detailed limit order book is a key advantage of utilizing an agent-based model as opposed to other generative techniques: since the simulation operates in the same manner as real markets, being composed of a large number of independent actors submitting instructions to a central matching engine, we not only allow for the highest level of customization and realism in the simulated market, but also reproduce all the finer details of the environment. This includes the full limit order book with _Level 3_ or _Market-by-order_ data, which records every individual event occurring within the market and is valuable (even necessary) for the development of advanced, high-frequency and data-intensive (e.g. AI) trading strategies, as well as in performing deep analyses of market behaviour for monitoring, surveillance and other regulatory purposes.

The simulator is designed to maintain any number of orderbooks simultaneously, ensuring statistical significance of the results observed by providing many realizations over which to verify the outcomes.

The simulator additionally includes implementations of the "background agents" which create the basic conditions of the markets in which distributed agents (miners) trade. It is the parameterization of the background model which these agents comprise that determines the high-level behaviour of the simulated orderbooks. The model implemented is based on well-established research in the field; for some background and details see [Chiarella et. al. 2007](https://arxiv.org/abs/0711.3581) and [Vuorenmaa & Wang 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2336772).

### Validator <span id="technical-validator"><span>

The validator functions in a sense as a "proxy" which allows the C++ simulator to communicate with participants in the subnet, and handles all the Bittensor-network-related tasks involved in validator operation - authenticating and distributing requests, validating and processing responses, calculating miner scores, setting weights and, ultimately, providing access to the subnet computational resources for external queries.

### Miner <span id="technical-miner"><span>

Miners receive state updates from validators, and respond with instructions as to how they would like to modify their positions in the simulated orderbook realizations by placing and cancelling orders. In this first version of the subnet, state updates are published at a parameterized interval throughout the simulation and miners are only able to submit instructions at each state publishing event - a planned future implementation will allow continuous bidirectional communication of state and instructions, in line with real exchange operation. The state update includes all L3 messages processed since the last update, so that strategies analysing the most detailed information are still possible to apply, being limited only in the time-scale at which the algorithm is able to act.

While awaiting the validator to request and receive responses from miners, the simulation is paused such that no events are processed during this time; in order to reward efficient computation as well as agent performance, the response time of miners is used to determine the "delay" or "latency" with which their instructions will be processed by the simulator. Longer response times thus imply more events that may take place before execution of their instructions, requiring realistic effects like price slippage to be accounted for. This incentivizes miners to be both fast and intelligent in their trading strategy, while also carefully managing their risk across all market state realizations.

Miner agents are otherwise treated in the simulator on the same footing as the background agents; their instructions are processed as if they would be submitted to a real exchange, with orders being traded or opened on the book in accordance with standard matching rules. Every agent's orders will interact with those of the background agents and miners - this ensures a proper and full accounting of the performance of the miner, including their market impact. Since agents actions directly affect the market structure, this must also be considered when making trading decisions.

---

<div style="page-break-after: always;"></div>

## Requirements <span id="requirements"><span>

Requirements are subject to change as the subnet matures and evolves; this section describes the recommended resources to be available for the initial simulation conditions. We currently manage 40 orderbooks in a simulation, each having around 1000 background agents, while the aim in the near- to mid-term is to reach 1,000+ simulated orderbooks in order to achieve a meaningful level of statistical significance in the evaluation of results.

### Validator <span id="requirements-validator"><span>

Validators need to host the C++ simulator as well as the Python validator. In the early days of the subnet, the number of orderbooks simulated as well as the count and type of background agents will be reduced so as to limit the requirements before the subnet matures and sufficient emissions are gained to justify the expense of hosting more powerful machinery. Basic requirements:

- 16GB RAM
- 8 CORE CPU
- Ubuntu >= 22.04
- g++ 14.

We hope to increase both major parameters significantly as soon as possible so that validators may wish to prepare a larger machine for easier expansion. It should be noted however that increasing the CPU resources available will result in a faster progression of simulations due to multi-threaded processing of the orderbook realizations. This should not inherently be a problem, but may cause divergences in scoring if there is a major discrepancy in resources with the other validators in the subnet. We plan to communicate the setup employed by our validator whenever changes are made, and will enable to configure the resources allocated for simulation processing if necessary.

### Miner <span id="requirements-miner"><span>

There are no set requirements for miners except that the basic Bittensor package and subnet miner tools occupy ~1GB of RAM per miner instance; resources needed will depend on the complexity and efficiency of the specific strategy implementation.

---

## Agents <span id="agents"><span>

In order to separate the basic network logic from the actual trading logic and allow to easily switch between different strategies, miners in this subnet define a separate class containing the agent logic which is referenced in the configuration of the miner and loaded for handling of simulation state updates. Some simple example agents are provided in the `agents` directory of this repository, and are copied to a directory `~/.taos/agents` if using the miner install script to prepare your environment. The objective in agent development is to produce logic which maximizes performance over all realizations in terms of the evaluation metrics applied by the validators. Currently assessment is based on an intraday Sharpe ratio of the changes in estimated total inventory value in conjunction with a requirement to maintain a certain level of cumulative trading volume; this will be continuously monitored and reviewed, and other relevant risk-adjusted performance measures incorporated if a need is observed.

Only some basic agents are immediately included as examples, designed to illustrate the fundamentals of reading the state updates and creating instructions. We expect miners to develop their own custom logic in order to compete in the subnet, but plan to release additional examples, tools and templates to facilitate implementation of certain common classes of trading strategies. An overview of the information needed to begin developing strategies is provided [here](agents/README.md). It is also possible to test agents offline against the background model on your local machine by following [these instructions](agents/proxy/README.md).

---

<div style="page-break-after: always;"></div>

## Install <span id="install"><span>

For convenience, this repository includes tools to prepare your environment with the necessary applications, build tools, dependencies and other prerequisites.

To get started, first clone the repository and enter the directory:

```console
git clone https://github.com/taos-im/sn-79
cd sn-79
```

### Docker <span id="install-docker"><span>

DOCKER DEPLOYMENT IMPLEMENTATION WILL BE COMPLETED IF REQUESTED BY VALIDATORS

### Validator <span id="install-validator"><span>

To prepare your environment for running a validator (including the C++ simulator), simply run the included script **as root user** (if unable to execute as root, please reach out to us for assistance, but may need to await Docker-based deploy in this case):

```console
./install_validator.sh
```

If prompted to restart any services, just hit "Enter" to proceed. You may need to re-open your shell session after installation completes before newly installed applications can be used.

This will install the following tools:

- **prometheus-node-exporter** : To enable resource usage monitoring via Grafana or similar
- **nvm + pm2** : For process management
- **tmux** : For multiplexing to allow simultaneous viewing of simulator and validator logs
- **pyenv** : For managing of Python version installations
- **Python 3.10.9** : This version of Python has been used in all testing; later versions will likely still work but have not been tested
- **τaos** : The Python component of the apparatus, containing the base validator and miner logic
- **vcpkg** : For C++ simulator dependency management
- **g++-14** : If on Ubuntu 22.04 rather than the latest 24.04, g++ 13.1 must be installed and used for compilation of the simulator (g++ 14.2 is already included in Ubuntu 24.04 install)
- **cmake-3.29.7** : Required to build and run the simulator
- **τaos.im simulator** : The C++/Pybind simulator application.

You can of course modify the install script if you wish to make changes to the installation, or use this as a guide to execute the steps by hand if you prefer. Note that the installation process takes quite a long time, often 2+ hours on Ubuntu 22.04, due to the need to compile specific cmake and g++ versions from source, so recommended to run in a multiplexer (e.g. screen/tmux) to prevent interruptions.

<div style="page-break-after: always;"></div>

### Miner <span id="install-miner"><span>

To prepare your environment for running a miner, simply execute the included script:

```console
./install_miner.sh
```

You may need to re-open your shell session after installation completes before newly installed applications can be used.
This will install the following tools:

- **prometheus-node-exporter** : To enable resource usage monitoring via Grafana or similar
- **nvm + pm2** : For process management
- **tmux** : For multiplexing to allow simultaneous viewing of simulator and validator logs
- **pyenv** : For managing of Python version installations
- **Python 3.10.9** : Fully tested version, others may work but have not been tested
- **τaos** : The Python component of the apparatus, containing the base validator and miner logic.

---

<div style="page-break-after: always;"></div>

## Run <span id="run"><span>

We include simple shell scripts to facilitate running of a validator or miner; it is also possible to run the applications directly yourself in the case of miner processes, though validators are strongly recommended to use the provided script in order to ensure that all components are updated properly.
If you wish to use the run scripts, first enter the directory where you have cloned this repo. You will of course also need to register the hotkey with which you wish to mine/validate on the subnet.

### Validator <span id="run-validator"><span>

To run a validator, you can use the provided `run_validator.sh` which accepts the following arguments:

- `-e` : The subtensor endpoint to which you will connect (default=`wss://entrypoint-finney.opentensor.ai:443`)
- `-p` : The path where your wallets are stored (default=`~/.bittensor/wallets/`)
- `-w` : The name of your coldkey (default=`taos`)
- `-h` : The name of your hotkey (default=`validator`)
- `-l` : Logging level for the validator, must be one of `error`, `warning`, `info`, `debug`, `trace` (default=`info`)
- `-d` : Pagerduty integration key; if you have a Pagerduty subscription, this allows to trigger alerts for critical failure scenarios (default=`""`)
- `-o` : Port on which Prometheus metrics will be published. If you use a different port than the default, please let us know so that your data will still appear at [taos.simulate.trading](https://taos.simulate.trading/?orgId=1) (default=`9001`).
- `-t` : Timeout for miner queries; this allows validators to tune the time allowed for miners to respond to account for differences in server geolocation or networking capability (default=`3.0`).
- `-s` : Flag to indicate that the simulator should not be restarted when performing the update; this allows to easily execute updates which only affect the Python validator operation (default=`0`; append `-s 1` to command to preserve running simulator during update).
- `-x` : Flag to indicate if wanting to launch tmux session for monitoring (default=`1`; append `-x 0` to command to disable tmux session creation).
<!-- - `-c` : If the simulator has been stopped for whatever reason, and you wish to resume the previous simulation rather than starting a new one, you can set this argument to the location of the `ckpt.json` found
in the output directory of the simulation.(default=`0` => start a new simulation) -->

The script will:

1. Pull and install the latest changes from the taos repository
2. Build the latest version of the simulator
3. Launch a validator under pm2 management as `validator` with the specified parameters
4. Start the simulator under pm2 management as `simulator`
5. Save the pm2 process list and configure for resurrection on restart
6. Open a `tmux` session with the logs for validator and simulator displayed in separate panes (if not disabled).

Example run command:

```
./run_validator.sh -e finney -p ~/.bittensor/wallets -w taos -h validator -u 79
```

**We recommend running as root to avoid any permissions issues.**

<div style="page-break-after: always;"></div>

<!-- CHECKPOINTING FUNCTIONALITY RESTORED SOON
To resume from a checkpoint, replace the last line with:
```
../build/src/cpp/taosim -c $CHECKPOINT
``` -->

### Miner <span id="run-miner"><span>

To run a miner, you can use the provided `run_miner.sh` which accepts the following arguments:

- `-e` : The subtensor endpoint to which you will connect (default=`wss://entrypoint-finney.opentensor.ai:443`)
- `-p` : The path where your wallets are stored (default=`~/.bittensor/wallets/`)
- `-w` : The name of your coldkey (default=`taos`)
- `-h` : The name of your hotkey (default=`miner`)
- `-u` : Netuid on which to serve the miner, for testing with localnet or testnet (default=`79`)
- `-a` : The port on which to serve the miner axon (default=`8091`)
- `-g` : The path at which your agent definition files are stored (default=`~/.taos/agents`)
- `-n` : The name of the file (and class) in the directory specified by `-g` of the agent logic which this miner will use (default=`SimpleRegressorAgent`)
- `-m` : Parameters for the active agent in the format `param_1=x param_2=y ..` (default=`min_quantity=0.1 max_quantity=1.0 expiry_period=200 model=PassiveAggressiveRegressor signal_threshold=0.0025`)
- `-l` : Logging level for the miner, must be one of `error`, `warning`, `info`, `debug`, `trace` (default=`info`)
  The script will:

1. Pull and install the latest changes from the taos repository
2. Launch a miner under pm2 management as `miner` with the specified parameters
3. Save the pm2 process list and configure for resurrection on restart
4. Display the logs of the running miner.

Example run command:

```
./run_miner.sh -e finney -p ~/.bittensor/wallets/ -w huc_q -h sn79-1 -u 79
```

To run manually without pm2, you can use the following commands from inside the repo directory:

```console
cd taos/im/neurons
python miner.py  --netuid 79 \
--subtensor.chain_endpoint $ENDPOINT \
--wallet.path $WALLET_PATH \
--wallet.name $WALLET_NAME \
--wallet.hotkey $HOTKEY_NAME \
--axon.port $AXON_PORT \
--logging.debug \
--agent.path $AGENT_PATH \
--agent.name $AGENT_NAME \
 --agent.params $AGENT_PARAMS
```

---
