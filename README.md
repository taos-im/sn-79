<div align="center">

# **MVTRX**: Bittensor SN79<!-- omit in toc -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---


**MVTRX** operates as a [Bittensor](https://bittensor.com) subnet at netuid 79 for decentralised market research and AI model training. It comprises three integrated components:

- **τaos**: agent-based simulation of automated trading strategies in intelligent markets, incentivising risk-managed, high-quality market participation
- **GenTRX**: distributed training of a shared order-book generative model, built on top of τaos simulation data (and future exchange data)
- **MVTRX Exchange**: off-chain limit order book exchange for Bittensor alpha tokens at [mvtrx.exchange](https://mvtrx.exchange), running the same C++ matching engine as the simulation. This release provides the tools and documentation to build and test agents against a **localnet** exchange; the mainnet exchange mechanism is a later release (see [Exchange](#exchange))

[![Website](https://img.shields.io/badge/website-black?logo=googlechrome
)](https://mvtrx.fi)
[![Exchange UI](https://img.shields.io/badge/exchange-black?logo=googlechrome
)](https://mvtrx.exchange)
[![Grafana](https://img.shields.io/badge/grafana-white?logo=grafana
)](https://taos.simulate.trading)
[![Simulation Terminal](https://img.shields.io/badge/dashboard-white?logo=grafana
)](https://mvtrx.simulate.trading)
[![Discord](https://img.shields.io/badge/discord-black?logo=discord
)](https://discord.com/channels/799672011265015819/1353733356470276096)
[![τaos Whitepaper](https://img.shields.io/badge/whitepaper-white?logo=proton
)](https://simulate.trading/taos-im-paper)

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
    - [GenTRX](#gentrx-technical)
3. [Requirements](#requirements)
    - [Validator](#requirements-validator)
    - [Miner](#requirements-miner)
4. [Agents](#agents)
5. [Exchange](#exchange)
6. [Install](#install)
    - [Validator](#install-validator)
    - [Miner](#install-miner)
    - [Docker](#install-docker)
7. [Run](#run)
    - [Registration](#run-registration)
    - [Validator](#run-validator)
    - [Miner](#run-miner)
8. [GenTRX Distributed Training](#gentrx)
---

<div style="page-break-after: always;"></div>

## Incentive Mechanism <span id="mechanism"><span>
The incentive mechanism described here covers the **τaos simulation** and **GenTRX training** components, which are the two that carry emissions on mainnet today. The **Exchange** runs as a separate subnet mechanism whose scoring is additional to these; it is not yet enabled on mainnet, and its reward dimensions will be published before it is. See [Exchange](#exchange) for what is available now.

For the τaos component: the mechanism is designed to promote intelligent, risk-managed trading logic to be applied by agents, in order that we are able to produce valid and valuable datasets mimicing the properties of a variety of different real-world asset classes and market conditions. See the [whitepaper](https://simulate.trading/taos-im-paper) for a detailed exploration of the background, goals and scope.

**Two reward pools.** Miner rewards are split across two incentive pools that run in parallel:

- **Trading pool** (~95% of rewards by default): scored on kappa and PnL from simulation trading. All miners registered on the simulation mechanism participate. Exchange trading is scored under its own mechanism with its own weights, not blended into this pool.
- **GenTRX training pool** (~5% by default, set by `--scoring.gentrx.simulation_share` on the validator): scored on gradient quality, assessed each round against held-out order-book data. Scales with active participation; unused training rewards return to the trading pool. Opt-in for both validators and miners; zero impact on trading rewards when not in use.

### Owner Role <span id="mechanism-owner"><span>
The subnet owners are tasked with ensuring fair, equitable and correct operation of the subnet mechanisms (as in all other subnets), while also being responsible for the design, refinement, tuning and publishing of the simulation parameters and logic.  This involves consistent monitoring, testing and development to expand the capabilities of the simulator and determine parameters which result in the most useful possible outputs being generated through the subnet's operation.  The owner must also ensure that the metrics utilized in determining miner rewards are chosen such that miners are incentivized to act fairly and in such a way that outputs are of optimal value in research, trading strategy development, market surveillance and other applications.

For the GenTRX component, the owners also operate the canonical **aggregator** (uid 0): a gradient server that evaluates all miner and sibling-validator proposals each round, applies the best-scoring delta to the shared model, and publishes the new checkpoint on-chain for all participants to download.

### Validator Role <span id="mechanism-validator"><span>
Validators in the subnet are responsible primarily for maintaining the state of the simulation, and rewarding agents (miners) which achieve the best results over all realizations of the simulated market.  They deploy two components:
- The C++ simulator, which handles all the computation necessary to simulate asset markets
- The Python validator, which receives state updates from the simulator, forwards these to miners, submits instructions received in response back to the simulator, and calculates miner scores based on their performance throughout the simulation.

Validators that opt in to **GenTRX** additionally run a gradient server (`GenTRX.src.gradient_server`) that scores miner gradients against held-out data each round and publishes aggregation proposals. They contribute to the training pool scoring alongside the trading pool, and their on-chain weights reflect both components. GenTRX is opt-in - validators that skip it run as pure trading validators with no change to their trading-pool behaviour.

### Miner Role <span id="mechanism-miner"><span>
Miners in the subnet function as trading agents in the distributed simulation; their role is to develop and host trading strategies which maximize their average risk-adjusted performance measures over all simulated market realizations, while also maintaining a sufficient level of trading activity.  There are no strict limitations on what strategies are able to be applied, but the simulation parameters and performance evaluation metrics will be continually reviewed, selected and adjusted with the intention of maximizing the utility of the output data, and promoting the use of intelligent, risk-averse and budget-constrained trading logic.

Miners can also opt in to **GenTRX distributed training** by running an agent that subclasses `GenTRXAgent`. Each round, the agent trains the shared model on its assigned slice of sim data, compresses the gradient, and uploads it to a personal S3 bucket. Validators score these gradients and the best delta updates the shared checkpoint. Miners earn from the training pool in addition to the trading pool, with no trade-off: training runs in a background thread concurrently with live trading.

---
<div style="page-break-after: always;"></div>

## Technical Operation <span id="technical"><span>
The description below covers the τaos simulation and GenTRX training components. The Exchange shares the same matching engine and the same agent interface, and is described in [Exchange](#exchange).

The subnet operates at technical level in the first implementation in quite familiar manner for the Bittensor ecosystem.  Validators construct requests containing the simulation state, which results from a series of computations by the simulator, and publishes these requests to miners at a pre-defined interval.  Miners must respond to validator requests within a reasonable timeframe in order for their instructions to be submitted to the simulation for execution.  Scores are calculated in general as a weighted sum of several risk-adjusted performance metrics; although, at least until others are required, only an intraday Kappa-3 ratio is evaluated.  Miners are also required to maintain a certain level of trading volume in order for their risk-adjusted performance score to be allocated in full - this prevents inactive miners from gaining incentives, and aligns with the objective of the project to encourage active automated trading rather than simple buy & hold or other very low-frequency strategies.

In the current approach, a new simulation configuration is intended to be deployed on approximately weekly basis, with each simulation being executed as an independent run where all miner agents begin with the same initial capital allocation.  Multiple runs of a particular configuration may be executed by validators before a new configuration is published, due to varying rate of progression resulting from differing resources deployed by validators. Miner scores are however calculated using a rolling window which is not cleared at the start of a new simulation, so that performance in previous races does still contribute to the miner's overall weighting.  Deregistrations are handled by resetting the account balance and positions of the agent associated with the UID which was newly registered to the configured starting values.

### Simulator <span id="technical-simulator"><span>
The bulk of the computation involved in running the simulations is handled by a C++ agent-based simulation engine, which is built on top of [MAXE](https://github.com/maxe-team/maxe).  This engine is deployed as a companion application to the validator logic, and handles the orderbook and account state maintenance necessary to simulate the high-frequency level microstructure of intelligent markets.  

The construction of a fully detailed limit order book is a key advantage of utilizing an agent-based model as opposed to other generative techniques: since the simulation operates in the same manner as real markets, being composed of a large number of independent actors submitting instructions to a central matching engine, we not only allow for the highest level of customization and realism in the simulated market, but also reproduce all the finer details of the environment.  This includes the full limit order book with _Level 3_ or _Market-by-order_ data, which records every individual event occurring within the market and is valuable (even necessary) for the development of advanced, high-frequency and data-intensive (e.g. AI) trading strategies, as well as in performing deep analyses of market behaviour for monitoring, surveillance and other regulatory purposes. 

The simulator is designed to maintain any number of orderbooks simultaneously, ensuring statistical significance of the results observed by providing many realizations over which to verify the outcomes.

The simulator additionally includes implementations of the "background agents" which create the basic conditions of the markets in which distributed agents (miners) trade.  It is the parameterization of the background model which these agents comprise that determines the high-level behaviour of the simulated orderbooks.  The model implemented is based on well-established research in the field; for some background and details see [Chiarella et. al. 2007](https://arxiv.org/abs/0711.3581) and [Vuorenmaa & Wang 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2336772).

### Validator <span id="technical-validator"><span>
The validator functions in a sense as a "proxy" which allows the C++ simulator to communicate with participants in the subnet, and handles all the Bittensor-network-related tasks involved in validator operation - authenticating and distributing requests, validating and processing responses, calculating miner scores, setting weights and, ultimately, providing access to the subnet computational resources for external queries.

Validators that opt in to GenTRX additionally run a **gradient server** sidecar process. The gradient server decomposes simulation data into per-miner training assignments, collects uploaded gradients, scores them against held-out order-book data, and publishes the best-scoring delta as the new model checkpoint each round. Trading and training scoring are computed in separate pools and merged at weight-setting time.

### Miner <span id="technical-miner"><span>
Miners receive state updates from validators, and respond with instructions as to how they would like to modify their positions in the simulated orderbook realizations by placing and cancelling orders.  In this first version of the subnet, state updates are published at a parameterized interval throughout the simulation and miners are only able to submit instructions at each state publishing event - a planned future implementation will allow continuous bidirectional communication of state and instructions, in line with real exchange operation.  The state update includes all L3 messages processed since the last update, so that strategies analysing the most detailed information are still possible to apply, being limited only in the time-scale at which the algorithm is able to act.

While awaiting the validator to request and receive responses from miners, the simulation is paused such that no events are processed during this time; in order to reward efficient computation as well as agent performance, the response time of miners is used to determine the "delay" or "latency" with which their instructions will be processed by the simulator.  Longer response times thus imply more events that may take place before execution of their instructions, requiring realistic effects like price slippage to be accounted for.   This incentivizes miners to be both fast and intelligent in their trading strategy, while also carefully managing their risk across all market state realizations.  

Miner agents are otherwise treated in the simulator on the same footing as the background agents; their instructions are processed as if they would be submitted to a real exchange, with orders being traded or opened on the book in accordance with standard matching rules.  Every agent's orders will interact with those of the background agents and miners - this ensures a proper and full accounting of the performance of the miner, including their market impact.  Since agents actions directly affect the market structure, this must also be considered when making trading decisions.

Miners that opt in to **GenTRX** extend their agent to also train the shared model each round. Training runs in a background thread concurrently with live trading. The agent downloads its assigned data slice, runs a forward/backward pass, compresses the gradient, and uploads it to a personal S3 bucket - all within the round window. A validator-run gradient server fetches and scores the upload; accepted gradients update the shared checkpoint, which the miner downloads at the start of the next round.

### GenTRX Distributed Training <span id="gentrx-technical"><span>

GenTRX adds a second incentive layer on top of τaos trading. Two processes work together each round:

**Validator**: every tick, the validator pushes simulation state to the gradient server (`POST /gentrx/state`). Once enough data has accumulated, it opens a new training round (`POST /gentrx/round`), creates per-miner assignments (book slice + time window), and delivers them to miners via dendrite. It then polls the gradient server for scores (`GET /gentrx/scores`) and merges them into its weight vector at weight-setting time alongside the trading-pool scores.

**Gradient server** (`run_gradients.sh` or `GenTRX.src.gradient_server`): a standalone sidecar process that:
- Replays incoming state ticks through an internal matching engine, accumulates order-book rows, and flushes parquet training files to the validator's S3 bucket
- Reads uploaded miner gradients from per-miner S3 buckets (discovered via chain commitments), double-scores each against both the miner's own assigned data and held-out validation books, applies an overfit penalty when own-data loss runs ahead of held-out loss, and rank-normalises with per-UID EMA
- Locally aggregates accepted gradients into a delta and publishes it to S3 as a proposal; the canonical aggregator (uid 0) evaluates all validator proposals and publishes the winning checkpoint
- On simulation end (`ESE` marker), automatically wipes the stale parquet data from S3 so the next simulation starts clean, without operator intervention

The gradient server can run on the same host as the validator (loopback, no API key needed) or on a dedicated GPU machine (`--bind 0.0.0.0`, shared API key). `run_gradients.sh` handles dependency installation, CUDA detection, pm2 lifecycle, and interactive credential setup. `run_validator.sh -G` orchestrates the full stack in one command.

**Two-pool scoring**: miner weights are a blend of a trading-pool score (kappa + PnL, ~95%) and a training-pool score (gradient quality, ~5%). Each pool is computed independently and merged at weight-setting time. Validators not participating in GenTRX contribute 100% to the trading pool; the training allocation scales with active participation and returns to trading when unused. The pool split is controlled by `--scoring.gentrx.simulation_share` on the validator.

---
<div style="page-break-after: always;"></div>

## Requirements <span id="requirements"><span>
Requirements are subject to change as the subnet matures and evolves; this section describes the recommended resources to be available for the initial simulation conditions.  We currently manage 40 orderbooks in a simulation, each having around 1000 background agents, while the aim in the near- to mid-term is to reach 1,000+ simulated orderbooks in order to achieve a meaningful level of statistical significance in the evaluation of results.

### Validator <span id="requirements-validator"><span>
Validators need to host the C++ simulator as well as the Python validator.  In the early days of the subnet, the number of orderbooks simulated as well as the count and type of background agents will be reduced so as to limit the requirements before the subnet matures and sufficient emissions are gained to justify the expense of hosting more powerful machinery.  Basic requirements:

- 32GB RAM
- 16 CORE CPU
- Ubuntu >= 22.04
- g++ 14.

We hope to increase both major parameters significantly so that validators may wish to prepare a larger machine for easier expansion.  It should be noted however that increasing the CPU resources available will result in a faster progression of simulations due to multi-threaded processing of the orderbook realizations.  This should not inherently be a problem, but may cause divergences in scoring if there is a major discrepancy in resources with the other validators in the subnet.  We plan to communicate the setup employed by our validator whenever changes are made, and will enable to configure the resources allocated for simulation processing if necessary.

**GenTRX (optional):** Validators that opt in to distributed training additionally require a GPU (NVIDIA 8-16 GB VRAM) for the gradient server process. See [`doc/gentrx/validator_setup.md`](doc/gentrx/validator_setup.md).

### Miner <span id="requirements-miner"><span>
There are no set requirements for miners except that the basic Bittensor package and subnet miner tools occupy ~1GB of RAM per miner instance; resources needed will depend on the complexity and efficiency of the specific strategy implementation.

**GenTRX (optional):** Miners that participate in distributed model training additionally benefit from a GPU (NVIDIA 6-8 GB VRAM minimum). CPU-only training is supported but may miss round deadlines. See [`doc/gentrx/miner_setup.md`](doc/gentrx/miner_setup.md).

---

## Agents <span id="agents"><span>
In order to separate the basic network logic from the actual trading logic and allow to easily switch between different strategies, miners in this subnet define a separate class containing the agent logic which is referenced in the configuration of the miner and loaded for handling of simulation state updates.  Some simple example agents are provided in the `agents` directory of this repository, and are copied to a directory `~/.taos/agents` if using the miner install script to prepare your environment.  The objective in agent development is to produce logic which maximizes performance over all realizations in terms of the evaluation metrics applied by the validators.  Currently assessment is based on an intraday Kappa-3 ratio in conjunction with a requirement to maintain a certain level of cumulative round-trip volume; this will be continuously monitored and reviewed, and other relevant risk-adjusted performance measures incorporated if a need is observed.

Only some basic agents are immediately included as examples, designed to illustrate the fundamentals of reading the state updates and creating instructions.  We expect miners to develop their own custom logic in order to compete in the subnet, but plan to release additional examples, tools and templates to facilitate implementation of certain common classes of trading strategies.  An overview of the information needed to begin developing strategies is provided [here](agents/README.md).  It is also possible to test agents offline against the background model on your local machine by following [these instructions](agents/proxy/README.md).

Agents serve both the simulation and the exchange from one class. Subclass `FinanceAgent` (or `GenTRXAgent`) and implement `respond(state)`; it is called in both modes, and you can override `respond_exchange` only if you want the exchange to behave differently. If you already run an agent, [`agents/MIGRATION_0.6.0.md`](agents/MIGRATION_0.6.0.md) covers what changes in this release (nothing required) and what opting in to the exchange will involve.

Miners participating in GenTRX distributed training define their agent by subclassing `GenTRXAgent`, which adds data collection, gradient compression, and S3 upload alongside the standard trading loop.  Example GenTRX-capable agents (`HybridTrainingAgent`, `RandomMakerAgent`, `RandomTakerAgent`) are included in the `agents` directory as starting points.  A guide to overriding the training hooks (`collect_row`, `select_training_files`, `train`) is provided in [`doc/gentrx/integration.md`](doc/gentrx/integration.md).

---
<div style="page-break-after: always;"></div>

## Exchange <span id="exchange"><span>

**MVTRX Exchange** is a live off-chain limit order book exchange for Bittensor alpha tokens, running at
[mvtrx.exchange](https://mvtrx.exchange). It uses the same C++ matching engine as the τaos simulation, so
an agent written against the simulation is written against the exchange too.

**What this release gives you.** 0.6.0 ships the agent interface, the wire protocol and the documentation
needed to **develop and test an exchange agent against a localnet exchange**. That is the supported scope
today. Note that running the exchange-mode engine itself requires components which are not included in
this repository.

The SN79 exchange **mechanism** on mainnet, under which miner exchange activity would be scored and
weighted on chain, is **not part of this release**: netuid 79 currently declares a single mechanism, and
there is nothing to register on. Mainnet exchange comes in a later release, with registration and scoring
details announced before it does.

**Your existing simulation agent is unaffected.** Nothing in 0.6.0 requires a change to an agent that
trades the simulation, and no exchange payload can reach a mainnet miner. See
[`agents/MIGRATION_0.6.0.md`](agents/MIGRATION_0.6.0.md).

**What differs from the simulation.** The interface is shared, but the venue is real, and three differences
cost people orders:

- **Prices land on a grid, and the grid truncates.** A price is snapped to a tick, downward, not rounded.
  A buy at a price you computed can therefore rest a tick below where you expected.
- **Leverage is refused.** The exchange runs with `maxLeverage=0`, so any order requesting leverage is
  rejected at placement rather than reduced. Gate leverage on the mode if your agent uses it.
- **A sell draws on one delegate.** Your sellable alpha is the largest individual delegate's stake, not
  the sum across your delegations and not anything specific to the querying validator, so free balance is
  not a sell limit.

Naming follows the venue: the simulation spells the pair **BASE** and **QUOTE**, the exchange spells it
**ALPHA** and **TAO**, and both spellings are accepted in both modes (`OrderCurrency.ALPHA`/`TAO` are
aliases of `BASE`/`QUOTE`).

| Resource | Link |
|---|---|
| Writing an exchange agent | [agents/README.md](agents/README.md), section "Exchange venue behaviours that catch people out" |
| Migrating an existing agent | [agents/MIGRATION_0.6.0.md](agents/MIGRATION_0.6.0.md) |
| Full agent guide | [agents/README.md](agents/README.md) |
| Exchange UI | [mvtrx.exchange](https://mvtrx.exchange) |
| Market data and terminal | [mvtrx.fi](https://mvtrx.fi) |

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
A containerised deployment is **not currently provided**; the native install
below (`./install_validator.sh` / `./install_miner.sh`) is the supported path.
If a Docker-based deploy would help your setup, please open an issue; it is on
the roadmap but not yet shipped, so do not rely on it being available today.

### Validator <span id="install-validator"><span>
To prepare your environment for running a validator (including the C++ simulator), simply run the included script **as root user** (if unable to execute as root, please reach out to us for assistance, but may need to await Docker-based deploy in this case):
```console
./install_validator.sh
```
If prompted to restart any services, just hit "Enter" to proceed.  You may need to re-open your shell session after installation completes before newly installed applications can be used.

This will install the following tools:
- **prometheus-node-exporter** : To enable resource usage monitoring via Grafana or similar
- **Node.js + pm2** : For process management (Node.js installed via NodeSource)
- **tmux** : For multiplexing to allow simultaneous viewing of simulator and validator logs
- **pyenv** : For managing of Python version installations
- **Python 3.10.9** : This version of Python has been used in all testing; later versions will likely still work but have not been tested
- **τaos** : The Python component of the apparatus, containing the base validator and miner logic
- **vcpkg** : For C++ simulator dependency management
- **g++-14** : If on Ubuntu 22.04 rather than the latest 24.04, g++ 14.1 must be installed and used for compilation of the simulator (g++ 14.2 is already included in Ubuntu 24.04 install)
- **cmake-3.29.7** : Required to build and run the simulator
- **τaos.im simulator** : The C++/Pybind simulator application.

You can of course modify the install script if you wish to make changes to the installation, or use this as a guide to execute the steps by hand if you prefer.  Note that the installation process takes quite a long time, often 2+ hours on Ubuntu 22.04, due to the need to compile specific cmake and g++ versions from source, so recommended to run in a multiplexer (e.g. screen/tmux) to prevent interruptions.

**GenTRX (optional):** The install script does not include GPU/CUDA driver setup. If you intend to run the gradient server, install NVIDIA drivers and CUDA beforehand (a standard step on most GPU cloud instances). GenTRX training dependencies (`boto3`, `transformers`, `polars`, etc.) are an optional extra: install them with `pip install -e ".[gentrx]"` before running the gradient server. The gradient server process is started automatically by `run_validator.sh -G`. Full setup guide: [`doc/gentrx/validator_setup.md`](doc/gentrx/validator_setup.md).

<div style="page-break-after: always;"></div>

### Miner <span id="install-miner"><span>
To prepare your environment for running a miner, simply execute the included script:
```console
./install_miner.sh
```
You may need to re-open your shell session after installation completes before newly installed applications can be used.
This will install the following tools:
- **Node.js + pm2** : For process management (Node.js installed via NodeSource)
- **tmux** : For multiplexing to allow convenient monitoring of miner logs
- **pyenv** : For managing of Python version installations
- **Python 3.10.9** : Fully tested version, others may work but have not been tested
- **τaos** : The Python component of the apparatus, containing the base validator and miner logic.

**GenTRX (optional):** GenTRX training dependencies are an optional extra; run the install script with `INSTALL_GENTRX=1` to have it install the `[gentrx]` extra, or install manually with `pip install -e ".[gentrx]"`. If you intend to train with GPU, install NVIDIA drivers and CUDA beforehand. Full setup guide: [`doc/gentrx/miner_setup.md`](doc/gentrx/miner_setup.md).

---
<div style="page-break-after: always;"></div>

## Run <span id="run"><span>
We include simple shell scripts to facilitate running of a validator or miner; it is also possible to run the applications directly yourself in the case of miner processes, though validators are strongly recommended to use the provided script in order to ensure that all components are updated properly.
If you wish to use the run scripts, first enter the directory where you have cloned this repo.

### Registration <span id="run-registration"><span>
Before running, you must have a Bittensor wallet (coldkey + hotkey) and register
that hotkey on the subnet. If you do not yet have a wallet:

```console
# Create a coldkey (keep this secured/offline; it controls funds) and a hotkey.
btcli wallet new_coldkey --wallet.name <coldkey>
btcli wallet new_hotkey  --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

Then register the hotkey on the subnet (netuid **79** on mainnet/finney; use the
testnet netuid with `--network test`). Registration burns a small amount of TAO:

```console
btcli subnet register --netuid 79 --wallet.name <coldkey> --wallet.hotkey <hotkey>
# testnet: btcli subnet register --netuid <testnet-netuid> --network test \
#          --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

Operational notes:
- The **coldkey** can stay on a secure/offline machine; only the **hotkey** needs
  to be present on the validator/miner host that serves the axon.
- Confirm registration with
  `btcli subnet metagraph --netuid 79` (your hotkey should appear with a UID).
- Wallets default to `~/.bittensor/wallets/`; pass `-p <path>` to the run scripts
  if yours live elsewhere.

### Which network to point at

- **Mainnet** (`netuid 79`, `wss://entrypoint-finney.opentensor.ai:443`) is the live subnet.
- **Public localnet** (`wss://localnet.mvtrx.exchange:443`) is open for testing against the
  exchange. Point the run scripts at it with `-e wss://localnet.mvtrx.exchange:443`.
- The **Bittensor test network** (`netuid 366`) carries no exchange, so it is not useful for
  exercising trading. Use the public localnet for that.

### Validator <span id="run-validator"><span>
To run a validator, you can use the provided `run_validator.sh` which accepts the following arguments:
- `-e` : The subtensor endpoint to which you will connect (default=`wss://entrypoint-finney.opentensor.ai:443`)
- `-E` : Path to an env file to source before the other flags are applied; values in the file override the in-script defaults, and CLI flags in turn override the file (default=*(none)*)
- `-p` : The path where your wallets are stored (default=`~/.bittensor/wallets/`)
- `-w` : The name of your coldkey (default=`taos`)
- `-h` : The name of your hotkey (default=`validator`)
- `-u` : The netuid on which to validate; if not passed, it is derived from the target network (default=`79` for mainnet, `366` for testnet)
- `-l` : Logging level for the validator, must be one of `error`, `warning`, `info`, `debug`, `trace` (default=`info`)
- `-d` : Pagerduty integration key; if you have a Pagerduty subscription, this allows to trigger alerts for critical failure scenarios (default=`""`)
- `-o` : Port on which Prometheus metrics will be published.  If you use a different port than the default, please let us know so that your data will still appear at [taos.simulate.trading](https://taos.simulate.trading/?orgId=1) (default=`9001`). Exchange market data is served via the MVTRX Data Service and UI at [mvtrx.fi](https://mvtrx.fi).
- `-P` : Port on which the validator's listener is served (this is where the C++ engine delivers its messages).  If not passed, defaults to `8000` for the simulation engine and `8001` for the exchange engine, so that the two can run side by side on one host.
- `-t` : Timeout for miner queries; this allows validators to tune the time allowed for miners to respond to account for differences in server geolocation or networking capability (default=`3.0`).
- `-s` : Flag to indicate that the simulator should not be restarted when performing the update; this allows to easily execute updates which only affect the Python validator operation (default=`0`; append `-s 1` to command to preserve running simulator during update).
- `-x` : Flag to indicate if wanting to launch tmux session for monitoring (default=`1`; append `-x 0` to command to disable tmux session creation).
- `-c` : If you wish to resume a previous simulation rather than starting a new one, you can set this argument to the location of the output directory of the simulation, or to `latest` to resume the most recently started simulation. (default=`0` => start a new simulation)
- `-M` : Mechanism id passed to the validator as `--neuron.mechid`; this selects which subnet mechanism the validator scores and sets weights on.  If not passed, defaults to `0` when running the simulation engine and `1` when running the exchange engine.
- `-G <mode>` : Enable GenTRX distributed training in the given mode. Disabled when the flag is not passed; regular validators pass `-G sibling`, which auto-starts a local gradient server that scores miner gradients alongside the τaos validator (a bare `-G` with no mode is also accepted and means `sibling`). `-G aggregator` is reserved for the subnet owner who operates the canonical uid-0 aggregator that publishes checkpoints on-chain.
- `-Q` : Gradient server URL; skip auto-start and use this address (e.g. for a remote GPU machine). Default *(auto)*.

The script will:
1. Pull and install the latest changes from the MVTRX repository
2. Build the latest version of the simulator
3. Launch a validator under pm2 management as `validator`
4. Start the simulator under pm2 management as `simulator`
5. Save the pm2 process list and configure for resurrection on restart
6. Open a `tmux` session with logs for validator and simulator (and gradient server logs if it is managed locally)

**Standard run:**
```bash
./run_validator.sh -w taos -h validator -u 79
```

**With GenTRX distributed training** (see [§GenTRX](#gentrx)):
```bash
./run_validator.sh -G sibling -w taos -h validator -u 79
```
On first run with `-G`: prompts for your S3 bucket credentials, detects a local GPU and starts the gradient server automatically (or guides you through connecting a remote GPU machine), and saves all configuration to `.env`. Subsequent runs restore saved config with no flags needed. Pass `-Q <url>` to point at an already-running gradient server without auto-starting one. Full setup guide: [`doc/gentrx/validator_setup.md`](doc/gentrx/validator_setup.md).

**S3 backend.** Each GenTRX validator needs its own writable S3 bucket for checkpoints, training data, and proposals; read-only credentials for the bucket are committed on-chain so miners and the aggregator can discover it. Supported providers are **Cloudflare R2**, **Storj**, and **Hippius S3** (auto-detected from the committed `account_id`; see [`GenTRX/src/chain.py`](GenTRX/src/chain.py)). The `-G` first-run wizard sets the required `GENTRX_VALIDATOR_S3_*` and `GENTRX_API_KEY` env vars into `.env` for you; for non-interactive setup, copy [`.env.example`](.env.example) to `.env` and fill in the values directly.

**Running the gradient server on a separate GPU machine** (or locally but managed independently):

Use the standalone `run_gradients.sh` script on the GPU host, then connect the validator to it with `-Q`:

```bash
# On the GPU machine: first run prompts for S3 credentials and API key,
# subsequent runs skip all prompts and restart the server.
./run_gradients.sh -G 

# On the validator host
./run_validator.sh -Q http://<gpu-host>:8100/gentrx -w taos -h validator -u 79
```

`run_gradients.sh` handles dependency installation (`git pull` + `pip install`), CUDA detection, pm2 process management, and opens a `gentrx` tmux session (htop / GPU monitor / logs). Key flags:

| Flag | Description | Default |
|------|-------------|---------|
| `-G`/`-m` | Run mode (sibling: score + propose) | `sibling` |
| `-b` | Bind address (`0.0.0.0` for remote access) | `127.0.0.1` |
| `-p` | Listen port | `8100` |
| `-e` | Subtensor endpoint | finney |
| `-u` | Netuid | `79` |
| `-k` | API key (required when `-b 0.0.0.0`) | *(wizard)* |

All credentials and configuration are saved to `.env` on first run so that `run_validator.sh` picks them up automatically on the same machine.

**We recommend running as root to avoid any permissions issues.**

To resume the validator from a simulation checkpoint, pass `-c latest` (most recent) or a log directory path:
```bash
./run_validator.sh -c latest -w taos -h validator -u 79
```

<div style="page-break-after: always;"></div>

### Miner <span id="run-miner"><span>
To run a miner, use the provided `run_miner.sh`:

| Flag | Description | Default |
|------|-------------|---------|
| `-e` | Subtensor endpoint | `wss://entrypoint-finney.opentensor.ai:443` |
| `-p` | Wallet directory | `~/.bittensor/wallets/` |
| `-w` | Coldkey name | `taos` |
| `-h` | Hotkey name | `miner` |
| `-u` | Netuid | `79` |
| `-a` | Axon port | `8091` |
| `-g` | Agent directory | `~/.taos/agents` |
| `-n` | Agent class name | `SimpleRegressorAgent` |
| `-m` | Agent params (`param=val ...`) | *(SimpleRegressorAgent defaults)* |
| `-l` | Log level (`error`/`warning`/`info`/`debug`/`trace`) | `info` |
| `-G` | Enable GenTRX distributed training | *(disabled)* |
| `-t` | GenTRX training params to override (`gtx_key=val ...`) | *(defaults)* |

The script will:
1. Pull and install the latest changes from the MVTRX repository
2. Launch a miner under pm2 management as `miner`
3. Save the pm2 process list and configure for resurrection on restart
4. Display the logs of the running miner (and training log if GenTRX enabled)

**Standard run:**
```bash
./run_miner.sh -w taos -h miner -u 79 -a 8091
```

**With GenTRX distributed training** (see [§GenTRX](#gentrx)):
```bash
./run_miner.sh -G -w taos -h miner -u 79 -a 8091
```
On first run with `-G`: prompts for your S3 bucket credentials, commits the read key on-chain, and saves all configuration to `.env`. It prints a complete reusable command at the end - useful as a template when running multiple UIDs (adjust `-w`/`-h`/`-a`). Subsequent runs restore saved config with no flags needed. To override training params without changing saved config: `./run_miner.sh -t "gtx_train_steps=100 gtx_train_batch_size=8"`. The default agent is `HybridTrainingAgent` - a template, not a finished strategy; tune before deploying seriously. Full setup guide: [`doc/gentrx/miner_setup.md`](doc/gentrx/miner_setup.md).

**S3 backend.** Each miner needs their own writable S3 bucket for gradient uploads; read-only credentials for the bucket are committed on-chain at miner startup so the gradient server can fetch the uploads. Supported providers are **Cloudflare R2**, **Storj**, and **Hippius S3** (auto-detected from the committed `account_id`). The `-G` first-run wizard sets the required `GENTRX_AGENT_S3_*` env vars into `.env` for you; for non-interactive setup, copy [`.env.example`](.env.example) to `.env` and fill in the values directly.

To run manually without pm2:
```bash
cd taos/im/neurons
python miner.py --netuid 79 \
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

## GenTRX Distributed Training <span id="gentrx"><span>

GenTRX trains a shared order-book generative model (~12M-parameter transformer) on τaos simulation data. Miners compute gradients locally each round and upload compressed deltas; validators score them against held-out data; the canonical checkpoint is published on-chain and available to all participants.

**Rewards:** 5% of miner rewards go to the training pool by default (set by `--scoring.gentrx.simulation_share` on the validator). The remaining 95% stays in the trading pool. The training allocation scales with active participation and returns to trading if no miners opt in. GenTRX is opt-in for both validators and miners - the run scripts for each role handle setup automatically (see [§Validator](#run-validator) and [§Miner](#run-miner) above).

| Resource | Link |
|---|---|
| Validator setup | [doc/gentrx/validator_setup.md](doc/gentrx/validator_setup.md) |
| Miner setup | [doc/gentrx/miner_setup.md](doc/gentrx/miner_setup.md) |
| Architecture & data flow | [doc/gentrx/overview.md](doc/gentrx/overview.md) |
| Integration (custom agents) | [doc/gentrx/integration.md](doc/gentrx/integration.md) |
| Local testing | [doc/gentrx/testing.md](doc/gentrx/testing.md) |

---

## Third-Party Licenses

This project includes components that are distributed under the MIT License:

| Project | File | Link |
|---------|------|------|
| Microsoft MarS | `LICENSES/mars_license.txt` | https://github.com/microsoft/MarS |
| Templar | `LICENSES/templar_license.txt` | https://github.com/tplr-ai/templar |

The primary project license is MIT (see `LICENSES/MIT.txt`). All third-party components are licensed under MIT as well, and their full license texts are provided in the `LICENSES/` directory.

## Acknowledgements

**[Microsoft MarS](https://github.com/microsoft/MarS):** The GenTRX model
architecture is inspired by MarS, which explored training transformers on
order book event sequences to model market microstructure. The GenTRX
implementation is independent original work. The order-book matching engine
in `GenTRX/src/orderbook.py` is adapted directly from MarS source code
(MIT license, see `LICENSES/mars_license.txt`).

**[Templar](https://github.com/tplr-ai/templar):** The distributed training
design references Templar's subnet architecture (MIT license, see
`LICENSES/templar_license.txt`).