# Miner Agent Testing

This directory contains tools allowing for the offline testing of miner trading agents against a local instance of the simulation.  Testing is achieved by the use of a "Proxy", which is a descendant of the main taos.im validator class with the Bittensor network functions and requirements removed so as to be run without any other preparations (e.g. localnet subtensor configuration) necessary.  The proxy receives state messages from the simulator and forwards them to agents running on the local machine, parses the instructions returned by agents and submits them to the simulation.  This enables miners to check, debug and evaluate the function and behaviour of their strategies in the base simulated market before deploying to mainnet where they will interact with a simulation having the same background configuration but including of course also the other miner agents.  Agents developed and tested using these tools can be immediately deployed in association with miners on testnet or mainnet.

> **GenTRX testing**: `agents/proxy/run` defaults to trading-only (proxy + agents + simulator). Pass `--train` to additionally wire up MinIO, the gradient server, and GenTRX training on top of `proxy.py`.

## Installation

In order to test agent implementations offline, the simulator must first be installed.  Run the `./install_simulator.sh` script **as root user** in the top directory of this repository to prepare all necessary build tools and dependencies for running the C++ simulator.  Note that this process takes a long time (a couple of hours of Ubuntu 22.04) as cmake and g++ are compiled from source.

You must also install the `taos` Python package by running the below command in the repository root:
```
pip install -e .
```

## Config
The distributed agent/proxy setup is configured by means of a JSON configuration file in this directory.  The default config loaded when running `./agents/proxy/run` without args is `config.json`.  Two canonical templates sit alongside it: `config.example.json` (trading only) and `config.train.example.json` (with GenTRX training); pass `--config <path>` to override the default.  The structure is:
- `proxy` : Proxy configuration options.
  - `port` : Port number on which proxy will listen for state updates from simulator (must match `Simulation.port` in simulation XML config). \[default=`8000`\]
  - `simulation_xml` : Path to the XML config used to launch the simulation.  Resolved relative to `agents/proxy/` when it begins with `./` or `..`, otherwise relative to the repo root. \[default=`"../../simulate/trading/run/config/simulation_0.xml"`\]
  - `timeout` : The number of seconds that proxy will await response from agent before moving on. \[default=`5`\]
  - `gradient_server_url` : Gradient server base URL for GenTRX training.  Required only when running with `--train`; omit (or leave empty) for trading-only runs.  Typically `"http://127.0.0.1:8100/gentrx"` for single-host setups.
- `agents` : Configuration for distributed agents.
  - `start_port` : Base port where first distributed agent will be hosted.  When multiple agents are configured, the port for each will be increased by 1. \[default=`8888`\]
  - `path` : Path where agent definition files are found. \[default=`..`\]
  - `<agent_file_name>` : Other entries in the `agents` config define the agents which are to be hosted; use the file name of the agent in the `agents.path` directory (without `.py` extension) as the key to specify a group of this class of agent.  This field contains a list of definitions for the variations/copies of this agent to be run in the experiment:
    - `params` : For each item in the list, define the strategy params to be used.
    - `count` : This number of the class of agents will be launched with the associated params.
- `taosim` : Simulator process options.
  - `bin` : Name (resolved on `$PATH`) or absolute path of the `taosim` binary launched by the runner. \[default=`"taosim"`\]
  - `delay` : Seconds the launcher waits between starting the proxy and starting the simulator, so the proxy's HTTP listener is ready before the simulator emits its first state update. \[default=`5`\]
- `training` : GenTRX training options (only consumed when `./agents/proxy/run --train` is used; ignored otherwise).
  - `n_agents` : Number of agent buckets to provision on the local MinIO (one bucket per agent slot).
  - `gradient_server` : Standalone gradient-server process configuration.
    - `port` : Port the gradient server binds to. \[default=`8100`, must match `proxy.gradient_server_url`\]
    - `checkpoint` : Path to the bootstrap checkpoint (relative to repo root). \[default=`"checkpoints/GenTRX/best.pt"`\]
    - `val_data` : Held-out validation parquet directory. \[default=`"data/proxy_test/server"`\]
    - `output` : Path the gradient server writes the latest checkpoint to. \[default=`"checkpoints/GenTRX/latest.pt"`\]
    - `interval` : Seconds between aggregation rounds. \[default=`30`\]
    - `min_score` : Reject gradients scoring below this threshold. \[default=`-0.1`\]
    - `mode` : Engine mode the gradient server is associated with (currently `"simulation"`).
  - `minio` : Local MinIO bring-up.
    - `port` : S3 API port. \[default=`9000`\]
    - `console` : Web console port. \[default=`9091`\]

## Proxy

The file `proxy.py` contains an implementation of a handler for processing messages published by the simulator via a `DistributedProxyAgent`.  In order to instruct the simulator to publish messages, you must include a node specifying to include the `<DistributedProxyAgent/>` in the simulation XML config under `<Simulation><Agents>`, and 1the `port` field in the `Simulation` node must match the `proxy.port` set in `config.json`.  If using the current active `simulation_0.xml` this is already present:
```
<Simulation ... host="localhost" port="{proxy.port}" bookStateEndpoint="/orderbook" generalMsgEndpoint="/account">
	<Agents>
		...
		<DistributedProxyAgent/>
	</Agents>
</Simulation>
```
This agent publishes the full state of the simulation to the configured port on localhost at an interval defined in simulation time via the `Simulation.step` field in the XML.  The Python proxy receives messages published from the simulator by this agent, parses them to a `MarketSimulationStateUpdate` synapse format, and forwards to the configured list of (locally hosted) distributed trading agents.  The proxy then awaits responses from the distributed agents, and when received will validate, parse to the correct format and return the instructions to the simulator for processing. 

For the full local test stack (proxy + agents + simulator and, with `--train`, MinIO + gradient server), use the orchestrated runner from the repo root: `./agents/proxy/run [--train]`.  To launch only the proxy FastAPI server directly (useful for debugging the proxy in isolation, without the simulator or agents), run from this directory:
```shell
python proxy.py --config <config_file='config.json'>
```

## Agents

Some example agent implementations are present in the directory above this one; all trading agents must inherit the `FinanceAgent` class (which serves both the simulation and the exchange; `FinanceSimulationAgent` is a pre-0.6.0 alias of its base and has no exchange path), and override two key functions:
- `initalize(self)` : Executed at the start of operation, used to construct and initialize any parameters or structures used throughout the agent operation.
- `respond(self, state)` : This function is called whenever a new `MarketSimulationStateUpdate` is received from the proxy, and should contain the core trading logic.

Instructions for order placement/cancellation are submitted by attachment to a response object; the `respond` function should always initialize an empty response as `response = self.make_response()`, with instructions then added in the logic via the convenience methods defined on it. `make_response()` returns a mode-aware response that finalizes to the type the current mechanism requires, so the same code serves both the simulation and the exchange. Constructing `FinanceAgentResponse(agent_id=self.uid)` directly still works but is simulation-only:
- `response.market_order(self, book_id : int, direction : OrderDirection, quantity : float, delay : int = 0, clientOrderId : int | None = None, stp : STP = STP.CANCEL_OLDEST, currency : OrderCurrency = OrderCurrency.BASE)` : Place a market order on the specified `book_id` for `quantity`. The direction of the order as buy or sell is specified via the `OrderDirection` enum as either `OrderDirection.BUY`  or `OrderDirection.SELL`.  Order will be submitted to book after `delay` simulation timesteps, using the `clientOrderId` and self-trade prevention rule passed.  By specifying `currency=OrderCurrency.QUOTE`, the `quantity` will be interpreted as the desired QUOTE value that should be executed in fulfilment of the order.
- `response.limit_order(self, book_id : int, direction : OrderDirection, quantity : float, price : float, delay : int = 0, clientOrderId : int | None = None, stp : STP = STP.CANCEL_OLDEST, postOnly : bool = False, timeInForce : TimeInForce = TimeInForce.GTC, expiryPeriod : int | None = None)` : Place a limit order on the specified `book_id` for `quantity`@`price`.  Order will be submitted to book after `delay` simulation timesteps, using the client id passed.  Advanced order options can also be supplied, including the self-trade prevention rule, post-only enforcement, time-in-force options (GTC, GTT, IOC or FOK), and an expiry period for use with `GTT` option.
- `response.cancel_order(self, book_id : int, order_id : int, quantity : float | None = None, delay : int = 0)` : Cancel the order with ID `order_id` on the specified `book_id` after the given simulation delay.
- `response.cancel_orders(self, book_id : int, order_ids : List[int], delay : int = 0)` : Cancel a list of orders by ID on the specified `book_id` after the given simulation delay.
The `respond` function must always return a `FinanceAgentResponse` class.

The recent events, open orders and account balances are updated when a new state is received via the `update` method on the base agent class.  These can be accessed via:
- `self.accounts` (`dict[int, Account]`) : A dictionary mapping `book_id` to the `Account` class representing the agent's balances on that book.
- `self.events` (`dict[int, FinanceEvent]`) : A dictionary mapping `book_id` to a list of `FinanceEvent` subclasses representing the events occurring for that agent since the last update (order placement/trade/cancellation).
- `self.simulation_config` (`MarketSimulationConfig`) : A class containing the parameters of the simulation (e.g. decimal precision of quantities).
- `self.history` : Contains the last 10 `MarketSimulationStateUpdate` updates received by the agent.

Agents should be created as a new .py file in the `agents` directory.  Agents take 3 command line parameters when launched:
- `--port` : Port number on which to host the listener for state updates from the proxy
- `--agent_id` : Unique integer identifier for the agent; this will be the ID of the agent also in the simulator
- `--params` : Custom strategy parameters, passed in format `<name1>=<val1> <name2>=<val2> ...`

To launch an agent directly, run the agent script from within the `agents` directory:
```shell
python <agent_file_name>.py --port <agent_port> --agent_id <agent_id> --params <parameters>
```
Agents developed in this way are immediately ready to be used in association with miners on testnet or mainnet; see the main repository readme for details on how to configure and run a miner.

## Testing

For development and testing of a new agent, first prepare the configuration for single agent testing:
```
{
	"proxy" : {
		"port" : 8000,
        "simulation_xml" : "../../run/config/simulation_0.xml"
	},
	"agents" : {
		"path" : "..",
		"start_port" : 8888,
		"MyAgent" : [
			{
				"params" : {"xx" : 0.1, "yy" : 1.0, "zz" : 200},
				"count" : 1
			}
		]
	}
}
```
This specifies that the proxy will listen on port `8000` for updates from the simulator started with config at `../../run/config/simulation_0.xml`, and forward the state updates to a `MyAgent` instance listening on port `8888`.

To test, follow the below procedure:
1. Launch the proxy `python proxy.py --config <config_file>`
2. Launch the agent `python MyAgent.py --port 8888 --agent_id 0 --params xx=0.1 yy=1.0 zz=200` (or use debugger to examine logic more closely)
3. Start the simulation (config file passed should match that specified to be used in `config.json`; recommended of course to test against the latest live simulation parameterization):
```cd /path/to/sn-79/simulate/trading/run 
../build/src/cpp/taosim -f config/simulation_0.xml
```

You should observe logs from the proxy indicating start of the simulation, receipt of state updates and responses from agents.   The agent logs received data, events and the instructions submitted.  Note that the simulator only starts publishing state updates to the proxy after the grace period (`Simulation.Agents.MultiBookExchangeAgent.gracePeriod`) has elapsed, so will not see state updates arriving immediately on simulation start.

## Automated launcher

The `run` script wires up the proxy, agents, simulator, and (in `--train` mode) MinIO + gradient server, then arranges them in a tmux session.

```bash
# Trading-only: proxy + simulator + agents, no S3 or gradient server
./agents/proxy/run

# Training mode: adds MinIO, gradient server, and GenTRX training
./agents/proxy/run --train

# Headless (no tmux), useful for CI or single-terminal debugging
./agents/proxy/run --no-tmux
./agents/proxy/run --train --no-tmux

# Override the config file or simulation XML
./agents/proxy/run --config /path/to/custom.json
./agents/proxy/run --sim-xml /path/to/custom_simulation.xml
```

The config defaults to `agents/proxy/config.json`. Reference examples are alongside it:

- `agents/proxy/config.example.json` : minimal trading-only template
- `agents/proxy/config.train.example.json` : full training-mode template

Tmux layout (training mode):

| Window | Panes |
|---|---|
| `minio` | MinIO logs |
| `proxy-sim` | proxy.py (left) \| taosim (right) |
| `grad-agents` | gradient server (top) \| one pane per agent (below) |

Tmux layout (trading mode): `proxy-sim` + `agents`.

Stop: `tmux kill-session -t proxy_test`

## Deployment

To deploy configurations with many distributed agents, a simple `launcher.py` script is included which automates the process.  Once the configuration has been appropriately defined with all agents and variations specified, to deploy the experiment simply run:
```shell
python launcher.py --config <config_file='config.json'>
```
The script launches the proxy and all agents as configured in the JSON configuration file, and prints the outputs to console.  All processes can be stopped by `Ctrl+C`.