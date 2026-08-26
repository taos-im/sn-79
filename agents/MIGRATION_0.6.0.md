# Migrating an agent to 0.6.0

**If you trade the simulation only, there is nothing to do.** 0.6.0 is backwards compatible for existing
agents, and the rest of this page explains why, then covers what to change if you want to trade the new
exchange mechanism as well.

## What 0.6.0 adds

The subnet now runs two mechanisms:

| mechanism | `mechid` | what it is | live on mainnet? |
|---|---|---|---|
| simulation | 0 | the agent-based market simulation, unchanged | yes, this is what you run today |
| exchange | 1 | an exchange with a real order book and AMM pools | **not on mainnet**; localnet only in 0.6.0 |

They are separate registrations with separate metagraphs and separate weights.

**Mainnet runs one mechanism, and 0.6.0 does not change that.** `MechanismCountCurrent(79)` reads `1` on
finney, so there is no exchange mechanism to register on and nothing to migrate to. What this release
provides is the interface, the wire types and the documentation to **build and test an exchange agent
against a localnet exchange**. Mainnet exchange is a later release; registration and scoring will be
announced before it arrives.

So the honest summary of this page is: **read it, change nothing yet.**

## Why your existing agent keeps working

Three things could have broken a deployed agent, and none of them do:

**Your registration is mechanism-scoped.** The validator builds its metagraph as
`subtensor.metagraph(netuid, mechid=...)`, and `--neuron.mechid` defaults to 0 for the simulation engine
and 1 for the exchange engine. An agent registered on mechanism 0 is not in the exchange validator's uid
set, so it is never sent an exchange state update. You opt in by registering, not by upgrading. And since
mainnet runs one mechanism today, there is currently no path by which an exchange state can reach you at
all.

**The base class kept its old name.** `FinanceSimulationAgent` was renamed to `FinanceAgentBase`, because
it never was the simulation-mode class: it holds the history, logging and event-hook machinery that both
modes use, and the mode dispatch lives one level down in `FinanceAgent`. The old name remains exported as
an alias of the same class object, so `from taos.im.agents import FinanceSimulationAgent` and
`class MyAgent(FinanceSimulationAgent)` both still work, and `issubclass` checks against it still hold.

**Your response object is still accepted as-is.** Agents written before 0.6.0 construct
`FinanceAgentResponse(agent_id=self.uid)` directly. 0.6.0 adds a mode-aware response from
`self.make_response()`, but the conversion step is conditional on that type, so a `FinanceAgentResponse`
you built yourself is returned unchanged.

## Preparing an exchange agent

You cannot trade the mainnet exchange in this release. You can develop and test against a localnet
exchange, and the agent you end up with is the one that will work later.

**1. Run against a localnet exchange.** See the "Exchange venue behaviours that catch people out"
section of [`README.md`](README.md) for the miner
command and the venue behaviours that catch people out (the price grid truncates, leverage is refused, and
a sell draws on a single delegate). Note that running the exchange-mode engine itself requires components
which are not included in this repository. When the mainnet mechanism arrives, participation will follow a
chain registration on that mechanism rather than any change to your code.

**2. Optionally, handle the exchange explicitly.** Your `respond(self, state)` is already called in both
modes, so an agent that does nothing else will trade the exchange with its existing logic. To branch, you
have three shapes available and can implement any one of them:

| you implement | called in simulation | called on the exchange |
|---|---|---|
| `respond(state)` | yes | yes |
| `respond_simulation(state)` | yes, instead of `respond` | no |
| `respond_exchange(state)` | no | yes, instead of `respond` |

`respond_exchange` falls back to `respond_simulation`, which falls back to `respond`, so adding one does
not oblige you to add the others.

If you do branch, build the response with **`self.make_response()`** rather than constructing a response
type directly. It returns an object with the same order API that finalizes to whichever type the current
mode requires, so one code path serves both. `self.exchange_mode` tells you which mode the current request
is in, and it is request-scoped, so it is correct even though one agent instance serves both validators.

To subclass fresh, use **`FinanceAgent`** (or `GenTRXAgent` for the model helpers). Do not subclass
`FinanceAgentBase` or its `FinanceSimulationAgent` alias for new work: it has no `respond_exchange` and no
exchange branch, so it is never served the exchange path.

## What differs in exchange mode

- **Balances are real.** The account carries settled and unsettled balances against live pools rather than
  simulated ones.
- **Naming.** The simulation spells the pair BASE and QUOTE, the exchange spells it ALPHA and TAO, and
  both spellings resolve in both modes (`OrderCurrency.ALPHA`/`TAO` are aliases of `BASE`/`QUOTE`), so
  porting an agent requires no renaming.
- **Leverage is unavailable.** The exchange runs with `maxLeverage=0`, so an order requesting leverage is
  refused at placement. Several of the example agents requested leverage unconditionally and were changed
  for this reason; if yours does, gate it on `self.exchange_mode`.
- **Positions cannot be closed.** `close_position` and `close_positions` settle a margin loan, and with no
  leverage there is none to settle: the exchange accepts three instructions in total (limit order, market
  order, cancel). Both calls log a warning and add nothing to your response rather than raising, and
  `onPositionClosed` / `onPositionCloseFailed` never fire. If your agent closes positions, gate it on
  `self.exchange_mode` so the difference is visible in your own code rather than as a silently missing
  instruction.
- **Notices.** Order acceptances, cancellations, fills and refusals arrive in `state.notices` for your uid,
  and the same `onOrderAccepted` / `onTrade` style hooks fire in both modes. Notices are delivered when
  they happen and fills are repeated for a short window, so an agent that is down misses events that
  occurred while it was away. Do not treat the notice stream as a ledger you can reconstruct from; the
  account on every state update is the authority.

## See also

- [`README.md`](README.md) for the full agent authoring guide, including the notice handlers.
- the "Exchange venue behaviours that catch people out" section of [`README.md`](README.md) for the
  exchange-specific wire types and behaviours.
