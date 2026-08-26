# Agent Development Guide

This document aims to provide some clarification and guidelines to assist miners in development of trading strategies for deployment in the subnet.  This is not intended as a comprehensive set of instructions, but rather provides an overview of the basic understanding and tools needed to begin the design and implementation of agent logic in the context of the τaos market simulation and exchange.

> **Already running an agent?** 0.6.0 introduces a second, exchange mechanism (localnet only for now;
> mainnet still runs the simulation mechanism). Your existing agent keeps working in the simulation with
> no changes. See [MIGRATION_0.6.0.md](MIGRATION_0.6.0.md) for why, and for the steps to trade the
> exchange as well.

---

**Table of Contents**
- [Data](#data)
  - [Processing State Updates](#processing-state-updates)
    - [Book States](#book-states)
    - [Notices](#notices)
- [Response](#response)
  - [The `FinanceAgentResponse` Class](#the-financeagentresponse-class)
    - [`market_order(...)`](#market_order)
    - [`limit_order(...)`](#limit_order)
    - [`cancel_order(...)`](#cancel_order)
    - [`cancel_orders(...)`](#cancel_orders)
  - [Timeout](#timeout)
  - [Latency](#latency)
    - [Response Time](#response-time)
    - [The `delay` Parameter](#the-delay-parameter)
  - [Trading Volume](#trading-volume)
    - [Contribution to Reward](#contribution-to-reward)
    - [Trading Limitation](#trading-limitation)
- [Agent Testing](#agent-testing)
  - [Local](#local)
  - [Testnet (Netuid 366)](#testnet-netuid-366)
  - [Mainnet (Netuid 79)](#mainnet-netuid-79)
- [Appendix](#appendix)
  - [The `MarketSimulationStateUpdate` Class](#the-marketsimulationstateupdate-class)

---

## Data

The first piece of the puzzle is understanding the format and content of the data published by validators.  Each request sent by a validator includes the latest (partial L3 + L2) state of all orderbooks in the simulation, a record of all events occurring in the simulation since the last update, and miner-specific information relating to the state of the agent's accounts and those events which involve the agent's orders.  The protocol class which defines how this data is represented practically can be found [here](../taos/im/protocol/__init__.py); the structure is expanded and documented for convenient reference in [The Appendix](#the-marketsimulationstateupdate-class).

### Processing State Updates

There are two main types of data that may need to be processed by an agent in deciding on their next trading actions: the first is the actual state data communicating the latest state of each of the simulated orderbooks, and the other is the notices which indicate when an event has occurred in relation to one of the agent's orders.

#### Book States

The latest state of the simulated orderbooks is contained in the `state.books` dictionary.  This dictionary maps the integer identifier for the book to a class structure which represents a snapshot of the top 21 levels of the book state together with a record of all events having occurred since the previous state was published.  In most cases, agents will need to iterate over the books, and apply the same logic to each of them in order to execute their strategy on all realizations of the simulated market.  As demonstrated in the examples, this is most straightforwardly achieved by a simple for loop:

```python
for book_id, book in state.books.items():
    # Analyze state data and generate instructions
```

For more complex and advanced strategies, it may be necessary to parallelize the processing of books in order to ensure to generate a response within the timeout.  Details of how to achieve this are for the miner to design in a manner suitable for their implementation and hosting configuration.

Each book object contains `bids` and `asks` arrays representing the top 21 levels on either side of the order book.  The arrays are ordered with higher indices corresponding to levels further from the midquote, meaning that the best bid (highest buy offer) and best ask (lowest sell offer) are at index 0.  Each level object has a `price` and `quantity` field indicating the price level and the total quantity of all resting orders composing the level.  The topmost levels include also an `orders` field which reveals the composition of the level in terms of the individual orders existing at that price.

```python
# Topmost LevelInfo objects
best_bid_level = book.bids[0]
best_ask_level = book.asks[0]
# Best bid price and volume
bid = best_bid_level.price
bid_vol = best_bid_level.quantity
# Best bid price and volume
ask = best_ask_level.price
ask_vol = best_ask_level.quantity
# Calculate spread
spread = ask - bid
```

The book object further contains an `events` field, which is populated with a complete listing of all the events which have occurred since the last state update.  This allows to reconstruct the complete, high-frequency history of the state of the orderbook throughout the previous publishing interval.  Full details of this procedure and usage are beyond the scope of this document, but the [ImbalanceAgent](ImbalanceAgent.py) sample agent demonstrates how to use this field and the associated tools to make use of a more complete, high-resolution record of the book state evolution.  An example of a simple use of this record is to obtain the price at which the last trade occurred; the `book` class includes a property method which allows to easily retrieve the latest `TradeInfo` object in the `events` record:

```python
last_trade : TradeInfo = book.last_trade
last_trade_price = last_trade.price
```

It may also be useful for certain strategies to obtain the trade price history over the preceding publishing interval:

```python
trade_price_history = {event.timestamp : event.price for event in book.events if event.type == 't'}
```

Each book object also exposes an `MTR` property giving the current **maker-taker ratio** for that book: the fraction of recent volume that was maker-side, in `[0, 1]` (typically around `0.5`).  It is computed by the exchange's dynamic fee policy and published with every book state, so agents can read it directly rather than reconstructing it from fills:

```python
for book_id, book in state.books.items():
    mtr = book.MTR
    if mtr is None:
        continue
    # e.g. provide more / tighter liquidity on books where making is scarce (low MTR)
```

Note that `MTR` is populated only under the **dynamic fee policy** used in simulation mode.  Under a tiered or zero-fee policy (including the zero-fee exchange) it does not apply and reports `0`, so treat a `0` or `None` as "no dynamic-fee signal for this book" rather than a genuine ratio.  This is the **book-wide** ratio across all participants; to track your own maker/taker split on a book, accumulate it from your own trade notices (see [Notices](#notices)): a trade where `Ma` equals your UID was maker-side for you, otherwise taker-side.

#### Notices

The `notices` field of the state update contains a dictionary mapping UIDs to a list of notifications corresponding to events that have occurred specifically in relation to that UID's previously submitted orders.  When received by a miner, this field contains only the notices corresponding to the receiving agent's actions.  If your agent does not override the default `FinanceAgentBase.update` method, specific event types can be handled by defining the appropriate functions in your agent code:

- `onStart(self, event : SimulationStartEvent)` : Triggered on start of a new simulation.
- `onOrderAccepted(self, event : OrderPlacementEvent)` : Triggered when an agent's order is accepted by the simulator.
- `onOrderRejected(self, event : OrderPlacementEvent)` : Triggered when an agent's order is rejected by the simulator (due to e.g. insufficient balance, invalid parameters etc).
- `onOrderCancelled(self, event : OrderCancellationEvent)` : Triggered when an agent's order is successfully cancelled in the simulator.
- `onOrderCancellationFailed(self, event : OrderCancellationEvent)` : Triggered when an agent's order fails to be cancelled in the simulator.
- `onTrade(self, event : TradeEvent)` : Triggered when an agent's order is involved in the trade.
- `onEnd(self, event : SimulationEndEvent)` : Triggered when a simulation ends.

To specify logic to be executed when a particular type of notice is received, simply define the handler function in your agent class:

```python
class MyTradingAgent(FinanceAgent):
    def initialize(self):
        ...

    def respond(self, state : MarketSimulationStateUpdate) -> FinanceAgentResponse:
        ...

    def onTrade(self, event : TradeEvent) -> None:
        print("{event}")
        # Do something - update internal records, trigger placement of a new order, recalculate statistics etc.
```

Note of course that these notices are part of the state update, and so all events occurring in the previous interval will be processed in sequence when the state is received (i.e. before your `respond` method is called).

These handlers fire in **both** simulation and exchange mode, so an agent that consumes notices this way needs no change when it moves to the exchange. Two mode differences are worth knowing: `onStart` and `onEnd` fire only where a simulation actually starts and ends, since an exchange does neither; and an exception raised inside your own handler is logged and skipped rather than aborting the rest of your notices, so one faulty handler costs you that handler and not your whole response.

## Running in both simulation and exchange mode

The subnet runs two mechanisms: a **simulation** and a live **exchange**. One agent can serve both, and
by default the same logic does.

### Which class to subclass

```
FinanceAgentBase            bookkeeping, logging, event hooks; defines respond
  └── FinanceAgent          adds respond_simulation + respond_exchange  <- subclass this
        └── GenTRXAgent     the same, plus GenTRX model support
```

Subclass **`FinanceAgent`** (or `GenTRXAgent` if you want the model helpers). All of the example agents in
this directory reach `FinanceAgent` through `GenTRXAgent`, so they serve both modes already.

`FinanceAgentBase` is the mode-agnostic base: it does history, logging and event dispatch, and defines
`respond`, but it has no `respond_exchange` and no exchange branch, so an agent built directly on it is
never served the exchange path. **It was called `FinanceSimulationAgent` before 0.6.0, and that name still
works** as an alias, so existing agents need no edit. New agents should name `FinanceAgent`.

### Which method is called

| you implement | called in simulation | called on the exchange |
|---|---|---|
| `respond(state)` | yes | yes |
| `respond_simulation(state)` | yes, instead of `respond` | no |
| `respond_exchange(state)` | no | yes, instead of `respond` |

`respond_simulation` and `respond_exchange` both fall back to `respond`, so **implementing `respond`
alone is enough for both modes**. Override one of the others only when you want genuinely different
behaviour in that mode.

```python
from taos.im.agents import FinanceAgent

class MyTradingAgent(FinanceAgent):
    def initialize(self):
        ...

    def respond(self, state):                 # serves BOTH modes
        r = self.make_response()
        ...
        return r
```

### Building the response

Use `self.make_response()`. It returns a response object with the same API in both modes and builds the
correct instruction types for whichever mode is being served. If you override `respond_simulation` or
`respond_exchange`, pass the mode explicitly (`self.make_response(exchange_mode=True)`) rather than
relying on the instance attribute, because concurrent requests from both validators share it.

### What differs between the modes

Most logic needs no mode branch: accounts and books are the same objects in both. Where you do need to
know, read `self.exchange_mode`.

* the state object is `MarketSimulationStateUpdate` in simulation and `ExchangeStateUpdate` on the
  exchange; both carry the same `books`, `accounts` and `notices` surfaces
* `onStart` and `onEnd` fire only in simulation, because an exchange does not start or end. Every other
  notice handler fires in both
* leverage is inert on the exchange (`maxLeverage` is 0), so leveraged orders are refused there
* `close_position` and `close_positions` are **simulation only**. Closing a position settles a margin
  loan, and the exchange has no leverage to settle: the exchange accepts three instructions in total
  (limit order, market order, cancel), so there is no close-position instruction to send. Calling either
  on the exchange logs a warning and adds nothing to your response, it does not raise. If your agent
  closes positions, guard the call with `if not self.exchange_mode:` so the intent is explicit in your
  own code. `onPositionClosed` and `onPositionCloseFailed` correspondingly never fire there
* `OrderCurrency` accepts BOTH spellings in BOTH trees: `BASE`/`QUOTE` are canonical and `ALPHA`/`TAO`
  are aliases of the same members, with the same values -- see the note under the order instructions above

## Exchange venue behaviours that catch people out

These are properties of the live venue, not of the agent API. They apply when your agent runs on the
exchange mechanism and are the things most likely to make a correct-looking order fail.

### Prices land on a grid, and the grid TRUNCATES

**Read this before you debug a fill that did not happen.**

The book has a fixed number of price decimals. A price you submit is truncated to that grid, not
rounded to it. That is deliberate: the grid is what makes resting orders comparable.

The consequence is not obvious, because binary floating point does not represent most decimal prices
exactly. A price computed in Python can sit a hair *below* the value you typed:

```python
>>> px = round(0.013637092231045785 * 0.5, 4)
>>> px
0.0068                                  # what you see
>>> from decimal import Decimal; Decimal(px)
Decimal('0.006799999999999999621136392846665330580435693264007568359375')
```

That is below `0.0068`, so truncation to four places yields **`0.0067`**, a whole tick lower than you
intended. Your order rests one tick away, and the quote reserved against it is smaller in proportion.
Measured on a real order: a 3,335,926.3957 buy reserved 22,350.71 TAO rather than the 22,684.30 the
price implied, a 1.5% difference caused entirely by the last decimal place.

Nothing rejects the order and nothing warns you. It rests at a legal price, just not the one you
meant.

This applies to **simulation as well as the exchange**: every numeric field arrives through the same
deserializer, so the behaviour is the same wherever you trade.

**Work in integer ticks and convert once.** Decide how many ticks from a reference price you want to
be, then build the price from that:

```python
TICK = 10 ** -price_decimals

def on_grid(price: float) -> float:
    """Snap to the grid the same way the book will, with the float error removed first."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-price_decimals)
    return float(Decimal(repr(price)).quantize(q, rounding=ROUND_HALF_UP))
```

`Decimal(repr(price))` is the important part: `repr` gives the shortest decimal that round-trips, so
`0.0068` stays `0.0068` instead of becoming `0.00679999…`. Quantize, then submit.

Check what you got rather than what you sent: the resting order reported back in the next state
carries the price the book actually holds. If it differs from your intent, this is why.

### A SELL draws on ONE delegate, so your free balance is not your sell limit

This is the constraint most likely to surprise you, because the number that looks authoritative is the
wrong one.

Your alpha on a book is held as **delegated stake**, and it can sit with several delegates at once.
`account.base_balance.free` is the **sum** across all of them. A SELL, though, is a pool swap against
**one** named delegate, so the most a single order can sell is the largest individual delegate's stake,
not the sum.

```python
acct = self.accounts[book_id]
acct.base_balance.free   # 1.49965345: the SUM across delegates
acct.delegate_stakes     # {'5AAA…': 0.76217809, '5BBB…': 0.73747536}
acct.sellable_alpha      # 0.76217809: the largest single delegate, your real cap
```

Size a SELL against `sellable_alpha`, not `base_balance.free`. Sizing against the sum produces an order
the engine cannot fill: it is refused as insufficient funds or, if you set `allow_partial=True`,
filled only up to that delegate's capacity, which is a short fill you did not ask for.

`delegate_stakes` being **empty means NOT REPORTED**, not "no stake". Fall back to `base_balance.free`
in that case rather than concluding you hold nothing.

### Leverage is refused on the exchange

Exchange mode runs with `maxLeverage=0`. A leveraged order is **rejected at placement**, not quietly
executed unleveraged, so an agent that requests leverage unconditionally works in simulation and places
nothing here. Branch on `self.exchange_mode`:

```python
def leverage(self):
    if self.exchange_mode:
        return 0.0
    return round(random.uniform(self.min_leverage, self.max_leverage), 2)
```

Returning it from one helper rather than at each call site also keeps any
`quantity() * (1 + leverage())` sizing consistent. The shipped examples do exactly this.

## Things that will save you time

| Symptom | Cause |
|---|---|
| never queried, no error | the axon address you published is not reachable from the validator |
| order rests a tick off | the grid truncates; see above |
| order rejected for size | below the minimum order size, or below the minimum stake |
| SELL rejected for funds while the balance looks sufficient | you sized against `base_balance.free` (the sum) instead of `sellable_alpha` (one delegate) |
| every order rejected, simulation was fine | you are requesting leverage; exchange mode runs `maxLeverage=0` |
| reserved less than expected | the reservation is quantity times the **resting** price, not your intended price |
| no fill on a crossing order | check the resting price first, then the available depth |

## Configuration

`initialize()` reads whatever you put in the `agents` section of your launcher config for your
class, so parameters belong there rather than hard-coded. `bin/agent_params.sh` lists the
parameters each shipped example expects.

Fields your agent reads off `self.simulation` (book count, price and volume decimals, initial price,
maximum open orders, publish interval) come from the exchange itself, so they are correct for the book
you are trading without you configuring anything.

## Response

Once a miner has received and analyzed the data, they must make a decision about what instructions they wish to submit to the simulation.  Any trading strategy is possible to implement, but note that the state is only published once every `config.publish_interval` simulation nanoseconds, and miners are only able to submit instructions in response to the state.  This implies that strategies must all operate at >= (`publish_interval / 1e9`) second timescale (we aim to lift/reduce this limitation in future).  Some simple example agent implementations can be found in this directory; **it is not expected that running any of the example agents without modification would lead to successful mining in the subnet**.

Miners are expected to develop their own custom agent logic and compete to improve their risk-adjusted performance.  This section documents and explains the usage of the tools involved in agent logic implementation; it does not intend to provide any guidance as to how to design a successful strategy.  However, our simulated markets aim to accurately approximate real markets, so that the same considerations should be applied when formulating strategies as in any trading scenario.  There are further some important limitations and restrictions imposed on miner agents which must be considered when designing a trading strategy, these are also reviewed and explained in the following sections.

### The `FinanceAgentResponse` Class

In order to submit instructions to the validator, a miner must respond to the validator request with an instance of the [`FinanceAgentResponse` class](../taos/im/protocol/response.py).  This class contains one property, `instructions`, which holds an array of `FinanceInstruction`, defined to encapsulate the four main instruction types which miners are able to execute : `PlaceMarketOrderInstruction`, `PlaceLimitOrderInstruction`, `CancelOrdersInstruction` and `ClosePositionsInstruction` (defined [here](../taos/im/protocol/instructions.py)).  The class additionally exposes convenience methods which allow to easily attach these instruction types to the `FinanceAgentResponse` instance:

---

#### Orders below the book's minimum size are rejected

Every book enforces a minimum order quantity. An order whose `quantity` falls below it is rejected
by the engine rather than partially accepted, and the rejection arrives as an order-rejected notice
rather than a fill.

The active value is on the state you already receive, so read it instead of assuming a number:

```python
min_size = float(getattr(state.config, "min_order_size", 0.0) or 0.0)
if quantity < min_size:
    return  # too small to be accepted
```

`min_order_size` is `0.0` only when a book sets no minimum. Shipped configurations set values as
large as `0.25`, so an agent that hardcodes a small fixed quantity can have every order rejected on
one book while trading normally on another. Size against the value on the state, per book.

---

#### `market_order(...)`

Place a **market order** to immediately buy or sell at the best available price.

##### **Signature**
```python
response.market_order(
    book_id: int,
    direction: OrderDirection,
    quantity: float,
    delay: int = 0,
    clientOrderId: int | None = None,
    stp: STP = STP.CANCEL_OLDEST,
    currency: OrderCurrency = OrderCurrency.BASE,
    leverage: float = 0.0,
    settlement_option: LoanSettlementOption | int = LoanSettlementOption.NONE
)
```

##### **Arguments**

| Parameter            | Type                                    | Description                                                                                                                               |
|----------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `book_id`            | `int`                                   | ID of the order book where the order will be placed.                                                                                     |
| `direction`          | `OrderDirection`                        | `OrderDirection.BUY` or `OrderDirection.SELL`.                                                                                            |
| `quantity`           | `float`                                 | Amount to buy/sell in `currency`.                                                                                                         |
| `delay`              | `int`, optional                         | Delay in simulation nanoseconds before the order reaches the market. This delay is added to the delay calculated based on response time. Defaults to `0`.                                                                      |
| `clientOrderId`      | `int` or `None`, optional               | Optional client-specified order ID for tracking.                                                                                          |
| `stp`                | `STP`, optional                         | Self-trade prevention strategy (`STP.NO_STP`, `STP.CANCEL_OLDEST`, `STP.CANCEL_NEWEST`, `STP.CANCEL_BOTH`, `STP.DECREASE_CANCEL`). Defaults to `STP.CANCEL_OLDEST`.                             |
| `currency`           | `OrderCurrency`, optional               | Currency to use for the order quantity (`OrderCurrency.BASE` or `OrderCurrency.QUOTE`). If set to `QUOTE`, the `quantity` will be interpreted as the amount of QUOTE currency to exchange. Defaults to `BASE`. On the exchange these are spelled `ALPHA` and `TAO` -- see [Currency names differ between the two modes](#currency-names-differ-between-the-two-modes). |
| `leverage`           | `float`, optional                       | Leverage multiplier to apply to the order. The effective order quantity will be `(1+leverage)`. For example, an order for 1.0 BASE with 0.5 leverage will be placed for 1.5 BASE total, where 0.5 is borrowed from the exchange. Must be non-negative. Defaults to `0.0` (no leverage).                                 |
| `settlement_option`  | `LoanSettlementOption` or `int`, optional | Strategy for settling outstanding margin loans using the proceeds of this order. Options: `LoanSettlementOption.NONE` (no loan repayments), `LoanSettlementOption.FIFO` (repay loans starting from oldest), or an integer order ID to repay the loan associated with a specific order. Defaults to `NONE`. Note: only unleveraged orders (`leverage=0`) can settle loans.                                 |

##### **Currency names differ between the two modes**

`taos.im.protocol` (simulation) and `taos.im.protocol.exchange` are parallel type trees, one per mode.
The enums line up by **value**, so an agent that passes `OrderDirection.BUY`, `TimeInForce.GTC` or an
`STP` member from the simulation tree keeps working when the same logic runs on the exchange.

`OrderCurrency` is the one exception, and it is a naming difference only:

| value | simulation | exchange | meaning |
|---|---|---|---|
| `0` | `OrderCurrency.BASE` | `OrderCurrency.ALPHA` | the subnet's alpha token |
| `1` | `OrderCurrency.QUOTE` | `OrderCurrency.TAO` | TAO |

Alpha is the base asset and TAO is the quote, so the two spellings mean the same thing -- and both
resolve in both trees. `OrderCurrency.BASE` and `OrderCurrency.ALPHA` are the same member, as are `QUOTE`
and `TAO`, so an agent can import the enum from either tree and use either spelling. Passing the integer
value works too, and is identical on both.

This was not always true: until 2026-08-20 the simulation tree had only `BASE`/`QUOTE` and the exchange
tree only `ALPHA`/`TAO`, so a dual-mode agent raised `AttributeError` on whichever tree it had not been
written against. If you are reading older agent code that carefully imports from one tree, that is why.

##### **Example**
```python
response.market_order(
    book_id=1,
    direction=OrderDirection.BUY,
    quantity=100.0,
    delay=50_000_000,  # 50ms delay
    leverage=0.5  # 50% leverage
)
```

---

#### `limit_order(...)`

Place a **limit order** at a specific price level.

##### **Signature**
```python
response.limit_order(
    book_id: int,
    direction: OrderDirection,
    quantity: float,
    price: float,
    delay: int = 0,
    clientOrderId: int | None = None,
    stp: STP = STP.CANCEL_OLDEST,
    postOnly: bool = False,
    timeInForce: TimeInForce = TimeInForce.GTC,
    expiryPeriod: int | None = None,
    leverage: float = 0.0,
    settlement_option: LoanSettlementOption | int = LoanSettlementOption.NONE
)
```

##### **Arguments**

| Parameter            | Type                                    | Description                                                                                                                                                                        |
|----------------------|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `book_id`            | `int`                                   | ID of the order book where the order will be placed.                                                                                                                              |
| `direction`          | `OrderDirection`                        | `OrderDirection.BUY` or `OrderDirection.SELL`.                                                                                                                                     |
| `quantity`           | `float`                                 | Quantity of the asset to trade.                                                                                                                                                    |
| `price`              | `float`                                 | Price at which to place the limit order.                                                                                                                                           |
| `delay`              | `int`, optional                         | Delay in simulation nanoseconds before the order reaches the market. This delay is added to the delay calculated based on response time. Defaults to `0`.                                                                                              |
| `clientOrderId`      | `int` or `None`, optional               | Optional client-specified order ID for tracking.                                                                                                                                   |
| `stp`                | `STP`, optional                         | Self-trade prevention strategy (`STP.NO_STP`, `STP.CANCEL_OLDEST`, `STP.CANCEL_NEWEST`, `STP.CANCEL_BOTH`, `STP.DECREASE_CANCEL`). Defaults to `STP.CANCEL_OLDEST`.                                                                                                                      |
| `postOnly`           | `bool`, optional                        | If True, prevents the order from matching immediately. If the limit order would match with any existing levels on the book at processing time, the instruction is rejected. Defaults to `False`.                                                                                                       |
| `timeInForce`        | `TimeInForce`, optional                 | Time-in-force option (`TimeInForce.GTC`, `TimeInForce.GTT`, `TimeInForce.IOC`, `TimeInForce.FOK`). **GTC** (Good Till Cancelled): Order remains on the book until cancelled or executed. **GTT** (Good Till Time): Order remains for `expiryPeriod` nanoseconds unless traded or cancelled. **IOC** (Immediate Or Cancel): Any part not immediately traded is cancelled. **FOK** (Fill Or Kill): Order is rejected if not executed in its entirety immediately. Defaults to `TimeInForce.GTC`.                                                                                                              |
| `expiryPeriod`       | `int` or `None`, optional               | Expiry period for `GTT` orders, in simulation nanoseconds. Required if `timeInForce` is `GTT`.                                                                                                                         |
| `leverage`           | `float`, optional                       | Leverage multiplier to apply to the order. The effective order quantity will be `(1+leverage)`. For example, an order for 1.0 BASE with 0.5 leverage will be placed for 1.5 BASE total, where 0.5 is borrowed from the exchange. Must be non-negative. Defaults to `0.0` (no leverage).                                 |
| `settlement_option`  | `LoanSettlementOption` or `int`, optional | Strategy for settling outstanding margin loans using the proceeds of this order. Options: `LoanSettlementOption.NONE` (no loan repayments), `LoanSettlementOption.FIFO` (repay loans starting from oldest), or an integer order ID to repay the loan associated with a specific order. Defaults to `NONE`. Note: only unleveraged orders (`leverage=0`) can settle loans.                                 |

##### **Example**
```python
response.limit_order(
    book_id=1,
    direction=OrderDirection.SELL,
    quantity=50,
    price=101.25,
    timeInForce=TimeInForce.GTT,
    expiryPeriod=10_000_000_000  # 10 seconds
)
```

##### **Notes**
- If `timeInForce` is `GTT`, `expiryPeriod` must be specified.
- If `timeInForce` is `IOC` or `FOK`, `postOnly` must be `False`.
- If `expiryPeriod` is specified but `timeInForce` is not `GTT`, expiry is ignored.
- You cannot hold leveraged positions on both sides of the book simultaneously.

---

#### `cancel_order(...)`

Cancel a single order.

##### **Signature**
```python
response.cancel_order(
    book_id: int,
    order_id: int,
    quantity: float | None = None,
    delay: int = 0
)
```

##### **Arguments**

| Parameter     | Type                | Description                                                                                             |
|---------------|---------------------|---------------------------------------------------------------------------------------------------------|
| `book_id`     | `int`               | ID of the order book where the order exists.                                                            |
| `order_id`    | `int`               | ID of the order to cancel.                                                                              |
| `quantity`    | `float` or `None`, optional | Quantity (in BASE) to cancel. If `None`, cancels the entire order. Defaults to `None`.        |
| `delay`       | `int`, optional     | Delay in simulation nanoseconds before the cancellation is processed. This delay is added to the delay calculated based on response time. Defaults to `0`.                                            |

##### **Example**
```python
response.cancel_order(book_id=1, order_id=42)
```

---

#### `cancel_orders(...)`

Cancel multiple orders at once.

##### **Signature**
```python
response.cancel_orders(
    book_id: int,
    order_ids: list[int],
    delay: int = 0
)
```

##### **Arguments**

| Parameter     | Type           | Description                                                                                                |
|---------------|----------------|------------------------------------------------------------------------------------------------------------|
| `book_id`     | `int`          | ID of the order book where the orders exist.                                                               |
| `order_ids`   | `list[int]`    | List of order IDs to cancel. Each order is fully cancelled.                                               |
| `delay`       | `int`, optional| Delay in simulation nanoseconds before the cancellations are processed. This delay is added to the delay calculated based on response time. Defaults to `0`.                                             |

##### **Example**
```python
response.cancel_orders(book_id=1, order_ids=[42, 43, 44])
```

---

#### `close_position(...)`

Close a **single leveraged position** and settle the associated loan.

##### **Signature**
```python
response.close_position(
    book_id: int,
    order_id: int,
    quantity: float | None = None,
    delay: int = 0
)
```

##### **Arguments**

| Parameter     | Type                | Description                                                                                                                               |
|---------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `book_id`     | `int`               | ID of the order book where the leveraged order exists.                                                                                   |
| `order_id`    | `int`               | ID of the leveraged order to close and settle.                                                                                            |
| `quantity`    | `float` or `None`, optional | Quantity (in BASE) to close. If `None`, closes the entire position associated with the specified order. Defaults to `None`.              |
| `delay`       | `int`, optional     | Delay in simulation nanoseconds before the instruction is processed at the exchange. This delay is added to the delay calculated based on response time. Defaults to `0`. |

##### **Example**
```python
response.close_position(
    book_id=1,
    order_id=123,
    quantity=0.5,  # Close half the position
    delay=20_000_000  # 20ms delay
)
```

---

#### `close_positions(...)`

Close **multiple leveraged positions** within the same order book.

##### **Signature**
```python
response.close_positions(
    book_id: int,
    order_ids: list[int],
    delay: int = 0
)
```

##### **Arguments**

| Parameter     | Type           | Description                                                                                                                                |
|---------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `book_id`     | `int`          | ID of the order book where the leveraged orders exist.                                                                                     |
| `order_ids`   | `list[int]`    | List of leveraged order IDs to close and settle. Each position is fully closed.                                                            |
| `delay`       | `int`, optional| Delay in simulation nanoseconds before the instruction is processed at the exchange. This delay is added to the delay calculated based on response time. Defaults to `0`. |

##### **Example**
```python
response.close_positions(
    book_id=1,
    order_ids=[101, 102, 103],
    delay=50_000_000  # 50ms delay
)
```


---

### Timeout

The query logic of validators enforces a timeout which specifies how long miners may take at maximum to respond to state updates.  The exact value of the timeout is subject to change, and is set in the [validator config](../taos/im/config/__init__.py) as `neuron.timeout` (see the `default` value for the current active setting).  If a response is not received within the timeout, no instructions will be submitted to the simulator for that agent.  It is the miner agent's responsibility to ensure that they receive, decompress and process the state update, as well as generate and return instructions, before the timeout expires.  This requires to allocate sufficient resources (CPU and network bandwidth) and optimize data analysis and other processes involved in trading decision making; it will also benefit the miner to locate nearby to key validators.

Parsing state updates is a major source of processing overhead for miners and can lead to timeouts even when the decision-making logic itself runs quickly. To reduce unnecessary work, the miner framework provides a configurable optimization called _lazy loading_. When enabled, deserialization defers the instantiation and validation of the Pydantic models that represent the state until their corresponding data fields are actually accessed.  This approach can drastically shorten the initial parsing time by skipping the construction and validation of unused data structures. As a result, agents that only interact with a subset of the state avoid the overhead of loading components they never use.

Miners can enable this optimization by adding `lazy_load=1` to their `--agent.params` when launching the miner.

---

### Latency

In real-world trading, there’s always some delay between submitting an order and it reaching the exchange. The simulator models this **latency** to make the simulation more realistic.

#### Response Time

When you submit instructions as a miner agent, they’re not executed immediately. Instead:  

- The simulator applies a **processing delay** to your instructions.  
- This delay is based on how quickly your agent responds to the validator’s state update.  
- **Faster responses mean your instructions are executed sooner** than those of slower agents.

The logic for calculating these delays is defined in the [`set_delays`](../taos/im/validator/reward.py) function.  

#### The `delay` Parameter

You can also manually add a **custom delay** to your instructions. This lets you schedule actions to happen later within the current simulation interval, rather than all at once:

```python
# Place an order immediately when the simulator receives your response
# This will be processed 0 simulation nanoseconds after the response time latency 
# determined by `set_delays` has elapsed
response.limit_order(
    book_id=0,
    direction=OrderDirection.BUY,
    quantity=1.0,
    price=301.25,
    delay=0
)

# Place another order 250ms after the instruction is received
response.limit_order(
    book_id=0,
    direction=OrderDirection.BUY,
    quantity=1.0,
    price=301.00,
    delay=250_000_000 # nanoseconds
)

# Place a third order 500ms after the instruction is received
response.limit_order(
    book_id=0,
    direction=OrderDirection.BUY,
    quantity=1.0,
    price=300.75,
    delay=500_000_000
)

# Place a fourth order 800ms after the instruction is received
response.limit_order(
    book_id=0,
    direction=OrderDirection.BUY,
    quantity=1.0,
    price=300.50,
    delay=800_000_000
)
```
This allows to attempt to take advantage of movements in price during the period between state updates, where otherwise miner agents are not able to participate.  The delay you specify is added on top of your agent’s response time latency. To maximize your advantage, you should aim to respond quickly as well as carefully scheduling your actions to take advantage of price movements within the publishing interval.

---

### Trading Volume

Trading volume plays an important role in determining the rewards assigned to a miner agent.  The trading volume of an agent is defined as the total value in QUOTE of the quantity of orders submitted by the agent which are matched by the simulator.  For example, if a miner submits an instruction to place an order with a quantity of `2.0`, and this order is matched at price `300.00`, their trading volume is increased by `600.00`.  This is true regardless of the role of the agent's order in the trade as either maker or taker.

#### Contribution to Reward

The primary component of incentive mechanism of the subnet is the risk-adjusted performance of the strategy, which is calculated using an intra-day Kappa-3 ratio where the returns are obtained as the difference in total inventory value held by the agent between subsequent state updates (see [rewarding logic](../taos/im/validator/reward.py)).  

In order to avoid rewarding miners who do not participate in active trading, as well as to incentivize the creation of volume in the simulated market, the calculated Kappa-3 values are then scaled by a factor derived using the agent's total traded volume over a [configured period](../taos/im/config/__init__.py) (see `scoring.activity.trade_volume_assessment_period`) of simulation time.  It should further be noted that, although any amount of traded volume within each `scoring.activity.trade_volume_sampling_interval` will trigger the volume factor to be assigned a value based on the traded volume during the assessment period, if no trades occur in the previous sampling interval then the volume factor will start to decay.  This is designed to prevent miners from producing a burst of trading activity once within the assessment period, and then stopping trading activity to benefit from the volume factor without maintaining consistent active trading. 

The inclusion of this factor in the scoring has the effect of magnifying the Kappa-3 ratios for miners which have executed more volume during the assessment period.  If the agent achieves a good Kappa-3 ratio while also trading significant volume, they will be rewarded more highly than a miner with the same performance at a lower traded volume.  Similarly, if a miner has high volume and poor Kappa-3 ratio, they will receive a worse score than a miner with the same performance and lower volume.  This discourages maximization of volume without sufficient regard for the performance, while also incentivizing the deployment of strategies which are both optimally risk-managed and highly active.

#### Trading Limitation

The validator logic also implements a cap on the maximum amount that can be traded within the assessment window.  This is intended to limit attempts to attain high volume-weighted scores during periods of good performance by recklessly trading purely for the sake of volume creation.  The limit is configured as a multiplier on the value of the initial capital allocated to miner agents; the multiplier is configured by validators using the `scoring.activity.capital_turnover_cap` parameter, where the value of initial capital allocated is configured in the simulator and can be read from the state update `config` field `miner_wealth`.  

Explicitly, if a miner has traded more than `scoring.activity.capital_turnover_cap * state.config.miner_wealth` in volume on a particular book over the `scoring.activity.trade_volume_assessment_period`, they will not be able to submit any more instructions to that book (other than cancellations) until their total volume over the preceding assessment period drops below this limit.

---

## Agent Testing

### Local

You can debug and test your agent logic offline before deploying to testnet or mainnet by making use of the facilities documented [here](proxy/README.md).  This setup allows to launch the simulator on your machine, and receive messages to the agent via a proxy which fulfils the role of the validator in a local setting.

For agents that participate in **GenTRX distributed training** (in addition
to trading), see:
- [`agents/proxy/README.md`](proxy/README.md) : proxy
  test with the full GenTRX gradient-server loop (no chain).
- [`doc/gentrx/miner_setup.md`](../doc/gentrx/miner_setup.md) : production
  miner setup (R2 bucket + on-chain commit).

### Public localnet

The exchange runs on a public localnet at `wss://localnet.mvtrx.exchange:443`, which is open for
testing. This is the place to exercise trading logic against a live exchange before you commit to
mainnet. Point your miner at it with `-e wss://localnet.mvtrx.exchange:443`.

### Testnet (Netuid 366)

Once you are satisfied that your agent logic works as intended, we recommend to register a UID on testnet (netuid 366) and deploy your miner as you intend to host it in mainnet environment.  This allows to confirm that all is properly configured for communication with validators, and the resources allocated to the miner are sufficient.  The test network carries no exchange, so use the public localnet above to exercise trading itself.  You can request testnet TAO at the [Bittensor Discord](https://discord.com/channels/799672011265015819/1389370202327748629).

### Mainnet (Netuid 79)

If all looks to be functioning well in testnet, register a UID on mainnet netuid 79 and restart your miner using the mainnet endpoint and your registered hotkey.  If you encounter issues, our team monitors the [τaos channel](https://discord.com/channels/799672011265015819/1353733356470276096) at BT Discord server.

Good luck!

---

## GenTRX Distributed Training <span id="gentrx"><span>

In addition to trading, miners can participate in **GenTRX**, a distributed training loop that builds a shared generative order-book model. Each training round, a miner downloads recent simulation data from the validator's S3 bucket, trains for a configurable number of steps, and uploads a compressed gradient delta. Validators score the gradient against held-out data; the aggregator (uid 0) aggregates accepted deltas and publishes the next checkpoint.

**Reward:** 5% of miner rewards are allocated to the GenTRX training pool by default (configurable by validators via `--scoring.gentrx.simulation_share`). This is separate from and additive to trading rewards: a miner that both trades well and trains well earns from both pools.

### GenTRX distributed training

All example agents support GenTRX distributed training. Training is **off by default** and activates only when `gtx_training_enabled=true` is passed in `--agent.params`. Agents run identically to prior behaviour without that flag.

| Agent | Trading logic | Notes |
|---|---|---|
| [`RandomMakerAgent`](RandomMakerAgent.py) | Random limit orders | Add `gtx_training_enabled=true` to enable |
| [`RandomTakerAgent`](RandomTakerAgent.py) | Random market orders | Add `gtx_training_enabled=true` to enable |
| [`ImbalanceAgent`](ImbalanceAgent.py) | LOB imbalance signal | Add `gtx_training_enabled=true` to enable |
| [`MovingHurstAgent`](MovingHurstAgent.py) | Hurst exponent momentum/reversion | Add `gtx_training_enabled=true` to enable |
| [`OrderOptionAgent`](OrderOptionAgent.py) | Advanced order options demo | Add `gtx_training_enabled=true` to enable |
| [`RevengAgent`](RevengAgent.py) | Volume-bucket momentum/reversion | Add `gtx_training_enabled=true` to enable |
| [`HybridTrainingAgent`](HybridTrainingAgent.py) | Imbalance-driven maker/taker | Training on by default; **template, not a finished strategy** |
| [`CustomTrainingAgent`](CustomTrainingAgent.py) | None (annotated template) | Override `_train_background` to plug in a custom training loop |

`GenTRXAgent` is the base class for all of the above; it lives in `taos.im.agents` (not in this directory). **To add GenTRX training to a custom strategy**, subclass `GenTRXAgent`, call `super().initialize()` and `super().respond(state)`, and the training loop is inherited. See [`CustomTrainingAgent.py`](CustomTrainingAgent.py) for an annotated example and [`doc/gentrx/integration.md`](../doc/gentrx/integration.md) for the full contract.

> **Note on `HybridTrainingAgent`:** Its docstring explicitly warns that deploying unmodified copies across many miners will cause them to interfere with each other. Use it as a starting point and customize the signal logic.

> **Note on dependencies:** the training path of these agents relies on the `[gentrx]` extra (torch, polars, pyarrow), which a plain install does not pull in. That applies to `CustomTrainingAgent` and `HybridTrainingAgent`, and to any of the others once you pass `gtx_training_enabled=true`. Run `pip install -e ".[gentrx]"` from the repository root before launching them. `RevengAgent`'s optuna optimizer is the exception: optuna is a core dependency, so that agent's trading logic runs after a plain `pip install -e .`.

### Quick setup

Full instructions are in [`doc/gentrx/miner_setup.md`](../doc/gentrx/miner_setup.md). In brief:

1. Create a Cloudflare R2 or Hippius bucket and generate write + read API tokens.
2. Run `python bin/setup_miner_bucket.py …` to verify tokens and commit read credentials on-chain.
3. Set `GENTRX_AGENT_S3_*` env vars in `.env`.
4. Run `bin/gentrx_preflight --role miner --env mainnet` to verify all components.
5. Run `./run_miner.sh -G` to launch with GenTRX training enabled.

Training is **enabled by default** (`gtx_training_enabled=true`). To opt out, pass `gtx_training_enabled=false` in `--agent.params`.

### Testing locally

| Test | What it covers | Runner |
|---|---|---|
| Proxy test (no chain) | Full GenTRX training loop, assignment lifecycle, scoring | [`agents/proxy/README.md`](proxy/README.md) |

---
---

## Appendix

### The `MarketSimulationStateUpdate` Class

---

- `version`

  This field is included to identify which version of the **taos** package the validator who sent the request is running.
  Miners do not need to worry about this generally; it is mainly to ensure backward compatibility during subnet development.

  **Type:** `int | None`

---

- `timestamp`

  The simulation timestamp at which the state was generated.
  This is represented as the number of nanoseconds since the start of the simulation.

  **Type:** `int`

---

- `config`

  Contains details of the simulation configuration ([`MarketSimulationConfig`](../taos/im/protocol/models.py)) used by the sending validator.  Includes simulation parameters, fee settings, and agent configurations.

  The fields which are important for miners are:

  - `config.baseDecimals` : Decimal precision of BASE currency values
  - `config.quoteDecimals` : Decimal precision of QUOTE currency values
  - `config.priceDecimals` : Decimal precision of prices (important when setting limit order price - input value will be rounded to this many decimals if specified to higher precision)
  - `config.volumeDecimals` : Decimal precision of volumes (important when setting order quantities - input value will be rounded to this many decimals if specified to higher precision)
  - `config.fee_policy` : The fee policy applied in the simulation
  - `config.max_open_orders` : The maximum number of orders that any agent in the simulation can simultaneously have open on the book.
  - `config.miner_wealth` : The total QUOTE value of the initial capital allocated to miners at start of simulation; this is used in determining the trading volume cap (see [Volume Limit](#volume-limit)).


  **Type:** `MarketSimulationConfig | str | None`

---

- `books`

  A dictionary mapping order book IDs to `Book` objects, which represent the state of each simulated order book at the time of the state publish event.

  **Type:** `dict[int, Book] | None`

  ---
  * **`Book`**

    Represents an order book at a specific point in time, including price levels and recent events.

    - `id`

      The unique identifier for this order book.

      **Type:** `int`

    - `MTR`

      The current maker-taker ratio for this order book (fraction of recent volume that was maker-side, in `[0, 1]`, typically around `0.5`).  Populated by the exchange's dynamic fee policy, so it carries a value under the dynamic fee policy used in simulation and reports `0` under a tiered or zero-fee policy (including the zero-fee exchange).  Accessed via the `book.MTR` property (the underlying serialized field key is `r`).

      **Type:** `float | None`

    - `bids`

      A list of bid price levels (`LevelInfo`) in descending price order (best/highest bid at index 0).

      **Type:** `list[LevelInfo]`

    - `asks`

      A list of ask price levels (`LevelInfo`) in ascending price order (best/lowest ask at index 0).

      **Type:** `list[LevelInfo]`

      ---

      **`LevelInfo`**
        - **`price`**
      
          The price of the bid level.

          **Type:** `float`

        - **`quantity`**
      
          The total quantity available at this bid price level.

          **Type:** `float`

        - **`orders`**
      
          A list of orders at this bid level (only present for top `config.detailedDepth` levels).

          **Type:** `list[Order] | None`

          ---
      
          **`Order`**
          - **`id`**

            The simulator-assigned ID of the order.

            **Type:** `int`

          - **`client_id`**

            The user-assigned client ID of the order.

            **Type:** `int | None`

          - **`timestamp`**

            The simulation timestamp at which the order was placed.

            **Type:** `int` 

          - **`quantity`**
        
            The remaining size of the order (in BASE).

            **Type:** `float` 

          - **`side`**

            The direction of the order, either `OrderDirection.BUY=0` or `OrderDirection.SELL=1`.

            **Type:** `int` 

          - **`price`**
            The price at which the order was placed.

            **Type:** `float` 

          - **`leverage`**
            The leverage applied to the order.  The effective order size is `(1+leverage)` times the base quantity, with the borrowed portion supplied by the exchange; `0.0` means unleveraged.

            **Type:** `float`

        ---

    ---

    - **`events`**

      A list of events that have occurred in the order book since the last snapshot.
      These may include `Order`, `TradeInfo`, and `Cancellation` entries.

      **Type:** `list[Order | TradeInfo | Cancellation] | None`

      ---

        **`Order`**

        - **`id`**

          The simulator-assigned ID of the order.

          **Type:** `int`

        - **`client_id`**
          The user-assigned client ID of the order.

          **Type:** `int | None`

        - **`timestamp`**
          The simulation timestamp at which the order was placed.

          **Type:** `int` 

        - **`quantity`**
          The remaining size of the order (in BASE).

          **Type:** `float` 

        - **`side`**
          The direction of the order, either `OrderDirection.BUY=0` or `OrderDirection.SELL=1`.

          **Type:** `int` 

        - **`price`**
          The price at which the order was placed.

          **Type:** `float` 

        - **`leverage`**
          The leverage applied to the order.  The effective order size is `(1+leverage)` times the base quantity, with the borrowed portion supplied by the exchange; `0.0` means unleveraged.

          **Type:** `float`

      ---
  
        **`TradeInfo`**

        - **`id`**  
          The simulator-assigned ID of the trade.  

          **Type:** `int`

        - **`side`**  
          Direction in which the trade was initiated.  
          `0` means BUY initiated, `1` means SELL initiated.  

          **Type:** `int`

        - **`timestamp`**  
          The simulation timestamp at which the trade occurred.  

          **Type:** `int`

        - **`taker_id`**  
          The ID of the aggressing order (the order initiating the trade).  

          **Type:** `int`

        - **`taker_agent_id`**  
          The ID of the agent who placed the aggressing order.  

          **Type:** `int`

        - **`taker_fee`**  
          The fee paid by the taker on this trade.

          **Type:** `float | None`

        - **`maker_id`**  
          The ID of the resting order (the order providing liquidity).  

          **Type:** `int`

        - **`maker_agent_id`**  
          The ID of the agent who placed the resting order.  

          **Type:** `int`

        - **`maker_fee`**  
          The fee paid by the maker on this trade.

          **Type:** `float | None`

        - **`quantity`**  
          The quantity traded (in base currency units).  

          **Type:** `float`

        - **`price`**  
          The price at which the trade occurred.  

          **Type:** `float`

      ---

      **`Cancellation`**

        - **`orderId`**  
          The ID of the cancelled order.  

          **Type:** `int`

        - **`timestamp`**  
          The simulation timestamp when the cancellation occurred.

          **Type:** `int`

        - **`price`**  
          The price of the cancelled order.

          **Type:** `float`

        - **`quantity`**  
          The quantity that was cancelled.

          **Type:** `float`
    ---

---

- `accounts`

  A dictionary mapping agent IDs to their trading accounts.
  Each agent maps to a dictionary of book IDs to `Account` objects.

  **Type:** `dict[int, dict[int, Account]] | None`

  ---
  * **`Account`**

    Represents an agent’s trading account on a specific order book.

    - **`agent_id`**

      The ID of the agent that owns this account.

      **Type:** `int`

    - **`book_id`**

      The ID of the order book where this account is active.

      **Type:** `int`

    - **`base_balance`**

      Represents the agent’s balance in the base currency.

      **Type:** `Balance`

    - **`quote_balance`**

      Represents the agent’s balance in the quote currency.

      **Type:** `Balance`

      ---

      **`Balance`**

      - **`currency`**

        The currency symbol (e.g., BTC).

        **Type:** `str`

      - **`total`**

        Total balance in this currency.

        **Type:** `float`

      - **`free`**

        Free balance available for trading.

        **Type:** `float`

      - **`reserved`**

        Reserved balance tied up in open orders.

        **Type:** `float`
    
      ---

    - **`orders`**

      A list of the agent’s currently open orders.

      **Type:** `list[Order]`

    - **`fees`**

      The fee rates applicable to this account.

      **Type:** `Fees | None`

      ---

      **`Fees`**

        - **`volume_traded`**
        
          Total volume traded by this agent for tiered fee assignment.

          **Type:** `float`

        - **`maker_fee_rate`**
        
          Current maker fee rate for the agent.

          **Type:** `float`

        - **`taker_fee_rate`**
        
          Current taker fee rate for the agent.

          **Type:** `float`

      ---

  ---

- `notices`

  A dictionary mapping agent IDs to a list of market events that occurred since the last update.

  **Type:** `dict[int, list[SimulationStartEvent | LimitOrderPlacementEvent | MarketOrderPlacementEvent | OrderCancellationsEvent | TradeEvent | ResetAgentsEvent | SimulationEndEvent]] | None`

  ---
  * **`SimulationStartEvent`**

    Represents the event generated on simulation start.

  ---

  * **`SimulationEndEvent`**

    Represents the event generated on simulation end.

  ---

  * **`OrderPlacementEvent`**

    Base class for events representing placement of an order in the simulation.

    - **`bookId`**

      The ID of the order book on which the order was attempted to be placed.

      **Type:** `int`

    - **`orderId`**

      The ID assigned to the order by the simulator.

      **Type:** `int`

    - **`clientOrderId`**

      Optional agent-assigned identifier for the order.

      **Type:** `int | None`

    - **`side`**

      The side of the book on which the order was attempted to be placed (`0=BID`, `1=ASK`).

      **Type:** `int`

    - **`quantity`**

      The size of the order in base currency.

      **Type:** `float`

    - **`success`**

      Flag indicating whether the order was successfully placed.

      **Type:** `bool`

    - **`message`**

      A message describing the result of the placement attempt (e.g., error reason).

      **Type:** `str`

  ---

  * **`LimitOrderPlacementEvent`**

    Represents the event generated on placement of a Limit Order.
    Inherits all fields from `OrderPlacementEvent`, plus the below:

    - **`price`**

      The price level at which the order was attempted to be placed.

      **Type:** `float`

  ---

  * **`MarketOrderPlacementEvent`**

    Represents the event generated on placement of a Market Order.
    Inherits all fields from `OrderPlacementEvent`.


  ---

  * **`OrderCancellationsEvent`**

    Represents cancellation of multiple orders.

    - **`bookId`**

      The ID of the order book where cancellations were attempted.

      **Type:** `int | None`

    - **`cancellations`**

      A list of events for each individual order cancellation.

      **Type:** `list[OrderCancellationEvent]`

      ---  
    
        **OrderCancellationEvent**

        Represents cancellation of a single order.

        - **`timestamp`**

          The simulation timestamp at which cancellation was attempted.

          **Type:** `int`

        - **`bookId`**

          The ID of the order book where the cancellation was attempted.

          **Type:** `int`

        - **`orderId`**

          The ID of the order being cancelled.

          **Type:** `int`

        - **`quantity`**

          The quantity to be cancelled in base currency.  
          If `None`, the entire remaining size of the order is cancelled.

          **Type:** `float | None`

        - **`success`**

          Flag indicating whether the cancellation was successful.

          **Type:** `bool`

        - **`message`**

          A message describing the result of the cancellation attempt.

          **Type:** `str`

        ---

  ---

  * **`TradeEvent`**

    Represents a trade that occurred in the simulation.

    - **`bookId`**

      The ID of the order book where the trade occurred.

      **Type:** `int | None`

    - **`tradeId`**

      The simulator-assigned ID of the trade.

      **Type:** `int`

    - **`clientOrderId`**

      Optional client-assigned ID of the resting order that was traded.

      **Type:** `int | None`

    - **`takerAgentId`**

      The ID of the agent that placed the aggressing order.

      **Type:** `int`

    - **`takerOrderId`**

      The ID of the aggressing order.

      **Type:** `int`

    - **`takerFee`**

      Fee paid by the taker.

      **Type:** `float`

    - **`makerAgentId`**

      The ID of the agent that placed the resting order.

      **Type:** `int`

    - **`makerOrderId`**

      The ID of the resting order.

      **Type:** `int`

    - **`makerFee`**

      Fee paid by the maker.

      **Type:** `float`

    - **`side`**

      Direction of the trade (`0=BUY initiated`, `1=SELL initiated`).

      **Type:** `int`

    - **`price`**

      The price at which the trade occurred.

      **Type:** `float`

    - **`quantity`**

      The quantity traded in base currency.

      **Type:** `float`

  ---

  * **`ResetAgentsEvent`**

    Represents a batch reset of multiple agent accounts.

    - **`resets`**

      A list of `ResetAgentEvent` objects, each describing an individual agent reset.

      **Type:** `list[ResetAgentEvent]`

      ---

      **`ResetAgentEvent`**

        Represents the event generated when a single agent account is reset.

        - **`success`**

          Flag indicating whether the agent’s account was successfully reset.

          **Type:** `bool`

        - **`message`**

          Message associated with the reset operation.

          **Type:** `str`

      ---

  ---

- `response`

  A mutable field populated by the miner agent with a response containing instructions for the simulation.

  **Type:** `FinanceAgentResponse | None`

---

- `compressed`

  A compressed representation of the state data, used to reduce transmission size.

  **Type:** `str | dict | None`

---

- `compression_engine`

  The library used for compressing the state data.
  Valid values: `"zlib"`, `"lz4"` (default=`lz4`).

  **Type:** `str`

---