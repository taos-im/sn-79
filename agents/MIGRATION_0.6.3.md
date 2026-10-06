# Agents on a multi-asset layout (0.6.3)

From 0.6.3 a simulation may carry more than one asset class: a multi-asset layout runs one background
market per class (each as one or more realizations), and every class quotes on its own price and volume
grid. The first production layout is eight markets of sixteen books, 128 books in all: books 0 to 95 are
the calibrated `simulation_0` class, books 96 to 127 a second class, `simulation_1`, at a tenth of the
price level. Both quote on 2 price decimals and 4 volume decimals, so on `simulation_1` one tick is ten
times larger relative to price, and its minimum order is 2.5364 units against 0.25.

Book ids run consecutively across the realizations in the layout's order (books 0 to 95, then 96 to 127), so an
agent written for 0.6.2 iterates them as before and still runs. On the second class it reads
`simulation_0`'s limits through the top-level configuration: orders below that class's minimum are
rejected, and parameters held in price units are ten times too large relative to price. Read the grids
and the minimum order per book.

## Telling the classes apart

The configuration that arrives with every state update exposes the layout, and the agent base class wraps
it. Everything is one call, keyed by the book id you already have in hand:

```python
self.asset_classes()           # [AssetClass(name, books, priceDecimals, volumeDecimals, config), ...]
self.class_of(book_id)         # "simulation_0" or "simulation_1" ("market" on a single market,
                               #  "exchange" on the exchange mechanism)
self.config_for(book_id)       # the market configuration governing that book (grids, agents, limits)
self.price_decimals(book_id)   # the book's price grid
self.volume_decimals(book_id)  # the book's volume grid
self.min_order_size(book_id)   # the book's minimum order
self.round_price(book_id, p)   # a price on the book's own grid
self.round_volume(book_id, q)  # a volume on the book's own grid
```

A single market is one class named `market` over every book, so the same lines run on a single-market
simulation, as mainnet is until the second class arrives at a run boundary.

## What to change

- Replace any `self.simulation_config.priceDecimals` or `volumeDecimals` (one grid for every book) with
  `self.round_price(book_id, ...)` and `self.round_volume(book_id, ...)`, inside the loop over
  `state.books.items()` where the book id is in hand.
- Size orders to `self.min_order_size(book_id)`, not to the top-level minimum.
- Scale any parameter you hold in price units (offsets, spreads, thresholds) to the book's price level.

The default agents (`SimpleRegressorAgent`, and `HybridTrainingAgent` under `-G`) and the market maker, taker,
arbitrage and random agents size to the book's own minimum and round on its grid from 0.6.3; treat the other
examples as templates and adapt them the same way. A layout may give its classes different
grids, and even on the same grid a class at a different price level has a different tick relative to price.

If your strategy is calibrated on `simulation_0`, treat the second class as a different market. It has the
same agent types and fundamental process, but its background traders are calibrated differently: its market
makers quote far less tightly and its other traders react on shorter horizons, so its spread, depth and fill
rate are not `simulation_0`'s. `self.class_of(book_id)` is the switch.

Capital is per book, as before: every book carries its own quote and base balance and its own volume
allowance; nothing moves between books or classes.

## Two other changes your agent may meet

- Your agent's `process` handler now runs on a live miner: the event notifications the validator sends
  outside the state update reach it. Simulation start and end still arrive in the state update and fire
  `onStart` and `onEnd` there. If you implemented `process`, review it; validators neither require nor accept
  a response.
- On the exchange mechanism a book id is the netuid of the pool it trades, so ids need not start at zero or
  be contiguous. Iterate `simulation_config.book_ids` rather than `range(book_count)`.

## Scoring notes

The skill bar is a share of the books being scored, the same share of each class's books: 15 of 96 on
`simulation_0` and 5 of 32 on `simulation_1`. Under the per-class emission weight each class is paid
through its own share of each half of the pool, mixed at the announced weights (95/5 at launch): an
account trading only the second class earns at most its weight, which is expected to step up as the class
proves itself on mainnet, and an account on both earns its share of each. Per-class gauges are on the
dashboard. The release's other scoring changes are described in the release announcement; none of them changes the
agent interface.
