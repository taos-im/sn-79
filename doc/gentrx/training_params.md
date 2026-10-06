# GenTRX Training Parameter Guide

The values below describe the model and training configuration as shipped in this repo. Tracking against a separate version number is intentionally avoided; the model evolves with the code.

## Model architecture

| Param | Value | Notes |
|---|---|---|
| `d_model` | 288 | Divisible by n_heads (288 / 8 = 36 per head) |
| `n_layers` | 8 | More depth gives better context modelling |
| `n_heads` | 8 | Standard for this model size |
| `d_ff` | 1152 | ~4× d_model |
| `dropout` | 0.1 | 0.15-0.2 also reasonable for stronger regularisation |
| `max_seq_len` | 2048 | Context window in orders |
| `film_layers` | (2, 5, 7) | FiLM conditioning injection points in backbone |
| `film_d_cond` | 64 | Hidden dim of FiLM projection MLP |
| **Total params** | **~12.1M** | Includes ~140K from FiLM and the richer time projection |

## Tokenizer bins

| Field | Bins | Range | Scale | Notes |
|---|---|---|---|---|
| `price` | 100 | [-500, 500] | symmetric log | Split 50 neg + 50 pos, log within each half. ~13 bins for ±5 ticks, dense near mid. |
| `vol_int` | 64 | [0, 100] | log | 81% in bin 0 (most orders vol = 0-1). Log helps the tail. |
| `vol_dec` | 8 | [0, 1] | linear | Fractional volume precision. |
| `interval` | 64 | [0, 50ms] | log | ~80% entropy across bins. |

**Symmetric log price binning.** `BinConfig(100, -500, 500, symmetric_log=True)`. Bin 50 is the zero band [-1, +1). Bins 0..49 cover negative (deep bid to near mid), bins 50..99 cover positive (near mid to deep ask). Log-spaced within each half gives ~13 bins per side for ±5 ticks.

Boundary behaviour:

- `vol_int` upper bound (100) clips ~0.4% of observations; may need adjustment for markets with markedly different volume distributions.
- `interval` upper bound (50 ms) clips ~4%; events with longer gaps collapse into the top bin.
- `price` range is fixed across symbols. An adaptive per-symbol range would help when tick sizes vary.

## Loss configuration

| Param | Value | Notes |
|---|---|---|
| Order type class weights | `[1.0, 1.0, 1.0, 1.0, 1.0]` (equal) | Five order-type classes: the three book events plus signed executions (a filled trade emits exec_buy or exec_sell). Weights are equal across classes. |
| Field loss weights | `order_type=2.0, price=1.5, interval=0.3, vol_int=0.5, vol_dec=0.5` | order_type and price carry the most actionable signal; interval is bin-quantised so its gradient is noisy and gets a smaller weight. |
| Label smoothing | none (0.05-0.1 reasonable) | Discourages overconfident predictions, helps generalisation. |

**Why these weights:**

- Order-type classes are weighted **equally**. Upweighting the rare directional and execution classes over the easy cancel-guess forced usability at the cost of generalisation, so the common cancel prediction is tolerated rather than penalised.
- `price=1.5`: price has 100 bins and the most complex distribution; extra signal helps.
- `interval=0.3`: interval is the noisiest field per gradient unit spent, partly because the bin scheme above quantises the distribution coarsely. The lower weight keeps it from dominating the loss while the model is still learning the easier fields.

## Standalone pretraining

Offline training of a seed checkpoint, separate from the live distributed loop. `--val-interval` here is a step counter (validate every N gradient steps), not a wall-clock cadence.

| Param | Quick test | Base checkpoint | Full training |
|---|---|---|---|
| `--lr` | 1e-4 | 1e-4 | 1e-4 |
| `--min-lr` | 1e-5 | 1e-6 | 1e-6 |
| `--warmup-steps` | 50 | 300 | 500 |
| `--batch-size` | 32 | 64 | 64 |
| `--seq-len` | 256 | 512 | 512 |
| `--max-books` | 3 | all | all |
| `--max-steps` | 500 | none | none |
| `--patience` | none | 3 | 5 |
| `--val-interval` | 100 | 500 | 500 |

## Distributed training (agent-side)

Training agents are triggered by an assignment arriving via dendrite. The assignment names the books, the exact page files, and the `model_version` to train against. Pages come from the validator bucket (credentials carried in the assignment payload). The agent downloads the named checkpoint from the validator's bucket (discovered via chain) when it is newer than the local one.

Training is **budget-driven**: the agent trains its assigned pages one at a time, recent first, and stops when the round budget runs out. A GPU clears several pages, a CPU trains a partial pass of the first. There is no fixed step count to tune.

| Param | Default | Notes |
|---|---|---|
| `gtx_round_budget_s` | 240 | Wall-clock training budget per round, in seconds. Keep below the round wallclock minus checkpoint/page download and gradient upload headroom. |
| `gtx_train_steps` | 0 | Optional fixed total-step cap per window. 0 means budget-governed (the default). Set a positive value only to pin step count for experiments. |
| `gtx_train_batch_size` | 16 on cuda, 4 on cpu | Bounded by device memory (attention is quadratic in seq×batch). Localnet / proxy launchers override to 8 for smaller GPUs. |
| `gtx_train_seq_len` | 512 | Matches the seed checkpoint's context and the gradient server's scoring loaders, so the model is trained and scored at the same context length. |
| `gtx_train_lr` | 1e-4 | Same as pretrain |
| `gtx_top_k_frac` | 0.10 | 10% retention, ~10× compression |
| `gtx_device` | `auto` | Device override. `auto` picks cuda if available else cpu. Set to `cpu` to force CPU even on GPU hosts (debugging, shared-host scenarios, memory analysis). The defaults above adjust based on the resolved device. |

**Key trade-off.** A larger `gtx_round_budget_s` lets a fast miner cover more pages per round, one pass over each; extra budget means more pages, not repeated passes over the same page. The held-out validation score the gradient server computes caps overfitting: a gradient that overfits its pages lowers the held-out score, so the budget self-limits in practice.

## Gradient server (aggregation)

| Param | Default | Notes |
|---|---|---|
| `--max-pending-rows-per-book` | `30000` | Page size: a book flushes a parquet once it reaches this many rows. This is the primary flush trigger, so active books emit uniform fixed-row pages. |
| `--books-per-miner` | 3 | Page files assigned per miner per round (one page per book). Miners train them incrementally, so weak hardware trains fewer. Each miner gets a random overlapping sample of books over a shared held-out window; the shared held-out set (not identical data) is what makes gradients comparable across miners. |
| `--val-fraction` | 0.10 | Size of the held-out scoring split. The validator picks the split each round, rotating which books are held out and keeping them disjoint from that round's training books, then pushes it to the server. So no book is trained and scored in the same round (no contamination), while every book is covered over time. This value is the server-side fallback size when no split is pushed. |
| `--parquet-interval-ns` | `300000000000` (5 min) | Sim-time fallback that tail-flushes a stalled book's partial page. Not the primary trigger. |
| `--min-score` | -0.1 | Stricter (e.g. -0.05 to 0.0) means fewer accepted gradients but safer. |
| `--rollback` | true | Always keep. Protects against regression. |
| `--max-val-batches` | 10 | More batches means more accurate scoring but slower. Range 10-30 is reasonable; raise if per-round score noise matters more than latency. |
| `--blocks-per-round` | 25 | Server-side estimate, used only by the heartbeat-loss fallback in `_round_complete`. Should match the validator's `--gentrx.blocks_per_round`. |
| `--block-time-s` | 12.0 | Assumed seconds per block on the target chain. |
| `--round-grace-s` | 30 | Grace seconds added to the heartbeat-loss estimate before force-closing a round. |

`--interval` (default 30 s) is a proxy / timer-mode knob and is ignored in block-synced production deployments; round closure is driven by `POST /gentrx/round` from the validator.

### Page size and sim grace period

The sim does not emit state immediately at startup. The shipped layouts (`multiasset_simulation_0.xml` and the per-class `simulation_0.xml` and `simulation_1.xml`) set `gracePeriod="600000000000"` nanoseconds (10 minutes), during which the exchange accepts connections but publishes no state. First state messages arrive at `t = 10 minutes` of sim time.

Data is then accumulated into fixed-row pages: a book flushes a parquet once it reaches `--max-pending-rows-per-book` rows (default 30 000). The gradient server cannot assign a book until its first page has flushed, so the first training round lands once an active book fills a page, which depends on the order rate rather than a fixed window. A quiet book that never fills a page is tail-flushed on the `--parquet-interval-ns` sim-time fallback.

| Setting | Default | Where |
|---|---|---|
| Sim grace period | 10 min (`600000000000` ns) | `MultiBookExchangeAgent.gracePeriod` in the simulation XML |
| Page size | 30 000 rows | `--max-pending-rows-per-book` on the gradient server |
| Tail-flush fallback | 5 min (`300000000000` ns) | `--parquet-interval-ns` on the gradient server |

Combined with **field-level weights** (order_type=2.0), the model gets 4x more gradient signal from "is the next order a bid or ask?" compared to "what's the volume decimal?"

---
