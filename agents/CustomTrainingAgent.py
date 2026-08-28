# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
CustomTrainingAgent — annotated template for custom GenTRX training and data
collection.

Two override points are demonstrated here.  Both have their full default logic
copied in so you can read, modify, and experiment without needing to trace into
the base class.

  train()                — one training window; runs on a background thread
                           after downloads complete.
  select_training_files() — choose which of the downloaded parquet files to
                           feed to the DataLoader.
  collect_row()          — build the row dict for one order event before it is
                           written to the local parquet buffer.

To use this template:
  1. Rename the class and this file to match your strategy.
  2. Add your trading signal to respond() below.
  3. Modify the three override methods as needed.
  4. Launch with gtx_training_enabled=true in --agent.params.

The full integration contract is in doc/gentrx/integration.md.
The data schema written by collect_row() is defined in
GenTRX/src/util/schema.py (order_stream_schema).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from taos.im.agents import GenTRXAgent

# Order-type constants and LOB types used in collect_row() overrides.
from GenTRX.src.util.schema import BID, ASK, CANCEL, LOB_DEPTH  # noqa: F401
from GenTRX.src.orderbook import LobSnapshot  # noqa: F401


class CustomTrainingAgent(GenTRXAgent):
    """Trading agent with fully custom training loop and data collection."""

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        # Opt-in guard: training is a no-op unless the operator explicitly
        # passes gtx_training_enabled=true in --agent.params.  Remove these
        # two lines if you always want training enabled.
        """Set up strategy state and register with the training server when configured."""
        if not hasattr(self.config, "gtx_training_enabled"):
            self.config.gtx_training_enabled = False
        if not hasattr(self.config, "gtx_collect_data"):
            self.config.gtx_collect_data = False

        super().initialize()

        # Read any strategy-specific config here.

    # ------------------------------------------------------------------ #
    # Trading signal — customize here                                     #
    # ------------------------------------------------------------------ #

    def respond(self, state):
        # MUST call super() first: runs data collection, inference, and
        # queues training if an assignment arrived this tick.
        """Produce this block's response from the custom-trained model's signal."""
        response = super().respond(state)

        # Add your order-placement logic here, modifying `response`.

        return response

    # ------------------------------------------------------------------ #
    # Data collection — what gets recorded per order event                #
    # ------------------------------------------------------------------ #

    def collect_row(
        self,
        book_id: int,
        ts: int,
        order_type: int,
        price_ticks: int,
        qty: float,
        snap: LobSnapshot,
        session_open_mid: int | None,
        *,
        interval_ns: int = 0,
    ) -> dict[str, Any] | None:
        """Build the row written to local parquet for one order event.

        Called once per BID, ASK, or CANCEL event when gtx_collect_data=true.
        Return the dict to buffer, or None to drop the event.

        The dict must satisfy order_stream_schema() — the fixed schema consumed
        by OrderDataset and expected by the model.  The required columns are the
        ones built below.  Extra keys are ignored at flush time; missing required
        keys raise at flush time.

        Customisation ideas:
          - Filter events by book: return None if book_id not in {12, 45}
          - Drop cancels:          return None if order_type == CANCEL
          - Normalise differently: change rel_price or volume encoding
          - Add extra columns:     they will be ignored by OrderDataset but are
                                   preserved in the parquet file for offline use

        Args:
            book_id: Book the event is on.
            ts: Event timestamp.
            order_type: BID, ASK or CANCEL.
            price_ticks: Price expressed in ticks.
            qty: Event quantity.
            snap: The book snapshot at the event.
            session_open_mid: Session-open mid price for normalisation.

        Returns:
            dict | None: The row to buffer, or None to drop the event.
        """
        # --- Drop any event type you don't want recorded, e.g.: ----------
        # if order_type == CANCEL:
        #     return None

        # --- Filter to specific books, e.g.: -----------------------------
        # if book_id not in {12, 45, 78}:
        #     return None

        # book_id is available here for per-market branching — see commented
        # examples above.
        _ = book_id

        mid = snap.mid_price
        ask_vols = snap.ask_volumes[:LOB_DEPTH] + [0] * max(0, LOB_DEPTH - len(snap.ask_volumes))
        bid_vols = snap.bid_volumes[:LOB_DEPTH] + [0] * max(0, LOB_DEPTH - len(snap.bid_volumes))

        row: dict[str, Any] = {
            "timestamp":       ts,
            "order_type":      order_type,
            "rel_price":       price_ticks - mid if mid > 0 else 0,
            "volume_int":      int(qty),
            "volume_dec":      qty - int(qty),
            "interval_ns":     interval_ns,
            "mid_price":       mid,
            "time_of_day_s":   int((ts // 1_000_000_000) % 86400),
            "mid_price_delta": int(mid - session_open_mid) if session_open_mid else 0,
        }
        for i in range(LOB_DEPTH):
            row[f"lob_ask_vol_{i + 1}"] = float(ask_vols[i]) / self._gtx.vol_scale
            row[f"lob_bid_vol_{i + 1}"] = float(bid_vols[i]) / self._gtx.vol_scale

        return row

    # ------------------------------------------------------------------ #
    # Training — which files to use                                       #
    # ------------------------------------------------------------------ #

    def select_training_files(
        self, parquet_files: list[Path], assignment: dict | None
    ) -> list[Path]:
        """Choose which downloaded parquet files to pass to the DataLoader.

        Called at the start of train() before the dataset is built.
        Return an empty list to skip the training window entirely.

        parquet_files: all local files downloaded for this round
        assignment:    the primary assignment dict (keys: round, books,
                       ts_start, ts_end, validator_uid, data, …)

        Customisation ideas:
          - Train only on specific books (assignment["books"] lists them)
          - Skip very small files that would produce degenerate batches
          - Prefer locally-collected files over S3-fetched ones

        Args:
            parquet_files: The downloaded training files.
            assignment: The training assignment being served.

        Returns:
            list: The files to train on; empty skips the window.
        """
        # assignment keys: round, books, ts_start, ts_end, validator_uid, data, …
        # Use them to drive filtering logic, e.g.:
        #   books = set(assignment.get("books", [])) if assignment else set()
        _ = assignment

        # --- Filter to specific books, e.g.: -----------------------------
        # target_books = {"12", "45"}
        # return [f for f in parquet_files if f.parent.parent.name in target_books]

        # --- Skip files smaller than a minimum row count, e.g.: ----------
        # MIN_BYTES = 64 * 1024  # 64 KB
        # return [f for f in parquet_files if f.stat().st_size >= MIN_BYTES]

        return parquet_files  # default: use everything

    # ------------------------------------------------------------------ #
    # Training — the loop itself                                          #
    # ------------------------------------------------------------------ #

    def train(
        self,
        parquet_files: list[Path],
        train_model,
        assignment: dict | None = None,
    ) -> None:
        """Run one training window and upload the compressed gradient.

        Called on a background thread after select_training_files() is applied.
        train_model is a deep copy of self._gtx.model — safe to mutate in place.

        This mirrors the base GenTRXAgent.train() one-pass contract:
          - Build one interleaved DataLoader over the assigned pages.
          - Train with train_incremental() — one budget-governed pass over the
            data, not a fixed step count — using self._make_window_config() so
            the trained model_version and the label-smoothing width the
            validator scores against are always stamped on the WindowConfig.
          - Hand the delta to self._submit_gradient(), which compresses,
            uploads, prunes old gradients, retries to a local file on S3
            failure, and advances the window counter.

        Customisation ideas:
          - Override ONLY the tunable knobs, e.g.
                win_cfg = self._make_window_config(n_steps=500, lr=self._gtx.train_lr * 0.5)
            n_steps > 0 caps total optimizer steps; leave it 0 (the default) for
            a full budget-governed pass. Do NOT hand-build WindowConfig() — that
            drops model_version / label_smooth_sigma, and the aggregator will
            reject the gradient (version mismatch) or misscore it (baseline
            trained at a different smoothing width).
          - Filter or reweight pages in select_training_files().
          - Keep _submit_gradient() as the publish surface — the validator reads
            that exact S3 path.

        Args:
            parquet_files: The files selected for this window.
            train_model: A deep copy of the model; safe to mutate.
            assignment: The training assignment being served.
        """
        from GenTRX.src.dataloader import OrderDataset, InterleaveSampler
        from GenTRX.src.distributed import train_incremental
        from torch.utils.data import DataLoader
        import pyarrow.parquet as _pq

        parquet_files = self.select_training_files(parquet_files, assignment)
        if not parquet_files:
            self._gtx.tlog.info("select_training_files returned empty list — skipping")
            return

        # Keep only pages with more rows than one training sequence; a shorter
        # page cannot yield a single sample and would build an empty loader.
        good = []
        for f in sorted(parquet_files):
            try:
                if _pq.read_metadata(f).num_rows > self._gtx.train_seq_len:
                    good.append(f)
            except Exception as exc:
                self._gtx.tlog.debug(f"skip page {f}: {exc}")
        if not good:
            self._gtx.tlog.info("no trainable pages, skipping")
            return

        block = int(getattr(self._gtx, "shuffle_block", 1024))
        alpha = float(getattr(self._gtx, "recency_alpha", 1.0))
        max_cached = min(len(good), int(getattr(self._gtx, "train_max_cached", 4)))
        try:
            ds = OrderDataset(
                good,
                seq_len=self._gtx.train_seq_len,
                tokenizer=self._gtx.tokenizer,
                max_cached=max_cached,
            )
        except Exception as exc:
            self._gtx.tlog.info(f"dataset build failed ({exc}), skipping")
            return
        loaders = [
            DataLoader(
                ds,
                batch_size=self._gtx.train_batch_size,
                sampler=InterleaveSampler(ds, shuffle=True, block=block, recency_alpha=alpha),
                num_workers=0,
            )
        ]
        self._gtx.tlog.info(
            f"{len(good)} files, block={block} recency_alpha={alpha} "
            f"max_cached={max_cached}, budget={self._gtx.round_budget_s:.0f}s..."
        )

        # Override only the tunable knobs, e.g.
        #   win_cfg = self._make_window_config(n_steps=500, lr=self._gtx.train_lr * 0.5)
        win_cfg = self._make_window_config()
        delta = train_incremental(train_model, loaders, win_cfg, self._gtx.device)
        self._gtx.tlog.info(
            f"training done: loss {delta.metadata.loss_before:.4f} → {delta.metadata.loss_after:.4f}"
        )

        # Publish surface shared with the base agent — compress + upload + prune
        # + retry-on-failure + window advance. Do not hand-roll this.
        self._submit_gradient(delta, assignment)


if __name__ == "__main__":
    """
    Example launch (paired with the local proxy/simulator in agents/proxy):

        python CustomTrainingAgent.py --port 8888 --agent_id 0 \
            --params gtx_training_enabled=true gtx_collect_data=true
    """
    from taos.common.agents import launch
    launch(CustomTrainingAgent)
