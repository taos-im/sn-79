# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Simple regressor agent: collects market data and predicts future returns using
a scikit-learn online regression model to drive order placement decisions.
"""

import time
import pandas as pd
import numpy as np
import bittensor as bt
from threading import Thread
from collections import defaultdict

from taos.common.agents import launch
from taos.im.agents.ai.regressor import FinanceSimulationAIRegressorAgent
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse

from sklearn.metrics import accuracy_score


class SimpleRegressorAgent(FinanceSimulationAIRegressorAgent):
    """Example: trades a linear-regressor prediction from the shared AI agent base."""
    def print_config(self):
        """Prints the agent's current strategy configuration."""
        bt.logging.info(f"""
---------------------------------------------------------------
Strategy Config
---------------------------------------------------------------
Order Quantity             : {self.quantity}
Order Expiry               : {self.expiry_period}ns
Signal Threshold           : {self.signal_threshold}
Model                      : {self.model}
Model Parameters:
""" + '\n'.join([f"\t{name} : {val}" for name, val in self.model_kwargs.items()]) + f"""
Checkpoint                 : {self.checkpoint if self.checkpoint else 'N/A'}
Pretraining Enabled        : {self.should_pretrain}
Sampling Interval          : {self.sampling_interval}s
Training Observations      : {self.train_n}
Training Interval          : {self.train_interval}
Minimum Training Runs      : {self.min_train_events}
Predictor Features         : {self.predKeys}
Target Features            : {self.targetKeys}
Output Directory           : {self.output_dir}
---------------------------------------------------------------""")

    def initialize(self):
        """
        Initializes the agent by setting configuration values,
        model parameters, and data containers.
        """
        # Quantity of BASE to attempt to buy/sell at each round
        self.quantity = self.config.quantity if hasattr(self.config,'quantity') else 1.0
        # Expiry period for limit orders in simulation nanoseconds
        self.expiry_period = self.config.expiry_period if hasattr(self.config,'expiry_period') else 120e9

        # Define the list of variables which will be used for prediction.  These are populated in the `update_predictors` method.
        # This listing contains several of the most commonly applied features.
        # Note that if modifying the features, you must define the calculation of any new features in `update_predictors`.
        self.predKeys = ['Open', 'High', 'Low', 'Close', 'Volume', 'Direction', 'TradeImbalance', 'OrderImbalance', 'AvgReturn']
        # Define the list of variables which will be predicted by the model.  These are also defined and populated in the `update_predictors` method
        # A common and natural target is to predict the return for the next step.
        self.targetKeys = ['LogReturn']
        
        # Model threshold and signal threshold
        # Signal threshold when the predicition is enough to place the orders
        # Model threshold determines when we trust the model enough, otherwise max the signal at the signal threshold (no action)
        self.signal_threshold = self.config.signal_threshold if hasattr(self.config,'signal_threshold') else 0.0025 
        self.model_threshold = self.config.model_threshold if hasattr(self.config, 'model_threshold') else self.signal_threshold * 2

        # Prepare common variables and execute pretraining if specified
        # Allowed options fpr the `model` parameter for this example strategy are
        # `ElasticNet`, `Lasso`, `PassiveAggressiveRegressor`, `MLP`
        # In the first two cases the selected model is used to apply penalty during Stochastic Gradient Descent for an SGD regressor model
        self.model = self.config.model if hasattr(self.config,'model') else "MLP"
        self.prepare(self.model)

        # Initialize storage for prediction and analysis
        self.predictors = {}
        self.target = {}
        self.last_signal = {}
        self.midquotes = {}
        self.signs = {}
        self.trueSigns = {}
        self.errors = {}
        self.book_event_history : dict[str, EventHistory | None] = {}

    def update_predictors(self, validator : str, book: Book, timestamp: int) -> None:
        """
        Gathers and processes historical event data to update predictors and targets.
        Features are computed based on sampling intervals.

        Args:
            book (Book): Book object from the state update.
            timestamp (int): Simulation timestamp of the associated state update.
        """
        if validator not in self.book_event_history or not self.book_event_history[validator]:
            lookback_minutes = max(
                (self.simulation_config.publish_interval // 1_000_000_000) // 60,
                self.sampling_interval * 2 // 60,
                1
            )
            self.book_event_history[validator] = book.event_history(timestamp, self.simulation_config, lookback_minutes)
        else:
            book.append_to_event_history(timestamp, self.book_event_history[validator], self.simulation_config)

        interval = self.sampling_interval

        # Aggregate data
        ohlc = self.book_event_history[validator].ohlc(interval)
        mean_price = self.book_event_history[validator].mean_trade_price(interval)
        trade_buckets = self.book_event_history[validator].bucket(self.book_event_history[validator].trades, interval)
        order_buckets = self.book_event_history[validator].bucket(self.book_event_history[validator].orders, interval)

        trade_volumes = {ts: sum(t.quantity for t in trades) for ts, trades in trade_buckets.items()}
        trade_imbalance = {
            ts: sum(t.quantity if t.side == OrderDirection.BUY else -t.quantity for t in trades)
            for ts, trades in trade_buckets.items()
        }
        order_volumes = {ts: sum(o.quantity for o in orders) for ts, orders in order_buckets.items()}
        order_imbalance = {
            ts: sum(o.quantity if o.side == OrderDirection.BUY else -o.quantity for o in orders)
            for ts, orders in order_buckets.items()
        }

        n_new = max((self.simulation_config.publish_interval // 1_000_000_000) // interval, 1)
        latest_timestamps = sorted(ohlc.keys())[-n_new:]

        new_predictors = defaultdict(list)
        new_targets = defaultdict(list)

        for ts in latest_timestamps:
            ohlc_data = ohlc.get(ts)
            if not ohlc_data:
                continue

            open_, high, low, close = ohlc_data.values()
            mean = mean_price.get(ts)
            t_vol = trade_volumes.get(ts, 0)
            t_imb = trade_imbalance.get(ts, 0)
            o_vol = order_volumes.get(ts, 0)
            o_imb = order_imbalance.get(ts, 0)

            new_predictors['Open'].append(open_)
            new_predictors['High'].append(high)
            new_predictors['Low'].append(low)
            new_predictors['Close'].append(close)
            new_predictors['Volume'].append(o_vol)
            new_predictors['Direction'].append(np.sign(close - open_))
            new_predictors['TradeImbalance'].append(t_imb / t_vol if t_vol else 0.0)
            new_predictors['OrderImbalance'].append(o_imb / o_vol if o_vol else 0.0)
            new_predictors['AvgReturn'].append(1 - mean / high if mean and high else 0.0)

            new_targets['LogReturn'].append(np.log(close / open_) if open_ > 0 else 0.0)

        book_id = book.id
        self.predictors[validator][book_id] = {
            k: self.predictors.get(validator, {}).get(book_id, {}).get(k, []) + new_predictors[k] for k in self.predKeys
        }
        self.target[validator][book_id] = {
            k: self.target.get(validator, {}).get(book_id, {}).get(k, []) + new_targets[k] for k in self.targetKeys
        }

        max_len = self.train_n + 3
        self.predictors[validator][book_id] = {k: v[-max_len:] for k, v in self.predictors[validator][book_id].items()}
        self.target[validator][book_id] = {k: v[-max_len:] for k, v in self.target[validator][book_id].items()}

        self.record_data(validator, book_id, {
            "predictors": dict(new_predictors),
            "target": dict(new_targets)
        })

    def signal(self, predictions: dict[str, float]) -> float:
        """
        Converts model predictions into a trading signal.
        Caps predictions to avoid overconfidence when the model is undertrained.

        Args:
            predictions (dict[str, float]): Dictionary mapping target name to the latest predicted value.
        """
        signal = predictions['LogReturn']
        if abs(signal) > self.model_threshold:
            signal = self.signal_threshold * np.sign(signal)
            bt.logging.info("Warning: Prediction magnitude exceeded threshold — more training required.")
        return signal

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """
        Called upon new market state updates. The core strategy loop:
        - Extracts market data
        - Updates predictors
        - Predicts returns
        - Places limit orders based on predicted signal

        Args:
            state (taos.im.protocol.MarketSimulationStateUpdate): The class representing the latest state of the simulation.

        Returns:
            taos.im.protocol.FinanceAgentResponse : The response which will be attached to the synapse for return to the querying validator.
        """
        response = self.make_response()  # mode-aware: emits exchange or simulation instructions
        start = time.time()

        for book_id, book in state.books.items():
            bestBid = book.bids[0].price if book.bids else 0.0
            bestAsk = book.asks[0].price if book.asks else bestBid + 10 ** (-self.simulation_config.priceDecimals)
            midquote = (bestBid + bestAsk) / 2            

            if state.dendrite.hotkey not in self.predictors:
                self.predictors[state.dendrite.hotkey] = {}
                self.target[state.dendrite.hotkey] = {}
                self.last_signal[state.dendrite.hotkey] = {}
                self.midquotes[state.dendrite.hotkey] = {}
                self.signs[state.dendrite.hotkey] = {}
                self.trueSigns[state.dendrite.hotkey] = {}
                self.errors[state.dendrite.hotkey] = {}
            # Initialization for unseen books
            if book_id not in self.predictors[state.dendrite.hotkey]:
                self.predictors[state.dendrite.hotkey][book_id] = {key: [] for key in self.predKeys}
                self.target[state.dendrite.hotkey][book_id] = {key: [] for key in self.targetKeys}
                self.last_signal[state.dendrite.hotkey][book_id] = 0.0
                self.midquotes[state.dendrite.hotkey][book_id] = 0.0
                self.signs[state.dendrite.hotkey][book_id] = []
                self.trueSigns[state.dendrite.hotkey][book_id] = []
                self.errors[state.dendrite.hotkey][book_id] = []
                self.init_book(state.dendrite.hotkey, book_id)

            self.update_predictors(state.dendrite.hotkey, book, state.timestamp)

            if not self.model_trained[state.dendrite.hotkey][book_id] or len(self.predictors[state.dendrite.hotkey][book_id][self.predKeys[0]]) < 1:
                bt.logging.info(f"BOOK {book_id}: Training Progress {self.trained_events[state.dendrite.hotkey][book_id]}/{self.min_train_events}")
                continue

            # Prepare latest predictor sample
            predictors = {key: self.predictors[state.dendrite.hotkey][book_id][key][-1] for key in self.predKeys}
            X = pd.DataFrame(predictors, index=[0])

            # Make predictions
            predictions = dict(zip(self.targetKeys, self.models[state.dendrite.hotkey][book_id].predict(X)))
            signal = self.signal(predictions)

            bt.logging.info(f"BOOK {book_id}: PREDICTION " +
                            ", ".join([f"{k}: {v:.4f}" for k, v in predictions.items()]) +
                            f" | SIGNAL {signal:.4f}")

            # Track performance
            if self.midquotes[state.dendrite.hotkey][book_id] != 0.0:
                curr_return = np.log(midquote / self.midquotes[state.dendrite.hotkey][book_id])
                self.errors[state.dendrite.hotkey][book_id].append((self.last_signal[state.dendrite.hotkey][book_id] - curr_return) ** 2)
                self.signs[state.dendrite.hotkey][book_id].append(np.sign(self.last_signal[state.dendrite.hotkey][book_id]))
                self.trueSigns[state.dendrite.hotkey][book_id].append(np.sign(curr_return))
                bt.logging.info(
                    f"VALIDATOR {state.dendrite.hotkey} | "
                    f"BOOK {book_id}: RETURN {curr_return:.4f} | "
                    f"MSE: {np.mean(self.errors[state.dendrite.hotkey][book_id]):.4f} | "
                    f"ACC: {accuracy_score(self.trueSigns[state.dendrite.hotkey][book_id], self.signs[state.dendrite.hotkey][book_id]):.2f}"
                )

            self.last_signal[state.dendrite.hotkey][book_id] = signal
            self.midquotes[state.dendrite.hotkey][book_id] = midquote

            # Trading logic
            if signal > self.signal_threshold:
                # If the signal is positive, firstly place a buy order just above the current best bid level
                response.limit_order(
                    book_id,
                    OrderDirection.BUY,
                    self.quantity,
                    round(bestBid + 10**(-self.simulation_config.priceDecimals), self.simulation_config.priceDecimals),
                    timeInForce=TimeInForce.GTT, expiryPeriod=self.expiry_period
                )
                # Place a sell order with distance from midquote proportional to the strength of the prediction
                response.limit_order(
                    book_id,
                    OrderDirection.SELL,
                    self.quantity,
                    round(midquote*np.exp(signal),self.simulation_config.priceDecimals),
                    timeInForce=TimeInForce.GTT, expiryPeriod=self.expiry_period
                )
            elif signal < -1* self.signal_threshold:
                # If the signal is negative, firstly place a sell order just below the current best ask level
                response.limit_order(
                    book_id,
                    OrderDirection.SELL,
                    self.quantity,
                    round(bestAsk - 10**(self.simulation_config.priceDecimals),self.simulation_config.priceDecimals),
                    timeInForce=TimeInForce.GTT, expiryPeriod=self.expiry_period
                )
                # Place a buy order with distance from midquote proportional to the strength of the prediction
                response.limit_order(
                    book_id,
                    OrderDirection.BUY,
                    self.quantity,
                    round(midquote*np.exp(signal),self.simulation_config.priceDecimals),
                    timeInForce=TimeInForce.GTT, expiryPeriod=self.expiry_period
                )

            # Launch asynchronous model update
            Thread(target=self.update_model, args=(state.dendrite.hotkey, book_id,)).start()

        bt.logging.info(f"Response Generated in {time.time() - start:.2f}s")
        return response


if __name__ == "__main__":
    """
    Example command for local standalone execution:
    python SimpleRegressorAgent.py --port 8888 --agent_id 0 --params quantity=10.0 expiry_period=120000000000 model=PassiveAggressiveRegressor sampling_interval=1 train_periods=60 train_n=100
    """
    launch(SimpleRegressorAgent)
