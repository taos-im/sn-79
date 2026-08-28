# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Scikit-learn regressor agent: online-learning finance simulation agent using
PassiveAggressiveRegressor, SGDRegressor, or MLPRegressor for order decisions.
"""

from copy import deepcopy
import glob
import os
import pickle
import bittensor as bt
import pandas as pd
import numpy as np

from bittensor.utils import is_valid_ss58_address

from taos.im.agents.ai import FinanceSimulationAIAgent
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor
from sklearn.neural_network import MLPRegressor


class FinanceSimulationAIRegressorAgent(FinanceSimulationAIAgent):
    """
    Extension of FinanceSimulationAIAgent using standard scikit-learn regressors.

    This agent applies regression models to financial simulation data, supporting online updates,
    checkpointing, pretraining, and MSE-based evaluation.
    """

    def print_config(self):
        """
        Print model and training configuration to the log.
        """
        bt.logging.info(f"""
---------------------------------------------------------------
Strategy Config
---------------------------------------------------------------
Model : {self.model}
Model Parameters :
""" + '\n'.join([f"\t{name} : {val}" for name, val in self.model_kwargs.items()]) + f"""
Checkpoint : {self.checkpoint if self.checkpoint else 'N/A'}
Pretraining : {self.should_pretrain}
Sampling Interval : {self.sampling_interval}s
Training Observations : {self.train_n}
Training Interval : {self.train_interval}
Minimum Training Requirement : {self.min_train_events} runs
Predictors : {self.predKeys}
Targets : {self.targetKeys}
Output Directory : {self.output_dir}
---------------------------------------------------------------""")

    def prepare(self, model: str):
        """
        Prepare the regression agent by selecting the appropriate sklearn model and initializing parameters.

        Args:
            model (str): Model name to use. Supported options include:
                         'PassiveAggressiveRegressor', 'MLP', 'Lasso', 'ElasticNet'.
        """
        self.model_kwargs = {}

        match model:
            case 'PassiveAggressiveRegressor':
                self.module = PassiveAggressiveRegressor
            case 'MLP':
                self.module = MLPRegressor
            case 'Lasso':
                self.module = SGDRegressor
            case 'ElasticNet':
                self.module = SGDRegressor
        self.get_module_parameters(self.module)
        match self.model:
            case 'Lasso':
                self.model_kwargs['penalty'] = 'l1'
            case 'ElasticNet':
                self.model_kwargs['penalty'] = 'elasticnet'
        super().prepare(model)

    def init_model(self, validator : str, book_id : int):
        """
        Instantiate a new model with the configured parameters.

        Args:
            book_id (int): ID of the order book.

        Returns:
            sklearn.base.BaseEstimator: A new model instance.
        """
        return self.module(**self.model_kwargs)

    def load_model(self, checkpoint: str):
        """
        Load a model from disk using pickle.

        Args:
            checkpoint (str): Path to the checkpoint file.

        Returns:
            sklearn.base.BaseEstimator: Loaded model.
        """
        return pickle.load(open(checkpoint, 'rb'))

    def save_model(self, model, location: str):
        """
        Serialize and save the model to disk using pickle.

        Args:
            model: The trained model object.
            location (str): Destination path to save the model.
        """
        try:
            pickle.dump(model, open(location, 'wb'))
            bt.logging.info(f"Model saved at {location}")
        except Exception as e:
            bt.logging.error(f"Failed to save model: {e}")

    def pretrain(self) -> bool:
        """
        Perform pretraining on all available saved data.

        This method:
        - Loads previously recorded features and targets.
        - Trains the model on historical data.
        - Saves a trained checkpoint.
        - Logs the model's evaluation performance (MSE).
        
        Returns:
            bool: Always True (assumes pretraining completes).
        """
        validators = os.listdir(self.output_dir)
        for validator in validators:
            if not is_valid_ss58_address(validator):
                continue
            pretrain_files = glob.glob(self.features_file(validator, '*'))
            for pretrain_file in pretrain_files:
                book_id = os.path.basename(pretrain_file).split('.')[1]
                bt.logging.info(f"Pre-Training Model for Book {book_id}...")

                self.init_book(validator, book_id)

                self.prevX = pd.read_csv(self.features_file(validator, book_id), header=None)
                self.prevY = pd.read_csv(self.targets_file(validator, book_id), header=None)
                self.prevX.columns = self.predKeys
                self.prevY.columns = self.targetKeys

                X_train, X_test, y_train, y_test = train_test_split(
                    self.prevX, self.prevY, test_size=0.1, random_state=42 + int(int(book_id))
                )
                y_train, y_test = np.ravel(y_train), np.ravel(y_test)

                self.models[validator][book_id] = self.models[validator][book_id].fit(X_train, y_train)
                self.save_model(self.models[validator][book_id], self.checkpoint_file(validator, book_id))

                y_pred = self.models[validator][book_id].predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                bt.logging.info(f"Pre-trained Model For Book {book_id} at Validator {validator} saved to {self.checkpoint_file(validator, book_id)} | Mean Squared Error = {mse}")
                self.model_trained[validator][book_id] = True

            return True

    def train(self, validator : str, book_id: int, timestamp: int, test: bool = False) -> bool:
        """
        Train the model on the most recent simulation data.

        If `test` is True, also computes MSE on a single hold-out prediction.

        Args:
            book_id (int): ID of the order book.
            timestamp (int): Current simulation timestamp.
            test (bool): Whether to perform a test evaluation step.

        Returns:
            bool: True if training was successful, False otherwise.
        """
        try:
            if len(self.predictors[validator][book_id][self.predKeys[0]]) < self.train_n + 3:
                bt.logging.info(f"BOOK {book_id} : ONLY {len(self.predictors[validator][book_id][self.predKeys[0]])} / {self.train_n + 3} OBSERVATIONS AVAILABLE FOR TRAINING")
                return False

            X_data = pd.DataFrame(self.predictors[validator][book_id])
            y_data = pd.DataFrame(self.target[validator][book_id])

            if test:
                X_train = X_data.iloc[-self.train_n - 3:-3]
                y_train = np.ravel(y_data.iloc[-self.train_n - 2:-2])
                X_test = X_data.iloc[-2].to_frame().T
                y_test = np.ravel(y_data.iloc[-1].to_frame().T)
            else:
                X_train = X_data.iloc[-self.train_n:-1]
                y_train = np.ravel(y_data.iloc[-self.train_n + 1:])

            model = deepcopy(self.models[validator][book_id])
            model = model.partial_fit(X_train, y_train)

            checkpoint_path = self.checkpoint_file(validator, book_id) + '.new'
            self.save_model(model, checkpoint_path)

            if test:
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                bt.logging.info(f"VALI {validator} BOOK {book_id} MODEL {self.model} : Mean Squared Error = {mse}")

            return True

        except Exception as e:
            bt.logging.error(f"Training failed for book {book_id} at validator {validator}: {e}")
            return False

    def record_data(self, validator : str, book_id: int, data: dict):
        """
        Persist new predictor and target data to CSV files for future training or pretraining.

        Args:
            book_id (int): ID of the order book.
            data (dict): Dictionary with 'predictors' and 'target' entries.
        """
        def append_to_csv(file_path: str, data: dict, include_header: bool = False):
            """Append one row to a CSV file, writing the header first when asked.

            Args:
                file_path (str): Destination file.
                data (dict): Column values for this row.
                include_header (bool): Write the header before the row.
            """
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, mode='a', header=include_header)

        append_to_csv(self.features_file(validator, book_id), data['predictors'])
        append_to_csv(self.targets_file(validator, book_id), data['target'])
