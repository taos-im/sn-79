# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Rayleigh Research

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import torch
import argparse
import bittensor as bt
from loguru import logger

from taos.common.config import add_validator_args

def add_im_validator_args(cls, parser):
    """Add validator specific arguments to the parser."""
    add_validator_args(cls, parser)
    
    parser.add_argument(
        "--repo.remote",
        type=str,
        help="Repository remote name.",
        default="origin",
    )
    
    parser.add_argument(
        "--simulation.seed_symbol",
        type=str,
        help="Binance spot market symbol for which price will be used to seed simulation price.",
        default="btcusdt",
    )

    parser.add_argument(
        "--simulation.xml_config",
        type=str,
        help="Path to XML file containing simulation configuration.",
        default="../../../simulate/trading/run/config/simulation_0.xml",
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Port number on which to serve validator listener.",
        default=8000,
    )

    parser.add_argument(
        "--scoring.max_instructions_per_book",
        type=int,
        help="Maximum number of instructions that can be submitted by miners for each book in a single response.",
        default=20,
    )

    parser.add_argument(
        "--scoring.sharpe.lookback",
        type=int,
        help="Number of previous liquidation value observations to use for Sharpe ratio calculation.",
        default=100,
    )

    parser.add_argument(
        "--scoring.sharpe.normalization_min",
        type=float,
        help="Sharpe values are normalized to fall within a range so as to produce non-negative value and facilitate scoring calculations.  This is the minimum value in the normalization range.",
        default=-10.0,
    )

    parser.add_argument(
        "--scoring.sharpe.normalization_max",
        type=float,
        help="Sharpe values are normalized to fall within a range so as to produce non-negative value and facilitate scoring calculations.  This is the maximum value in the normalization range.",
        default=10.0,
    )
    
    parser.add_argument(
        "--scoring.activity.trade_volume_sampling_interval",
        type=int,
        help="The simulation time interval at which miner agent trading volume history is sampled.",
        default=600_000_000_000,
    )
    
    parser.add_argument(
        "--scoring.activity.trade_volume_assessment_period",
        type=int,
        help="The period in simulation timesteps over which agent trading volumes are aggregated when evaluating activity.",
        default=86400_000_000_000,
    )

    parser.add_argument(
        "--scoring.activity.capital_turnover_rate",
        type=float,
        help="The number of times within each `trade_volume_assessment_period` that miner agents must trade the equivalent in volume to their initial capital allocation value in order to receive full rewards for their risk-adjusted performance.",
        default=1.0,
    )

    parser.add_argument(
        "--scoring.max_delay",
        type=int,
        help="Maximum simulation timestamp delay to be applied to miner responses.",
        default=500_000_000,
    )

    parser.add_argument(
        "--scoring.min_delay",
        type=int,
        help="Maximum simulation timestamp delay to be applied to miner responses.",
        default=10_000_000,
    )
    
    parser.add_argument(
        "--reporting.disabled",
        action="store_true",
        help="If set, the validator will not publish metrics.",
        default=False,
    )
