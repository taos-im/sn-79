# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Rayleigh Research

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
"""
Legacy Sharpe-based reward computation (unrealized + realized), retained for
reference. Active scoring uses reward.py (Kappa-3-based).
"""

import torch
import random
import bittensor as bt
import numpy as np
from typing import Dict, Tuple
from taos.im.neurons.validator import Validator
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.utils.sharpe import sharpe, batch_sharpe

def score_uid(validator_data: Dict, uid: int) -> float:
    """
    Calculates the combined unrealized + realized Sharpe score for a specific UID.

    Args:
        validator_data (Dict): Dictionary containing validator state, including
            sharpe_values, activity_factors, activity_factors_realized,
            compact_volumes, compact_roundtrip_volumes, config, simulation_config,
            reward_weights, and simulation_timestamp.
        uid (int): UID of the miner to score.

    Returns:
        float: The combined Sharpe score for the given UID.
    """
    sharpe_values = validator_data['sharpe_values']
    activity_factors = validator_data['activity_factors']
    activity_factors_realized = validator_data['activity_factors_realized']
    compact_volumes = validator_data['compact_volumes']
    compact_roundtrip_volumes = validator_data['compact_roundtrip_volumes']
    config = validator_data['config']['scoring']
    simulation_config = validator_data['simulation_config']
    reward_weights = validator_data['reward_weights']

    simulation_timestamp = validator_data['simulation_timestamp']
    simulation_config['publish_interval']

    if not sharpe_values[uid]:
        return 0.0

    uid_sharpe = sharpe_values[uid]
    sharpes_unrealized = uid_sharpe['books']
    sharpes_realized = uid_sharpe['books_realized']
    
    norm_min = config['sharpe']['normalization_min']
    norm_max = config['sharpe']['normalization_max']
    norm_range = norm_max - norm_min
    norm_range_inv = 1.0 / norm_range if norm_range != 0 else 0.0
    
    normalized_sharpes_unrealized = {
        book_id: max(0.0, min(1.0, (sharpe_val - norm_min) * norm_range_inv))
        for book_id, sharpe_val in sharpes_unrealized.items()
    }
    normalized_sharpes_realized = {
        book_id: (max(0.0, min(1.0, (sharpe_val - norm_min) * norm_range_inv)) if sharpe_val is not None else 0.0)
        for book_id, sharpe_val in sharpes_realized.items()
    }

    volume_cap = round(
        config['activity']['capital_turnover_cap'] * simulation_config['miner_wealth'],
        simulation_config['volumeDecimals']
    )
    volume_cap_inv = 1.0 / volume_cap

    lookback = config['sharpe']['lookback']
    
    decay_grace_period = config['activity'].get('decay_grace_period', 600_000_000_000)
    activity_impact = config['activity'].get('impact', 0.33)
    time_acceleration_power = 2.0
    
    scoring_interval_seconds = config['interval'] / 1e9
    total_intervals = lookback // scoring_interval_seconds
    grace_intervals = decay_grace_period / config['interval']
    decay_window_intervals = total_intervals - grace_intervals
    base_decay_factor = 2 ** (-1 / decay_window_intervals)
    
    decay_window_ns = (lookback * config['interval']) - decay_grace_period
    decay_window_ns_inv = 1.0 / decay_window_ns if decay_window_ns > 0 else 0.0

    compact_volumes_uid = compact_volumes[uid]
    miner_volumes = {book_id: data['lookback_volume'] for book_id, data in compact_volumes_uid.items()}
    latest_volumes = {book_id: data['latest_volume'] for book_id, data in compact_volumes_uid.items()}
    latest_timestamps = {book_id: data['latest_timestamp'] for book_id, data in compact_volumes_uid.items()}
    
    # Calculate the activity factors to be multiplied onto the unrealized Sharpes to obtain the final values for assessment
    # If the miner has traded in the previous Sharpe assessment window, the factor is equal to the ratio of the miner trading volume to the cap multipled by the impact factor
    # If the miner has not traded, their existing activity factor is decayed by the factor defined above, with accelerating decay after the grace period
    activity_factors_uid = activity_factors[uid]
    for book_id, miner_volume in miner_volumes.items():
        if latest_volumes[book_id] > 0:
            activity_factors_uid[book_id] = min(1 + ((miner_volume * volume_cap_inv) * activity_impact), 2.0)
        else:
            latest_time = latest_timestamps[book_id]
            
            if latest_time > 0:
                inactive_time = max(0, simulation_timestamp - latest_time)
            else:
                inactive_time = simulation_timestamp
            
            current_factor = activity_factors_uid[book_id]

            activity_multiplier = max(current_factor, 1.0)

            if inactive_time <= decay_grace_period:
                time_acceleration = 1.0
            else:
                time_beyond_grace = inactive_time - decay_grace_period
                time_ratio = time_beyond_grace * decay_window_ns_inv
                time_acceleration = 1 + (time_ratio ** time_acceleration_power)
            
            total_acceleration = activity_multiplier * time_acceleration
            decay_factor = base_decay_factor ** total_acceleration
            activity_factors_uid[book_id] *= decay_factor
    
    compact_roundtrip_volumes_uid = compact_roundtrip_volumes[uid]
    miner_roundtrip_volumes = {
        book_id: data['lookback_roundtrip_volume'] 
        for book_id, data in compact_roundtrip_volumes_uid.items()
    }
    latest_roundtrip_volumes = {
        book_id: data['latest_roundtrip_volume'] 
        for book_id, data in compact_roundtrip_volumes_uid.items()
    }
    latest_roundtrip_timestamps = {
        book_id: data['latest_roundtrip_timestamp']
        for book_id, data in compact_roundtrip_volumes_uid.items()
    }
    
    # Calculate the activity factors to be multiplied onto the realized Sharpes to obtain the final values for assessment
    # If the miner has traded in the previous Sharpe assessment window, the factor is equal to the ratio of the miner trading volume to the cap multipled by the impact factor
    # If the miner has not traded, their existing activity factor is decayed by the factor defined above, with accelerating decay after the grace period
    activity_factors_realized_uid = activity_factors_realized[uid]
    for book_id, roundtrip_volume in miner_roundtrip_volumes.items():
        if latest_roundtrip_volumes[book_id] > 0:
            activity_factors_realized_uid[book_id] = min(1 + ((roundtrip_volume * volume_cap_inv) * activity_impact), 2.0)
        else:
            latest_time = latest_roundtrip_timestamps[book_id]
            
            if latest_time > 0:
                inactive_time = max(0, simulation_timestamp - latest_time)
            else:
                inactive_time = simulation_timestamp
            
            current_factor = activity_factors_realized_uid[book_id]
            activity_multiplier = max(current_factor, 1.0)

            if inactive_time <= decay_grace_period:
                time_acceleration = 1.0
            else:
                time_beyond_grace = inactive_time - decay_grace_period
                time_ratio = time_beyond_grace * decay_window_ns_inv
                time_acceleration = 1 + (time_ratio ** time_acceleration_power)
            
            total_acceleration = activity_multiplier * time_acceleration
            decay_factor = base_decay_factor ** total_acceleration
            activity_factors_realized_uid[book_id] *= decay_factor
    
    # Calculate activity weighted normalized sharpes
    activity_weighted_normalized_sharpes_unrealized = []
    book_ids_unrealized = []
    for book_id, activity_factor in activity_factors_uid.items():
        norm_sharpe = normalized_sharpes_unrealized[book_id]
        if activity_factor < 1 or norm_sharpe > 0.5:
            weighted = activity_factor * norm_sharpe
        else:
            weighted = (2 - activity_factor) * norm_sharpe
        activity_weighted_normalized_sharpes_unrealized.append(min(weighted, 1))
        book_ids_unrealized.append(book_id)

    activity_weighted_normalized_sharpes_realized = []
    book_ids_realized = []
    for book_id, activity_factor_realized in activity_factors_realized_uid.items():
        norm_sharpe_realized = normalized_sharpes_realized[book_id]
        if activity_factor_realized < 1 or norm_sharpe_realized > 0.5:
            weighted_realized = activity_factor_realized * norm_sharpe_realized
        else:
            weighted_realized = (2 - activity_factor_realized) * norm_sharpe_realized
        activity_weighted_normalized_sharpes_realized.append(min(weighted_realized, 1))
        book_ids_realized.append(book_id)

    uid_sharpe['books_weighted'] = {
        book_id: weighted_sharpe
        for book_id, weighted_sharpe in zip(book_ids_unrealized, activity_weighted_normalized_sharpes_unrealized)
    }
    uid_sharpe['books_weighted_realized'] = {
        book_id: weighted_sharpe
        for book_id, weighted_sharpe in zip(book_ids_realized, activity_weighted_normalized_sharpes_realized)
    }
    
    # Use the 1.5 rule to detect left-hand outliers in the activity-weighted Sharpes    
    data_unrealized = np.array(activity_weighted_normalized_sharpes_unrealized)
    q1_unrealized, q3_unrealized = np.percentile(data_unrealized, [25, 75])
    iqr_unrealized = q3_unrealized - q1_unrealized
    # Apply minimum IQR and scale penalty to reward consistency
    min_iqr = 0.01

    effective_iqr = max(iqr_unrealized, min_iqr)
    lower_threshold_unrealized = q1_unrealized - 1.5 * effective_iqr
    outliers_unrealized = data_unrealized[data_unrealized < lower_threshold_unrealized]
    
    # Outliers detected here are activity-weighted Sharpes which are significantly lower than those achieved on other books
    # A penalty equal to 67% of the difference between the mean outlier value and the value at the centre of the possible activity weighted Sharpe values is calculated
    # Penalty is scaled by consistency: tight clusters (low IQR) get reduced penalty to reward consistent performance
    if len(outliers_unrealized) > 0 and np.median(outliers_unrealized) < 0.5:
        base_penalty = (0.5 - np.median(outliers_unrealized)) / 1.5
        consistency_bonus = 1.0 - np.exp(-5 * iqr_unrealized)  # Sigmoid-like scaling
        outlier_penalty_unrealized = base_penalty * consistency_bonus
    else:
        outlier_penalty_unrealized = 0
    
    # The median of the activity weighted Sharpes provides the base score for the miner
    activity_weighted_normalized_median_unrealized = np.median(data_unrealized)
    # The penalty factor is subtracted from the base score to punish particularly poor performance on any particular book
    sharpe_score_unrealized = max(
        activity_weighted_normalized_median_unrealized - abs(outlier_penalty_unrealized), 
        0.0
    )

    data_realized = np.array(activity_weighted_normalized_sharpes_realized)
    q1_realized, q3_realized = np.percentile(data_realized, [25, 75])
    iqr_realized = q3_realized - q1_realized
    # Apply minimum IQR and scale penalty to reward consistency
    effective_iqr_realized = max(iqr_realized, min_iqr)
    lower_threshold_realized = q1_realized - 1.5 * effective_iqr_realized
    outliers_realized = data_realized[data_realized < lower_threshold_realized]
    # Outliers detected here are activity-weighted Sharpes which are significantly lower than those achieved on other books
    # A penalty equal to 67% of the difference between the mean outlier value and the value at the centre of the possible activity weighted Sharpe values is calculated
    # Penalty is scaled by consistency: tight clusters (low IQR) get reduced penalty to reward consistent performance
    if len(outliers_realized) > 0 and np.median(outliers_realized) < 0.5:
        base_penalty_realized = (0.5 - np.median(outliers_realized)) / 1.5
        consistency_bonus_realized = 1.0 - np.exp(-5 * iqr_realized)
        outlier_penalty_realized = base_penalty_realized * consistency_bonus_realized
    else:
        outlier_penalty_realized = 0
    
    # The median of the activity weighted Sharpes provides the base score for the miner
    activity_weighted_normalized_median_realized = np.median(data_realized)
    # The penalty factor is subtracted from the base score to punish particularly poor performance on any particular book
    sharpe_score_realized = max(
        activity_weighted_normalized_median_realized - abs(outlier_penalty_realized), 
        0.0
    )

    uid_sharpe['activity_weighted_normalized_median'] = activity_weighted_normalized_median_unrealized
    uid_sharpe['penalty'] = abs(outlier_penalty_unrealized)
    uid_sharpe['activity_weighted_normalized_median_realized'] = activity_weighted_normalized_median_realized
    uid_sharpe['penalty_realized'] = abs(outlier_penalty_realized)
    uid_sharpe['score_realized'] = sharpe_score_realized

    balance_ratio_multiplier = 1.0 
    uid_sharpe['balance_ratio_multiplier'] = balance_ratio_multiplier

    sharpe_score_unrealized_adjusted = sharpe_score_unrealized * balance_ratio_multiplier
    uid_sharpe['score_unrealized'] = sharpe_score_unrealized_adjusted    

    combined_score = (
        reward_weights['sharpe'] * sharpe_score_unrealized_adjusted + 
        reward_weights['sharpe_realized'] * sharpe_score_realized
    )
    uid_sharpe['score'] = combined_score
    
    return combined_score

def score_uids(validator_data: Dict, inventory_values: Dict) -> Dict:
    """
    Calculates the new score value for all UIDs by computing both unrealized and realized Sharpe ratios.

    This function orchestrates the Sharpe calculation process:
    1. Extracts inventory history (for unrealized Sharpe)
    2. Extracts realized P&L history (for realized Sharpe)
    3. Calls sharpe() or batch_sharpe() to compute both metrics
    4. Calls score_uid() to combine scores with activity weighting

    Args:
        validator_data (Dict): Dictionary containing validator state with keys:
            - sharpe_values: Storage for Sharpe metrics
            - realized_pnl_history: Realized P&L from completed trades
            - config: Scoring configuration (includes min_realized_observations)
            - uids: List of UIDs to process
            - deregistered_uids: UIDs pending reset
            - simulation_config: Simulation parameters
        inventory_values (Dict): Inventory value history for all UIDs
            Format: {uid: {timestamp: {book_id: value}}}

    Returns:
        Dict: Final inventory scores for all UIDs after combining unrealized and realized Sharpe
            Format: {uid: combined_score}
    """
    config = validator_data['config']['scoring']
    uids = validator_data['uids']
    deregistered_uids = validator_data['deregistered_uids']
    simulation_config = validator_data['simulation_config']
    sharpe_values = validator_data['sharpe_values']
    realized_pnl_history = validator_data['realized_pnl_history']

    if config['sharpe']['parallel_workers'] == 0:
        sharpe_values.update({
            uid: sharpe(
                uid, 
                inventory_values[uid],
                realized_pnl_history.get(uid, {}),
                config['sharpe']['lookback'],
                config['sharpe']['normalization_min'],
                config['sharpe']['normalization_max'],
                config['sharpe']['min_lookback'],
                config['sharpe']['min_realized_observations'],
                simulation_config['grace_period'],
                deregistered_uids
            )
            for uid in uids
        })
    else:
        if config['sharpe']['parallel_workers'] == -1:
            num_processes = len(config['sharpe']['reward_cores'])
        else:
            num_processes = config['sharpe']['parallel_workers']
        batch_size = int(256 / num_processes)
        batches = [uids[i:i+batch_size] for i in range(0, 256, batch_size)]
        sharpe_values.update(batch_sharpe(
            inventory_values,
            realized_pnl_history,
            batches,
            config['sharpe']['lookback'],
            config['sharpe']['normalization_min'],
            config['sharpe']['normalization_max'],
            config['sharpe']['min_lookback'],
            config['sharpe']['min_realized_observations'],
            simulation_config['grace_period'],
            deregistered_uids,
            cores=config['sharpe']['reward_cores'][:num_processes]
        ))

    validator_data['sharpe_values'] = sharpe_values

    uid_scores = {
        uid: score_uid(validator_data, uid)
        for uid in uids
    }
    return uid_scores

def distribute_rewards(rewards: list, config: Dict) -> torch.FloatTensor:
    """
    Distributes rewards using a Pareto distribution to create variance in reward allocation.

    Args:
        rewards (list): List of raw reward scores for each UID
        config (Dict): Configuration dictionary containing rewarding parameters with keys:
            - rewarding.seed (int): Random seed for reproducibility
            - rewarding.pareto.shape (float): Shape parameter for Pareto distribution
            - rewarding.pareto.scale (float): Scale parameter for Pareto distribution

    Returns:
        torch.FloatTensor: Tensor of distributed rewards maintaining the original order of UIDs
    """
    rng = np.random.default_rng(config['rewarding']['seed'])
    num_uids = len(rewards)
    distribution = torch.FloatTensor(sorted(
        config['rewarding']['pareto']['scale'] *
        rng.pareto(config['rewarding']['pareto']['shape'], num_uids)
    ))
    rewards_tensor = torch.FloatTensor(rewards)
    sorted_rewards, sorted_indices = rewards_tensor.sort()
    distributed_rewards = distribution * sorted_rewards
    return torch.gather(distributed_rewards, 0, sorted_indices.argsort())

def get_rewards(validator_data: Dict) -> Tuple[torch.FloatTensor, Dict]:
    """
    Calculate rewards using pre-computed inventory history and compact volumes.

    Args:
        validator_data (Dict): Dictionary containing validator state, compact volumes,
                              and compact round-trip volumes

    Returns:
        Tuple[torch.FloatTensor, Dict]: (rewards, updated_data)
    """
    inventory_history = validator_data['inventory_history']
    simulation_timestamp = validator_data['simulation_timestamp']
    validator_data['simulation_timestamp'] = simulation_timestamp

    if 'activity_factors_realized' not in validator_data:
        validator_data['activity_factors_realized'] = {
            uid: {book_id: 0.0 for book_id in range(len(validator_data['activity_factors'][uid]))}
            for uid in validator_data['uids']
        }
    
    uid_scores = score_uids(validator_data, inventory_history)
    rewards = list(uid_scores.values())
    device = validator_data.get('device', 'cpu')
    distributed_rewards = distribute_rewards(rewards, validator_data['config']).to(device)
    
    updated_data = {
        'sharpe_values': validator_data['sharpe_values'],
        'activity_factors': validator_data['activity_factors'],
        'activity_factors_realized': validator_data['activity_factors_realized'],
        'simulation_timestamp': validator_data['simulation_timestamp'],
    }
    
    return distributed_rewards, updated_data

def set_delays(self: Validator, synapse_responses: dict[int, MarketSimulationStateUpdate]) -> list[FinanceAgentResponse]:
    """
    Applies base delay based on process time using an exponential mapping,
    and adds a per-book Gaussian-distributed random latency instruction_delay to instructions,
    with zero instruction_delay applied to the first instruction per book.

    Args:
        self (taos.im.neurons.validator.Validator): Validator instance.
        synapse_responses (dict[int, MarketSimulationStateUpdate]): Latest state updates.

    Returns:
        list[FinanceAgentResponse]: Delayed finance responses.
    """
    responses = []
    timeout = self.config.neuron.timeout
    min_delay = self.config.scoring.min_delay
    max_delay = self.config.scoring.max_delay
    min_instruction_delay = self.config.scoring.min_instruction_delay
    max_instruction_delay = self.config.scoring.max_instruction_delay

    def compute_delay(p_time: float) -> int:
        """Exponential scaling of process time into delay."""
        t = p_time / timeout
        exp_scale = 5
        delay_frac = (np.exp(exp_scale * t) - 1) / (np.exp(exp_scale) - 1)
        delay = min_delay + delay_frac * (max_delay - min_delay)
        return int(delay)

    for _uid, synapse_response in synapse_responses.items():
        response = synapse_response.response
        if response:
            base_delay = compute_delay(synapse_response.dendrite.process_time)

            seen_books = set()
            for instruction in response.instructions:
                book_id = instruction.bookId

                if book_id not in seen_books:
                    instruction_delay = 0
                    seen_books.add(book_id)
                else:
                    instruction_delay = random.randint(min_instruction_delay, max_instruction_delay)

                instruction.delay += base_delay + instruction_delay

            responses.append(response)
            bt.logging.info(
                f"UID {response.agent_id} responded with {len(response.instructions)} instructions "
                f"after {synapse_response.dendrite.process_time:.4f}s – base delay {base_delay}{self.simulation.time_unit}"
            )

    return responses