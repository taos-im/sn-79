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

import sys
import torch
import argparse
import bittensor as bt
from loguru import logger

from taos.common.config import add_validator_args
from taos.im.config.simulation import add_simulation_args
# taos.im.config.exchange is loaded lazily inside add_im_validator_args when
# engine='exchange' is requested. Not part of this tree; import is guarded.


def _detect_engine_mode() -> str:
    """Pre-scan sys.argv to detect --engine value before the full argparse pass."""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--engine' and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith('--engine='):
            return arg.split('=', 1)[1]
    return 'simulation'


def _flag(value) -> bool:
    """Boolean option that can be turned OFF on the command line: `--x`, `--x true` and the default are
    on, `--x false|0|no|off` is off. A store_true flag with default=True has no off switch."""
    return str(value).strip().lower() not in ('0', 'false', 'no', 'off')


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
        '--benchmark.enabled',
        type=bool,
        default=False,
        help='Enable benchmark agents'
    )

    parser.add_argument(
        '--benchmark.agents',
        type=str,
        default='../config/benchmark_agents.json',
        help='JSON file path with benchmark agent configurations'
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Port number on which to serve validator listener.",
        default=8000,
    )
    
    parser.add_argument(
        "--compression.engine",
        choices=['zlib', 'lz4', 'zstd'],
        help="Compression engine to apply, either `zlib` or `lz4` or `zstd`.",
        default="lz4",
    )
    
    parser.add_argument(
        "--compression.level",
        type=int,
        help="Compression level.",
        default=1,
    )

    parser.add_argument(
        "--compression.parallel_workers",
        type=int,
        help="Number of parallel workers to use in synapse compression. (0 => no parallelization, -1 => auto [half available cores])",
        default=-1,
    )
    
    parser.add_argument(
        "--scoring.interval",
        type=int,
        help="The simulation time interval at which reward calculation is executed.",
        default=5_000_000_000,
    )
    
    parser.add_argument(
        "--scoring.score_ema_halflife",
        type=int,
        help="Half-life in simulation nanoseconds of the per-UID track-record EMA applied to the "
             "trading score BEFORE the reward floor + Pareto allocation. Standing is earned across "
             "multiple windows rather than from one: a single strong window converts into standing "
             "only gradually, and later weak windows forfeit unearned standing — the market "
             "analogue of multi-period track records / deferred compensation. Expressed as sim-time "
             "so it is independent of the scoring cadence; 0 disables. Default equals "
             "scoring.kappa.lookback (the 3h assessment window): the standing's memory horizon "
             "matches the evidence horizon it is built on. Distinct from "
             "neuron.moving_average_alpha, which smooths the POST-Pareto weight signal at the "
             "scoring-cycle scale and does not affect rankings.",
        default=10_800_000_000_000,
    )

    parser.add_argument(
        "--scoring.max_instructions_per_book",
        type=int,
        help="Maximum number of instructions that can be submitted by miners for each book in a single response.",
        default=5,
    )
    
    parser.add_argument(
        "--scoring.max_inactive_books",
        type=float,
        help="Maximum ratio of books that can be neglected without affecting score.  This number of books will be excluded from the scoring calculation (selected as lowest performing).",
        default=0.375,
    )

    # The three trading weights below are the CURRENT LADDER RUNG, set as defaults so a deployment
    # is a file copy. The 0.6.1 de-beta ladder stands at its final rung, (0, 0, 1), the
    # characterised full replacement. At kappa weight 0 the kappa-3 batch is not computed (its
    # carrier is a stub) and an empty de-beta cycle carries the previous map rather than scoring the
    # board 0. Rehearsal rung was (0.79, 0.21, 0.0), rung 1 (0.5925, 0.1575, 0.25), rung 2
    # (0.395, 0.105, 0.50). They must sum to 1, checked at validator init.
    parser.add_argument(
        "--scoring.kappa.weight",
        type=float,
        help="Weight applied to Kappa evaluation in final score calculation",
        default=0.0,
    )

    parser.add_argument(
        "--scoring.kappa.parallel_workers",
        type=int,
        help="Number of parallel workers to use in Kappa-3 calculation. (0 => no parallelization, -1 => auto [half available cores])",
        default=-1,
    )

    parser.add_argument(
        "--scoring.kappa.min_lookback",
        type=int,
        help="Minimum period of observations in simulation nanoseconds required for Kappa calculation.",
        default=5400_000_000_000,
    )

    parser.add_argument(
        "--scoring.kappa.lookback",
        type=int,
        help="Window in simulation nanoseconds of realized P&L observations to use for Kappa-3 ratio calculation.",
        default=10800_000_000_000,
    )

    parser.add_argument(
        "--scoring.kappa.tau",
        type=float,
        help="Threshold return parameter for Kappa-3 calculation (minimum acceptable return per period).",
        default=0.0,
    )
    
    parser.add_argument(
        "--scoring.kappa.min_realized_observations",
        type=int,
        help="The minimum number of realized P&L observations (round-trips) required in the assessment window for Kappa-3 score to be assigned.",
        default=3,
    )

    parser.add_argument(
        "--scoring.kappa.normalization_min",
        type=float,
        help="Kappa-3 values are normalized to fall within a range so as to produce non-negative value and facilitate scoring calculations. This is the minimum value in the normalization range.",
        default=-2.5,
    )

    parser.add_argument(
        "--scoring.kappa.normalization_max",
        type=float,
        help="Kappa-3 values are normalized to fall within a range so as to produce non-negative value and facilitate scoring calculations. This is the maximum value in the normalization range.",
        default=2.5,
    )
    
    parser.add_argument(
        "--scoring.kappa.pnl.impact",
        type=float,
        help="Multiplied onto normalized Kappa-3 values to modify the impact of realized PnL in scoring calculations.",
        default=0.0,
    )

    parser.add_argument(
        "--scoring.pnl.weight",
        type=float,
        help="Weight applied to Realized PnL evaluation in final score calculation",
        default=0.0,
    )

    parser.add_argument(
        "--scoring.debeta.enabled",
        action="store_true",
        help="DEPRECATED and ignored since scoring.debeta.weight became the only dial (weight 0 == the old disabled, plus the decomposition stays published). Parsed so existing launch lines do not crash; removed in 0.6.2",
        default=False,
    )

    parser.add_argument(
        "--scoring.debeta.weight",
        type=float,
        default=1.0,
        help="The de-beta component's share of the FLAT trading score: trading = kappa.weight*kappa + "
             "pnl.weight*pnl + debeta.weight*debeta, the three weights validated to sum to 1 at init. "
             "Default is the current ladder rung (see the kappa.weight comment). At 0.0 emissions are "
             "legacy with the de-beta decomposition always computed and published "
             "(permanent rehearsal visibility). (0, 0, 1) is the characterized full replacement; a "
             "component is computed only when its own weight is nonzero, so that rung also ends the "
             "kappa-3 compute. Migration curve (front-loaded reallocation, no floor-zeroing "
             "below ~0.5): step 0.1 -> 0.25 -> 0.5 -> 1.0, scaling kappa/pnl down proportionally, each "
             "step reversible",
    )
    parser.add_argument(
        "--scoring.debeta.publish_book_gauges",
        type=_flag,
        nargs="?",
        const=True,
        help="Publish the PER-BOOK de-beta gauges (debeta_capture_buy/sell, debeta_book_making, "
             "debeta_alpha) for dashboard drill-down. Default ON for the 0.6.1 ladder: the per-UID "
             "gauges say WHAT a miner scored, these say on WHICH book and WHY (one-sided capture, "
             "alpha under the floor), which is what the ladder gates are read against. The cost is "
             "cardinality uid x book: on a 259-uid 128-book board each gauge is ~33k series, so the "
             "four add ~133k series and roughly 24MB to every /metrics scrape that already runs "
             "154MB. Pass `false` to turn them off where the scrape body is the binding constraint.",
        default=True,
    )

    parser.add_argument(
        "--scoring.debeta.making_pool",
        type=str,
        choices=["rank", "proportional", "proportional_blended", "proportional_both"],
        default="proportional_both",
        help="How the trading pool pays the two de-beta legs. 'proportional_both' (default since 0.6.2): both halves "
             "are paid in proportion. The making half in proportion to each uid's making credit after its counterparty "
             "factor and tether; the skill half in proportion to each uid's net alpha over the books it filled, times "
             "its skill counterparty factor (skill_p11_strength), among uids with positive skill on at least the skill "
             "bar of qualifying books. Proportional pay makes an operator's total equal its share of what it provides, "
             "so splitting a strategy across more uids gains nothing. If no uid qualifies for the skill half, that "
             "half is paid through the Pareto ladder. 'proportional': the making half as above, the skill half through "
             "the ladder. 'proportional_blended': the making half as above, the ladder ranking the full blended score. "
             "'rank': the blended rank score through the ladder, the rule through 0.6.1. Every setting publishes "
             "debeta_making_share and debeta_ladder_input per uid."
    )

    parser.add_argument(
        "--scoring.debeta.w_make",
        type=float,
        help="De-beta operator dial: weight on the making (liquidity) rank vs (1-w_make) on the "
             "drift-stripped skill rank. Launched at 0.30 (skill-led); 0.50 from the second rung of the "
             "ladder so the liquidity leg carries the weight added at that rung.",
        default=0.50,
    )

    parser.add_argument(
        "--scoring.debeta.centered_window",
        type=int,
        help="Half-window (in trades) for the non-lagging centered mid used by the making "
             "spread-capture component.",
        default=15,
    )

    parser.add_argument(
        "--scoring.debeta.floor_scale",
        type=float,
        help="E5 magnitude floor for kappa-of-alpha: a per-book |alpha| must clear "
             "floor_scale*median(|alpha|) to count (kappa is magnitude-blind; kills tiny-consistent "
             "spam). 0 (default since 0.6.2) disables the pool-relative floor in favour of the "
             "size-neutral hurdle (skill_hurdle_bps); 0.5 is the basis shipped through 0.6.1.",
        default=0.0,
    )

    parser.add_argument(
        "--scoring.debeta.min_books",
        type=int,
        help="Activation guard ONLY: if fewer than this many miners receive a positive de-beta score "
             "in a cycle, that cycle scores on the legacy path (warmup / cold-accumulator safety). "
             "It does NOT set the per-miner qualifying-book requirement: the skill leg's own minimum "
             "book count is fixed at 4 inside kappa_of_alpha/kappa_floored and is not configurable.",
        default=4,
    )

    parser.add_argument(
        "--scoring.debeta.p11_strength",
        type=float,
        help="Counterparty-diversity discount on the making leg: making *= (1 - strength * max(0, "
             "excess_concentration)), where excess_concentration is how much of a maker's fills come from its largest "
             "takers beyond those takers' share of everyone else's flow. 0 disables; 1.0, the default, removes the "
             "making credit of a maker served by a dedicated taker and leaves a maker served by the market at large "
             "untouched.",
        default=1.0,
    )

    parser.add_argument(
        "--scoring.debeta.p11_topk",
        type=int,
        help="How many of a maker's largest takers the counterparty concentration measure reads, for both the making "
             "discount and the skill-leg factor. 10, the default. The measure subtracts the same takers' share of "
             "everyone else's flow, so takers common to the whole market cancel and only concentration particular to "
             "the maker counts.",
        default=10,  # 0.6.2 launch value: a many-taker ring is exposed at 10 (2 caught only a dedicated feeder)
    )

    parser.add_argument(
        "--scoring.debeta.s3_k",
        type=float,
        help="Tether on miner-sourced making credit: on the counterparty volumes the P11 factor reads, a maker's "
             "making credit is scaled by (background + min(miner-sourced, k x background)) / total, after the P11 "
             "factor. 4, the default since 1 October 2026 (0.6.3), leaves every maker with at least a fifth of its "
             "maker volume against the background market untouched; a maker served mostly by other miners keeps at "
             "most k times its background share. 0 disables. Market analogue: liquidity programmes pay for liquidity "
             "provided to public order flow, not for flow arranged among participants.",
        default=4.0,
    )

    parser.add_argument(
        "--scoring.debeta.making_basis",
        type=str,
        choices=["captured", "realized"],
        help="Which basis the making pool pays on. captured: the two-sided spread of a uid's maker fills against the "
             "centred mid at the moment of each fill. realized (default since 0.6.3): the same captured credit "
             "multiplied by the uid's markout quality, the share of that spread its maker fills still hold "
             "making_horizon_s of simulation time later, summed over all its fills and held between 0 and 1. Only the "
             "maker's side of a fill is judged, so taking never earns making credit, and nobody is paid more than on "
             "the captured basis. The mark at the horizon is the size-weighted average of the prints with the "
             "background market on at least one side. Both bases are accumulated and published at every setting; this "
             "dial selects which one pays. Market analogue: maker programmes graded on markout.",
        default="realized",
    )

    parser.add_argument(
        "--scoring.debeta.making_horizon_s",
        type=float,
        help="Simulation seconds after a fill at which its realized spread is read for markout quality (the market's "
             "own clock; real time in exchange mode). 20, the default: past about 20 seconds the mark mostly measures "
             "the trend, which the skill half already prices with the drift removed.",
        default=20.0,
    )

    parser.add_argument(
        "--scoring.debeta.making_floor_scale",
        type=float,
        help="Magnitude floor for the MAKING rank, the making-side twin of floor_scale: a uid's "
             "two-sided capture must clear making_floor_scale*median(positive making) to enter the "
             "making rank, otherwise it ranks as 0 (rank is magnitude-blind; a negligible two-sided "
             "quoter ranked beside the field's real makers on testnet). 0, the default, disables it: "
             "on an earlier testnet board it moved the combined Gini from 0.64 to 0.74 for the "
             "removal of four small makers, so it stays off until observed. Skill is always clamped "
             "at 0 before ranking, so only positive directional skill earns skill rank.",
        default=0.0,
    )

    parser.add_argument(
        "--scoring.debeta.skill_rank_scope",
        type=str,
        choices=["positives", "whole"],
        help="How the skill leg is ranked. 'positives' (default): the positive skills are ranked among "
             "themselves, lowest positive 0, highest 1, non-positive 0. 'whole': the clamped skills are "
             "ranked over the whole pool (the earlier rule, kept for rollback), under which the smallest "
             "positive skill inherits the rank of the whole non-positive block, so a negligible skill "
             "collects a mid score and near-zero skills square-wave as their sign flips.",
        default="positives",
    )

    parser.add_argument(
        "--scoring.debeta.making_rank_scope",
        type=str,
        choices=["positives", "whole"],
        help="How the making leg is ranked, the twin of skill_rank_scope. 'positives' (default): the "
             "positive makings are ranked among themselves, lowest positive 0, highest 1, zero making 0. "
             "'whole': ranked over the whole pool (the rule shipped previously, kept for rollback), "
             "under which the smallest positive making inherits the rank of the whole zero-maker block. "
             "Most of the pool makes nothing, so negligible two-sided capture is materially overpaid.",
        default="positives",
    )

    parser.add_argument(
        "--scoring.debeta.presence_gate",
        type=int,
        choices=[0, 1],
        help="De-beta presence gate. 1 (default): a uid whose last presence_window queries all failed "
             "(no HTTP 200) is not scorable that cycle; its legs stay published with present 0 and its "
             "accumulators keep running, so it resumes at full standing when it answers again. 0: off. "
             "Without it, an agent that has stopped answering keeps earning on the timing of resting "
             "fills already inside the scoring window.",
        default=1,
    )

    parser.add_argument(
        "--scoring.debeta.presence_window",
        type=int,
        help="Number of most recent validator queries a uid must have failed in a row to count as absent "
             "for the presence gate; fewer outcomes than this (fresh restart) count as present.",
        default=50,
    )

    # 0.6.2 skill-leg forms. Every default reproduces the 0.6.1 leg exactly; the new quantities are
    # published as gauges at every setting so they can be read on a live board before they score.
    parser.add_argument(
        "--scoring.debeta.skill_variant",
        type=str,
        choices=["drift", "held"],
        help="Which drift strip the skill leg's alpha uses. drift (default): the window's whole drift, "
             "so a position closed inside the window keeps being re-marked by later prints. held: the "
             "drift over the prints on which the agent held inventory, so a closed round trip is scored "
             "once and a single constant position scores exactly zero. The other variant is published "
             "alongside as debeta_skill_held or debeta_skill_drift.",
        default="drift",
    )
    parser.add_argument(
        "--scoring.debeta.skill_subwindows",
        type=int,
        help="Temporal consistency: split the skill window into this many equal sub-windows and assess "
             "skill with skill_subwindow_form. 1 (default) is the plain windowed leg. The three-way "
             "weakest kappa is published as debeta_skill_weakest3 whatever the setting.",
        default=1,
    )
    parser.add_argument(
        "--scoring.debeta.skill_subwindow_form",
        type=str,
        choices=["weakest", "sign_gated"],
        help="weakest: skill is the smallest floored kappa across the sub-windows. sign_gated (default): "
             "a book enters the full-window kappa only when its sub-window alphas agree in sign with its "
             "full-window alpha on at least skill_subwindow_min_agree sub-windows.",
        default="sign_gated",
    )
    parser.add_argument(
        "--scoring.debeta.skill_subwindow_min_agree",
        type=int,
        help="Sub-windows whose alpha sign must agree with the full-window alpha for a book to count under "
             "sign_gated.",
        default=2,
    )
    parser.add_argument(
        "--scoring.debeta.skill_hurdle_bps",
        type=float,
        help="Absolute skill hurdle in basis points of the agent's filled notional on a book: a book's "
             "alpha counts toward skill only above it. 2.3 (default since 0.6.2) reproduces the "
             "membership the 0.5 floor gave on both a young and a mature field and cannot be moved by "
             "anyone else's size; 0 keeps the pool-relative floor alone.",
        default=2.3,
    )
    parser.add_argument(
        "--scoring.debeta.skill_hurdle_books",
        type=int,
        help="Books an agent must clear the hurdle on to stay in the skill pool used by skill_pool_scaling.",
        default=4,
    )
    parser.add_argument(
        "--scoring.debeta.skill_min_books",
        type=int,
        help="Qualifying books the skill leg needs before kappa is computed at all; below it skill is "
             "0. The shipped value 4 was chosen to stop single-book scoring and carries more weight "
             "than that: kappa divides by the median absolute deviation of the per-book alphas, so it "
             "is maximised at its smallest admissible sample, and an agent whose magnitude floor "
             "leaves four survivors out of a hundred-odd traded books scores a consistency ratio on "
             "three per cent of its own evidence. Raising it is a statement about statistical "
             "validity, not about size. Never falls below 4. From 0.6.3 the count is 0 by default and "
             "derives from scoring.debeta.skill_min_books_share against the books being scored; a "
             "positive value here is an explicit override.",
        default=0,  # 0.6.2 launched at 20 on 128 books; 0.6.3 derives it from the share below (0.15625 x 128 = 20)
    )
    parser.add_argument(
        "--scoring.debeta.class_weights",
        type=str,
        help="Per-asset-class emission weights under a multi-asset layout, one per background in document order, comma "
             "separated; normalised to sum 1. Each class's making and skill shares are normalised within the class and "
             "mixed at these weights, so a new class starts with a small share of each half. Empty is the flat rule: a "
             "class weighs whatever share of the field's credit and alpha it produces. Ignored on a single market; on "
             "a layout whose background count differs from the dial's length the scorer uses the flat rule for the "
             "round and warns. '0.95,0.05', the default, is the setting for the shipped two-class layout "
             "(simulation_0, simulation_1).",
        default="0.95,0.05",
    )
    parser.add_argument(
        "--scoring.debeta.skill_min_books_share",
        type=float,
        help="The skill bar as a share of the books the validator scores (the canonical ids under "
             "multi-asset), rounded up and never below the four-book floor. 0.15625 is the 0.6.2 launch "
             "bar of 20 on 128 books; the same count would be 71 per cent of a 28-book layout and "
             "unreachable for a four-book asset class, which is why it is a share. An unknown layout "
             "keeps the launch count of 20.",
        default=20 / 128,
    )

    parser.add_argument(
        "--scoring.debeta.skill_max_inactive_books",
        type=float,
        help="The share of the field's books an agent may carry no qualifying alpha on without penalty: beyond it the "
             "skill leg is scaled by min(1, books filled / ((1 - this) x field books)). 0.375, the default, mirrors "
             "scoring.max_inactive_books; 0 disables. The skill bar (skill_min_books_share) remains the main "
             "constraint; this bounds narrowness, not activity. debeta_skill_coverage_factor publishes the multiplier "
             "per agent.",
        default=0.375,  # 0.6.2 launch value: coverage scaling below 62.5% of the field's books
    )

    parser.add_argument(
        "--scoring.debeta.skill_p11_strength",
        type=float,
        help="The counterparty factor applied to the skill leg: skill *= max(0, 1 - strength x EC+), with EC+ the same "
             "excess counterparty concentration p11_strength applies to making, taken at unit strength so it does not "
             "depend on the making discount. Alpha earned against a concentrated counterparty is discounted as making "
             "credit is; a uid whose fills come from the market at large keeps 1.0. 1.0, the default; 0 disables. "
             "debeta_skill_p11_factor publishes the multiplier per agent.",
        default=1.0,  # 0.6.2 launch value: the counterparty factor reaches the skill leg at full strength
    )

    parser.add_argument(
        "--scoring.debeta.skill_pool_scaling",
        type=int,
        choices=[0, 1],
        help="1: scale the skill leg's rank by the share of the skill pool that clears the hurdle on "
             "skill_hurdle_books books, so an emptying pool pays less rather than the same to fewer. "
             "0 (default): off. Published as debeta_skill_pool_factor either way.",
        default=0,
    )
    parser.add_argument(
        "--scoring.debeta.presence_share_weighting",
        type=int,
        choices=[0, 1],
        help="1: multiply the de-beta score by the agent's share of successful responses over the presence "
             "window. 0 (default): the absent set alone gates. The share is published as "
             "debeta_presence_share either way.",
        default=0,
    )
    parser.add_argument(
        "--scoring.debeta.presence_min_share",
        type=float,
        help="Agents whose success share over the presence window is below this are absent outright. "
             "0 (default): off.",
        default=0.0,
    )

    parser.add_argument(
        "--scoring.debeta.mark_mode",
        type=str,
        choices=["last", "vwap", "median"],
        help="M1 settlement-style marking: value inventory MTM on a rolling reference over the "
             "last mark_window prints instead of the last trade (the settlement-window analogue "
             "used by real venues), so a single manufactured print cannot revalue a position. "
             "'vwap' = volume-weighted mean (movable by one large wash print - measured); "
             "'median' = window median (robust: moving it needs a sustained majority of prints). "
             "Default 'last': last-trade marking, byte-identical to 0.6.0.",
        default="last",
    )

    parser.add_argument(
        "--scoring.debeta.mark_window",
        type=int,
        help="Window length in prints for the rolling settlement mark (mark_mode vwap/median). "
             "Inert while mark_mode=last.",
        default=200,
    )

    parser.add_argument(
        "--scoring.pnl.lookback",
        type=int,
        help="Window in simulation nanoseconds of realized P&L observations used for the "
             "PnL-score component. Independent of scoring.kappa.lookback; defaults to the "
             "same 3h so behaviour is unchanged unless explicitly tuned.",
        default=10800_000_000_000,
    )

    parser.add_argument(
        "--scoring.pnl.normalization.method",
        type=str,
        help="Method for normalizing P&L: 'daily_return'",
        default="daily_return",
    )

    parser.add_argument(
        "--scoring.pnl.normalization.min_daily_return",
        type=float,
        help="Floor for daily return ratio.",
        default=-1.0,
    )

    parser.add_argument(
        "--scoring.pnl.normalization.max_daily_return",
        type=float,
        help="Cap for daily return ratio.",
        default=1.0,
    )

    parser.add_argument(
        "--scoring.gentrx.simulation_share",
        type=float,
        help="Share of miner rewards reserved for GenTRX gradient submitters. The default 0.05 means rewards split "
             "95%% to trading (the de-beta trading score) and up to 5%% to training, scaled by participation (N_active "
             "/ N_registered_miners). The unused training portion returns to trading. When GenTRX is not running, no "
             "gradients are submitted and 100%% of rewards go to trading regardless of this setting.",
        default=0.05,
    )

    parser.add_argument(
        "--scoring.gentrx.ema_alpha",
        type=float,
        help="Per-UID EMA alpha applied inside score_uid to smooth the "
             "rank-normalized gentrx score across rounds before the slow "
             "validator-level moving average. Smaller = more smoothing.",
        default=0.1,
    )

    # ---- GenTRX distributed training ----
    # GenTRX is now HTTP-only — the gradient server runs as a separate process
    # (typically a sibling on the same host for single-machine setups, talking
    # over loopback). All aggregator / scoring / book-distribution tunables
    # live on the standalone gradient server's CLI; the validator side just
    # configures how to reach it.
    parser.add_argument(
        "--gentrx.enabled",
        action="store_true",
        help="Enable GenTRX: push sim state to the gradient server, deliver "
             "assignments to miners via dendrite, expose scores to weight calc.",
        default=False,
    )
    parser.add_argument(
        "--gentrx.gradient_server_url",
        type=str,
        help="Gradient server base URL (e.g. http://127.0.0.1:8100/gentrx for "
             "single-machine setups). REQUIRED when --gentrx.enabled is set.",
        default="",
    )
    parser.add_argument(
        "--gentrx.api_key",
        type=str,
        help="Shared secret for validator↔gradient server auth (also "
             "GENTRX_API_KEY env var). Required when the gradient server "
             "binds to a non-loopback interface.",
        default="",
    )
    parser.add_argument(
        "--gentrx.interval",
        type=int,
        help="Poll interval in seconds for score polls and round cadence in "
             "timer mode (blocks_per_round=0). In block-synced mode, round "
             "cadence is driven by the chain.",
        default=30,
    )
    parser.add_argument(
        "--gentrx.blocks_per_round",
        type=int,
        help="Block-synced round cadence: round = block // blocks_per_round. "
             "The validator derives the round from the chain and pushes "
             "POST /gentrx/round to the gradient server — the server itself "
             "has no block-sync config. Default 25 ≈ 5min at mainnet 12s/block, "
             "matches the 5min training window. Pass 0 for timer mode (proxy only).",
        default=25,
    )
    parser.add_argument(
        "--gentrx.books_per_miner",
        type=int,
        help="Books (pages) assigned per miner per round. More pages = more "
             "data per window (a bigger one-pass batch), less overfit.",
        default=3,
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
        "--scoring.activity.impact",
        type=float,
        help="Multiplied onto activity factors to modify the impact of volume weighting in scoring calculations.",
        default=0.0,
    )
    
    parser.add_argument(
        "--scoring.activity.decay_grace_period",
        type=int,
        help="The period in simulation timesteps for which the decay factor is unaccelerated. After this duration of not trading/round-tripping, activity factor decay accelerates.",
        default=600_000_000_000,
    )
    
    parser.add_argument(
        "--scoring.activity.decay_rate",
        type=float,
        help="Rate of the decay applied to activity factor when no trading has occurred.",
        default=0.0,
    )

    parser.add_argument(
        "--scoring.activity.capital_turnover_cap",
        type=float,
        help="The number of times within each `trade_volume_assessment_period` that miner agents are able to trade the equivalent in volume to their initial capital allocation value before they are restricted from further activity.",
        default=10.0,
    )

    parser.add_argument(
        "--scoring.inventory.min_balance_ratio_multiplier",
        type=float,
        help="The minimum value for the multiplier applied to Kappa-3 scores to penalize holding of small ratio of BASE currency.",
        default=0.5,
    )

    parser.add_argument(
        "--scoring.inventory.max_balance_ratio_multiplier",
        type=float,
        help="The maximum value for the multiplier applied to Kappa-3 scores to reward holding larger ratio of BASE currency.",
        default=1.2,
    )

    parser.add_argument(
        "--scoring.min_delay",
        type=int,
        help="Minimum simulation timestamp delay that may be applied to miner responses.",
        default=10_000_000,
    )

    parser.add_argument(
        "--scoring.max_delay",
        type=int,
        help="Maximum simulation timestamp delay to may be applied to miner responses.",
        default=1000_000_000,
    )

    parser.add_argument(
        "--scoring.min_instruction_delay",
        type=int,
        help="Minimum additive simulation timestamp delay to be applied to subsequent instructions sent in the same response.",
        default=5_000_000,
    )

    parser.add_argument(
        "--scoring.max_instruction_delay",
        type=int,
        help="Maximum additive simulation timestamp delay to be applied to subsequent instructions sent in the same response.",
        default=25_000_000,
    )

    parser.add_argument(
        "--rewarding.seed",
        type=int,
        help="Seed to use in generating distribution for rewards.",
        default=898746039182,
    )

    parser.add_argument(
        "--rewarding.pareto.scale",
        type=float,
        help="Scale parameter for Pareto distribution used in allocating rewards.",
        default=1.0,
    )

    parser.add_argument(
        "--rewarding.pareto.shape",
        type=float,
        help="Shape parameter for Pareto distribution used in allocating rewards. Lower "
             "= steeper payout curve concentrated on top performers. 1.0 concentrated ~51%% of "
             "reward on the top-5 UIDs, over-amplifying whichever strategy currently tops Kappa; "
             "the default spreads that to about a third and restores mid-tier reward at negligible "
             "cost to the highest genuine earners. Sharpen it again once the top of the board is "
             "reliably skill-driven.",
             
        default=1.42,
    )

    parser.add_argument(
        "--rewarding.floor.enabled",
        type=bool,
        help="Enable the soft score floor: taper below-percentile trading scores toward "
             "zero before Pareto allocation, so merely-adequate UID fleets stop earning "
             "and rewards concentrate on genuine performers. Enabled by default; the "
             "taper is gentle enough that a genuinely-improving newcomer clears it well "
             "inside the 24 h immunity window.",
        default=True,
    )

    parser.add_argument(
        "--rewarding.floor.percentile",
        type=float,
        help="Percentile of active (positive) trading scores below which the soft floor "
             "tapers rewards toward zero. Default 50 (median).",
        default=50.0,
    )

    parser.add_argument(
        "--rewarding.floor.softness",
        type=float,
        help="Soft-floor taper width in (0, 1]: scores ramp linearly from 0 at "
             "threshold*(1-softness) up to full at the threshold. Smaller = sharper "
             "(→ hard cliff); 1.0 = gentlest. Default 0.5.",
        default=0.5,
    )

    parser.add_argument(
        "--reporting.disabled",
        action="store_true",
        help="If set, the validator will not publish metrics.",
        default=False,
    )

    parser.add_argument(
        "--neuron.mechid",
        type=int,
        help="Bittensor submechanism ID for weight submission. Defaults to 0 for simulation engine, 1 for exchange engine.",
        default=None,
    )

    parser.add_argument(
        "--engine",
        type=str,
        choices=["simulation", "exchange"],
        help="Validator engine mode: 'simulation' (default) or 'exchange'.",
        default="simulation",
    )

    parser.add_argument(
        "--neuron.fill_notice_window_seconds",
        type=float,
        default=900.0,
        help=(
            "Exchange mode: how long a settled-fill notice is re-sent on every state update, so a miner "
            "that was unreachable when its fill settled still learns of it. There is no acknowledgement, "
            "so a reachable miner receives each fill on every update inside the window; the agent base "
            "drops the repeats by trade id."
        ),
    )

    parser.add_argument(
        "--neuron.fill_notice_max_per_uid",
        type=int,
        default=500,
        help="Exchange mode: cap on parked settled-fill notices per miner inside the redelivery window.",
    )

    parser.add_argument(
        "--neuron.proxy_min_balance_tao",
        type=float,
        default=0.003,
        help=(
            "Exchange mode: the free TAO a miner's settlement proxy wallet must hold for the miner's "
            "placements to be included in a batch at all. A proxy below it has every placement refused "
            "with a notice naming the proxy, its balance and this requirement; cancels still pass. The "
            "default covers one settlement fee (about 0.002) plus the executor's 0.001 reserve. The "
            "executor still checks the exact fee at settlement, so this is the gate, not the ledger. With "
            "PROXY_AUTOFUND_TAO set (testbeds) a proxy closes the gate only after the executor reports a "
            "failed fee pre-flight, since an empty proxy is funded at its first settlement there."
        ),
    )

    parser.add_argument(
        "--neuron.proxy_funding_refresh_seconds",
        type=float,
        default=30.0,
        help=(
            "Exchange mode: how often the validator re-reads every proxy wallet's free balance for the "
            "placement gate. A settlement that fails for want of fee balance marks the proxy unfunded "
            "at once, without waiting for the next read."
        ),
    )

    parser.add_argument(
        "--neuron.observe",
        action="store_true",
        default=False,
        help=(
            "Observe mode: collect chain state and push to the data service for UI display, "
            "but skip all miner queries and weight submission. "
            "Useful for connecting to mainnet/testnet to preview the UI without any chain interaction."
        ),
    )

    # ── Engine-specific arguments (loaded conditionally) ──────────────────────
    engine_mode = _detect_engine_mode()
    if engine_mode == 'exchange':
        try:
            from taos.im.config.exchange import add_exchange_args
        except ImportError as e:
            raise RuntimeError(
                "engine='exchange' requested but taos.im.config.exchange is not "
                "available: exchange-mode engine runs require components that are "
                "not part of this repository."
            ) from e
        add_exchange_args(cls, parser)
    else:
        add_simulation_args(cls, parser)
