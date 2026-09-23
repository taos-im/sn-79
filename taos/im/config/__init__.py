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
        default="rank",
        help="How the making leg's share of emission (debeta.weight * debeta.w_make) is paid. "
             "'rank' (default, the shipped behaviour): the making rank enters the blended score and "
             "the whole score goes through the Pareto sort-multiply. 'proportional': the making leg "
             "comes out of the ladder and its share is paid in proportion to each uid's captured "
             "spread, with the ladder kept for the skill leg. The ladder pays rank POSITIONS with a "
             "steep top, so an operator whose accounts occupy the top positions collects many "
             "top-of-curve weights whatever its aggregate service; seen on a mainnet board where "
             "2026, one coldkey with 35 per cent of the board's captured spread took 64 per cent of "
             "emission. Proportional pay makes an operator's total equal its share of the liquidity "
             "actually provided, so splitting a strategy across more uids gains nothing and no "
             "identity rule is needed. 'proportional_blended': the same pool on the same shares, "
             "but the ladder keeps ranking the full blended score, so an account with no maker "
             "volume is capped by its making rank as it is under 'rank'. Plain 'proportional' leaves "
             "the ladder on the skill leg alone, which at w_make 0.50 makes half of emission a pot "
             "decided on skill only; on twelve mainnet boards of 22 September 2026 zero-maker "
             "accounts took 42.1 per cent of emission under it, against 10.4 with the pool off and "
             "5.2 under the blended setting. The blended setting's cost is that making is paid on "
             "both surfaces, so the largest operator takes 59.5 per cent against 26.9 under plain "
             "proportional, and cloning the same capture across 16 uids gains 1.18x against 1.04x "
             "(4.00x under 'rank'). 'proportional_both': both halves additive. The making half as "
             "under 'proportional'; the skill half in proportion to each uid's net alpha over the books "
             "it filled, times its counterparty factor (skill_p11_strength), among uids with positive "
             "skill on at least skill_min_books qualifying books. The skill ladder over kappa is a "
             "tournament: kappa is magnitude-blind, so one predictor split sixteen ways gained 14.03x on "
             "real alphas (22 September 2026); net alpha is additive, so this share is cloning-invariant "
             "by construction (1.00x measured). Real markets pay traders on realised P&L and let "
             "consistency decide who is allocated capital. Falls back to the ladder for the skill half "
             "when nobody is eligible. All settings publish debeta_making_share and "
             "debeta_ladder_input per uid, so the alternative is visible before it is enabled. "
             "Switching is announced and dated on the scoring page.",
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
             "spam). 0 disables the floor.",
        default=0.5,
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
        help="P11 counterparty-diversity discount strength on the making leg: making *= "
             "(1 - strength*max(0,excess_concentration)). 0 disables. 1.0, the default, fully removes a "
             "dedicated-feeder maker's making credit; a diverse maker is untouched. Closes the E3 "
             "sacrificial-feeder hole in the making metric.",
        default=1.0,
    )

    parser.add_argument(
        "--scoring.debeta.p11_topk",
        type=int,
        help="How many of a maker's largest takers the P11 excess-concentration measure looks at, for "
             "both the making discount and the skill-leg factor. 2, the default, is the shipped scorer "
             "and catches a dedicated feeder. A ring that spreads the same feeding over ten or twenty "
             "takers reads as diverse at 2 (each feeder a few per cent of the maker's fills) and is "
             "exposed at 10 or more, because the measure subtracts the same takers' share of everyone "
             "else's flow: common takers cancel, private feeders do not. On a later mainnet "
             "2026 mainnet tape at 10: fed makers 0.25 to 0.68, honest makers 0.90 to 0.97.",
        default=10,  # 0.6.2 launch value: a many-taker ring is exposed at 10 (2 caught only a dedicated feeder)
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
             "alpha counts toward skill only above it. 0 (default) keeps the pool-relative floor alone.",
        default=0.0,
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
             "validity, not about size. Never falls below 4.",
        default=20,  # 0.6.2 launch value (23 Sep 2026): the skill leg needs a real sample; 4 is the mechanism's own floor
    )

    parser.add_argument(
        "--scoring.debeta.skill_max_inactive_books",
        type=float,
        help="The skill leg's twin of scoring.max_inactive_books: the share of the field's books an "
             "agent may carry no qualifying alpha on without penalty. Below that its skill is scaled "
             "by coverage / (1 - this) * field books, before ranking. kappa is normalised by the "
             "median absolute deviation of the per-book alphas, so it is maximised at its smallest "
             "admissible sample and kappa_floored's four-book minimum is a floor to sit on rather "
             "than a bar to clear. The kappa leg pads neglected books into its average as zero; that "
             "shape cannot be reused here because padding a MAD-normalised ratio saturates it "
             "instead of collapsing it, so the penalty is multiplicative. The counts are books the "
             "agent FILLED, pre-floor, because the post-floor count falls with size per book rather "
             "than with breadth. Presence in a book is cheap, so this bounds narrowness and not "
             "activity; skill_min_books is the load-bearing constraint. 0 (default) disables and "
             "reproduces the 0.6.1 leg exactly. debeta_skill_coverage_factor publishes the "
             "multiplier per agent either way.",
        default=0.375,  # 0.6.2 launch value: coverage scaling below 62.5% of the field's books
    )

    parser.add_argument(
        "--scoring.debeta.skill_p11_strength",
        type=float,
        help="The counterparty factor applied to the SKILL leg: skill *= max(0, 1 - strength * EC+), "
             "with EC+ the same excess top-2 counterparty concentration p11_strength discounts making "
             "by, taken at unit strength so this does not depend on the making discount being on. "
             "Feeding a maker at chosen prices manufactures per-book alpha on the fed side, and on 22 "
             "September 2026 that is where a same-operator feeding ring was paid: fed makers with making "
             "ranks of 0.003 to 0.36 held skill ranks of 0.94 to 1.00 on four to nine books, 6.75 per "
             "cent of incentive. Against real alphas the factor left every genuine skill account at "
             "exactly 1.0 and moved no uid outside the pattern by more than 0.20 of rank; it is "
             "cloning-invariant because it is a property of the flow, not the account count. Only partly "
             "effective on the rank ladder (kappa 9.9 x 0.25 is still top-decile), fully effective under "
             "making_pool=proportional_both, which is not to be enabled without it. Calibrated on the "
             "simulation, where background agents make concentration visible; exchange mode needs its "
             "own calibration as P11 on making did. 0 (default) disables. debeta_skill_p11_factor "
             "publishes the multiplier per agent either way.",
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
        help="Share of miner rewards reserved for GenTRX gradient submitters. "
             "The default 0.05 means rewards split 95%% to trading "
             "(kappa+pnl) and up to 5%% to training, scaled by participation "
             "(N_active / N_registered_miners). The unused training portion "
             "returns to trading. When GenTRX is not running, no gradients "
             "are submitted and 100%% of rewards go to trading regardless "
             "of this setting.",
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
             "= steeper payout curve concentrated on top performers. 1.0 concentrated ~51% of "
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
