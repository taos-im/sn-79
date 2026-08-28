<div align="center">

# **MVTRX**: Bittensor SN79<!-- omit in toc -->
### **Decentralized Simulation of Automated Trading in Intelligent Markets:** <!-- omit in toc -->
### **Risk-Averse Agent Optimization** <!-- omit in toc -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
---
# Frequently Asked Questions

</div>

#### 1. How does MVTRX differ from other finance-related subnets?

Other finance-related subnets, at least to our knowledge at time of writing, focus on incentivizing the creation and deployment of trading strategies which act against particular real-world markets, and seek to extract value from the trading signals produced by miners.  While this approach has some promise, τaos has more general aspirations to provide value across a broad spectrum of use cases within the financial industry.  By providing an environment where miners trade in many statistically similar but independently evolving simulated markets simultaneously, we not only encourage the study and development of much more robust trading strategies, but also produce vast quantities of high-resolution, maximally detailed data which can be used by traders, researchers, institutions and regulators to better understand and account for the underlying risks present in all markets.

#### 2. How are miners evaluated?

While the exact details of the incentive mechanism are subject to change over time, the key objective for miners is always to maximize average _risk-adjusted_ performance over time across all simulated orderbooks.  The central metric applied in measuring performance is the Kappa-3 ratio, which measures risk-adjusted returns based on realized profits and losses from completed round-trip trades.  Kappa-3 is defined as K₃(τ) = (μ - τ) / [LPM₃(τ)]^(1/3), where μ is the mean return, τ is the target threshold (typically zero), and LPM₃ is the third lower partial moment measuring downside risk.  This metric emphasizes consistent profitability while heavily penalizing downside volatility, making it particularly suitable for evaluating trading strategies.

The returns for the Kappa-3 calculation are based on actual realized P&L from closed positions (completed round-trip trades), and the measurement is taken over a rolling window of length defined in the [validator config](/taos/im/config/__init__.py) as `--scoring.kappa.lookback`.  Note that the actual simulation time period of the assessment is related to this through the `Simulation.step` in the [simulation config](/simulate/trading/run/config/simulation_0.xml) - a new observation is obtained every `Simulation.step` simulation nanoseconds.

There are then a few additional transformations applied to avoid manipulation and encourage active trading; miners' scores are scaled in proportion to their round-trip trading volume over the assessment window, with a decay factor being applied during any inactive periods.  A penalty is applied to miners scores in cases where significant outliers in terms of performance exist between the simulated books.  The full details of the implementation can be understood by studying the [reward logic](/taos/im/validator/reward.py).  The overall score of a miner is determined as an exponential moving average of the score calculated at each observation, with the period of the EMA being set via the `--neuron.moving_average_alpha` parameter which is applied in the [base validator logic](/taos/common/neurons/validator.py).

#### 3. How do I get started mining in the subnet?

This FAQ is a good entry point, after which it is recommended to go through the [README](/README.md).  Once you are familiar with the subnet function and vision, and decide you want to get involved, you can check out the [agents readme](/agents/README.md) for more detailed information on how to design and develop strategies for the τaos framework.  Before getting into mining on testnet or mainnet, we recommend also to set up a local testing environment using the ["proxy" validator](/agents/proxy/README.md) tools which enable to launch a local instance of the simulation engine and confirm how your strategy behaves and performs against the background market.

Once you have developed and tested your strategy locally, you may next wish to inspect the current behaviour and performance of existing miners in the subnet via the [taos.simulate.trading dashboard](https://taos.simulate.trading) (an updated version of this dashboard as well as documentation to assist in interpreting the visualizations are upcoming).  To get an idea of the expected performance of your strategy and validate your hosting and networking configuration, you can request testnet TAO via the [Bittensor Discord](https://discord.com/channels/799672011265015819/1389370202327748629), register a UID in our test netuid 366 and deploy and [monitor](https://testnet.simulate.trading) your miner here.  We continuously run a validator in testnet using the latest code and configurations.  Once you are confident that your agent has what it takes, register to mainnet and join the competition!  If you have any questions or concerns, reach out to us at our [Discord Channel](https://discord.com/channels/799672011265015819/1353733356470276096).

**Note that the example miners given in the /agents directory are not expected to perform well in the subnet - you need to develop a smart custom algorithm to compete**

#### 4. Is there an MVTRX testnet?

Yes, netuid 366 with monitoring via [testnet.simulate.trading](https://testnet.simulate.trading).

The Bittensor test network carries no exchange, so netuid 366 is not the place to exercise trading.
For that there is a public localnet at `wss://localnet.mvtrx.exchange:443`, which runs the exchange
and is open for testing. Point the run scripts at it with `-e wss://localnet.mvtrx.exchange:443`.

#### 5. How do I monitor my miner?

Data reflecting the current state of the simulation, including miner performance and activity, is published by validators at [taos.simulate.trading](https://taos.simulate.trading).  You can access detailed information for your specific miner by clicking on your UID in the "Agent" column of the "Agents" table - this view reports detailed metrics related to your agent's behaviour and performance.  An identical dashboard for testnet is available at [testnet.simulate.trading](https://testnet.simulate.trading).

#### 6. What is the update schedule and procedure for the subnet?

We target to release an update every week on Wednesday.  The content of the update will be announced ahead of time, usually earlier in the week or latest earlier in the day on Wednesday, with code being pushed to repo and deployed to testnet around 15:00 UTC.  Assuming no issues, the new version is deployed to mainnet at around 17:00 UTC.

In most cases the updates do not require changes from miners, in situations where the code changes require miners to update this will be clearly communicated.  Changes having a wider impact will be announced further ahead of time and run on testnet for an extended period to ensure a smooth deployment to live.

#### 7. What is "simulation time" and why is it different from real world time?

Being that our markets are synthetic and generated through a powerful C++ engine which creates many statistically realistic limit order books simultaneously, the simulation maintains an internal "clock" which tracks to nanosecond precision the time elapsed since the start of simulation in a manner consistent with the evolution.  Due mostly to constraints related to the requirement to query miners with the latest state information for all books and await responses at regular intervals, the time in the simulation typically progresses more slowly than actual time - that is, over 1 hour of real time, the simulation may only progress by 20 minutes (though it should be kept in mind that this represents the evolution of many simulated books over a 20 minute period).  We work to reduce this discrepancy and decrease time taken in the query process, while we have already started planning the next iteration of the subnet which would enable real-time communication between miners and validators, eliminating this query process delay entirely.

#### 8. My miner seems to be receiving requests and responding, but I don't see any activity and my score is not increasing.  What's going on?

If you have just registered the miner, note that scores are not assigned until sufficient time has passed to allow calculating a meaningful Kappa-3 ratio with enough realized trades.  If the situation persists, you will need to check your UID at the [Agents Dashboard](https://taos.simulate.trading/d/edy6vxytuud4wd/agents) and confirm a few critical things:

- Do you see recent trades for all book IDs?  Miners must trade on every book in order to receive score.
- Under the Requests plot, do you see a large proportion of failures or timeouts?  If you are not seeing mostly success, usually this is due to taking too long to respond - validators allow a maximum of `--neuron.timeout` seconds (defined in the [base validator config](/taos/common/config/__init__.py)) for miners to respond.  This can be addressed by increasing resources, optimizing your strategy logic and ensuring sufficient network connectivity; you may also want to consider geolocating your miner nearby to the biggest validators for the best possible latency.
- Pay attention to the Kappa-3 Score and Kappa Penalty, these are the primary metrics used in determining miner score.

#### 9. As a miner, I've hit the trading volume limit and can no longer submit instructions.  How is this limit enforced and what can I do now?

In order to prevent miners from attempting to exploit the volume-weighting of scores and overloading the simulations with excessive careless trading activity, a "cap" is enforced on the total QUOTE volume allowed to be traded in a given period of simulation time.  Trading volume is calculated in QUOTE as the sum of price multiplied by quantity over all trades in which the miner is involved.  The period over which the trading volume limit is assessed is defined in the [validator config](/taos/im/config/__init__.py) as `--scoring.activity.trade_volume_assessment_period` (specified in simulation nanoseconds), where this is checked every `--scoring.activity.trade_volume_sampling_interval` (simulation nanoseconds) against the limit which is calculated as `--scoring.activity.capital_turnover_cap` multiplied by the initial wealth allocated to miners defined in the [simulation config](/simulate/trading/run/config/simulation_0.xml) as `Simulation.Agents.MultiBookExchangeAgent.Balances.wealth`.  

If a miner trades more than `Simulation.Agents.MultiBookExchangeAgent.Balances.wealth` * `--scoring.activity.capital_turnover_cap` in a period of `--scoring.activity.trade_volume_assessment_period` simulation nanoseconds, no more instructions will be accepted on that book (except cancellations) until the total QUOTE volume traded in the most recent `trade_volume_assessment_period` drops below the limit.  Miners need to consider this limitation when designing and testing strategies in order to maximize volume and Kappa-3 ratio without exceeding the limit.  If the cap is hit early in the first 24 hours, there is a high risk of deregistration as no further actions will be possible until at least 24 simulation hours have elapsed.  Your current total traded volume is included in the state update for easy reference, accessible in code via `self.accounts[book_id]['traded_volume]`.

Note also that the trading volumes used in the assessment are not reset when a new simulation begins; the trading volumes are determined based on a 24 hour period which may span multiple simulations.

#### 10. Why do validators in the subnet exhibit discrepancies in vTrust?

While we have made changes to attempt to better align the weights assigned to miners by different validators, the nature of the subnet operation does require that there exist a group of miners which consistently outperform others across all simulated markets in order that validators would agree on the top scoring UIDs.  The discrepancies result from a combination of inconsistent performance by miners, networking considerations leading to different success rates and latencies between miners and validators, and the fact that the simulation hosted by each validator is different from others due to the stochastic nature of the background model.  We continue to seek ways to improve this situation, but have to consider also the impact of these changes in relation to the utility of the subnet - if all validators are evaluating the same exact conditions, this eliminates a key element guaranteeing the robustness of the top scoring miners to perform well in all environments.

#### 11. Do you currently burn any miner emissions, or have any plans to implement this mechanism?

No, we do not burn miner emissions and do not currently have any plans to implement this.  Though this seems it may make sense in some other subnets, we do not see that this would be the case for us.  If in future we see need to apply such, this will not be done without careful consideration and consultation with all participants.

#### 12. Why does the current scoring system seem to favor passive market making over active trading?

The transition from inventory-based Sharpe ratios to realized P&L-based Sharpe and then Kappa-3 ratios addresses this concern directly. Kappa-3 measures performance based on actual completed round-trip trades, which inherently accounts for all trading costs including fees, spreads, and slippage. Standing limit orders that never get filled do not contribute to the Kappa-3 calculation, eliminating the advantage previously seen by passive strategies with stable but inactive inventory positions.

The Kappa-3 metric focuses on realized profitability rather than mark-to-market inventory changes, ensuring that scores reflect genuine trading skill and execution quality. Combined with round-trip volume weighting and activity decay mechanisms, this approach naturally rewards active, profitable trading while discouraging passive "standing still" behavior.

#### 13. How does the scoring system account for trading costs like fees and spreads?

Kappa-3 is calculated from realized P&L values which explicitly include all trading costs. When positions are opened and closed (round-trip trades), the realized profit or loss incorporates maker/taker fees paid or rebates received, as well as the effective spread captured or paid during execution. This means trading costs directly impact the Kappa-3 calculation through the mean return in the numerator, and poor execution that incurs excessive costs will naturally reduce a miner's score. The [Dynamic Incentive Structure](https://simulate.trading/taos-im-dis-paper) (DIS) further amplifies the impact of execution quality by adjusting fees based on market conditions, encouraging miners to provide liquidity when needed and take liquidity appropriately.

#### 14. Does the scoring system penalize order cancellations or repeated re-posting?

Currently, the scoring framework does not directly penalize order cancellations or the repeated submission of identical orders. The system focuses on realized profitability and round-trip trading volume. However, we recognize that excessive cancellations or unchanging re-posts can place unnecessary load on the simulation and may represent inefficient behavior rather than legitimate market-making.

Future refinements may apply operational-efficiency considerations, such as penalties for high cancel-to-fill ratios or for repeatedly submitting identical orders that do not produce new executions. We may also consider simpler guardrails at the agent level, for example raising minimum order sizes, reducing the number of instructions permitted per round, or further limiting the maximum number of open orders.

#### 15. How does the scoring system encourage participation across all books?

The current system enforces participation across all books through activity factors with associated decay mechanisms, and applies an outlier penalty that reduces a miner's score when Kappa-3 performance diverges significantly across books. Miners must actively trade and generate realized profits on all books to maintain high scores, as books with no round-trip trading activity contribute zero to the overall Kappa-3 assessment.

The round-trip volume weighting ensures that meaningful trading activity is rewarded, while the activity decay mechanism penalizes miners who abandon books or fail to maintain consistent profitable trading. The outlier penalty further discourages strategies that specialize in only a subset of books at the expense of others.

#### 16. Will the scoring system move toward execution-based metrics rather than inventory-based metrics?

The scoring system has already transitioned to execution-based metrics through the adoption of realized Sharpe and then Kappa-3 ratios calculated on realized P&L from completed round-trip trades. This shift addresses the limitations of inventory-delta approaches by explicitly rewarding actual trading profits after all costs, rather than unrealized mark-to-market changes.

The Kappa-3 metric inherently emphasizes execution quality, as realized P&L captures the timing and pricing of trade execution. Future enhancements may incorporate additional execution-focused measures such as time-to-fill efficiency, more sophisticated downside-risk measures, and operational penalties based on cancel-to-fill ratios to further refine the assessment of trading quality.

#### 17. Why are some miners able to maintain high scores with minimal trading activity?

With the transition to Kappa-3 scoring based on realized P&L, minimal trading activity results in insufficient data to calculate meaningful scores. Miners must complete round-trip trades to generate the realized profits that feed into the Kappa-3 calculation. The system requires a minimum number of non-zero realized P&L observations (`--scoring.kappa.min_realized_observations`) before assigning scores, and applies activity decay to reduce scores during periods without recent round-trip trading activity.

This execution-based approach naturally addresses the previous issue where stable inventory positions could generate strong scores without meaningful trading, as only actual completed trades contribute to performance measurement.

#### 18. Are identical re-posts treated as no-ops in the simulator?

No. Identical re-posts are fully recorded as cancel/placement events. They do not generate new fills unless market conditions change, but they still consume simulator resources and count toward operational activity. Any future scoring penalties aimed at discouraging churn will rely on these recorded events. Excessive cancel/repost cycles often add load without improving liquidity or execution quality - future scoring revisions may apply modest penalties based on recorded operational counts (e.g., cancel-to-fill ratios, repeated identical reposts) to discourage wasteful operational behavior while still allowing legitimate quote updates.

#### 19. How will a cost-aware, execution-focused scoring model treat taker trades?

The Kappa-3 scoring model already treats taker trades in a cost-aware manner, as realized P&L from any trade (maker or taker) includes all associated fees. Taker trades that pay fees are evaluated based on whether the resulting round-trip trade generates positive realized profit after costs. Strategic liquidity taking that creates net-positive round-trips improves Kappa-3 scores, while excessive taking that incurs costs without generating alpha will reduce scores. The DIS framework further guides appropriate liquidity-taking behavior by adjusting fee schedules based on market conditions.

#### 20. How will scoring handle time-weighted quoting quality (e.g., being near the top of book)?

While Kappa-3 scoring focuses on realized profitability from completed trades, we recognize that continuous provision of competitive quotes contributes to market quality even when fill rates are low. Future enhancements may explore time-weighted measures of quote quality, such as time spent quoting within a certain percentage of the best bid/ask, to complement execution-based metrics. These would serve as soft multipliers on round-trip volume or activity factors, ensuring that high-quality liquidity provision in quiet markets is appropriately valued without replacing the core realized-profitability focus.

#### 21. How will the system ensure miners don't just optimize for scoring rather than real liquidity?

The Kappa-3 scoring system aligns incentives with genuine market quality by measuring actual realized profits from completed trades, which inherently requires providing competitive quotes that result in executions. Gaming through passive behavior or manipulation is naturally discouraged because:

1. Only completed round-trip trades contribute to scores
2. All trading costs (fees, spreads) are fully reflected in realized P&L
3. Round-trip volume weighting rewards active, profitable trading
4. Activity decay penalizes strategies that abandon books or stop trading
5. Outlier penalties discourage specialized strategies that ignore some books

This execution-focused approach ensures that high scores require genuine trading skill and execution quality rather than optimization around scoring artifacts.