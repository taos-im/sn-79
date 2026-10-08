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

The details of the incentive mechanism change over time; the objective stays the same: sustained, risk-aware trading performance across all simulated orderbooks. Most of a miner's reward comes from the trading pool, scored on the simulation; a share (5% by default) goes to [GenTRX](/doc/gentrx/overview.md) training. The trading pool is paid in two equal halves.

**Making** pays for liquidity supplied to the market. Each maker fill earns the spread it captured against the centred mid, and the credit on a book is twice the smaller of its buy and sell sides, so only two-sided making counts. From 0.6.3 that credit is scaled by markout quality, the share of the spread the fills still hold 20 simulation seconds later, so quotes the market runs over earn nothing. Credit concentrated on a few counterparties is discounted, and from 0.6.3 a maker's credit is scaled down when its fills against other miners exceed four times its fills against the background market. The half is paid in proportion to each miner's credit.

**Skill** pays for trading profit the market's drift does not explain. On each book a miner's mark-to-market P&L is reduced by its average inventory times the book's price move; what remains is its alpha, and a book counts once its alpha clears a small hurdle relative to the notional traded there. A miner qualifies when its alpha is consistently positive across books, measured by a downside-adjusted consistency ratio, on at least the required number of books (20 of 128 on a single market; from 0.6.3, 15 of the 96 `simulation_0` books and 5 of the 32 `simulation_1` books). Qualifying miners share the half in proportion to their net alpha, after the same counterparty discount.

The trading score is smoothed over a track record of about three simulated hours (`--scoring.score_ema_halflife`), a miner that stops answering validators is not scored until it answers again, and from 0.6.3 each asset class is paid through its own share. Validators set weights from a moving average of each miner's score (`--neuron.moving_average_alpha`, in the [base validator logic](/taos/common/neurons/validator.py)). The rules are stated in the dial help of the [validator config](/taos/im/config/__init__.py) and implemented in the [reward logic](/taos/im/validator/reward.py).

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

A new miner is scored from its first cycle, but its score builds over the three-hour track record, and the skill half pays nothing until the miner has qualifying alpha on enough books.  If the situation persists, you will need to check your UID at the [Agents Dashboard](https://taos.simulate.trading/d/edy6vxytuud4wd/agents) and confirm a few critical things:

- Do you see recent trades on many book IDs?  The skill half needs qualifying alpha on a set share of the books (20 of 128 on a single market; from 0.6.3, 15 of the 96 `simulation_0` books and 5 of the 32 `simulation_1` books) and is scaled down when more than about three eighths of the books carry none.
- Under the Requests plot, do you see a large proportion of failures or timeouts?  If you are not seeing mostly success, usually this is due to taking too long to respond - validators allow a maximum of `--neuron.timeout` seconds (defined in the [base validator config](/taos/common/config/__init__.py)) for miners to respond.  This can be addressed by increasing resources, optimizing your strategy logic and ensuring sufficient network connectivity; you may also want to consider geolocating your miner nearby to the biggest validators for the best possible latency.
- Check De-beta Eligible, Scored Books, Making Rank, Making Share, Skill Rank and the counterparty and tether factors in the Agents table, and the de-beta panels on your Agent page.  Making credit needs fills on both sides of a book; skill needs qualifying alpha on enough books.

#### 9. As a miner, I've hit the trading volume limit and can no longer submit instructions.  How is this limit enforced and what can I do now?

To keep traded volume tied to the capital behind it, and to keep careless trading from overloading the simulations, a "cap" is enforced on the total QUOTE volume allowed to be traded in a given period of simulation time.  Trading volume is calculated in QUOTE as the sum of price multiplied by quantity over all trades in which the miner is involved.  The period over which the trading volume limit is assessed is defined in the [validator config](/taos/im/config/__init__.py) as `--scoring.activity.trade_volume_assessment_period` (specified in simulation nanoseconds), where this is checked every `--scoring.activity.trade_volume_sampling_interval` (simulation nanoseconds) against the limit which is calculated as `--scoring.activity.capital_turnover_cap` multiplied by the initial wealth allocated to miners defined in the [simulation config](/simulate/trading/run/config/simulation_0.xml) as `Simulation.Agents.MultiBookExchangeAgent.Balances.wealth` (from 0.6.3, in each asset class's own configuration).  

If a miner trades more than `Simulation.Agents.MultiBookExchangeAgent.Balances.wealth` * `--scoring.activity.capital_turnover_cap` in a period of `--scoring.activity.trade_volume_assessment_period` simulation nanoseconds, no more instructions will be accepted on that book (except cancellations) until the total QUOTE volume traded in the most recent `trade_volume_assessment_period` drops below the limit.  Miners need to consider this limitation when designing and testing strategies; volume on its own earns nothing in scoring.  If the cap is hit early in the first 24 hours, there is a high risk of deregistration as no further actions will be possible until at least 24 simulation hours have elapsed.  Your current total traded volume is included in the state update for easy reference, accessible in code via `self.accounts[book_id].traded_volume`.

Note also that the trading volumes used in the assessment are not reset when a new simulation begins; the trading volumes are determined based on a 24 hour period which may span multiple simulations.

#### 10. Why do validators in the subnet exhibit discrepancies in vTrust?

While we have made changes to attempt to better align the weights assigned to miners by different validators, the nature of the subnet operation does require that there exist a group of miners which consistently outperform others across all simulated markets in order that validators would agree on the top scoring UIDs.  The discrepancies result from a combination of inconsistent performance by miners, networking considerations leading to different success rates and latencies between miners and validators, and the fact that the simulation hosted by each validator is different from others due to the stochastic nature of the background model.  We continue to seek ways to improve this situation, but have to consider also the impact of these changes in relation to the utility of the subnet - if all validators are evaluating the same exact conditions, this eliminates a key element guaranteeing the robustness of the top scoring miners to perform well in all environments.

#### 11. Do you currently burn any miner emissions, or have any plans to implement this mechanism?

No, we do not burn miner emissions and do not currently have any plans to implement this.  Though this seems it may make sense in some other subnets, we do not see that this would be the case for us.  If in future we see need to apply such, this will not be done without careful consideration and consultation with all participants.

#### 12. Does the scoring favour passive market making over active trading?

Making is one of the two halves by design, because liquidity supplied to the market is what the simulation needs. It is not paid for passivity: a resting order earns nothing until it is filled, credit requires fills on both sides of a book, and from 0.6.3 a fill the market runs over within seconds keeps none of its credit. The skill half pays active trading, on profit with the market's drift removed.

#### 13. How does the scoring system account for trading costs like fees and spreads?

Fees and the spread paid are in the P&L the skill half measures, so poor execution lowers a miner's alpha. The making half measures the spread captured on maker fills. The [Dynamic Incentive Structure](https://simulate.trading/taos-im-dis-paper) (DIS) fee policy adjusts maker and taker rates to market conditions, rewarding liquidity where it is scarce.

#### 14. Does the scoring system penalize order cancellations or repeated re-posting?

Not directly. Each round accepts a limited number of instructions per book (`--scoring.max_instructions_per_book`) and the simulation limits open orders, so heavy churn mostly costs the miner its own capacity. If cancel and re-post cycles come to load the simulation, these operational limits may be tightened.

#### 15. How does the scoring system encourage participation across all books?

The skill half needs qualifying alpha on a minimum number of books, and scales skill down when more than about three eighths of the books carry none (`--scoring.debeta.skill_max_inactive_books`). The making half is summed book by book, so each book with genuine two-sided making adds credit. A strategy confined to a few books earns little of either half.

#### 16. Is scoring based on execution or on inventory?

On execution. The making half measures spread captured on actual fills, judged from 0.6.3 by where the price is shortly afterwards. The skill half measures mark-to-market profit from trading with the effect of the market's drift on held inventory removed, so holding a position through a trend is not counted as skill.

#### 17. Can a miner keep a high score with little trading?

No. Making credit needs fills on both sides of a book, skill needs qualifying alpha on enough books, and both are measured over the scoring window, so a miner that stops trading loses both. The track-record smoothing also means a single strong window raises a score only gradually.

#### 18. Are identical re-posts treated as no-ops in the simulator?

No. Identical re-posts are fully recorded as cancel and placement events. They do not generate new fills unless market conditions change, but they still consume simulator resources and count toward the instruction limit per book.

#### 19. How are taker trades treated?

Taking never earns making credit: from 0.6.3 only the maker's side of a fill is judged. A taker's trades count in the skill half like any other: fees and the spread paid are in its P&L, and what counts is the alpha left after the market's drift is removed. The DIS fee policy adjusts taker rates to market conditions.

#### 20. How is quoting quality judged?

When quotes are filled. From 0.6.3 a maker's credit is scaled by how much of the spread its fills still hold 20 simulation seconds later, so quoting width, and how long a quote is left in front of flow, are what is priced. Resting time on its own is not paid.

#### 21. How does the system keep miners from optimizing for the score rather than for real liquidity?

The rules pay what real liquidity and real skill produce, and discount what can be manufactured:

1. Making credit comes only from fills, on both sides of a book.
2. From 0.6.3, credit for fills the market runs over is removed, and the price a fill is judged against cannot be moved by trades arranged between miners.
3. Credit concentrated on a few counterparties is discounted, and from 0.6.3 credit from other miners' flow is tied to what a maker supplies to the background market.
4. Skill is measured after removing the market's drift on held inventory, so riding a trend is not skill.
5. Pay follows performance, not volume or age: the same performance pays the same.
