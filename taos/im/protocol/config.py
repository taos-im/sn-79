# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Simulation configuration: fee tiers, the fee policy and the market simulation config.

Split out of taos.im.protocol.models, which re-exports every name here; import from either.
"""
from xml.etree.ElementTree import Element
from pydantic import Field
from taos.common.protocol import BaseModel
from taos.im.protocol.orders import Order


def _req(el: Element | None, tag: str) -> Element:
    """Return the required child <tag> of ``el``, raising on malformed config.

    Replaces chained ``el.find(tag).find(...)`` access where a missing element
    would otherwise surface as an opaque ``AttributeError`` on ``None``.
    """
    child = el.find(tag) if el is not None else None
    if child is None:
        raise ValueError(f"Malformed simulation config: missing <{tag}>")
    return child


def _balance_fields(
    bcfg: Element, init_price: float, quote_decimals: int
) -> tuple[str, float | None, float | None, float]:
    """Derive (capital_type, base_balance, quote_balance, wealth) for one agent
    group's <Balances> element, matching the original per-group logic: when a
    <Base> child is present the group uses static balances and computes wealth
    from <Quote>+<Base>*price; otherwise it reads a flat ``wealth`` attribute.
    """
    base_el = bcfg.find("Base")
    quote_el = bcfg.find("Quote")
    capital_type = "static" if base_el is not None else bcfg.attrib["type"]
    base_balance = float(base_el.attrib["total"]) if base_el is not None else None
    quote_balance = float(quote_el.attrib["total"]) if quote_el is not None else None
    if base_el is not None:
        # Original logic assumes <Quote> accompanies <Base>; assert rather than
        # AttributeError so a malformed config fails with a clear message.
        assert quote_el is not None, "Malformed config: <Base> present without <Quote>"
        wealth = round(
            float(quote_el.attrib["total"]) + float(base_el.attrib["total"]) * init_price,
            quote_decimals,
        )
    else:
        # `pareto` carries the per-agent total as a flat `wealth`. The range-drawn types
        # (`uniform-50`, `pareto-50`) carry bounds, and the engine renamed those bounds
        # `wealth`/`cap` -> `wealthMin`/`wealthMax` while keeping the old names working
        # (Balances.cpp requiredWealthAttr). Accept either spelling of the lower bound,
        # preferring `wealth` so every config that still has it parses exactly as before.
        if "wealth" in bcfg.attrib:
            wealth = float(bcfg.attrib["wealth"])
        elif "wealthMin" in bcfg.attrib:
            wealth = float(bcfg.attrib["wealthMin"])
        else:
            raise KeyError(
                f"<Balances type={capital_type!r}> has no <Base> child and no wealth "
                f"attribute; expected 'wealth' or 'wealthMin'"
            )
    return capital_type, base_balance, quote_balance, wealth


class FeeTier(BaseModel):
    """One tier of a volume-tiered fee schedule.

    Attributes:
        volume_required (float): Rolling traded volume required to qualify for this tier.
        maker_fee (float): Maker fee rate charged at this tier.
        taker_fee (float): Taker fee rate charged at this tier.
    """
    volume_required : float
    maker_fee : float
    taker_fee : float


class FeePolicy(BaseModel):
    """The venue's fee schedule as published on the state update.

    Attributes:
        fee_type (str): Which fee scheme is active.
        params (dict): Scheme-specific parameters, exactly as the engine published them.
        tiers (list[FeeTier]): Volume tiers when the scheme is tiered; empty otherwise.
    """
    fee_type : str
    params : dict
    tiers : list[FeeTier]

    @classmethod
    def from_xml(cls, xml : Element):
        """
        Constructs an instance of the class from the XML simulation configuration element.
        """
        if xml:
            fee_policy = FeePolicy(fee_type=xml.attrib['type'], params={k : v for k, v in xml.attrib.items() if k != 'type'}, tiers=[FeeTier(volume_required=0, maker_fee=0.0, taker_fee=0.0 )])
            match fee_policy.fee_type:
                case 'static':
                    fee_policy.tiers = [FeeTier(volume_required=0, maker_fee=xml.attrib['makerFee'], taker_fee=xml.attrib['takerFee'] )]
                case 'tiered':
                    fee_policy.tiers = [FeeTier(volume_required=tier.attrib['volumeRequired'], maker_fee=tier.attrib['makerFee'], taker_fee=tier.attrib['takerFee']) for tier in xml.findall("Tier")]
        else:
            fee_policy = FeePolicy(fee_type=xml.attrib['type'], params={k : v for k, v in xml.attrib.items() if k != 'type'}, tiers=[FeeTier(volume_required=0, maker_fee=0.0, taker_fee=0.0)]  )
            fee_policy.tiers = [FeeTier(volume_required=0, maker_fee=0.0, taker_fee=0.0 )]
        return fee_policy

    def to_prom_info(self) -> dict:
        """
        Creates a dictionary containing the details of the fee policy specification in format suitable for publishing via Prometheus Info metric
        """
        prometheus_info = {}
        prometheus_info['simulation_fee_policy_type'] = self.fee_type
        for name, value in self.params.items():
            prometheus_info[f'simulation_fee_policy_{name}'] = str(value)
        if self.fee_type == 'tiered':
            for i, tier in enumerate(self.tiers):
                prometheus_info[f'simulation_fee_policy_tier_{i}_volume_required'] = f"{tier.volume_required:.2f}"
                prometheus_info[f'simulation_fee_policy_tier_{i}_maker_rate'] = f"{tier.maker_fee * 100:.4f}"
                prometheus_info[f'simulation_fee_policy_tier_{i}_taker_rate'] = f"{tier.taker_fee * 100:.4f}"
        return prometheus_info


class MarketSimulationConfig(BaseModel):
    """
    Class to represent the configuration of an intelligent markets simulation.

    Attributes:
        simulation_id (str | None): Unique identifier for the simulation instance.
        logDir (str | None): Directory where simulation logs are saved.

        remoteAgentCount (int | None): Number of remote agents in simulation
        block_count (int): Number of parallel "blocks" of simulation runs (related to parallelization implementation).

        time_unit (str): Unit of time used in the simulation (e.g., 'ns' for nanoseconds). Default is 'ns'.
        duration (int): Total simulation time in the given time_unit.
        grace_period (int): Time period at start of simulation which must elapse before miner agents are able to submit instructions.
        publish_interval (int): Interval at which the simulation state is published.
        log_window (int | None): Size of the time window for logs.

        books_per_block (int): Number of order books simulated in each parallelization block.
        book_count (int): Total number of order books in the simulation.
        book_levels (int): Number of levels included for each book in the state update (containing price and volume data).
        detailed_book_levels (int): Number of levels for which full book level data is included (level composition in terms of orders).

        baseDecimals (int): Decimal precision for base currency values.
        quoteDecimals (int): Decimal precision for quote currency values.
        priceDecimals (int): Decimal precision for price values.
        volumeDecimals (int): Decimal precision for order volumes.

        fee_policy (FeePolicy | None): The fee policy applied to trades.

        min_order_size (float): Minimum order volume accepted by the engine; orders below this size are rejected. Default 0.0 (no minimum).
        max_open_orders (int | None): Maximum number of open orders per agent.

        max_leverage (float): Maximum leverage allowed for agents.
        max_loan (float): Maximum loan amount agents can take.
        maintenance_margin (float): Maintenance margin ratio required for agents to avoid liquidation.

        miner_capital_type (str): Capital allocation strategy for miners ('static' or 'pareto').
        miner_base_balance (float | None): Initial base currency balance for miners.
        miner_quote_balance (float | None): Initial quote currency balance for miners.
        miner_wealth (float): Total wealth allocated to miners (QUOTE value of initial BASE balance at initial price + initial QUOTE balance).

        init_price (float): Initial market price for the simulation.

        # Fundamental Price (FP) parameters
        fp_update_period (int | None): Period for updating the fundamental price.
        fp_seed_interval (int | None): Interval for reseeding the fundamental process.
        fp_mu (float | None): Drift term in the fundamental price process.
        fp_sigma (float | None): Volatility in the fundamental price process.
        fp_lambda (float | None): Intensity of price jumps.
        fp_mu_jump (float | None): Mean size of price jumps.
        fp_sigma_jump (float | None): Volatility of price jumps.

        # Initialization Agent Configuration
        # Initialization Agents are triggered only once at the start of the simulation to provide an initial random state of the orderbook
        # After some time, when the other background agents have had time to create a sensible orderbook structure, their orders are cancelled.
        init_agent_count (int): Number of initialization agents.
        init_agent_capital_type (str): Capital allocation strategy for initialization agents.
        init_agent_base_balance (float | None): Base currency balance for initialization agents.
        init_agent_quote_balance (float | None): Quote currency balance for initialization agents.
        init_agent_wealth (float): Total wealth allocated to initialization agents.
        init_agent_tau (int): Time period after which orders placed by initialization agents are cancelled.

        # High-Frequency Trader (HFT) agents
        # HFT Agents function somewhat like market makers in real markets.
        # https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2336772
        hft_agent_count (int): Number of HFT agents.
        hft_agent_capital_type (str): Capital allocation strategy for HFT agents.
        hft_agent_base_balance (float | None): Base currency balance for HFT agents.
        hft_agent_quote_balance (float | None): Quote currency balance for HFT agents.
        hft_agent_wealth (float): Total wealth allocated to HFT agents.

        hft_agent_feed_latency_min (int): Minimum market data feed latency for HFT agents.
        hft_agent_order_latency_min (int): Minimum order placement latency for HFT agents.
        hft_agent_order_latency_max (int): Maximum order placement latency for HFT agents.
        hft_agent_order_latency_scale (float): Scaling factor for HFT order latencies.

        hft_agent_tau (int): Latency parameter for HFT agents.
        hft_agent_delta (int): Sensitivity parameter for HFT agents.
        hft_agent_psi (float): Probability weighting factor for HFT decisions.
        hft_agent_gHFT (float): Aggressiveness factor in HFT strategies.
        hft_agent_kappa (float): Inventory control parameter for HFT agents.
        hft_agent_spread (float): Target bid-ask spread for HFT agents.
        hft_agent_order_size_mean (float): Mean size of orders placed by HFT agents.
        hft_agent_price_noise (float): Noise applied to HFT agent pricing decisions.
        hft_agent_price_shift (float): Systematic price shift applied by HFT agents.

        # Stylized Trader Agent (STA) configuration
        # STA agents aim to approximate the behaviour of several interacting classes of traders.
        # https://arxiv.org/abs/0711.3581
        sta_agent_count (int): Number of STA agents.
        sta_agent_capital_type (str): Capital allocation strategy for STA agents.
        sta_agent_base_balance (float | None): Base currency balance for STA agents.
        sta_agent_quote_balance (float | None): Quote currency balance for STA agents.
        sta_agent_wealth (float): Total wealth allocated to STA agents.

        sta_agent_feed_latency_min (int): Minimum market data feed latency for STA agents. Default is 0.
        sta_agent_feed_latency_mean (int): Mean feed latency for STA agents.
        sta_agent_feed_latency_std (int): Standard deviation of feed latency for STA agents.
        sta_agent_order_latency_min (int): Minimum order placement latency for STA agents.
        sta_agent_order_latency_max (int): Maximum order placement latency for STA agents.
        sta_agent_order_latency_scale (float): Scaling factor for STA order latencies.
        sta_agent_decision_latency_mean (int): Mean decision-making latency for STA agents.
        sta_agent_decision_latency_std (int): Standard deviation of decision latency for STA agents.
        sta_agent_selection_scale (float): Scale factor influencing STA selection preferences.

        sta_agent_noise_weight (float): Weight for noise component in STA decision making.
        sta_agent_chartist_weight (float): Weight for chartist component in STA agents.
        sta_agent_fundamentalist_weight (float): Weight for fundamentalist component in STA agents.

        sta_agent_tau (int): Decision interval for STA agents.
        sta_agent_tauHist (int): Historical observation window size for STA agents.
        sta_agent_tauF (int): Forecast horizon for STA agents, truncated. WIRE-COMPATIBLE: see
            sta_agent_tauF_exact for the configured value.
        sta_agent_tauF_exact (float | None): The configured forecast horizon, which may be
            fractional. Added rather than retyping sta_agent_tauF, which miners validate.
        sta_agent_sigmaEps (float): Volatility parameter in STA forecasting.
        sta_agent_r_aversion (float): Risk aversion parameter for STA agents.

        # Futures Agent configuration
        # The Futures Agent aims to bring real-world connection into the simulation dynamics
        # These agents make trading decisions based on external signals obtained from live futures markets
        futures_agent_count (int | None): Number of futures agents.
        futures_agent_capital_type (str | None): Capital allocation strategy for futures agents.
        futures_agent_base_balance (float | None): Base currency balance for futures agents.
        futures_agent_quote_balance (float | None): Quote currency balance for futures agents.
        futures_agent_wealth (float | None): Total wealth allocated to futures agents.

        futures_agent_volume (float | None): Typical trade volume for futures agents.
        futures_agent_sigmaEps (float | None): Noise level in futures agent decisions.
        futures_agent_lambda (float | None): Order arrival intensity for futures agents.
        futures_agent_feed_latency_mean (int | None): Mean market data latency for futures agents.
        futures_agent_feed_latency_std (int | None): Standard deviation of feed latency for futures agents.
        futures_agent_order_latency_min (int | None): Minimum order latency for futures agents.
        futures_agent_order_latency_max (int | None): Maximum order latency for futures agents.
        futures_agent_selection_scale (float | None): Scale factor for futures agent selection.
    """
    simulation_id : str | None = None
    logDir : str | None = None
    
    remoteAgentCount : int | None = None
    block_count : int

    time_unit : str = 'ns'
    duration : int
    grace_period : int
    publish_interval : int
    log_window : int | None = None

    books_per_block : int
    book_count : int
    book_levels : int
    detailed_book_levels : int = 5

    baseDecimals : int
    quoteDecimals : int
    priceDecimals : int
    volumeDecimals : int

    fee_policy : FeePolicy | None = None

    min_order_size : float = 0.0
    max_open_orders : int | None = None

    max_leverage : float
    max_loan : float
    maintenance_margin : float

    miner_capital_type : str
    miner_base_balance : float | None
    miner_quote_balance : float | None
    miner_wealth : float

    init_price : float

    fp_update_period : int | None = None
    fp_seed_interval : int | None = None
    fp_mu : float | None = None
    fp_sigma : float | None = None
    fp_lambda : float | None = None
    fp_mu_jump : float | None = None
    fp_sigma_jump : float | None = None

    init_agent_count : int
    init_agent_capital_type : str
    init_agent_base_balance : float | None
    init_agent_quote_balance : float | None
    init_agent_wealth : float

    init_agent_tau : int

    hft_agent_count : int
    hft_agent_capital_type : str
    hft_agent_base_balance : float | None
    hft_agent_quote_balance : float | None
    hft_agent_wealth : float

    hft_agent_feed_latency_min : int
    hft_agent_order_latency_min : int
    hft_agent_order_latency_max : int
    hft_agent_order_latency_scale : float

    hft_agent_tau : float
    hft_agent_delta : int
    hft_agent_psi : float
    hft_agent_gHFT : float
    hft_agent_kappa : float
    hft_agent_spread : float
    hft_agent_order_size_mean : float
    hft_agent_price_noise : float
    hft_agent_price_shift : float

    sta_agent_count : int
    sta_agent_capital_type : str
    sta_agent_base_balance : float | None
    sta_agent_quote_balance : float | None
    sta_agent_wealth : float

    sta_agent_feed_latency_min : int = 0
    sta_agent_feed_latency_mean : int
    sta_agent_feed_latency_std : int
    sta_agent_order_latency_min : int
    sta_agent_order_latency_max : int
    sta_agent_order_latency_scale : float
    sta_agent_decision_latency_mean : int
    sta_agent_decision_latency_std : int
    sta_agent_selection_scale : float

    sta_agent_noise_weight : float
    sta_agent_chartist_weight : float
    sta_agent_fundamentalist_weight : float

    sta_agent_tau : int
    sta_agent_tauHist : int
    sta_agent_tauF : int
    sta_agent_tauF_exact : float | None = None
    sta_agent_sigmaEps : float
    sta_agent_r_aversion : float

    futures_agent_count : int | None = None
    futures_agent_capital_type : str | None = None
    futures_agent_base_balance : float | None = None
    futures_agent_quote_balance : float | None = None
    futures_agent_wealth : float | None = None

    futures_agent_volume : float | None = None
    futures_agent_sigmaEps : float | None = None
    futures_agent_lambda : float | None = None
    futures_agent_feed_latency_mean : int | None = None
    futures_agent_feed_latency_std : int | None = None
    futures_agent_order_latency_min : int | None = None
    futures_agent_order_latency_max : int | None = None
    futures_agent_selection_scale : float | None = None

    @property
    def book_ids(self) -> list[int]:
        """The book ids of this simulation, dense: the same surface ExchangeConfig exposes, so the validator's
        query, report and scoring loops iterate one attribute whichever config class the run carries."""
        return list(range(self.book_count))

    @classmethod
    def from_xml(cls, xml : Element):
        """
        Constructs an instance of the class from the XML simulation configuration.
        """
        agents_config = _req(xml, "Agents")
        MBE_config = _req(agents_config, "MultiBookExchangeAgent")
        books_config = _req(MBE_config, "Books")
        processes_config = _req(books_config, "Processes")
        FP_config = _req(processes_config, "FundamentalPrice")
        balances_config = _req(MBE_config, "Balances")
        fees_config = _req(MBE_config, "FeePolicy")

        init_config = _req(agents_config, "InitializationAgent")
        init_balances_config = init_config.find("Balances")
        init_balances_config = init_balances_config if init_balances_config is not None else balances_config
        STA_config = _req(agents_config, "StylizedTraderAgent")
        STA_balances_config = STA_config.find("Balances")
        STA_balances_config = STA_balances_config if STA_balances_config is not None else balances_config
        HFT_config = _req(agents_config, "HighFrequencyTraderAgent")
        HFT_balances_config = HFT_config.find("Balances")
        HFT_balances_config = HFT_balances_config if HFT_balances_config is not None else balances_config
        Futures_config = _req(agents_config, "FuturesTraderAgent")
        Futures_balances_config = Futures_config.find("Balances")
        Futures_balances_config = Futures_balances_config if Futures_balances_config is not None else balances_config

        init_price = float(MBE_config.attrib["initialPrice"])
        quote_decimals = int(MBE_config.attrib["quoteDecimals"])
        miner_capital_type, miner_base_balance, miner_quote_balance, miner_wealth = _balance_fields(
            balances_config, init_price, quote_decimals
        )
        init_capital_type, init_base_balance, init_quote_balance, init_wealth = _balance_fields(
            init_balances_config, init_price, quote_decimals
        )
        hft_capital_type, hft_base_balance, hft_quote_balance, hft_wealth = _balance_fields(
            HFT_balances_config, init_price, quote_decimals
        )
        sta_capital_type, sta_base_balance, sta_quote_balance, sta_wealth = _balance_fields(
            STA_balances_config, init_price, quote_decimals
        )
        futures_capital_type, futures_base_balance, futures_quote_balance, futures_wealth = _balance_fields(
            Futures_balances_config, init_price, quote_decimals
        )
        return MarketSimulationConfig(
            remoteAgentCount=int(MBE_config.attrib['remoteAgentCount']),
            block_count=int(xml.attrib['blockCount']),

            time_unit = str(xml.attrib['timescale']),
            duration = int(xml.attrib['duration']),
            grace_period = int(MBE_config.attrib['gracePeriod']),
            publish_interval = int(xml.attrib['step']),
            log_window = int(xml.attrib['logWindow']),

            books_per_block = int(books_config.attrib['instanceCount']),
            book_count = int(xml.attrib['blockCount']) * int(books_config.attrib['instanceCount']),
            book_levels = int(books_config.attrib['maxDepth']),
            # DEFAULTED, like scaleR below, because a bare attrib[] here is a CRASH LOOP.
            # from_xml runs inside SimulationEngine.start(), which runs inside the validator's
            # __init__, so a KeyError raises before the validator exists, pm2 restarts it, and it
            # raises again. simulation_0_RS.xml and simulation_0_mngr.xml carry no detailedDepth and
            # would take down any validator launched against them. 0 reads as "no detailed depth
            # requested", which is what a config that never mentions it is asking for.
            detailed_book_levels = int(
                books_config.attrib['detailedDepth'] if 'detailedDepth' in books_config.attrib else 0
            ),

            baseDecimals = int(MBE_config.attrib['baseDecimals']),
            quoteDecimals = int(MBE_config.attrib['quoteDecimals']),
            priceDecimals = int(MBE_config.attrib['priceDecimals']),
            volumeDecimals = int(MBE_config.attrib['volumeDecimals']),

            fee_policy=FeePolicy.from_xml(fees_config),

            min_order_size=float(MBE_config.attrib.get('minOrderSize', 0.0)),
            max_open_orders=int(MBE_config.attrib['maxOpenOrders']),

            max_leverage = float(MBE_config.attrib['maxLeverage']),
            max_loan = float(MBE_config.attrib['maxLoan']),
            maintenance_margin = float(MBE_config.attrib['maintenanceMargin']),

            miner_capital_type=miner_capital_type,
            miner_base_balance=miner_base_balance,
            miner_quote_balance=miner_quote_balance,
            miner_wealth=miner_wealth,

            init_price = init_price,

            fp_update_period = int(FP_config.attrib['updatePeriod']) + 1,
            fp_seed_interval = int(FP_config.attrib['seedInterval']),
            fp_mu = float(FP_config.attrib['mu']),
            fp_sigma = float(FP_config.attrib['sigma']),
            fp_lambda = float(FP_config.attrib['lambda']),
            fp_mu_jump = float(FP_config.attrib['muJump']),
            fp_sigma_jump = float(FP_config.attrib['sigmaJump']),

            init_agent_count = int(init_config.attrib['instanceCount']),
            init_agent_capital_type = init_capital_type,
            init_agent_base_balance = init_base_balance,
            init_agent_quote_balance = init_quote_balance,
            init_agent_wealth = init_wealth,

            init_agent_tau = int(init_config.attrib['tau']),

            hft_agent_count = int(HFT_config.attrib['instanceCount']),
            hft_agent_capital_type = hft_capital_type,
            hft_agent_base_balance = hft_base_balance,
            hft_agent_quote_balance = hft_quote_balance,
            hft_agent_wealth = hft_wealth,

            hft_agent_feed_latency_min = int(HFT_config.attrib['minMFLatency']),
            hft_agent_order_latency_min = int(HFT_config.attrib['minOPLatency']),
            hft_agent_order_latency_max = int(HFT_config.attrib['maxOPLatency']),
            hft_agent_order_latency_scale = float(HFT_config.attrib['opLatencyScaleRay']),

            hft_agent_tau = float(HFT_config.attrib['tau']),
            hft_agent_delta = int(HFT_config.attrib['delta']),
            # BOTH SPELLINGS. 84b6e061 renamed this attribute psiHFT_constant -> psiHFT_quotes in
            # simulation_0.xml. A bare lookup here is not a degraded field: from_xml runs inside
            # SimulationEngine.start(), which runs inside the validator's __init__, so it raises
            # KeyError before the validator exists and pm2 restarts it into the same failure. The
            # validator ran on its already-parsed copy for ninety minutes and only died when a chain
            # recycle restarted it -- 54 restarts before it was caught.
            #
            # Accepting only the NEW name would move the crash rather than fix it: ten other configs
            # in that directory still carry the old one, including the acceptance config the engine
            # itself runs. Same defensive shape as scaleR below, which is why scaleR never bit.
            hft_agent_psi = float(
                HFT_config.attrib['psiHFT_quotes'] if 'psiHFT_quotes' in HFT_config.attrib
                else HFT_config.attrib['psiHFT_constant'] if 'psiHFT_constant' in HFT_config.attrib
                else 0.0
            ),
            hft_agent_gHFT = float(HFT_config.attrib['gHFT']),
            hft_agent_kappa = float(HFT_config.attrib['kappa']),
            hft_agent_spread = float(HFT_config.attrib['spread']),
            hft_agent_order_size_mean = float(HFT_config.attrib['orderMean']),
            hft_agent_price_noise = float(HFT_config.attrib['noiseRay']),
            hft_agent_price_shift = float(HFT_config.attrib['shiftPercentage']),

            sta_agent_count = int(STA_config.attrib['instanceCount']),
            sta_agent_capital_type = sta_capital_type,
            sta_agent_base_balance = sta_base_balance,
            sta_agent_quote_balance = sta_quote_balance,
            sta_agent_wealth = sta_wealth,

            sta_agent_feed_latency_mean = int(STA_config.attrib['MFLmean']),
            sta_agent_feed_latency_std = int(STA_config.attrib['MFLstd']),
            sta_agent_order_latency_min = int(STA_config.attrib['minOPLatency']),
            sta_agent_order_latency_max = int(STA_config.attrib['maxOPLatency']),
            sta_agent_order_latency_scale = float(STA_config.attrib['opLatencyScaleRay']),
            sta_agent_decision_latency_mean = int(STA_config.attrib['delayMean']),
            sta_agent_decision_latency_std = int(STA_config.attrib['delaySTD']),
            sta_agent_selection_scale = float(STA_config.attrib['scaleR'] if 'scaleR' in STA_config.attrib else 0.0),

            sta_agent_noise_weight = float(STA_config.attrib['sigmaN']),
            sta_agent_chartist_weight = float(STA_config.attrib['sigmaC']),
            sta_agent_fundamentalist_weight = float(STA_config.attrib['sigmaF']),

            sta_agent_tau = int(STA_config.attrib['tau']),
            sta_agent_tauHist = int(STA_config.attrib['tauHist']),
            # THIS FIELD IS A WIRE CONTRACT WITH MINERS WE DO NOT CONTROL, so its TYPE may not
            # change. 84b6e061 set tauF="1.4"; int() on that literal raises ValueError and
            # fails from_xml inside the validator's __init__, so truncate instead of casting
            # directly. Retyping this to float would have been correct and unshippable: a
            # miner running an older model validates sta_agent_tauF as int, rejects the WHOLE
            # MarketSimulationStateUpdate on a fractional value, leaves simulation_config as
            # None, and dies on `'NoneType' has no attribute 'book_count'` -- a symptom that
            # names nothing to do with the config. Both acceptance miners did exactly that on
            # before they were restarted onto the new model.
            #
            # Unknown fields are IGNORED by the model, so ADDING one is safe where retyping is
            # not. The configured value therefore goes alongside rather than replacing.
            sta_agent_tauF = int(float(STA_config.attrib['tauF'])),
            sta_agent_tauF_exact = float(STA_config.attrib['tauF']),
            sta_agent_sigmaEps = float(STA_config.attrib['sigmaEps']),
            sta_agent_r_aversion = float(STA_config.attrib['r_aversion']),

            futures_agent_count = int(Futures_config.attrib['instanceCount']),
            futures_agent_capital_type = futures_capital_type,
            futures_agent_base_balance = futures_base_balance,
            futures_agent_quote_balance = futures_quote_balance,
            futures_agent_wealth = futures_wealth,

            futures_agent_volume = float(Futures_config.attrib['volume']),
            futures_agent_sigmaEps = float(Futures_config.attrib['sigmaEps']),
            futures_agent_lambda = float(Futures_config.attrib['lambda'] if 'lambda' in Futures_config.attrib else 0.0),
            futures_agent_feed_latency_mean = int(Futures_config.attrib['MFLmean']),
            futures_agent_feed_latency_std = int(Futures_config.attrib['MFLstd']),
            futures_agent_order_latency_min = int(Futures_config.attrib['minOPLatency']),
            futures_agent_order_latency_max = int(Futures_config.attrib['maxOPLatency']),
            futures_agent_selection_scale = float(Futures_config.attrib['scaleR'] if 'scaleR' in Futures_config.attrib else 0.0),
        )

    def label(self) -> str:
        """
        Function to generate a unique label based on the config parameters for a simulation.
        This is used to ensure that simulation-specific data is reset when a new simulation config is deployed.
        """
        return f"du{self.duration}{self.time_unit}_gr{self.grace_period}-bo{self.book_count}-{self.miner_capital_type}_{self.miner_wealth}-" + \
            f"pd{self.priceDecimals}_vd{self.volumeDecimals}_bd{self.baseDecimals}_qd{self.quoteDecimals}-" + \
            f"ip{self.init_price}-" + \
            f"ina_{self.init_agent_count}_{self.init_agent_capital_type}_{self.init_agent_wealth}_" + \
            f"sta_{self.sta_agent_count}_{self.sta_agent_capital_type}_{self.sta_agent_wealth}_" + \
            f"wn{self.sta_agent_noise_weight}_wc{self.sta_agent_chartist_weight}_wf{self.sta_agent_fundamentalist_weight}_" + \
            f"hft_{self.hft_agent_count}_{self.hft_agent_capital_type}_{self.hft_agent_wealth}_" + \
            f"ta{self.hft_agent_tau}_de{self.hft_agent_delta}_ps{self.hft_agent_psi}"
