/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/agent/HighFrequencyTraderAgent.hpp>

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MessagePayload.hpp>

#include "DistributionFactory.hpp"
#include "RayleighDistribution.hpp"
#include "GBMValuationModel.hpp"
#include "Simulation.hpp"

#include <boost/algorithm/string/regex.hpp>

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

HighFrequencyTraderAgent::HighFrequencyTraderAgent(Simulation *simulation) noexcept
    : Agent{simulation}
{}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::configure(const pugi::xml_node &node)
{
    Agent::configure(node);

    m_rng = &simulation()->rng();

    m_wealthFrac = 0.99;

    pugi::xml_attribute attr;
    static constexpr auto ctx = std::source_location::current().function_name();

    if (attr = node.attribute("exchange"); attr.empty()) {
        throw std::invalid_argument(fmt::format(
            "{}: missing required attribute 'exchange'", ctx));
    }
    m_exchange = attr.as_string();

    if (simulation()->exchange() == nullptr) {
        throw std::runtime_error(fmt::format(
            "{}: exchange must be configured a priori", ctx));
    }
    m_bookCount = simulation()->exchange()->books().size();

    if (attr = node.attribute("tau"); attr.empty() || attr.as_double() <= 0.0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'tau' should have a value greater than 0.0", ctx));
    }
    m_tau = attr.as_double();

    if (attr = node.attribute("gHFT"); attr.empty() || attr.as_double() == 0.0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'gHFT' should have a value greater than 0.0", ctx));
    }
    double meanGamma= attr.as_double();
    attr = node.attribute("gHFTstd"); 
    double stdGamma = (attr.empty() || attr.as_double() == 0.0) ? 0.075 :attr.as_double();
    std::normal_distribution<double> riskDist(meanGamma, stdGamma);
    m_gHFT = riskDist(*m_rng);
    
    if (attr = node.attribute("delta"); attr.empty() || attr.as_double() == 0.0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'delta' should have a value greater than 0.0", ctx));
    }
    double averageDelta = attr.as_double();
    m_delta = std::clamp( averageDelta*(1+(meanGamma-m_gHFT)/m_gHFT), averageDelta*0.67,averageDelta*1.34 );
    if (attr = node.attribute("kappa"); attr.empty() || attr.as_double() == 0.0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'kappa' should have a value greater than 0.0", ctx));
    }
    m_kappa = attr.as_double();

    if (attr = node.attribute("spread"); attr.empty() || attr.as_double() == 0.0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'spread' should have a value greater than 0.0", ctx));
    }
    m_spread = attr.as_double();

    m_priceInit = taosim::util::decimal2double(simulation()->exchange()->config2().initialPrice);

    if (attr = node.attribute("minOPLatency"); attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'minOPLatency' should have a value greater than 0", ctx));
    }
    m_opl.min = attr.as_ullong();
    if (attr = node.attribute("maxOPLatency"); attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'maxOPLatency' should have a value greater than 0", ctx));
    }
    m_opl.max = attr.as_ullong();
    if (m_opl.min >= m_opl.max) {
        throw std::invalid_argument(fmt::format(
            "{}: minD ({}) should be strictly less maxD ({})", ctx, m_opl.min, m_opl.max));
    }

    const double psiQuotes = node.attribute("psiHFT_quotes").as_double(0.0);
    if (psiQuotes > 0.0) {
        const double orderMean = node.attribute("orderMean").as_double();
        const double orderSTD = node.attribute("orderSTD").as_double();
        const double meanQuote = std::exp(orderMean + 0.5 * orderSTD * orderSTD);
        const double dispersion = node.attribute("psiHFT_dispersion").as_double(0.1);
        std::normal_distribution<double> inventoryControlDist(psiQuotes, psiQuotes * dispersion);
        m_psi = std::max(inventoryControlDist(*m_rng), 1e-9) * meanQuote;
    }
    else {
        if (attr = node.attribute("psiHFT_constant"); attr.empty()) {
            throw std::invalid_argument(fmt::format(
                "{}: needs 'psiHFT_quotes' (full inventory in units of the agent's own average "
                "quote), or the legacy 'psiHFT_constant'", ctx));
        }
        std::normal_distribution<double> inventoryControlDist(attr.as_double(), 10.0);
        m_psi = inventoryControlDist(*m_rng);
    }
    m_topLevel = std::vector<TopLevelWithVolumes>(m_bookCount, TopLevelWithVolumes{});
    m_baseFree = std::vector<double>(m_bookCount, 0.);
    m_quoteFree = std::vector<double>(m_bookCount, 0.);
    m_inventory = std::vector<double>(m_bookCount, 0.);
    m_restingBid = std::vector<RestingQuote>(m_bookCount);
    m_restingAsk = std::vector<RestingQuote>(m_bookCount);
    m_pendingBid = std::vector<bool>(m_bookCount, false);
    m_pendingAsk = std::vector<bool>(m_bookCount, false);
    m_fillWakeScheduled = std::vector<bool>(m_bookCount, false);
    m_fillRequotes = std::vector<uint64_t>(m_bookCount, 0);
    m_rebalances = std::vector<uint64_t>(m_bookCount, 0);

    m_baseName = [&] {
        std::string res = name();
        boost::algorithm::erase_regex(res, boost::regex("(_\\d+)$"));
        return res;
    }();
    m_catUId = [&] {
        const std::string numStr = std::string{name()}.substr(m_baseName.size() + 1);
        return static_cast<uint32_t>(std::stoul(numStr));
    }();

    // Defaulted rather than required, so configs that predate the chain keep loading. The
    // values put the CLASS mean interval near 200 ms, which at ten instances gives each
    // maker a routine refresh roughly every two seconds. That split is the whole point and
    // it is a starting position to tune, not a calibration.
    m_acdDelayDist = std::weibull_distribution<float>{1.0, 1.0};
    m_omegaDu = node.attribute("acdOmega").as_float(3.9f);
    m_alphaDu = node.attribute("acdAlpha").as_float(0.15f);
    m_betaDu = node.attribute("acdBeta").as_float(0.65f);
    m_minDelay = node.attribute("minDMD").as_ullong(10'000'000);
    m_maxDelay = node.attribute("maxDMD").as_ullong(5'000'000'000);

    // Seeded with the configured initial price rather than left at zero. On an empty book the
    // maker falls back to this for both sides, and at zero makeOrder rejects every quote, so
    // the class contributed nothing at all until somebody else established a price. With an
    // InitializationAgent present that never showed; without one it makes the maker mute
    // exactly when the book most needs a two-sided quote.
    m_lastPrice.assign(m_bookCount, TimestampedPrice{.timestamp = 0, .price = m_priceInit});

    attr = node.attribute("opLatencyScaleRay"); 
    const double scale = (attr.empty() || attr.as_double() == 0.0) ? 0.235 : attr.as_double();
    const double percentile = 1-std::exp(-1/(2*scale*scale));
    m_orderPlacementLatencyDistribution =  std::make_unique<taosim::stats::RayleighDistribution>(scale, percentile); 

    if (attr = node.attribute("orderMean"); attr.empty()) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'orderMean' should have a value that make sense for the distribution", ctx));
    }
    m_orderMean = attr.as_double();
    attr = node.attribute("orderSTD");
    m_orderSTD = (attr.empty() || attr.as_double() < 0.0f)  ? 1.0 : attr.as_double();

    std::lognormal_distribution<double> orderDist(m_orderMean, m_orderSTD);
    for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {
        const double seed = orderDist(*m_rng);
        m_orderSizeBid.push_back(seed);
        m_orderSizeAsk.push_back(orderDist(*m_rng));
        m_lastInvSign.push_back(0);
    }

    // The variance ratio now comes off the shared bar clock rather than a private estimator,
    // so the two horizons are stated in bars instead of in halflives. The defaults match what
    // the estimator's halflives worked out to: a fast view of about half a minute against a
    // slow reference of about a quarter hour.
    m_varFastBars = std::max(node.attribute("varFastBars").as_uint(30u), 1u);
    m_varSlowBars = std::max(node.attribute("varSlowBars").as_uint(900u), m_varFastBars + 1);
    m_varGain = std::clamp(node.attribute("varGain").as_double(0.0), 0.0, 1.0);
    m_varRatioCap = std::max(node.attribute("varRatioCap").as_double(4.0), 1.0);
    m_undercutTicks = std::max(node.attribute("undercutTicks").as_int(0), 0);
    m_noiseRay = node.attribute("noiseRay").as_double();
    m_priceShiftDistribution =  std::make_unique<taosim::stats::RayleighDistribution>(m_noiseRay);
    m_minMFLatency = node.attribute("minMFLatency").as_ullong();
    m_shiftPercentage = node.attribute("shiftPercentage").as_double();

    attr = node.attribute("sigmaSqr");
    m_sigmaSqr =  (attr.empty() || attr.as_double() < 0.0f) ? 0.00001 : attr.as_double();
    m_debug = node.attribute("debug").as_bool();

    m_priceIncrement =
        1 / std::pow(10, simulation()->exchange()->config().parameters().priceIncrementDecimals);
    m_volumeIncrement =
        1 / std::pow(10, simulation()->exchange()->config().parameters().volumeIncrementDecimals);
    m_maxLeverage = taosim::util::decimal2double(simulation()->exchange()->getMaxLeverage());
    m_maxRate = node.attribute("rateMax").as_double(0.0075); 
    // Doubles as the fee-term width and as both bounds of the inventory-probability clamp
    // below, so above 0.5 the lower bound exceeds the upper and std::clamp is undefined.
    // Clamped here rather than at the use site so the value is one thing everywhere.
    m_sigmaMargin = std::clamp(node.attribute("marginNoiseSTD").as_double(0.00002), 0.0, 0.5);
    m_rateSensitivity = node.attribute("sensitivityCoef").as_double(100.0);
    m_spreadSensitivityExp= node.attribute("spreadSensitivityExp").as_double(2.07);
    m_spreadSensitivityBase= node.attribute("spreadSensitivityBase").as_double(0.00119);
    m_maxLoan = taosim::util::decimal2double(simulation()->exchange()->getMaxLoan());

    m_minSpreadTicks = node.attribute("minSpreadTicks").as_int(0);
    m_numLevels = std::max(1, node.attribute("numLevels").as_int(1));
    m_levelSpacingTicks = std::max(1, node.attribute("levelSpacing").as_int(1));
    m_spreadRefPrice = node.attribute("spreadRefPrice").as_double(0.0);

    m_quoteSizeVolumeFrac = std::max(node.attribute("quoteSizeVolumeFrac").as_double(0.0), 0.0);
    m_quoteSizeWindowBars = std::max(node.attribute("quoteSizeWindowBars").as_uint(30u), 1u);

    const Timestamp gracePeriod =
        node.parent().child("MultiBookExchangeAgent").attribute("gracePeriod").as_ullong();
    m_entryDelay = node.attribute("entryDelay").as_ullong(gracePeriod / 2);

    // Dimensionless multiplier, not a real flattening time; folded into gHFT and sigmaSqr.
    m_inventoryHorizon = node.attribute("inventoryHorizon").as_double(0.5);
    m_rebalanceGateMode = node.attribute("rebalanceGateMode").as_int(0);

    m_pRes.assign(m_bookCount, m_priceInit);
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::receiveMessage(Message::Ptr msg)
{
    if (msg->type == "EVENT_SIMULATION_START") {
        handleSimulationStart();
    }
    else if (msg->type == "EVENT_SIMULATION_END") {
        handleSimulationStop();
    }
    else if (msg->type == "RESPONSE_SUBSCRIBE_EVENT_TRADE"
        || msg->type == "RESPONSE_SUBSCRIBE_EVENT_TRADE_OWN") {
        handleTradeSubscriptionResponse();
    }
    else if (msg->type == "WAKEUP") {
        handleWakeup(msg);
    }
    else if (msg->type == "WAKEUP_HFT_FILL") {
        handleFillWakeup(msg);
    }
    else if (msg->type == "RESPONSE_PLACE_ORDER_LIMIT") {
        handleLimitOrderPlacementResponse(msg);
    }
    else if (msg->type == "ERROR_RESPONSE_PLACE_ORDER_LIMIT") {
        handleLimitOrderPlacementErrorResponse(msg);
    }
    else if (msg->type == "RESPONSE_PLACE_ORDER_MARKET") {
        handleMarketOrderPlacementResponse(msg);
    }
    else if (msg->type == "ERROR_RESPONSE_PLACE_ORDER_MARKET") {
        handleMarketOrderPlacementErrorResponse(msg);
    }
    else if (msg->type == "RESPONSE_CANCEL_ORDERS") {
        handleCancelOrdersResponse(msg);
    }
    else if (msg->type == "ERROR_RESPONSE_CANCEL_ORDERS") {
        handleCancelOrdersErrorResponse(msg);
    }
    else if (msg->type == "EVENT_TRADE") {
        handleTrade(msg);
    }
    else {
        simulation()->logDebug("{}", msg->type);
    }
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleSimulationStart()
{
    m_id = simulation()->exchange()->accounts().lookupLocalAgentId(name());
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        0,
        name(),
        m_exchange,
        "SUBSCRIBE_EVENT_TRADE_OWN");
    for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {
        if (m_catUId != 0) continue;
        // The ROUTINE cancel-and-requote rides a token: one maker refreshes at a time, so
        // the class's churn rate stops scaling with how many makers are configured. Being
        // hit is handled per instance and does not wait for a turn.
        //
        // No instance quotes before its first turn, so the entry delay is expressed once, on
        // the token. The class then arrives staggered over a few turns rather than all at
        // the same instant.
        auto& chains = simulation()->exchange()->wakeupChains().at(bookId);
        const Timestamp now = simulation()->currentTimestamp();
        // Registered as of the entry, or the watchdog would judge the chain overdue during
        // the wait and reseed a class that has not started yet.
        chains.registerChain(
            m_baseName,
            now + m_entryDelay,
            m_maxDelay,
            simulation()->localAgentManager()->roster()->at(m_baseName));
        const Timestamp initDelay = m_entryDelay + decisionMakingDelay(bookId);
        if (chains.arm(m_baseName, now, initDelay)) {
            simulation()->dispatchMessage(
                now,
                initDelay,
                name(),
                fmt::format("{}_{}", m_baseName, selectTurn()),
                "WAKEUP",
                MessagePayload::create<WakeupPayload>(bookId));
        }
    }
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleSimulationStop()
{
    for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {
        const auto* chain = simulation()->exchange()->wakeupChains().at(bookId).find(m_baseName);
        fmt::println(
            "AGENTDIAG {{\"agent\":\"{}\",\"book\":{},\"fill_requotes\":{},"
            "\"rebalances\":{},"
            "\"chain_turns\":{},\"inventory\":{},\"mid\":{}}}",
            name(), simulation()->bookIdCanon(bookId), m_fillRequotes[bookId],
            m_rebalances[bookId],
            chain != nullptr ? chain->wakeCount : 0, m_inventory[bookId],
            m_lastPrice[bookId].price);
    }
    std::fflush(stdout);
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleTradeSubscriptionResponse()
{
    
}

//-------------------------------------------------------------------------

// The wake IS the quote cycle: the book is read straight off the exchange at this cadence.
// The chain turn. Cancels both sides and requotes both, unconditionally: the routine churn
// lives here, so there is no comparison against what is already resting and no per-level
// bookkeeping to keep in step with the book.
void HighFrequencyTraderAgent::handleWakeup(Message::Ptr msg)
{
    // Guards a null payload, not a type mismatch: static_pointer_cast cannot fail.
    const auto payload = std::static_pointer_cast<WakeupPayload>(msg->payload);

    const BookId bookId = payload->bookId;
    const Timestamp now = simulation()->currentTimestamp();

    auto& chains = simulation()->exchange()->wakeupChains().at(bookId);
    chains.noteWake(m_baseName, now);
    // Hand the token on before quoting, so the chain's continuation is already in the queue
    // before anything that can fail runs. Against an early return, which the affordability
    // checks do take; a throw ends the run, since nothing between here and
    // Simulation::simulate catches one.
    const auto chosenAgent = selectTurn();
    const Timestamp delay = decisionMakingDelay(bookId);
    if (chains.arm(m_baseName, now, delay)) {
        simulation()->dispatchMessage(
            now,
            delay,
            name(),
            fmt::format("{}_{}", m_baseName, chosenAgent),
            "WAKEUP",
            MessagePayload::create<WakeupPayload>(bookId));
    }

    requote(bookId, true, true);
}

//-------------------------------------------------------------------------

// The reaction to having been hit, on its own clock rather than the chain's. Deliberately
// delayed: a maker learning of its own fill and requoting in the same instant is not a
// latency any venue offers.
void HighFrequencyTraderAgent::handleFillWakeup(Message::Ptr msg)
{
    const auto payload = std::static_pointer_cast<WakeupPayload>(msg->payload);
    if (payload == nullptr) return;
    const BookId bookId = payload->bookId;

    const bool doBid = m_pendingBid[bookId];
    const bool doAsk = m_pendingAsk[bookId];
    m_pendingBid[bookId] = false;
    m_pendingAsk[bookId] = false;
    m_fillWakeScheduled[bookId] = false;
    if (!doBid && !doAsk) return;

    ++m_fillRequotes[bookId];
    requote(bookId, doBid, doAsk);
}

//-------------------------------------------------------------------------

// Both triggers land here. Cancel whatever is still resting on the sides being requoted,
// refresh everything the quote is priced off, and place again. The rebalance branch inside
// placeOrder is reached on both paths, which is what gives a maker being run over a chance
// to act on the fill rather than waiting for its turn.
void HighFrequencyTraderAgent::requote(BookId bookId, bool doBid, bool doAsk)
{
    const Timestamp now = simulation()->currentTimestamp();

    cancelResting(bookId, doBid, doAsk);


    const auto l1 = simulation()->exchange()->statsHub()->l1(bookId);
    auto& topLevel = m_topLevel.at(bookId);
    topLevel.bid = taosim::util::decimal2double(l1.bestBidPrice);
    topLevel.bidQty = taosim::util::decimal2double(l1.bestBidVolume);
    topLevel.ask = taosim::util::decimal2double(l1.bestAskPrice);
    topLevel.askQty = taosim::util::decimal2double(l1.bestAskVolume);

    if (topLevel.bid == 0.0)
        topLevel.bid = m_lastPrice.at(bookId).price;
    if (topLevel.ask == 0.0)
        topLevel.ask = m_lastPrice.at(bookId).price;

    const double midquote = (topLevel.bid + topLevel.ask) / 2;
    m_lastPrice.at(bookId) = TimestampedPrice{.timestamp=now, .price=midquote};

    m_baseFree[bookId] = m_wealthFrac *
        taosim::util::decimal2double(simulation()->exchange()->account(name()).at(bookId).base.getFree());
    m_quoteFree[bookId] = m_wealthFrac *
        taosim::util::decimal2double(simulation()->exchange()->account(name()).at(bookId).quote->getFree());

    // The inventory skew is a price offset on the same reference scale as the spread, so it
    // takes the same conversion. See placeOrder.
    m_pRes[bookId] = midquote
        - m_gHFT * m_inventory[bookId] * effectiveSigmaSqr(bookId) * m_inventoryHorizon
            * priceScaleFactor(midquote);

    // A side with a placement still in flight already has its next quote on the way. Placing
    // another would put two out and leave the agent knowing the id of only one of them.
    placeOrder(
        bookId,
        topLevel,
        doBid && !m_restingBid[bookId].inFlight,
        doAsk && !m_restingAsk[bookId].inFlight);
}

//-------------------------------------------------------------------------

// Cancels only what is still resting. A quote the market took in full is already gone, and
// asking the exchange to cancel it would buy an error response and nothing else.
void HighFrequencyTraderAgent::cancelResting(BookId bookId, bool doBid, bool doAsk)
{
    auto cancelSide = [&](OrderDirection direction) {
        auto& quote = resting(bookId, direction);
        if (!quote.live) return;
        quote.live = false;
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            orderPlacementLatency(),
            name(),
            m_exchange,
            "CANCEL_ORDERS",
            MessagePayload::create<CancelOrdersPayload>(
                std::vector{taosim::event::Cancellation(quote.id)}, bookId));
    };

    if (doBid) cancelSide(OrderDirection::BUY);
    if (doAsk) cancelSide(OrderDirection::SELL);
}

//-------------------------------------------------------------------------

HighFrequencyTraderAgent::RestingQuote& HighFrequencyTraderAgent::resting(
    BookId bookId, OrderDirection direction) noexcept
{
    return direction == OrderDirection::BUY ? m_restingBid[bookId] : m_restingAsk[bookId];
}

//-------------------------------------------------------------------------

uint64_t HighFrequencyTraderAgent::selectTurn()
{
    const auto& counts = simulation()->localAgentManager()->roster()->baseNamesToCounts();
    return std::uniform_int_distribution<uint64_t>{0, counts.at(m_baseName) - 1}(*m_rng);
}

//-------------------------------------------------------------------------

// The class's churn clock, an ACD recursion on its own realized durations, the same shape
// StylizedTrader and NoiseTrader use.
Timestamp HighFrequencyTraderAgent::decisionMakingDelay(BookId bookId)
{
    auto& clocks = simulation()->exchange()->acdClocks().at(bookId);
    const auto last = clocks.get(m_baseName);
    float psiNext = m_omegaDu + m_alphaDu * last.delay + m_betaDu * last.psi;
    if (!std::isfinite(psiNext)) {
        psiNext = m_omegaDu / (1.0f - m_alphaDu - m_betaDu);
    }
    double delay = static_cast<double>(std::exp(psiNext)) * m_acdDelayDist(*m_rng);
    if (!std::isfinite(delay) || delay > static_cast<double>(m_maxDelay)) {
        delay = static_cast<double>(m_maxDelay);
    }
    const Timestamp clamped = std::clamp(static_cast<Timestamp>(delay), m_minDelay, m_maxDelay);
    clocks.insert(
        m_baseName,
        taosim::book::AcdClock{.delay = std::log(static_cast<float>(clamped)), .psi = psiNext});
    return clamped;
}

//-------------------------------------------------------------------------

// How long it takes the maker to learn it was hit and get a replacement back to the book.
// The feed leg is its configured floor; the order leg is the same draw every placement uses.
Timestamp HighFrequencyTraderAgent::fillReactionLatency()
{
    return m_minMFLatency + orderPlacementLatency();
}

//-------------------------------------------------------------------------

//-------------------------------------------------------------------------

// Where the agent learns the id of what it just put up. There is no cancel scheduled here
// any more: a quote now leaves the book because it was hit or because the chain came round,
// and `tau` no longer sets a lifetime.
void HighFrequencyTraderAgent::handleLimitOrderPlacementResponse(Message::Ptr msg)
{
    const auto payload = std::static_pointer_cast<PlaceOrderLimitResponsePayload>(msg->payload);
    const auto& request = *payload->requestPayload;

    resting(request.bookId, request.direction) = RestingQuote{
        .id = payload->id,
        .volume = taosim::util::decimal2double(request.volume),
        .live = true,
        .inFlight = false
    };
}

//-------------------------------------------------------------------------

// A rejected quote leaves the side empty, so the flag has to come down or that side is
// never quoted again.
void HighFrequencyTraderAgent::handleLimitOrderPlacementErrorResponse(Message::Ptr msg)
{
    const auto payload =
        std::static_pointer_cast<PlaceOrderLimitErrorResponsePayload>(msg->payload);
    const auto& request = *payload->requestPayload;
    resting(request.bookId, request.direction).inFlight = false;
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleMarketOrderPlacementResponse(Message::Ptr msg)
{
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleMarketOrderPlacementErrorResponse(Message::Ptr msg)
{
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleCancelOrdersResponse(Message::Ptr msg)
{
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleCancelOrdersErrorResponse(Message::Ptr msg)
{
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleTrade(Message::Ptr msg)
{
    const auto payload = std::static_pointer_cast<EventTradePayload>(msg->payload);
    const BookId bookId = payload->bookId;

    if (m_id == payload->context.aggressingAgentId) {
        m_inventory[bookId] += payload->trade.direction() == OrderDirection::BUY ?
            taosim::util::decimal2double(payload->trade.volume()) :
            taosim::util::decimal2double(-payload->trade.volume());
    }
    if (m_id == payload->context.restingAgentId) {
        m_inventory[bookId] += payload->trade.direction() == OrderDirection::BUY ?
            taosim::util::decimal2double(-payload->trade.volume()) :
            taosim::util::decimal2double(payload->trade.volume());
        // An aggressor buying lifted the resting ASK; an aggressor selling hit the BID.
        const auto side = payload->trade.direction() == OrderDirection::BUY
            ? OrderDirection::SELL
            : OrderDirection::BUY;
        noteOwnFill(bookId, side, payload->trade);
    }

}

//-------------------------------------------------------------------------

// Nets the fill off the resting quote and arms the delayed reaction. Sides are coalesced,
// so a burst of fills inside one reaction delay produces one requote and not one per fill.
void HighFrequencyTraderAgent::noteOwnFill(
    BookId bookId, OrderDirection side, const Trade& trade)
{
    auto& quote = resting(bookId, side);
    if (quote.live && quote.id == trade.restingOrderID()) {
        quote.volume -= taosim::util::decimal2double(trade.volume());
        // Nothing left to cancel when the requote comes round.
        if (quote.volume <= 0.0) quote.live = false;
    }

    (side == OrderDirection::BUY ? m_pendingBid : m_pendingAsk)[bookId] = true;
    if (m_fillWakeScheduled[bookId]) return;
    m_fillWakeScheduled[bookId] = true;
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        fillReactionLatency(),
        name(),
        name(),
        "WAKEUP_HFT_FILL",
        MessagePayload::create<WakeupPayload>(bookId));
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::sendOrder(std::optional<PlaceOrderLimitPayload::Ptr> payload) {
    
    if (payload.has_value()) {
        resting(payload.value()->bookId, payload.value()->direction).inFlight = true;
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            orderPlacementLatency(),
            name(),
            m_exchange,
            "PLACE_ORDER_LIMIT",
            payload.value());
    }
}

//-------------------------------------------------------------------------

std::optional<PlaceOrderLimitPayload::Ptr> HighFrequencyTraderAgent::makeOrder(BookId bookId, OrderDirection direction,
    double volume, double limitPrice, double wealth) {
    
    if (limitPrice <= 0 || volume <= 0 || wealth <= 0) {
        return std::nullopt;
    }

    double leverage = (volume * limitPrice - wealth) / wealth;
    if (leverage > 0) {
        if (leverage > m_maxLeverage) {
            leverage = m_maxLeverage;
        }
        volume = volume / (1. + leverage);
    } else {
        leverage = 0.;
    }
    
    return std::make_optional(MessagePayload::create<PlaceOrderLimitPayload>(
        direction,
        taosim::util::double2decimal(volume),
        taosim::util::double2decimal(limitPrice),
        taosim::util::double2decimal(leverage),
        bookId));
 }

//-------------------------------------------------------------------------

std::lognormal_distribution<double> HighFrequencyTraderAgent::quoteSizeDistribution(
    BookId bookId) const
{
    if (m_quoteSizeVolumeFrac > 0.0) {
        const auto window =
            simulation()->exchange()->statsHub()->window(bookId, m_quoteSizeWindowBars);
        if (window.seconds > 0.0 && window.volume > 0.0) {
            const double target = m_quoteSizeVolumeFrac * (window.volume / window.seconds);
            if (target > 0.0) {
                // Chosen so the MEAN of the draw is the target: a lognormal's mean is
                // exp(mu + sigma^2/2), not exp(mu).
                return std::lognormal_distribution<double>{
                    std::log(target) - 0.5 * m_orderSTD * m_orderSTD, m_orderSTD};
            }
        }
    }
    return std::lognormal_distribution<double>{m_orderMean, m_orderSTD};
}

//-------------------------------------------------------------------------

// Converts a price-space quantity fitted at the reference price to the current price level.
// One is the identity, which is what it returns at the reference, so a run starting there
// begins identically to one before this existed and only diverges as price moves away.
double HighFrequencyTraderAgent::priceScaleFactor(double midquote) const noexcept
{
    const double refPrice = (m_spreadRefPrice > 0.0) ? m_spreadRefPrice : m_priceInit;
    if (!(refPrice > 0.0) || !(midquote > 0.0)) return 1.0;
    return midquote / refPrice;
}

//-------------------------------------------------------------------------

// The calibrated `sigmaSqr` sets the LEVEL and the measurement supplies only the time
// variation, as a dimensionless fast-over-slow ratio. That is what keeps this insensitive to
// the fact that `sigmaSqr` is not a variance in any real units.
//
// Both horizons come off the shared bar clock, so two makers reading at the same instant get
// the same number. While the slow window is not yet filled the ratio is one and the whole term
// is inert, which is the behaviour the estimator's `primed` flag used to give.
double HighFrequencyTraderAgent::effectiveSigmaSqr(BookId bookId) const
{
    if (m_varGain <= 0.0) return m_sigmaSqr;

    const auto* hub = simulation()->exchange()->statsHub().get();
    const auto slow = hub->window(bookId, m_varSlowBars);
    const double slowVariance = slow.variancePerSecond();
    if (slow.bars <= m_varFastBars || !(slowVariance > 0.0)) return m_sigmaSqr;

    const double ratio = std::clamp(
        hub->window(bookId, m_varFastBars).variancePerSecond() / slowVariance,
        1.0 / m_varRatioCap,
        m_varRatioCap);
    return m_sigmaSqr * (1.0 + m_varGain * (ratio - 1.0));
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::placeOrder(
    BookId bookId, const TopLevelWithVolumes& topLevel, bool doBid, bool doAsk) {
    
    const double currentInventory = m_inventory[bookId];
    const double actualSpread = topLevel.ask - topLevel.bid;
    const double midquote = (topLevel.ask + topLevel.bid)/2;
    double relativeSpread = actualSpread/midquote;
    if (isnan(relativeSpread)) {
        // error recovery
        relativeSpread = m_spread;
    }
    const int invSign = (currentInventory > 0.0) - (currentInventory < 0.0);
    if (invSign != 0 && invSign != m_lastInvSign.at(bookId)) {
        auto lognormalDist = quoteSizeDistribution(bookId);
        // Floored at what the venue will actually accept. Sizing to the flow means a thin
        // book gets small quotes, which is the point, but below the exchange minimum the
        // order is rejected outright and the maker is simply absent.
        const double floorSize = taosim::util::decimal2double(
            simulation()->exchange()->config2().minOrderSize);
        m_orderSizeBid.at(bookId) = std::max(lognormalDist(*m_rng), floorSize);
        m_orderSizeAsk.at(bookId) = std::max(lognormalDist(*m_rng), floorSize);
        m_lastInvSign.at(bookId) = invSign;
    }

    double skipProb = std::exp(-1.0*std::pow(relativeSpread/m_spreadSensitivityBase, m_spreadSensitivityExp));
    double makerRate = taosim::util::decimal2double(simulation()->exchange()->clearingManager().feePolicy()->getRates(bookId,m_id).maker);

    // Whether to shed inventory aggressively this round, decided HERE rather than after the
    // quote prices are built, because the quotes have to be priced against the inventory the
    // maker expects to be left holding. See the rebalance block below.
    const double rateProb = m_rebalanceGateMode == 1
        ? 1.0 / (1.0 + std::exp((makerRate - m_maxRate) / std::sqrt(m_sigmaMargin)))
        : std::exp(-std::pow((makerRate - m_maxRate), 2.0) / (2 * m_sigmaMargin));
    const double inventoryProb = std::clamp(
        std::abs(currentInventory)/m_psi, m_sigmaMargin, 1 - m_sigmaMargin);
    const bool rebalanceNow =
        std::bernoulli_distribution{skipProb*rateProb*inventoryProb}(*m_rng)
        && std::abs(currentInventory) > 0.1;

    
    const double rayleighShift =  m_noiseRay * std::sqrt(-2.0 * std::log(1.0 - m_shiftPercentage));

    // Fitted coefficients producing a price at the reference level, not Avellaneda-Stoikov
    // in its own units: `sigmaSqr` is not a log-return variance. See parameters.md.
    const double optimalSpread = effectiveSigmaSqr(bookId)*m_gHFT*m_inventoryHorizon
     + 2/m_gHFT * std::log(1 + m_gHFT/m_kappa);
    // Applied to every price-space term: spread, noise AND inventory skew.
    double quoteInventory = currentInventory;
    if (rebalanceNow) {
        ++m_rebalances[bookId];
        const OrderDirection direction =
            currentInventory <= 0 ? OrderDirection::BUY : OrderDirection::SELL;
        const double maxQtyTop = currentInventory <= 0 ? topLevel.askQty : topLevel.bidQty;
        const double rebalanceQty = std::uniform_real_distribution<double>{
            0.1, std::min(std::abs(currentInventory), maxQtyTop)}(*m_rng);
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            orderPlacementLatency(),
            name(),
            m_exchange,
            "PLACE_ORDER_MARKET",
            MessagePayload::create<PlaceOrderMarketPayload>(
                direction,
                taosim::util::double2decimal(
                    rebalanceQty,
                    simulation()->exchange()->config().parameters().volumeIncrementDecimals),
                bookId));
        // Price the quotes against what the maker expects to be left holding. Without this the
        // passive side repeats the offload the aggressive side is already doing: a long maker
        // sells the top of book AND shows an aggressive ask, and if both fill it lands short,
        // which is the shed-and-overshoot that keeps a hot potato moving rather than ending it.
        quoteInventory += (direction == OrderDirection::BUY ? rebalanceQty : -rebalanceQty);
    }

    // Recomputed from `quoteInventory`; `m_pRes` stays the reservation price implied by the
    // inventory actually held, which is what the diagnostics and tests read.
    const double pRes = rebalanceNow
        ? midquote - m_gHFT * quoteInventory * effectiveSigmaSqr(bookId) * m_inventoryHorizon
              * priceScaleFactor(midquote)
        : m_pRes.at(bookId);

    const double priceScale = priceScaleFactor(midquote);
    double spread = optimalSpread * (1 + makerRate*m_rateSensitivity) * priceScale;
    if (m_minSpreadTicks > 0) {
        spread = std::max(spread, static_cast<double>(m_minSpreadTicks) * m_priceIncrement);
    }

    const double noiseScale = priceScale;

    double wealthBid = topLevel.ask * m_baseFree[bookId] + m_quoteFree[bookId];
    double orderVolumeBid = m_orderSizeBid.at(bookId);
    double orderVolumeAsk = m_orderSizeAsk.at(bookId);
    double noiseBid = (m_priceShiftDistribution->sample(*m_rng) - rayleighShift) * noiseScale;
    double priceOrderBid = pRes - (spread / 2.0) - noiseBid;

    double wealthAsk = topLevel.bid * m_baseFree[bookId] + m_quoteFree[bookId];
    double noiseAsk = (m_priceShiftDistribution->sample(*m_rng) - rayleighShift) * noiseScale;
    double priceOrderAsk = pRes + (spread / 2.0) + noiseAsk;

    if (m_undercutTicks > 0) {
        const double step = static_cast<double>(m_undercutTicks) * m_priceIncrement;
        if (topLevel.ask > 0.0) {
            const double limitAsk = pRes + (spread / 2.0);
            priceOrderAsk = std::max(limitAsk, std::min(priceOrderAsk, topLevel.ask - step));
        }
        if (topLevel.bid > 0.0) {
            const double limitBid = pRes - (spread / 2.0);
            priceOrderBid = std::min(limitBid, std::max(priceOrderBid, topLevel.bid + step));
        }
        if (priceOrderBid >= priceOrderAsk) {
            priceOrderBid = pRes - (spread / 2.0);
            priceOrderAsk = pRes + (spread / 2.0);
        }
    }
    const double bidVolPerLevel = orderVolumeBid / static_cast<double>(m_numLevels);
    const double askVolPerLevel = orderVolumeAsk / static_cast<double>(m_numLevels);
    for (int lvl = 0; lvl < m_numLevels; ++lvl) {
        const double offset = static_cast<double>(lvl) * static_cast<double>(m_levelSpacingTicks) * m_priceIncrement;
        const double bidPx = std::round((priceOrderBid - offset) / m_priceIncrement) * m_priceIncrement;
        const double askPx = std::round((priceOrderAsk + offset) / m_priceIncrement) * m_priceIncrement;
        if (doAsk) {
            sendOrder(makeOrder(bookId, OrderDirection::SELL, askVolPerLevel, askPx, wealthAsk));
        }
        if (doBid) {
            sendOrder(makeOrder(bookId, OrderDirection::BUY, bidVolPerLevel, bidPx, wealthBid));
        }
    }
}

//-------------------------------------------------------------------------

Timestamp HighFrequencyTraderAgent::orderPlacementLatency()
{
    return static_cast<Timestamp>(std::lerp(m_opl.min, m_opl.max, m_orderPlacementLatencyDistribution->sample(*m_rng)));
}

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
