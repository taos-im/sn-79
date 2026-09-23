/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/agent/FuturesTraderAgent.hpp>

#include <taosim/process/FuturesSignal.hpp>
#include "DistributionFactory.hpp"
#include "RayleighDistribution.hpp"
#include "Simulation.hpp"

#include <boost/algorithm/string/regex.hpp>

#include <algorithm>

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

FuturesTraderAgent::FuturesTraderAgent(Simulation* simulation) noexcept
    : Agent{simulation}
{}

//-------------------------------------------------------------------------

void FuturesTraderAgent::configure(const pugi::xml_node& node)
{
    Agent::configure(node);

    m_rng = &simulation()->rng();

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

    attr = node.attribute("sigmaN");
    m_sigmaN = ( attr.empty() || attr.as_double() < 0.0f) ? 0.7 : attr.as_double();

    if (attr = node.attribute("sigmaEps"); attr.empty() || attr.as_double() <= 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'sigmaEps' should have a value greater than 0.0f", ctx));
    }
    m_sigmaEps = attr.as_double();

    if (attr = node.attribute("minOPLatency"); attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'minLatency' should have a value greater than 0", ctx));
    }
    m_opl.min = attr.as_ullong();
    if (attr = node.attribute("maxOPLatency"); attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'maxLatency' should have a value greater than 0", ctx));
    }
    m_opl.max = attr.as_ullong();
    if (m_opl.min >= m_opl.max) {
        throw std::invalid_argument(fmt::format(
            "{}: 'minOPLatency' ({}) should be strictly less 'maxOPLatency' ({})",
            ctx, m_opl.min, m_opl.max));
    }

    m_volumeFrac = std::max(node.attribute("volumeFrac").as_double(0.0), 0.0);
    m_volumeWindowBars = std::max(node.attribute("volumeWindowBars").as_uint(30u), 1u);
    attr = node.attribute("volume");
    m_volume = (attr.empty() || attr.as_double() <= 0.0) ? 1.0 : attr.as_double();

    attr = node.attribute("tau");
    m_tau = (attr.empty() || attr.as_ullong() == 0) ? 120'000'000'000 : attr.as_ullong();

    
    attr = node.attribute("orderTypeProb");
    m_orderTypeProb = (attr.empty() || attr.as_float() <= 0.0f) ? 0.5f : attr.as_float();

    m_lastUpdate = std::vector<Timestamp>(m_bookCount,0);
    
    m_orderFlag = std::vector<bool>(m_bookCount, false);

    // Cache "external" FuturesSignal per book — eliminates per-tick string
    // lookup + RTTI cast in placeOrder/getProcessDetails hot paths.
    m_externalSignal.reserve(m_bookCount);
    for (BookId b = 0; b < m_bookCount; ++b) {
        m_externalSignal.push_back(dynamic_cast<process::FuturesSignal*>(
            simulation()->exchange()->process("external", b)));
    }

    m_priceIncrement = 1 / std::pow(10, simulation()->exchange()->config().parameters().priceIncrementDecimals);
    m_volumeIncrement = 1 / std::pow(10, simulation()->exchange()->config().parameters().volumeIncrementDecimals);

    m_marketFeedLatencyDistribution = std::normal_distribution<double>{
        [&] {
            static constexpr const char* name = "MFLmean";
            if (auto attr = node.attribute(name); attr.empty()) {
                throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute '{}'", ctx, name)};
            } else {
                return attr.as_double();
            }
        }(),
        [&] {
            static constexpr const char* name = "MFLstd";
            if (auto attr = node.attribute(name); attr.empty()) {
                throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute '{}'", ctx, name)};
            } else {
                return attr.as_double();
            }
        }()
    };
    m_decisionMakingDelayDistribution = std::normal_distribution<double>{
        [&] {
            static constexpr const char* name = "delayMean";
            if (auto attr = node.attribute(name); attr.empty()) {
                throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute '{}'", ctx, name)};
            } else {
                return attr.as_double();
            }
        }(),
        [&] {
            static constexpr const char* name = "delaySTD";
            if (auto attr = node.attribute(name); attr.empty()) {
                throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute '{}'", ctx, name)};
            } else {
                return attr.as_double();
            }
        }()
    };

    
    attr = node.attribute("opLatencyScaleRay"); 
    const double scale = (attr.empty() || attr.as_double() == 0.0) ? 0.235 : attr.as_double();
    const double percentile = 1-std::exp(-1/(2*scale*scale));
    m_orderPlacementLatencyDistribution = std::make_unique<taosim::stats::RayleighDistribution>(scale, percentile); 

    m_baseName = [&] {
        std::string res = name();
        boost::algorithm::erase_regex(res, boost::regex("(_\\d+)$"));
        return res;
    }();

    size_t pos = name().find_last_not_of("0123456789");
    if (pos != std::string::npos && pos + 1 < name().size()) {
        std::string numStr = name().substr(pos + 1);
        m_catUId = static_cast<uint32_t>(std::stoul(numStr));
    }
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::receiveMessage(Message::Ptr msg)
{

    if (msg->type == "EVENT_SIMULATION_START") {
        handleSimulationStart();
    }
    else if (msg->type == "EVENT_SIMULATION_END") {
        handleSimulationStop();
    }
    else if (msg->type == "RESPONSE_SUBSCRIBE_EVENT_TRADE") {
        handleTradeSubscriptionResponse();
    }
    else if (msg->type == "RESPONSE_PLACE_ORDER_MARKET") {
        handleMarketOrderPlacementResponse(msg);
    }
    else if (msg->type == "ERROR_RESPONSE_PLACE_ORDER_MARKET") {
        handleMarketOrderPlacementErrorResponse(msg);
    }
    else if (msg->type == "RESPONSE_PLACE_ORDER_LIMIT") {
        handleLimitOrderPlacementResponse(msg);
    }
    else if (msg->type == "ERROR_RESPONSE_PLACE_ORDER_LIMIT") {
        handleLimitOrderPlacementErrorResponse(msg);
    }
    else if (msg->type == "EVENT_TRADE") {
        handleTrade(msg);
    }
    else if (msg->type == "WAKEUP") {
        handleWakeup(msg);
    }
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleSimulationStart()
{
    if (m_catUId == 0) {
        for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {

            auto chosenAgent = selectTurn();
            const Timestamp initDelay = decisionMakingDelay();

            auto& chains = simulation()->exchange()->wakeupChains().at(bookId);
            // The bound the watchdog waits out is the widest hop delay PLUS the L1 round
            // trip, because the token is consumed at the wake and handed on only once the
            // response lands.
            chains.registerChain(
                m_baseName,
                simulation()->currentTimestamp(),
                decisionMakingDelayBound() + marketFeedLatencyBound(),
                simulation()->localAgentManager()->roster()->at(m_baseName));
            if (chains.arm(m_baseName, simulation()->currentTimestamp(), initDelay)) {
                simulation()->dispatchMessage(
                    simulation()->currentTimestamp(),
                    initDelay,
                    name(),
                    fmt::format("{}_{}", m_baseName, chosenAgent),
                    "WAKEUP",
                    MessagePayload::create<WakeupPayload>(bookId));
            }
        }
    }
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleSimulationStop()
{}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleTradeSubscriptionResponse()
{

}

//-------------------------------------------------------------------------

uint64_t FuturesTraderAgent::selectTurn() {
    const auto& agentBaseNamesToCounts = simulation()->localAgentManager()->roster()->baseNamesToCounts();
    return  std::uniform_int_distribution<uint64_t>{0, agentBaseNamesToCounts.at(m_baseName) - 1}(*m_rng);
}

//------------------------------------------------------------------------

void FuturesTraderAgent::handleWakeup(Message::Ptr &msg)
{
    const auto payload = std::static_pointer_cast<WakeupPayload>(msg->payload);

    const BookId bookId = payload->bookId;
    const Timestamp now = simulation()->currentTimestamp();

    simulation()->exchange()->wakeupChains().at(bookId).noteWake(m_baseName, now);

    // Drawn FIRST, and included in the hop below. The gap between two wakes is the market
    // feed latency plus the decision delay, which is the arrival intensity this class was
    // calibrated at; drawing before selectTurn fixes the per-hop rng sequence.
    const Timestamp marketFeedDelay = marketFeedLatency();

    uint64_t chosenOne = selectTurn();
    const Timestamp hop = decisionMakingDelay() + marketFeedDelay;
    if (simulation()->exchange()->wakeupChains().at(bookId).arm(m_baseName, now, hop)) {
        simulation()->dispatchMessage(
            now,
            hop,
            name(),
            fmt::format("{}_{}", m_baseName, chosenOne),
            "WAKEUP",
            MessagePayload::create<WakeupPayload>(bookId));
    }

    if (m_orderFlag[bookId]) return;

    const auto l1 = simulation()->exchange()->statsHub()->l1(bookId);
    const double bestBid = taosim::util::decimal2double(l1.bestBidPrice);
    const double bestAsk = taosim::util::decimal2double(l1.bestAskPrice);
    placeOrder(bookId, bestAsk, bestBid);
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleMarketOrderPlacementResponse(Message::Ptr msg)
{
    const auto payload = std::static_pointer_cast<PlaceOrderMarketResponsePayload>(msg->payload);
    m_orderFlag[payload->requestPayload->bookId] = false;
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleMarketOrderPlacementErrorResponse(Message::Ptr msg)
{
    const auto payload =
        std::static_pointer_cast<PlaceOrderMarketErrorResponsePayload>(msg->payload);

    const BookId bookId = payload->requestPayload->bookId;

    m_orderFlag[bookId] = false;
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleLimitOrderPlacementResponse(Message::Ptr msg)
{
    const auto payload = std::static_pointer_cast<PlaceOrderLimitResponsePayload>(msg->payload);

    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        m_tau,
        name(),
        m_exchange,
        "CANCEL_ORDERS",
        MessagePayload::create<CancelOrdersPayload>(
            std::vector{taosim::event::Cancellation(payload->id)}, payload->requestPayload->bookId));

    m_orderFlag[payload->requestPayload->bookId] = false;
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleLimitOrderPlacementErrorResponse(Message::Ptr msg)
{
    const auto payload =
        std::static_pointer_cast<PlaceOrderLimitErrorResponsePayload>(msg->payload);

    const BookId bookId = payload->requestPayload->bookId;

    m_orderFlag[bookId] = false;
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleCancelOrdersResponse(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleCancelOrdersErrorResponse(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleTrade(Message::Ptr msg)
{
    const auto payload = std::static_pointer_cast<EventTradePayload>(msg->payload);
}


//-------------------------------------------------------------------------

void FuturesTraderAgent::placeOrder(BookId bookId, double bestAsk, double bestBid)
{
    const auto futuresDetails = getProcessDetails(bookId, "external");
    const double logReturn = futuresDetails.logReturn; 
    if (logReturn == 0.0 || futuresDetails.volumeFactor == 0.0) return;
    const double sign = logReturn < 0.0 ? -1.0 : 1.0;
    const double epsilon = std::normal_distribution{0.0, m_sigmaEps}(*m_rng);
    const double forecast =  sign + epsilon;
    
    // `volumeFrac` states the clip against the book's traded volume per second, so the same
    // signal is proportionate at any price level. Absent, the legacy absolute `volume`
    // applies, which is exponential in the signal strength rather than proportional.
    const double signalStrength = futuresDetails.volumeFactor;
    const double anchoredMean = [&] -> double {
        if (m_volumeFrac <= 0.0) return 0.0;
        const auto window =
            simulation()->exchange()->statsHub()->window(bookId, m_volumeWindowBars);
        if (!(window.seconds > 0.0) || !(window.volume > 0.0)) return 0.0;
        return m_volumeFrac * (window.volume / window.seconds) * signalStrength;
    }();
    // sigma is 1, so a lognormal's mean is exp(mu + 1/2); solve for the mu that lands the mean
    // on the target.
    const double newMean = anchoredMean > 0.0
        ? std::log(anchoredMean) - 0.5
        : std::log(m_volume) * signalStrength;
    std::lognormal_distribution<> lognormalDist(newMean, 1); 
    double volume = lognormalDist(*m_rng);
    // NOTE: this ties the limit offset to the ORDER SIZE, with a threshold at one base unit and
    // a sign flip below it. It was already odd when the size was an absolute constant; with the
    // size anchored to the book's flow the threshold has no fixed meaning at all, since a clip
    // of "one" is whatever a second of volume happens to be. Left alone here because changing
    // it moves where the class posts rather than how much, which is a separate decision.
    const double priceShift =  volume > 1.0 ? std::floor(volume)*m_priceIncrement : -1*std::floor(volume/m_priceIncrement)*m_priceIncrement;
    volume =  std::floor(volume / m_volumeIncrement) * m_volumeIncrement;
    if (volume == 0) return;
    const bool draw = std::bernoulli_distribution{m_orderTypeProb}(*m_rng);
    if (forecast > 0) {
        if (draw) {
            placeBuy(bookId, volume);
        } else {
            placeBid(bookId, volume, bestBid - priceShift);
        }
    }
    else if (forecast < 0) {
        if (draw) {
            placeSell(bookId, volume);
        } else {
            placeAsk(bookId, volume, bestAsk + priceShift);
        }
    }
}
//-------------------------------------------------------------------------

void FuturesTraderAgent::placeBid(BookId bookId, double volume, double price)
{
    m_orderFlag[bookId] = true;

    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        orderPlacementLatency(),
        name(),
        m_exchange,
        "PLACE_ORDER_LIMIT",
        MessagePayload::create<PlaceOrderLimitPayload>(
            OrderDirection::BUY,
            taosim::util::double2decimal(volume),
            taosim::util::double2decimal(price),
            bookId));
}
//-------------------------------------------------------------------------

void FuturesTraderAgent::placeBuy(BookId bookId, double volume)
{
    m_orderFlag[bookId] = true;

    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        orderPlacementLatency(),
        name(),
        m_exchange,
        "PLACE_ORDER_MARKET",
        MessagePayload::create<PlaceOrderMarketPayload>(
            OrderDirection::BUY,
            taosim::util::double2decimal(volume),
            bookId));
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::placeAsk(BookId bookId, double volume, double price)
{
    m_orderFlag[bookId] = true;
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        orderPlacementLatency(),
        name(),
        m_exchange,
        "PLACE_ORDER_LIMIT",
        MessagePayload::create<PlaceOrderLimitPayload>(
            OrderDirection::SELL,
            taosim::util::double2decimal(volume),
            taosim::util::double2decimal(price),
            bookId));
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::placeSell(BookId bookId, double volume)
{
    m_orderFlag[bookId] = true;
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        orderPlacementLatency(),
        name(),
        m_exchange,
        "PLACE_ORDER_MARKET",
        MessagePayload::create<PlaceOrderMarketPayload>(
            OrderDirection::SELL,
            taosim::util::double2decimal(volume),
            bookId));
}

//-------------------------------------------------------------------------

Timestamp FuturesTraderAgent::orderPlacementLatency()
{
    return static_cast<Timestamp>(
        std::lerp(m_opl.min, m_opl.max, m_orderPlacementLatencyDistribution->sample(*m_rng)));
}

//-------------------------------------------------------------------------

Timestamp FuturesTraderAgent::marketFeedLatency()
{
    return static_cast<Timestamp>(std::min(std::abs(m_marketFeedLatencyDistribution(*m_rng)),
            m_marketFeedLatencyDistribution.mean() + 3 * m_marketFeedLatencyDistribution.stddev()));
}

//-------------------------------------------------------------------------

Timestamp FuturesTraderAgent::marketFeedLatencyBound() const
{
    return static_cast<Timestamp>(
        m_marketFeedLatencyDistribution.mean() + 3 * m_marketFeedLatencyDistribution.stddev());
}

//-------------------------------------------------------------------------

Timestamp FuturesTraderAgent::decisionMakingDelayBound() const
{
    return static_cast<Timestamp>(
        m_decisionMakingDelayDistribution.mean()
        + 3.0 * m_decisionMakingDelayDistribution.stddev());
}

//-------------------------------------------------------------------------

Timestamp FuturesTraderAgent::decisionMakingDelay()
{
    return static_cast<Timestamp>(std::min(std::abs(m_decisionMakingDelayDistribution(*m_rng)),
            m_decisionMakingDelayDistribution.mean()
            + 3.0 * m_decisionMakingDelayDistribution.stddev()));
}

//-------------------------------------------------------------------------

double FuturesTraderAgent::getProcessValue(BookId bookId, const std::string& name)
{
    return simulation()->exchange()->process(name, bookId)->value();
}

//-------------------------------------------------------------------------

FuturesTraderAgent::FuturesDetails FuturesTraderAgent::getProcessDetails(
    BookId bookId, const std::string& name)
{
    // Fast path: placeOrder's hot loop only ever asks for "external".
    const auto externalProcess = (name == "external")
        ? m_externalSignal[bookId]
        : dynamic_cast<process::FuturesSignal*>(simulation()->exchange()->process(name, bookId));
    return {
        .logReturn = externalProcess->state().logReturn,
        .volumeFactor = externalProcess->state().volumeFactor
    };
}

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------