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

    if (attr = node.attribute("psiHFT_constant"); attr.empty()) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'psiHFT_constant' should have a value greater than or equal to 0.0", ctx));
    }
    std::normal_distribution<double> inventoryControlDist(attr.as_double(), 10.0);
    m_psi = inventoryControlDist(*m_rng);
    m_topLevel = std::vector<TopLevelWithVolumes>(m_bookCount, TopLevelWithVolumes{});
    m_baseFree = std::vector<double>(m_bookCount, 0.);
    m_quoteFree = std::vector<double>(m_bookCount, 0.);
    m_inventory = std::vector<double>(m_bookCount, 0.);
    m_deltaHFT = std::vector<double>(m_bookCount, 0.);
    m_tauHFT = std::vector<Timestamp>(m_bookCount, Timestamp{});

    m_lastPrice.resize(m_bookCount);

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

    m_varHalflifeSeconds = node.attribute("varHalflifeSeconds").as_double(30.0);
    m_varJumpRobust = node.attribute("varJumpRobust").as_bool(true);
    m_varGain = std::clamp(node.attribute("varGain").as_double(0.0), 0.0, 1.0);
    m_varRatioCap = std::max(node.attribute("varRatioCap").as_double(4.0), 1.0);
    m_undercutTicks = std::max(node.attribute("undercutTicks").as_int(0), 0);
    m_varEst = std::vector<taosim::util::TradeStatsEstimator>(m_bookCount);
    for (auto& est : m_varEst) {
        est.configure(m_varHalflifeSeconds, m_varJumpRobust,
            node.attribute("varSlowHalflifeMultiple").as_double(30.0));
    }

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
    m_sigmaMargin = node.attribute("marginNoiseSTD").as_double(0.00002);
    m_rateSensitivity = node.attribute("sensitivityCoef").as_double(100.0);
    m_spreadSensitivityExp= node.attribute("spreadSensitivityExp").as_double(2.07);
    m_spreadSensitivityBase= node.attribute("spreadSensitivityBase").as_double(0.00119);
    m_maxLoan = taosim::util::decimal2double(simulation()->exchange()->getMaxLoan());

    m_spreadMode = node.attribute("spreadMode").as_int(0);
    m_minSpreadTicks = node.attribute("minSpreadTicks").as_int(0);
    m_numLevels = std::max(1, node.attribute("numLevels").as_int(1));
    m_levelSpacingTicks = std::max(1, node.attribute("levelSpacing").as_int(1));
    m_spreadRefPrice = node.attribute("spreadRefPrice").as_double(0.0);
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
    else if (msg->type == "RESPONSE_RETRIEVE_L1_EXT") {
        handleRetrieveL1ExtResponse(msg);
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
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            static_cast<Timestamp>(m_deltaHFT[bookId]),
            name(),
            m_exchange,
            "RETRIEVE_L1_EXT",
            MessagePayload::create<RetrieveL1ExtPayload>(bookId));
    }
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleSimulationStop()
{
    simulation()->logDebug("-----The simulation ends now----");
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleTradeSubscriptionResponse()
{
    
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleRetrieveL1ExtResponse(Message::Ptr msg)
{
    // Guards a null payload, not a type mismatch: static_pointer_cast cannot fail.
    const auto payload = std::static_pointer_cast<RetrieveL1ExtResponsePayload>(msg->payload);
    if (payload == nullptr) return;
    m_varEst.at(payload->bookId).update(payload->tradeStats, simulation()->currentTimestamp());
    const BookId bookId = payload->bookId;
    m_deltaHFT[bookId] = m_delta / (1.0 + std::exp(std::abs(m_inventory[bookId]) - m_psi));
    m_tauHFT[bookId] = std::max(
        static_cast<Timestamp>(m_tau * m_minMFLatency),
        static_cast<Timestamp>(std::ceil(m_tau * m_deltaHFT[bookId]))
    );
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        std::max(static_cast<Timestamp>(m_deltaHFT[bookId]), static_cast<Timestamp>(m_minMFLatency)),
        name(),
        m_exchange,
        "RETRIEVE_L1_EXT",
        MessagePayload::create<RetrieveL1ExtPayload>(bookId));

    auto& topLevel = m_topLevel.at(bookId);
    topLevel.bid = taosim::util::decimal2double(payload->bestBidPrice);
    topLevel.bidQty = taosim::util::decimal2double(payload->bestBidVolume);
    topLevel.ask = taosim::util::decimal2double(payload->bestAskPrice);
    topLevel.askQty = taosim::util::decimal2double(payload->bestAskVolume);
    
    if (topLevel.bid == 0.0)
        topLevel.bid = m_lastPrice.at(payload->bookId).price;
    if (topLevel.ask == 0.0) 
        topLevel.ask = m_lastPrice.at(payload->bookId).price;

    const double midquote = (topLevel.bid + topLevel.ask) / 2;
    m_lastPrice.at(payload->bookId) = TimestampedPrice{.timestamp=simulation()->currentTimestamp(), .price=midquote};
    
    m_baseFree[bookId] = m_wealthFrac * 
        taosim::util::decimal2double(simulation()->exchange()->account(name()).at(bookId).base.getFree());
    m_quoteFree[bookId] = m_wealthFrac * 
        taosim::util::decimal2double(simulation()->exchange()->account(name()).at(bookId).quote->getFree());

    double timescaling = 1-(simulation()->currentTimestamp()/ m_delta)/(simulation()->duration() / m_delta);
    m_pRes[bookId] = midquote - m_gHFT * m_inventory[bookId] * effectiveSigmaSqr(bookId) * timescaling;

    const double skew = std::abs(m_pRes[bookId] - midquote);

    placeOrder(bookId, topLevel);
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleLimitOrderPlacementResponse(Message::Ptr msg)
{
    const auto payload = std::static_pointer_cast<PlaceOrderLimitResponsePayload>(msg->payload);
    const BookId bookId = payload->requestPayload->bookId;

    m_deltaHFT[bookId] = m_delta / (1.0 + std::exp(std::abs(m_inventory[bookId]) - m_psi));
    m_tauHFT[bookId] = std::max(
        static_cast<Timestamp>(m_tau * m_minMFLatency),
        static_cast<Timestamp>(std::ceil(m_tau * m_deltaHFT[bookId]))
    );

    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        m_tauHFT[bookId],
        name(),
        m_exchange,
        "CANCEL_ORDERS",
        MessagePayload::create<CancelOrdersPayload>(
            std::vector{taosim::event::Cancellation(payload->id)}, payload->requestPayload->bookId));
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::handleLimitOrderPlacementErrorResponse(Message::Ptr msg)
{
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
    }

}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::sendOrder(std::optional<PlaceOrderLimitPayload::Ptr> payload) {
    
    if (payload.has_value()) {
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

double HighFrequencyTraderAgent::effectiveSigmaSqr(BookId bookId) const
{
    if (m_varGain <= 0.0) return m_sigmaSqr;
    const double ratio = std::clamp(
        m_varEst.at(bookId).ratio(), 1.0 / m_varRatioCap, m_varRatioCap);
    return m_sigmaSqr * (1.0 + m_varGain * (ratio - 1.0));
}

//-------------------------------------------------------------------------

void HighFrequencyTraderAgent::placeOrder(BookId bookId, const TopLevelWithVolumes& topLevel) {
    
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
        std::lognormal_distribution<double> lognormalDist(m_orderMean, m_orderSTD);
        m_orderSizeBid.at(bookId) = lognormalDist(*m_rng);
        m_orderSizeAsk.at(bookId) = lognormalDist(*m_rng);
        m_lastInvSign.at(bookId) = invSign;
    }

    double skipProb = std::exp(-1.0*std::pow(relativeSpread/m_spreadSensitivityBase, m_spreadSensitivityExp));
    double makerRate = taosim::util::decimal2double(simulation()->exchange()->clearingManager().feePolicy()->getRates(bookId,m_id).maker);
    
    const double rayleighShift =  m_noiseRay * std::sqrt(-2.0 * std::log(1.0 - m_shiftPercentage));
    const double optimalSpread = effectiveSigmaSqr(bookId)*m_gHFT*(1-(simulation()->currentTimestamp()/ m_delta)/(simulation()->duration()/m_delta))
     + 2/m_gHFT * std::log(1 + m_gHFT/m_kappa);
    double spread = optimalSpread*(1+makerRate*m_rateSensitivity);
    if (m_spreadMode == 1) {
        const double refPrice = (m_spreadRefPrice > 0.0) ? m_spreadRefPrice : m_priceInit;
        if (refPrice > 0.0) {
            spread *= midquote / refPrice;
        }
    }
    if (m_minSpreadTicks > 0) {
        spread = std::max(spread, static_cast<double>(m_minSpreadTicks) * m_priceIncrement);
    }

    double noiseScale = 1.0;
    if (m_spreadMode == 1) {
        const double refPrice = (m_spreadRefPrice > 0.0) ? m_spreadRefPrice : m_priceInit;
        if (refPrice > 0.0) noiseScale = midquote / refPrice;
    }

    double wealthBid = topLevel.ask * m_baseFree[bookId] + m_quoteFree[bookId];
    double orderVolumeBid = m_orderSizeBid.at(bookId);
    double orderVolumeAsk = m_orderSizeAsk.at(bookId);
    double noiseBid = (m_priceShiftDistribution->sample(*m_rng) - rayleighShift) * noiseScale;
    double priceOrderBid = m_pRes.at(bookId) - (spread / 2.0) - noiseBid;

    double wealthAsk = topLevel.bid * m_baseFree[bookId] + m_quoteFree[bookId];
    double noiseAsk = (m_priceShiftDistribution->sample(*m_rng) - rayleighShift) * noiseScale;
    double priceOrderAsk = m_pRes.at(bookId) + (spread / 2.0) + noiseAsk;

    if (m_undercutTicks > 0) {
        const double step = static_cast<double>(m_undercutTicks) * m_priceIncrement;
        if (topLevel.ask > 0.0) {
            const double limitAsk = m_pRes.at(bookId) + (spread / 2.0);
            priceOrderAsk = std::max(limitAsk, std::min(priceOrderAsk, topLevel.ask - step));
        }
        if (topLevel.bid > 0.0) {
            const double limitBid = m_pRes.at(bookId) - (spread / 2.0);
            priceOrderBid = std::min(limitBid, std::max(priceOrderBid, topLevel.bid + step));
        }
        if (priceOrderBid >= priceOrderAsk) {
            priceOrderBid = m_pRes.at(bookId) - (spread / 2.0);
            priceOrderAsk = m_pRes.at(bookId) + (spread / 2.0);
        }
    }
    double rateProb;
    if (m_rebalanceGateMode == 1) {
        rateProb = 1.0 / (1.0 + std::exp((makerRate - m_maxRate) / std::sqrt(m_sigmaMargin)));
    } else {
        rateProb = std::exp(-std::pow((makerRate - m_maxRate), 2.0)/(2*m_sigmaMargin));
    }
    double inventoryProb = std::clamp(std::abs(currentInventory)/m_psi, m_sigmaMargin, 1- m_sigmaMargin);
    const bool rebalanceDraw =
        std::bernoulli_distribution{skipProb*rateProb*inventoryProb} (*m_rng);
    const bool rebalanceInvOk = std::abs(currentInventory) > 0.1;
    if (rebalanceDraw && rebalanceInvOk) {
        OrderDirection direction = currentInventory <= 0 ? OrderDirection::BUY : OrderDirection::SELL;
        double maxQtyTop = currentInventory <=0 ? topLevel.askQty : topLevel.bidQty;
        double rebalanceQty = std::uniform_real_distribution<double>{0.1,std::min(std::abs(currentInventory),maxQtyTop)} (*m_rng);
        simulation()->dispatchMessage(
                simulation()->currentTimestamp(),
                orderPlacementLatency(),
                name(),
                m_exchange,
                "PLACE_ORDER_MARKET",
                MessagePayload::create<PlaceOrderMarketPayload>(
                direction, taosim::util::double2decimal(rebalanceQty,simulation()->exchange()->config().parameters().volumeIncrementDecimals), bookId));
    } else {
        const double bidVolPerLevel = orderVolumeBid / static_cast<double>(m_numLevels);
        const double askVolPerLevel = orderVolumeAsk / static_cast<double>(m_numLevels);
        for (int lvl = 0; lvl < m_numLevels; ++lvl) {
            const double offset = static_cast<double>(lvl) * static_cast<double>(m_levelSpacingTicks) * m_priceIncrement;
            const double bidPx = std::round((priceOrderBid - offset) / m_priceIncrement) * m_priceIncrement;
            const double askPx = std::round((priceOrderAsk + offset) / m_priceIncrement) * m_priceIncrement;
            sendOrder(makeOrder(bookId, OrderDirection::SELL, askVolPerLevel, askPx, wealthAsk));
            sendOrder(makeOrder(bookId, OrderDirection::BUY,  bidVolPerLevel, bidPx, wealthBid));
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
