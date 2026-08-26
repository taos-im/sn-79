/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/agent/NoiseTraderAgent.hpp>

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MessagePayload.hpp>
#include <taosim/process/MagneticField.hpp>
#include "DistributionFactory.hpp"
#include "RayleighDistribution.hpp"
#include "Simulation.hpp"
#include <taosim/util/RootFinding.hpp>

#include <boost/algorithm/string/regex.hpp>
#include <boost/bimap.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/random.hpp>

#include <algorithm>

//-------------------------------------------------------------------------

namespace taosim::agent
{

inline auto investmentPosition = [](double price, double forecast, double variance, double base, double quote) {
    return (std::log(forecast/price) + variance)/(variance*price);
};

NoiseTraderAgent::NoiseTraderAgent(Simulation* simulation) noexcept
    : Agent{simulation}
{}

//-------------------------------------------------------------------------

void NoiseTraderAgent::configure(const pugi::xml_node& node)
{
    Agent::configure(node);

    m_rng = &simulation()->rng();

    pugi::xml_attribute attr;
    static constexpr auto ctx = std::source_location::current().function_name();

    // -- BEGIN Config general
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
    m_priceIncrement = 1 / std::pow(10, simulation()->exchange()->config().parameters().priceIncrementDecimals);
    m_volumeIncrement = 1 / std::pow(10, simulation()->exchange()->config().parameters().volumeIncrementDecimals);

    m_debug = node.attribute("debug").as_bool();

    m_baseName = [&] {
        std::string res = name();
        boost::algorithm::erase_regex(res, boost::regex("(_\\d+)$"));
        return res;
    }();
    // -- END


    // -- BEGIN Latencies OPL AND MFL
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
    attr = node.attribute("opLatencyScaleRay"); 
    const double scale = (attr.empty() || attr.as_double() == 0.0) ? 0.235 : attr.as_double();
    const double percentile = 1-std::exp(-1/(2*scale*scale));
    m_orderPlacementLatencyDistribution =  std::make_unique<taosim::stats::RayleighDistribution>(scale, percentile); 


    // ACD PARAMETERS
    m_acdDelayDist = std::weibull_distribution<float>{1.0, 1.0}; 
    if (attr = node.attribute("acdOmega"); attr.empty()) {
        throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute 'acdOmega'", ctx)};
    }
    m_omegaDu = attr.as_float();
    if (attr = node.attribute("acdAlpha"); attr.empty() || attr.as_float() < 0.0f) {
            throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute 'acdAlpha'", ctx)};
    }
    m_alphaDu =  attr.as_float(); 
    if (attr = node.attribute("acdBeta"); attr.empty() || attr.as_float() < 0.0f || m_alphaDu + attr.as_float() > 1) {
            throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute 'acdBeta' or values need to be checked", ctx)};
    }
    m_betaDu =  attr.as_float(); 
    m_gammaDu = node.attribute("acdGamma").as_float(1.0);
    attr = node.attribute("maxDMD");
    m_maxDelay = (attr.empty() || attr.as_ullong() < 1'000'000'000) ? static_cast<Timestamp>(450'000'000'000) : attr.as_ullong();
    attr = node.attribute("minDMD");
    m_minDelay = (attr.empty() || attr.as_ullong() < 100'000'000) ? static_cast<Timestamp>(100'000'000) : attr.as_ullong();

    // BEGIN Order details
    attr = node.attribute("meanVolume");
    double meanVol = (attr.empty() || attr.as_double() <= 0.0) ? 1.0 : attr.as_double();
    std::lognormal_distribution<> lognormalDist(meanVol, 1.0); 
    m_volumeConst = lognormalDist(*m_rng);

    attr = node.attribute("balanceCoef");
    m_balanceCoef=  (attr.empty() || attr.as_double() <= 0.0) ? 0.5 : attr.as_double();

    m_sigma = node.attribute("sigmaExp").as_double(0.00001);
    m_feeReserveFrac = std::clamp(node.attribute("feeReserveFrac").as_double(0.01), 0.0, 0.5);
    m_mWeight = node.attribute("weight").as_double(0.1);

    try {
        (void)simulation()->exchange()->process("magneticfield", 0);
    } catch (const std::exception&) {
        throw std::invalid_argument(fmt::format(
            "{}: requires a Books process named 'magneticfield' (used for the Ising field "
            "and to hold ACD wakeup state)", name()));
    }

    attr = node.attribute("tau");
    m_tau = (attr.empty() || attr.as_ullong() == 0) ? 120'000'000'000 : attr.as_ullong();

    m_state.orderFlag = std::vector<bool>(m_bookCount, false);

    m_magneticField.reserve(m_bookCount);
    for (BookId b = 0; b < m_bookCount; ++b) {
        m_magneticField.push_back(dynamic_cast<process::MagneticField*>(
            simulation()->exchange()->process("magneticfield", b)));
    }

    size_t pos = name().find_last_not_of("0123456789");
    if (pos != std::string::npos && pos + 1 < name().size()) {
        std::string numStr = name().substr(pos + 1);
        m_catUId = static_cast<uint32_t>(std::stoul(numStr));
    }

    m_logFlag = node.attribute("log").as_bool();

}

//-------------------------------------------------------------------------

void NoiseTraderAgent::receiveMessage(Message::Ptr msg)
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
    else if (msg->type == "RESPONSE_RETRIEVE_L1") {
        handleRetrieveL1Response(msg);
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
    }  else if (msg->type == "WAKEUP") {
        handleWakeup(msg);
    }
}

//-------------------------------------------------------------------------

void NoiseTraderAgent::handleSimulationStart()
{
    if (m_catUId == 0) {
        for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {

            auto chosenAgent = selectTurn();
            simulation()->dispatchMessage(
                simulation()->currentTimestamp(),
                marketFeedLatency(),
                name(),
                fmt::format("{}_{}", m_baseName, chosenAgent),
                "WAKEUP",
                MessagePayload::create<RetrieveL1Payload>(bookId));

            const auto field = m_magneticField[bookId];
            const float initPsi = m_omegaDu / (1.0f - m_alphaDu - m_betaDu);
            field->insertDurationComp(m_baseName, process::DurationComp{.delay=initPsi, .psi=initPsi});
        }
    }
}

//-------------------------------------------------------------------------

void NoiseTraderAgent::handleSimulationStop()
{
    if (m_catUId != 0) return;
    for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {
        const auto field = dynamic_cast<process::MagneticField*>(
            simulation()->exchange()->process("magneticfield", bookId));
        if (field) field->emitDiagnostics(m_baseName, bookId);
    }
}

//-------------------------------------------------------------------------

void NoiseTraderAgent::handleTradeSubscriptionResponse()
{

}

//------------------------------------------------------------------------

uint64_t NoiseTraderAgent::selectTurn()
{
    const auto& agentBaseNamesToCounts = simulation()->localAgentManager()->roster()->baseNamesToCounts();
    return  std::uniform_int_distribution<uint64_t>{0, agentBaseNamesToCounts.at(m_baseName) - 1}(*m_rng);
}

//------------------------------------------------------------------------

void NoiseTraderAgent::handleWakeup(Message::Ptr &msg)
{
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        marketFeedLatency(),
        name(),
        m_exchange,
        "RETRIEVE_L1",
        msg->payload);
}

//-------------------------------------------------------------------------

void NoiseTraderAgent::handleRetrieveL1Response(Message::Ptr msg)
{
    const auto payload = std::static_pointer_cast<RetrieveL1ResponsePayload>(msg->payload);

    const BookId bookId = payload->bookId;

    uint64_t chosenOne = selectTurn();
    const auto field = m_magneticField[bookId];
    double avgMagnetism = std::abs(field->avgMagnetism());
    const auto lastDurationComp = field->getDurationComp(m_baseName);
    float lastDelay = lastDurationComp.delay;
    float psi_prev = lastDurationComp.psi;
    float psi_next = m_omegaDu + m_alphaDu * lastDelay + m_betaDu *psi_prev + m_gammaDu*std::log(1-avgMagnetism);
    if (!std::isfinite(psi_next)) {
        psi_next = m_omegaDu/(1-m_alphaDu - m_betaDu);
    }
    double delayRaw = static_cast<double>(std::exp(psi_next)) * m_acdDelayDist(*m_rng);
    if (!std::isfinite(delayRaw) || delayRaw > static_cast<double>(m_maxDelay)) {
        delayRaw = static_cast<double>(m_maxDelay);
    }
    Timestamp delay_timestamped = std::clamp(static_cast<Timestamp>(delayRaw), m_minDelay, m_maxDelay);
    const float delay = static_cast<float>(delay_timestamped);
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        delay_timestamped,
        name(),
        fmt::format("{}_{}", m_baseName, chosenOne),
        "WAKEUP",
        MessagePayload::create<RetrieveL1Payload>(bookId));

    field->insertDurationComp(m_baseName, process::DurationComp{.delay=std::log(delay), .psi=psi_next});

    double bestBid = taosim::util::decimal2double(payload->bestBidPrice);
    double bestAsk = taosim::util::decimal2double(payload->bestAskPrice);    
    if  (bestBid == 0.0) bestBid = m_price; 
    if  (bestAsk == 0.0) bestAsk = bestBid + m_priceIncrement; 
    const double midQuote = 0.5*(bestAsk + bestBid);
    m_price = midQuote;
    placeOrder(bookId);
}

//-------------------------------------------------------------------------

void NoiseTraderAgent::handleMarketOrderPlacementResponse(Message::Ptr msg)
{
    const auto payload = std::static_pointer_cast<PlaceOrderMarketResponsePayload>(msg->payload);
    m_state.orderFlag[payload->requestPayload->bookId] = false;
}

//-------------------------------------------------------------------------

void NoiseTraderAgent::handleMarketOrderPlacementErrorResponse(Message::Ptr msg)
{
    const auto payload =
        std::static_pointer_cast<PlaceOrderMarketErrorResponsePayload>(msg->payload);

    const BookId bookId = payload->requestPayload->bookId;

    m_state.orderFlag[bookId] = false;
}

//-------------------------------------------------------------------------

void NoiseTraderAgent::handleLimitOrderPlacementResponse(Message::Ptr msg)
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

    m_state.orderFlag[payload->requestPayload->bookId] = false;
}

//-------------------------------------------------------------------------

void NoiseTraderAgent::handleLimitOrderPlacementErrorResponse(Message::Ptr msg)
{
    const auto payload =
        std::static_pointer_cast<PlaceOrderLimitErrorResponsePayload>(msg->payload);

    const BookId bookId = payload->requestPayload->bookId;

    m_state.orderFlag[bookId] = false;
}

//-------------------------------------------------------------------------

void NoiseTraderAgent::handleCancelOrdersResponse(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void NoiseTraderAgent::handleCancelOrdersErrorResponse(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void NoiseTraderAgent::handleTrade(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void NoiseTraderAgent::placeOrder(BookId bookId)
{
    const auto field = m_magneticField[bookId];
    const int sign = field->signAt(m_catUId);
    if (m_logFlag) field->logState(simulation()->currentTimestamp(), m_catUId);
    const float magnetism = field->avgMagnetism();
    const float avgMagnetism = std::abs(magnetism);
    double balance =m_balanceCoef*(1+sign*magnetism)/2; 
    double volume = balance*m_volumeConst;
    const auto freeBase =
        taosim::util::decimal2double(simulation()->account(name()).at(bookId).base.getFree());
    const auto freeQuote =
        taosim::util::decimal2double(simulation()->account(name()).at(bookId).quote->getFree());
    float adjustedRet = m_sigma +  m_mWeight*magnetism + field->magnetismReturn();
    ForecastResult forecastResult = {.price= m_price*std::exp(adjustedRet), .varianceOfLastLogReturns=m_sigma};
    const auto [indifferencePrice, indifferencePriceConverged] =
        calculateIndifferencePrice(forecastResult, freeBase, freeQuote);
    if (!indifferencePriceConverged) return;

    const auto [minimumPrice, minimumPriceConverged] =
        calculateMinimumPrice(forecastResult, freeBase, freeQuote, indifferencePrice);
    if (!minimumPriceConverged) return;

    const auto maximumPrice = forecastResult.price;
    double weight; 
    if (sign*magnetism > 0) {
        if (std::bernoulli_distribution(avgMagnetism)(*m_rng)) {
            if (sign > 0) {
                volume = balance*(freeQuote/(m_price*(1.0 + m_sigma*3.0)));
                placeBuy(bookId, volume);
            } else if (sign < 0) {
                volume = balance*(freeBase);
                placeSell(bookId, volume);
            }
            return;
        }
        weight = std::abs(0.5-avgMagnetism);
    } else {
        weight = 1- avgMagnetism;
    }
    const double sampleLow = std::max(minimumPrice, m_priceIncrement);
    const double sampleHigh = maximumPrice;
    if (sampleLow >= sampleHigh) return;

    const double sampledPrice = samplePrice(sampleLow, indifferencePrice, sampleHigh, sign, weight);
    const double price = std::round(sampledPrice / m_priceIncrement) * m_priceIncrement;
    if (price <= 0.0) return;
    if (sampledPrice < indifferencePrice) {
        volume = calcPositionPrice(forecastResult,sampledPrice,freeBase,freeQuote) - freeBase;
        volume = std::min(volume, freeQuote / price * (1.0 - m_feeReserveFrac));
        placeBid(bookId,volume,price);
        field->setValAt(m_catUId, 1);
    } else if (sampledPrice > indifferencePrice) {
        volume = freeBase - calcPositionPrice(forecastResult, sampledPrice,freeBase,freeQuote);
        volume = std::min(volume, freeBase);
        placeAsk(bookId, volume, price);
        field->setValAt(m_catUId, -1);
    }
}
//-------------------------------------------------------------------------
double NoiseTraderAgent::samplePrice(double minP, double indiffP, double maxP,
                   int sign, double weight)
{
    if (!(maxP > minP)) {
        return minP;
    }
    const double i = std::clamp((indiffP - minP) / (maxP - minP), 0.0, 1.0);

    double mode;
    if (sign >= 0) {
        mode = i * (1.0 - weight);
    }
    else {
        mode = i + (1.0 - i) * weight;
    }
    mode = std::clamp(mode, 0.0, 1.0);

    double s = 6.0;
    double alpha = mode * (s - 2.0) + 1.0;
    double beta  = (1.0 - mode) * (s - 2.0) + 1.0;
    std::gamma_distribution<double> distA(alpha, 1.0);
    std::gamma_distribution<double> distB(beta, 1.0);

    double x = distA(*m_rng);
    double y = distB(*m_rng);
    double betaValue = x / (x + y);
    return minP + betaValue * (maxP - minP);
}
//-------------------------------------------------------------------------

NoiseTraderAgent::OptimizationResult NoiseTraderAgent::calculateIndifferencePrice(
    const NoiseTraderAgent::ForecastResult& forecastResult, double freeBase, double freeQuote)
{
    auto residual = [&](double x) {
        return investmentPosition(x, forecastResult.price,
            forecastResult.varianceOfLastLogReturns, freeBase, freeQuote) - freeBase;
    };
    const auto upperBound =
        forecastResult.price * std::exp(forecastResult.varianceOfLastLogReturns);
    const auto root = util::solveScalarBracketed(residual, m_priceIncrement, upperBound);
    return {.value = root.value, .converged = root.converged};
}

//-------------------------------------------------------------------------

NoiseTraderAgent::OptimizationResult NoiseTraderAgent::calculateMinimumPrice(
    const NoiseTraderAgent::ForecastResult& forecastResult,
    double freeBase,
    double freeQuote,
    double indifferencePrice)
{
    auto residual = [&](double x) {
        return x * (investmentPosition(x, forecastResult.price,
            forecastResult.varianceOfLastLogReturns, freeBase, freeQuote) - freeBase) - freeQuote;
    };
    const auto root = util::solveScalarBracketed(residual, m_priceIncrement, indifferencePrice);
    if (!root.converged && root.status == util::RootStatus::SameSignNegative) {
        return {.value = m_priceIncrement, .converged = true};
    }
    return {.value = root.value, .converged = root.converged};
}

// -------------------------------------------------------------------------
double NoiseTraderAgent::calcPositionPrice(const NoiseTraderAgent::ForecastResult& forecastResult, double price, double freeBase, double freeQuote) {
    return investmentPosition(price, forecastResult.price, forecastResult.varianceOfLastLogReturns, freeBase, freeQuote);
}
void NoiseTraderAgent::placeBid(BookId bookId, double volume, double price)
{
    volume = std::floor(volume / m_volumeIncrement) * m_volumeIncrement;
    if (volume == 0) return;
    m_state.orderFlag[bookId] = true;

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

void NoiseTraderAgent::placeBuy(BookId bookId, double volume)
{
    m_state.orderFlag[bookId] = true;

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

void NoiseTraderAgent::placeAsk(BookId bookId, double volume, double price)
{
    volume = std::floor(volume / m_volumeIncrement) * m_volumeIncrement;
    if (volume == 0) return;
    m_state.orderFlag[bookId] = true;
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

void NoiseTraderAgent::placeSell(BookId bookId, double volume)
{
    m_state.orderFlag[bookId] = true;
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

Timestamp NoiseTraderAgent::orderPlacementLatency()
{
    return static_cast<Timestamp>(
        std::lerp(m_opl.min, m_opl.max, m_orderPlacementLatencyDistribution->sample(*m_rng)));
}

//-------------------------------------------------------------------------

Timestamp NoiseTraderAgent::marketFeedLatency()
{
    return static_cast<Timestamp>(std::min(
        std::abs(m_marketFeedLatencyDistribution(*m_rng)),
        m_marketFeedLatencyDistribution.mean() + 3 * m_marketFeedLatencyDistribution.stddev()));
}

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------