/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/common.hpp>
#include "Agent.hpp"
#include "GBMValuationModel.hpp"
#include "Distribution.hpp"
#include "Order.hpp"

#include <cmath>
#include <random>

//-------------------------------------------------------------------------

namespace taosim::process { class FuturesSignal; }

namespace taosim::agent
{

//-------------------------------------------------------------------------

class FuturesTraderAgent : public Agent
{
public:
    FuturesTraderAgent() noexcept = default;
    explicit FuturesTraderAgent(Simulation* simulation) noexcept;

    // TODO: Wrap state into a struct and provide a single access point here.
    [[nodiscard]] auto&& volumeFactor(this auto&& self) noexcept { return self.m_volumeFactor; }
    [[nodiscard]] auto&& factorCounter(this auto&& self) noexcept { return self.m_factorCounter; }
    [[nodiscard]] auto&& lastUpdate(this auto&& self) noexcept { return self.m_lastUpdate; }
    [[nodiscard]] auto&& orderFlag(this auto&& self) noexcept { return self.m_orderFlag; }
    [[nodiscard]] auto&& tradePrice(this auto&& self) noexcept { return self.m_tradePrice; }

    virtual void configure(const pugi::xml_node& node) override;
    virtual void receiveMessage(Message::Ptr msg) override;

private:
    struct FuturesDetails
    {
        double logReturn;
        double volumeFactor;
    };

    void handleSimulationStart();
    void handleSimulationStop();
    void handleTradeSubscriptionResponse();
    void handleWakeup(Message::Ptr &msg);
    void handleMarketOrderPlacementResponse(Message::Ptr msg);
    void handleMarketOrderPlacementErrorResponse(Message::Ptr msg);
    void handleLimitOrderPlacementResponse(Message::Ptr msg);
    void handleLimitOrderPlacementErrorResponse(Message::Ptr msg);
    void handleCancelOrdersResponse(Message::Ptr msg);
    void handleCancelOrdersErrorResponse(Message::Ptr msg);
    void handleTrade(Message::Ptr msg);
    uint64_t selectTurn();

    void placeOrder(BookId bookId, double bestAsk, double bestBid);
    void placeBid(BookId bookId,double volume, double price);
    void placeBuy(BookId bookId,double volume);
    void placeAsk(BookId bookId, double volume, double price);
    void placeSell(BookId bookId, double volume);
    double getProcessValue(BookId bookId, const std::string& name);
    FuturesDetails getProcessDetails(BookId bookId, const std::string& name);
    Timestamp orderPlacementLatency();
    Timestamp marketFeedLatency();
    Timestamp decisionMakingDelay();
    // The clamps the two delay draws apply, without drawing from the rng.
    [[nodiscard]] Timestamp marketFeedLatencyBound() const;
    [[nodiscard]] Timestamp decisionMakingDelayBound() const;

    // Parameters, injections.
    std::mt19937* m_rng;
    std::string m_exchange;
    uint32_t m_bookCount;
    double m_sigmaN;
    double m_sigmaEps;
    DelayBounds m_opl;
    double m_volume;
    // Clip as a multiple of the book's traded volume per second, times the signal strength.
    // Zero keeps the absolute `volume`.
    double m_volumeFrac{};
    uint32_t m_volumeWindowBars{};
    float m_lambda;
    Timestamp m_tau;
    float m_orderTypeProb;
    Timestamp m_historySize;
    double m_priceIncrement;
    double m_volumeIncrement;
    std::normal_distribution<double> m_marketFeedLatencyDistribution;
    std::normal_distribution<double> m_decisionMakingDelayDistribution;
    std::unique_ptr<taosim::stats::Distribution> m_orderPlacementLatencyDistribution;
    std::string m_baseName;
    uint32_t m_catUId;

    // State.
    std::vector<float> m_volumeFactor;
    std::vector<uint32_t> m_factorCounter;
    std::vector<Timestamp> m_lastUpdate;
    std::vector<bool> m_orderFlag;
    // Cached "external" FuturesSignal pointer per bookId.
    std::vector<taosim::process::FuturesSignal*> m_externalSignal;
    // std::vector<TimestampedPrice> m_tradePrice;
};

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
