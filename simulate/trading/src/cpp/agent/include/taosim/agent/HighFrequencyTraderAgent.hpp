/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/common.hpp>
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include "Distribution.hpp"
#include "Agent.hpp"
#include "Order.hpp"

#include <boost/circular_buffer.hpp>

#include <taosim/util/TradeStatsEstimator.hpp>

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

class HighFrequencyTraderAgent : public Agent
{
public:
    struct RecordedOrder
    {
        OrderID orderId;
        double price;
        double volume;
        OrderDirection direction;
        bool traded;
        bool canceled;
    };

    HighFrequencyTraderAgent() noexcept = default;
    HighFrequencyTraderAgent(Simulation* simulation) noexcept;

    // TODO: Wrap state into a struct and provide a single access point here.
    [[nodiscard]] auto&& topLevel(this auto&& self) noexcept { return self.m_topLevel; }
    [[nodiscard]] auto&& inventory(this auto&& self) noexcept { return self.m_inventory; }
    [[nodiscard]] auto&& baseFree(this auto&& self) noexcept { return self.m_baseFree; }
    [[nodiscard]] auto&& quoteFree(this auto&& self) noexcept { return self.m_quoteFree; }
    [[nodiscard]] auto&& orderFlag(this auto&& self) noexcept { return self.m_orderFlag; }
    [[nodiscard]] auto&& recordedOrders(this auto&& self) noexcept { return self.m_recordedOrders; }
    [[nodiscard]] auto&& deltaHFT(this auto&& self) noexcept { return self.m_deltaHFT; }
    [[nodiscard]] auto&& tauHFT(this auto&& self) noexcept { return self.m_tauHFT; }
    [[nodiscard]] auto&& lastPrice(this auto&& self) noexcept { return self.m_lastPrice; }
    [[nodiscard]] auto&& id(this auto&& self) noexcept { return self.m_id; }
    [[nodiscard]] auto&& pRes(this auto&& self) noexcept { return self.m_pRes; }

    virtual void configure(const pugi::xml_node& node) override;
    virtual void receiveMessage(Message::Ptr msg) override;

private:

    void handleSimulationStart();
    void handleSimulationStop();
    void handleTradeSubscriptionResponse();
    void handleRetrieveL1ExtResponse(Message::Ptr msg);
    void handleLimitOrderPlacementResponse(Message::Ptr msg);
    void handleLimitOrderPlacementErrorResponse(Message::Ptr msg);
    void handleMarketOrderPlacementResponse(Message::Ptr msg);
    void handleMarketOrderPlacementErrorResponse(Message::Ptr msg);
    void handleCancelOrdersResponse(Message::Ptr msg);
    void handleCancelOrdersErrorResponse(Message::Ptr msg);
    void handleTrade(Message::Ptr msg);

    [[nodiscard]] double effectiveSigmaSqr(BookId bookId) const;
    void placeOrder(BookId bookId, const TopLevelWithVolumes& topLevel);
    std::optional<PlaceOrderLimitPayload::Ptr> makeOrder(
        BookId bookId, OrderDirection direction, double volume, double limitPrice, double wealth);
    void sendOrder(std::optional<PlaceOrderLimitPayload::Ptr> payload);
    Timestamp orderPlacementLatency();

    // Parameters, injections.
    std::mt19937* m_rng;
    double m_wealthFrac;
    std::string m_exchange;
    uint32_t m_bookCount;
    double m_tau;
    double m_gHFT;
    double m_delta;
    double m_kappa;
    double m_spread;
    double m_priceInit;
    DelayBounds m_opl;
    double m_psi;
    std::unique_ptr<taosim::stats::Distribution> m_orderPlacementLatencyDistribution;
    double m_orderMean;
    double m_orderSTD;
    std::vector<double> m_orderSizeBid;
    std::vector<double> m_orderSizeAsk;
    std::vector<int> m_lastInvSign;
    std::vector<taosim::util::TradeStatsEstimator> m_varEst;
    double m_varHalflifeSeconds{};
    bool m_varJumpRobust{};
    double m_varGain{};
    double m_varRatioCap{};
    int m_undercutTicks{};
    double m_noiseRay;
    std::unique_ptr<taosim::stats::Distribution> m_priceShiftDistribution;
    Timestamp m_minMFLatency;
    double m_shiftPercentage;
    double m_sigmaSqr;
    double m_priceIncrement;
    double m_volumeIncrement;
    double m_maxLeverage;
    double m_maxRate;
    double m_sigmaMargin;
    double m_rateSensitivity;
    double m_spreadSensitivityExp;
    double m_spreadSensitivityBase;
    double m_maxLoan;
    bool m_debug;
    int m_spreadMode;
    int m_minSpreadTicks;
    int m_numLevels;
    int m_levelSpacingTicks;
    double m_spreadRefPrice;
    int m_rebalanceGateMode{0};
    // State.
    std::vector<TopLevelWithVolumes> m_topLevel;
    std::vector<double> m_inventory;
    std::vector<double> m_baseFree;
    std::vector<double> m_quoteFree;
    std::vector<bool> m_orderFlag;
    std::vector<double> m_deltaHFT;
    std::vector<Timestamp> m_tauHFT;
    std::vector<TimestampedPrice> m_lastPrice;
    std::vector<double> m_pRes;
    AgentId m_id;
};

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
