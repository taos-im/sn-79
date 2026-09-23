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

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

class HighFrequencyTraderAgent : public Agent
{
public:
    // The agent's own quote on one side of one book. It keeps exactly one a side, so this
    // is the whole ledger: two entries per book, and no per-level record of anything.
    struct RestingQuote
    {
        OrderID id{};
        // What is still resting, so a fill can be netted off it and a fully-filled quote is
        // not pointlessly cancelled.
        double volume{};
        bool live{};
        // A placement dispatched whose response has not come back yet. Without this the
        // agent cannot tell "nothing out on this side" from "something out that I do not
        // know the id of", and every requote inside a round trip adds another order it can
        // never cancel.
        bool inFlight{};

        MSGPACK_DEFINE_MAP(id, volume, live, inFlight);
    };

    HighFrequencyTraderAgent() noexcept = default;
    HighFrequencyTraderAgent(Simulation* simulation) noexcept;

    // TODO: Wrap state into a struct and provide a single access point here.
    [[nodiscard]] auto&& topLevel(this auto&& self) noexcept { return self.m_topLevel; }
    [[nodiscard]] auto&& inventory(this auto&& self) noexcept { return self.m_inventory; }
    [[nodiscard]] auto&& baseFree(this auto&& self) noexcept { return self.m_baseFree; }
    [[nodiscard]] auto&& quoteFree(this auto&& self) noexcept { return self.m_quoteFree; }
    [[nodiscard]] auto&& restingBid(this auto&& self) noexcept { return self.m_restingBid; }
    [[nodiscard]] auto&& restingAsk(this auto&& self) noexcept { return self.m_restingAsk; }
    [[nodiscard]] auto&& pendingBid(this auto&& self) noexcept { return self.m_pendingBid; }
    [[nodiscard]] auto&& pendingAsk(this auto&& self) noexcept { return self.m_pendingAsk; }
    [[nodiscard]] auto&& fillWakeScheduled(this auto&& self) noexcept { return self.m_fillWakeScheduled; }
    // How many requotes came from being hit rather than from a chain turn. The split between
    // the two is the thing being tuned, so it is a diagnostic and not just a test hook.
    [[nodiscard]] auto&& fillRequotes(this auto&& self) noexcept { return self.m_fillRequotes; }
    // Aggressive inventory sheds, per book. Diagnostic: the rebalance is what closes the
    // hot-potato loop, so a test asserting it stays quoting needs to know it fired at all.
    [[nodiscard]] auto&& rebalances(this auto&& self) noexcept { return self.m_rebalances; }
    [[nodiscard]] auto&& lastPrice(this auto&& self) noexcept { return self.m_lastPrice; }
    [[nodiscard]] auto&& id(this auto&& self) noexcept { return self.m_id; }
    [[nodiscard]] auto&& pRes(this auto&& self) noexcept { return self.m_pRes; }
    // What the rebalance gate divides inventory by, in base units.
    [[nodiscard]] auto&& psi(this auto&& self) noexcept { return self.m_psi; }

    virtual void configure(const pugi::xml_node& node) override;
    virtual void receiveMessage(Message::Ptr msg) override;

private:

    void handleSimulationStart();
    void handleSimulationStop();
    void handleTradeSubscriptionResponse();
    // The chain turn: cancel both sides and requote both.
    void handleWakeup(Message::Ptr msg);
    // The delayed reaction to having been hit, requoting only the sides that were.
    void handleFillWakeup(Message::Ptr msg);
    void requote(BookId bookId, bool doBid, bool doAsk);
    void cancelResting(BookId bookId, bool doBid, bool doAsk);
    [[nodiscard]] RestingQuote& resting(BookId bookId, OrderDirection direction) noexcept;
    Timestamp decisionMakingDelay(BookId bookId);
    Timestamp fillReactionLatency();
    void noteOwnFill(BookId bookId, OrderDirection side, const Trade& trade);
    uint64_t selectTurn();
    void handleLimitOrderPlacementResponse(Message::Ptr msg);
    void handleLimitOrderPlacementErrorResponse(Message::Ptr msg);
    void handleMarketOrderPlacementResponse(Message::Ptr msg);
    void handleMarketOrderPlacementErrorResponse(Message::Ptr msg);
    void handleCancelOrdersResponse(Message::Ptr msg);
    void handleCancelOrdersErrorResponse(Message::Ptr msg);
    void handleTrade(Message::Ptr msg);

    [[nodiscard]] double effectiveSigmaSqr(BookId bookId) const;
    [[nodiscard]] double priceScaleFactor(double midquote) const noexcept;
    [[nodiscard]] std::lognormal_distribution<double> quoteSizeDistribution(BookId bookId) const;
    void placeOrder(BookId bookId, const TopLevelWithVolumes& topLevel, bool doBid, bool doAsk);
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
    // Full-inventory scale, in base units either way: `psiHFT_quotes` times the agent's own
    // average quote, or the legacy absolute value.
    double m_psi;
    std::unique_ptr<taosim::stats::Distribution> m_orderPlacementLatencyDistribution;
    double m_orderMean;
    double m_orderSTD;
    // Quote size as a multiple of the traded volume per second; zero keeps the absolute form.
    double m_quoteSizeVolumeFrac{};
    uint32_t m_quoteSizeWindowBars{};
    std::vector<double> m_orderSizeBid;
    std::vector<double> m_orderSizeAsk;
    std::vector<int> m_lastInvSign;
    uint32_t m_varFastBars{};
    uint32_t m_varSlowBars{};
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
    int m_minSpreadTicks;
    int m_numLevels;
    int m_levelSpacingTicks;
    double m_spreadRefPrice;
    double m_inventoryHorizon{};
    // How long after the run starts the class begins quoting.
    Timestamp m_entryDelay{};
    int m_rebalanceGateMode{0};
    // Chain clock. The class churns on one shared token per book, so the routine
    // cancel-and-requote rate is a property of the class rather than of how many instances
    // are configured.
    std::string m_baseName;
    uint32_t m_catUId{};
    float m_omegaDu{}, m_alphaDu{}, m_betaDu{};
    Timestamp m_minDelay{}, m_maxDelay{};
    std::weibull_distribution<float> m_acdDelayDist;

    // State.
    std::vector<TopLevelWithVolumes> m_topLevel;
    std::vector<double> m_inventory;
    std::vector<double> m_baseFree;
    std::vector<double> m_quoteFree;
    std::vector<TimestampedPrice> m_lastPrice;
    std::vector<double> m_pRes;
    std::vector<RestingQuote> m_restingBid;
    std::vector<RestingQuote> m_restingAsk;
    // Sides hit since the last reaction, coalesced so a burst of fills inside one reaction
    // delay produces one requote rather than one per fill.
    std::vector<bool> m_pendingBid;
    std::vector<bool> m_pendingAsk;
    std::vector<bool> m_fillWakeScheduled;
    std::vector<uint64_t> m_fillRequotes;
    std::vector<uint64_t> m_rebalances;
    AgentId m_id;
};

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
