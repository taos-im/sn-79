/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/matching/OrderPlacementValidator.hpp>

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"

#include <algorithm>
#include <limits>

//-------------------------------------------------------------------------

namespace taosim::matching
{

namespace
{

//-------------------------------------------------------------------------

// A reservation is sized by predicting how far an order will sweep. With a price band active that
// prediction is unsafe in a way that cannot be repaired by capping it: the band can stop the sweep at
// MATCH time, after the reservation is set, leaving the volume resting at the order's OWN price and
// later filled there. Two failures were traced to exactly that -
//   book #118 #6702 : BUY 43.0093@330.69 reserved at the 300.19 ask, filled at 330.69
//   book #19  #11229: BUY 12.6981@371.06 reserved at the 300.70 ask, filled at 371.06
// - each ending in "freeing X exceeding reservation of Y" and a terminated engine.
//
// So when a band is active a BUY predicts NO sweep and reserves its whole volume at its limit price,
// which is the most it can ever pay per unit and therefore covers every execution path the band can
// force. Sells reserve BASE (volume, price-independent), so they only need the band-aware floor.
//
// Band OFF is bit-for-bit unchanged: bandLimit returns the numeric_limits sentinels, so the order's
// own price is returned and the prediction runs exactly as before. That property is what keeps every
// band-off determinism result and all three organic pairs valid, and detprove checks it.
[[nodiscard]] decimal_t sweepCap(
    const book::Book::Ptr& book, decimal_t price, OrderDirection direction) noexcept
{
    if (direction == OrderDirection::BUY) {
        const auto cap = book->bandLimit(true);
        if (cap == std::numeric_limits<decimal_t>::max()) return price;   // band off
        // With the band now TWO-SIDED (both sweep loops enforce [lo, hi]), the earlier narrowing -
        // "an order priced within the band cannot be stopped by it" - is false: a buy sweeping asks
        // that rest BELOW lo is stopped by the new lower bound regardless of its own price, the
        // predicted cheap fills never execute, the volume rests at the order's price and the
        // reservation blows up exactly as before (measured: FreeException at sim step 0, freeing
        // 1587.96 against a 22.76 reservation). So under an ACTIVE band every buy predicts no sweep
        // and reserves its full volume at its limit price. Band off remains bit-for-bit unchanged.
        return 0_dec;
    }
    const auto floor = book->bandLimit(false);
    if (floor == std::numeric_limits<decimal_t>::min()) return price;    // band off
    return std::max(price, floor);
}

//-------------------------------------------------------------------------

}  // namespace


//-------------------------------------------------------------------------

OrderPlacementValidator::OrderPlacementValidator(
    const OrderPlacementValidator::Parameters& params, MultiBookExchangeAgent* exchange) noexcept
    : m_params{params},
      m_exchange{exchange}
{}

//-------------------------------------------------------------------------

OrderPlacementValidator::ExpectedResult
    OrderPlacementValidator::validateMarketOrderPlacement(
        const accounting::Account& account,
        taosim::book::Book::Ptr book,
        PlaceOrderMarketPayload::Ptr payload,
        FeePolicyWrapper& feePolicy,
        decimal_t maxLeverage,
        decimal_t maxLoan,
        AgentId agentId) const
{
    // A market order is valid if the initiating account has either
    //   - sufficient funds to at least partially collect the requested shares from the book (buy)
    //   - enough inventory *and* the book can at least partially fill the order (sell)
    // AND
    //   - the order volume respects the minimum increment
    
    if (payload->leverage < 0_dec || payload->leverage > maxLeverage) {
        return std::unexpected{OrderErrorCode::INVALID_LEVERAGE};
    }
    if (payload->volume <= 0_dec) {
        return std::unexpected{OrderErrorCode::INVALID_VOLUME};
    }
    
    payload->volume = util::round(payload->volume, 
        payload->currency == Currency::BASE ? m_params.volumeIncrementDecimals : m_params.quoteIncrementDecimals);
    payload->leverage = util::round(payload->leverage, m_params.volumeIncrementDecimals);

    if (payload->volume <= 0_dec) {
        return std::unexpected{OrderErrorCode::INVALID_VOLUME};
    }

    if (!payload->skipMinSizeCheck &&
            payload->currency == Currency::BASE &&
            payload->volume < m_exchange->config2().minOrderSize) {
        return std::unexpected{OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION};
    }

    const decimal_t payloadTotalAmount = util::round(payload->volume * util::dec1p(payload->leverage),
        payload->currency == Currency::BASE ? m_params.volumeIncrementDecimals : m_params.quoteIncrementDecimals);

    const auto& balances = account.at(book->id());
    const auto& baseBalance = balances.base;
    const auto& quoteBalance = balances.quote;
    
    decimal_t orderSize{};
    bool instantTrade = false;

    if (payload->direction == OrderDirection::BUY) {
        if (book->sellQueue().empty()) {
            return std::unexpected{OrderErrorCode::EMPTY_BOOK};
        }
        if (payload->leverage > 0_dec && (balances.m_baseLoan > 0_dec || !balances.m_sellLeverages.empty())){
            return std::unexpected{OrderErrorCode::DUAL_POSITION};
        }
        decimal_t volumeWeightedPrice{};
        decimal_t volume{};
        if (payload->currency == Currency::BASE){
            bool done = false;
            for (auto it = book->sellQueue().cbegin(); it != book->sellQueue().cend(); ++it) {
                const auto& level = *it;
                for (const auto tick : level) {
                    if (book->orderToClientInfo().at(tick->id()).agentId == agentId){    // STP
                        if (payload->stpFlag == STPFlag::CO || payload->stpFlag == STPFlag::CN || payload->stpFlag == STPFlag::CB)
                            continue;
                    }
                    
                    decimal_t tickVolume = util::round(tick->totalVolume(), m_params.volumeIncrementDecimals);
                    if (volume + tickVolume >= payloadTotalAmount) {
                        const decimal_t partialVolume = payloadTotalAmount - volume;
                        volume += partialVolume;
                        decimal_t tradeCost = util::round(tick->price() * partialVolume, m_params.quoteIncrementDecimals);
                        volumeWeightedPrice += tradeCost;
                        m_exchange->simulation()->logDebug(
                            "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} QUOTE ({}*{}) FOR TRADE OF BUY VOLUME-BASED ORDER {}x{}@MARKET AGAINST {}@{}",
                            m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), 
                            tradeCost, partialVolume, tick->price(), util::dec1p(payload->leverage), payload->volume, tickVolume, tick->price());
                        done = true;
                        break;
                    }
                    volume += tickVolume;
                    const decimal_t tradeCost = util::round(tick->price() * tickVolume, m_params.quoteIncrementDecimals);
                    volumeWeightedPrice += tradeCost;
                    m_exchange->simulation()->logDebug(
                        "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} QUOTE ({}*{}) FOR TRADE OF BUY VOLUME-BASED ORDER {}x{}@MARKET AGAINST {}@{}",
                        m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), tradeCost, tickVolume, tick->price(),
                        util::dec1p(payload->leverage), payload->volume, tickVolume, tick->price());
                }
                if (done) break;
            }
            orderSize = payload->volume;
        } 
        else if (payload->currency == Currency::QUOTE){
            bool done = false;
            for (auto it = book->sellQueue().cbegin(); it != book->sellQueue().cend(); ++it) {
                const auto& level = *it;
                for (const auto tick : level) {
                    if (book->orderToClientInfo().at(tick->id()).agentId == agentId){    // STP
                        if (payload->stpFlag == STPFlag::CO || payload->stpFlag == STPFlag::CN || payload->stpFlag == STPFlag::CB)
                            continue;
                    }
                    
                    decimal_t tickVolume = util::round(tick->totalVolume(), m_params.volumeIncrementDecimals);
                    if (volumeWeightedPrice + tickVolume * tick->price() >= payloadTotalAmount) {
                        const decimal_t partialQuote = payloadTotalAmount - volumeWeightedPrice;
                        volumeWeightedPrice += partialQuote;
                        volume += util::round(partialQuote / tick->price(), m_params.baseIncrementDecimals);
                        done = true;
                        break;
                    }
                    volumeWeightedPrice += util::round(tickVolume * tick->price(), m_params.quoteIncrementDecimals);
                    volume += tickVolume;
                }
                if (done) break;
            }

            if (!payload->skipMinSizeCheck && volume < m_exchange->config2().minOrderSize) {
                return std::unexpected{OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION};
            }
            orderSize = util::round(volume / util::dec1p(payload->leverage), m_params.baseIncrementDecimals);
        }

        volumeWeightedPrice = util::round(volumeWeightedPrice, m_params.quoteIncrementDecimals);
        if (volumeWeightedPrice > 0_dec) instantTrade = true;

        // Both currency branches above have accumulated the quote total this buy would consume, which
        // is the amount that has to settle on chain. A walk that consumed nothing settles nothing, so
        // it is left to the paths that already handle an unfillable order.
        if (!payload->skipMinSizeCheck && volumeWeightedPrice > 0_dec
                && !checkMinQuoteOrderSize(agentId, volumeWeightedPrice)) {
            return std::unexpected{OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION};
        }

        if (payload->leverage == 0_dec){
            if (!quoteBalance->canReserve(volumeWeightedPrice)) {
                instantTrade = false;
                return std::unexpected{OrderErrorCode::INSUFFICIENT_QUOTE};
            }
        } else {
            volumeWeightedPrice = util::round(volumeWeightedPrice / util::dec1p(payload->leverage), m_params.quoteIncrementDecimals);
            const decimal_t price = book->bestAsk();
            if (volumeWeightedPrice <= 0_dec){
                return std::unexpected{OrderErrorCode::INVALID_VOLUME};
            }
            if (!balances.canBorrow(volumeWeightedPrice, price, payload->direction) ||
                volumeWeightedPrice * payload->leverage + balances.totalLoanInQuote(price) > maxLoan) {

                instantTrade = false;
                return std::unexpected{OrderErrorCode::EXCEEDING_LOAN};
            }
        }

        return OrderPlacementValidator::ExpectedResult{
            Result{
                .direction = payload->direction,
                .amount = volumeWeightedPrice,
                .leverage = payload->leverage,
                .orderSize = orderSize,
                .instantTrade = instantTrade
            }
        };

    } else {
        if (book->buyQueue().empty()) {
            return std::unexpected{OrderErrorCode::EMPTY_BOOK};
        }
        if (payload->leverage > 0_dec && (balances.m_quoteLoan > 0_dec || !balances.m_buyLeverages.empty())){
            return std::unexpected{OrderErrorCode::DUAL_POSITION};
        }
        decimal_t volume{};
        if (payload->currency == Currency::BASE){
            volume = payload->volume * util::dec1p(payload->leverage);
            orderSize = payload->volume;
            // This branch never walks the book, so the proceeds are estimated at the best bid rather
            // than by replaying the match. That over-states what a deeper walk would realise, making
            // the floor permissive rather than strict: it refuses orders that cannot possibly settle,
            // and does not refuse one that merely might. A strict version would have to duplicate the
            // matching walk here.
            if (!payload->skipMinSizeCheck && book->bestBid() > 0_dec
                    && !checkMinQuoteOrderSize(agentId, volume * book->bestBid())) {
                return std::unexpected{OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION};
            }
            m_exchange->simulation()->logDebug(
                "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} BASE FOR SELL VOLUME-BASED ORDER {}x{}@MARKET",
                m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), volume,
                util::dec1p(payload->leverage), payload->volume);
        }
        else if (payload->currency == Currency::QUOTE){
            decimal_t volumeWeightedPrice{};
            bool done = false;
            for (auto it = book->buyQueue().crbegin(); it != book->buyQueue().crend(); ++it) {
                const auto& level = *it;
                for (const auto tick : level) {
                    if (book->orderToClientInfo().at(tick->id()).agentId == agentId){    // STP
                        if (payload->stpFlag == STPFlag::CO || payload->stpFlag == STPFlag::CN || payload->stpFlag == STPFlag::CB)
                            continue;
                    }

                    decimal_t tickVolume = util::round(tick->totalVolume(), m_params.volumeIncrementDecimals);
                    if (volumeWeightedPrice + tick->price() * tickVolume >= payloadTotalAmount) {
                        const decimal_t partialQuote = payloadTotalAmount - volumeWeightedPrice;
                        volumeWeightedPrice += partialQuote;
                        volume += util::round(partialQuote / tick->price(), m_params.baseIncrementDecimals);
                        m_exchange->simulation()->logDebug(
                            "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} BASE ({}*{}) FOR TRADE OF SELL QUOTE-BASED ORDER {}x{}@MARKET AGAINST {}@{}",
                            m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), 
                            util::round(partialQuote / tick->price(), m_params.baseIncrementDecimals),
                            util::dec1p(payload->leverage), tick->volume(),
                            util::dec1p(payload->leverage), payload->volume, tickVolume, tick->price());
                        done = true;
                        break;
                    }
                    volumeWeightedPrice += util::round(tick->price() * tickVolume, m_params.quoteIncrementDecimals);
                    volume += util::round(tickVolume, m_params.baseIncrementDecimals);
                    m_exchange->simulation()->logDebug(
                        "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} BASE ({}*{}) FOR TRADE OF SELL QUOTE-BASED ORDER {}x{}@MARKET AGAINST {}@{}",
                        m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), 
                        util::round(tickVolume, m_params.baseIncrementDecimals),
                        util::dec1p(payload->leverage), tick->volume(),
                        util::dec1p(payload->leverage), payload->volume, tickVolume, tick->price());
                }
                if (done) break;
            }

            if (!payload->skipMinSizeCheck && volume < m_exchange->config2().minOrderSize) {
                return std::unexpected{OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION};
            }
            // The proceeds this walk would realise, and so the amount that has to settle on chain.
            if (!payload->skipMinSizeCheck && !checkMinQuoteOrderSize(agentId, volumeWeightedPrice)) {
                return std::unexpected{OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION};
            }

            orderSize = util::round(volume / util::dec1p(payload->leverage), m_params.baseIncrementDecimals);
        }

        volume = util::round(volume, m_params.baseIncrementDecimals);
        if (volume > 0_dec) instantTrade = true;

        if (payload->leverage == 0_dec){
            if (!baseBalance.canReserve(volume)) {
                instantTrade = false;
                return std::unexpected{OrderErrorCode::INSUFFICIENT_BASE};
            }
        } else {
            volume = util::round(volume / util::dec1p(payload->leverage), m_params.baseIncrementDecimals);
            const decimal_t price = book->bestBid();
            if (volume <= 0_dec){
                return std::unexpected{OrderErrorCode::INVALID_VOLUME};
            }
            if (!balances.canBorrow(volume, price, payload->direction) ||
                volume * price * payload->leverage + balances.totalLoanInQuote(price) > maxLoan) {

                instantTrade = false;
                return std::unexpected{OrderErrorCode::EXCEEDING_LOAN};
            }
        }
        return OrderPlacementValidator::ExpectedResult{
            Result{
                .direction = payload->direction,
                .amount = volume,
                .leverage = payload->leverage,
                .orderSize = orderSize,
                .instantTrade = instantTrade
            }
        };
        
    }
}

//-------------------------------------------------------------------------

OrderPlacementValidator::ExpectedResult
    OrderPlacementValidator::validateLimitOrderPlacement(
        const accounting::Account& account,
        taosim::book::Book::Ptr book,
        PlaceOrderLimitPayload::Ptr payload,
        FeePolicyWrapper& feePolicy,
        decimal_t maxLeverage,
        decimal_t maxLoan,
        AgentId agentId) const
{
    // A limit order is valid if the initiating account has either
    //   - sufficient funds to place the order (limit buy)
    //   - sufficient inventory available to cover the to-be-sold volume (limit sell)
    // AND
    //   - the price and volume of the order are in accord with their respective minimum increments

    if (payload->leverage < 0_dec || payload->leverage > maxLeverage) {
        return std::unexpected{OrderErrorCode::INVALID_LEVERAGE};
    }
    if (payload->volume <= 0_dec) {
        return std::unexpected{OrderErrorCode::INVALID_VOLUME};
    }
    if (payload->price <= 0_dec) {
        return std::unexpected{OrderErrorCode::INVALID_PRICE};
    }
    if (account.activeOrders().at(book->id()).size() >= m_exchange->config2().maxOpenOrders) {
        return std::unexpected{OrderErrorCode::EXCEEDING_MAX_ORDERS};
    }

    payload->price = util::round(payload->price, m_params.priceIncrementDecimals);
    payload->volume = util::round(payload->volume, 
        payload->currency == Currency::BASE ? m_params.volumeIncrementDecimals : m_params.quoteIncrementDecimals);
    payload->leverage = util::round(payload->leverage, m_params.volumeIncrementDecimals);  

    if (payload->volume <= 0_dec) {
        return std::unexpected{OrderErrorCode::INVALID_VOLUME};
    }
    if (payload->price <= 0_dec) {
        return std::unexpected{OrderErrorCode::INVALID_PRICE};
    }

    if (!checkMinOrderSizeLimit(payload)) {
        return std::unexpected{OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION};
    }

    // A BASE-currency order states alpha, so its notional needs the price; a QUOTE-currency order
    // states the notional directly. Checked here rather than inside checkMinOrderSizeLimit because the
    // floor is agent-scoped and only this scope knows who placed the order. There is no
    // skipMinSizeCheck on a limit payload, so this matches checkMinOrderSizeLimit above in applying
    // unconditionally.
    {
        const auto totalAmount = util::round(payload->volume * util::dec1p(payload->leverage),
            payload->currency == Currency::BASE
                ? m_params.volumeIncrementDecimals : m_params.quoteIncrementDecimals);
        const auto quoteNotional = payload->currency == Currency::BASE
            ? totalAmount * payload->price
            : totalAmount;
        if (!checkMinQuoteOrderSize(agentId, quoteNotional)) {
            return std::unexpected{OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION};
        }
    }


    if (!checkTimeInForce(book, payload, agentId, 0_dec)) { //###
        return std::unexpected{OrderErrorCode::CONTRACT_VIOLATION};
    }
    if (payload->postOnly && !checkPostOnly(book, payload, agentId, 0_dec)) { //###
        return std::unexpected{OrderErrorCode::CONTRACT_VIOLATION};
    }

    const auto payloadTotalAmount = util::round(payload->volume * util::dec1p(payload->leverage), 
        payload->currency == Currency::BASE ? m_params.volumeIncrementDecimals : m_params.quoteIncrementDecimals); 

    const auto& balances = account.at(book->id());
    const auto& baseBalance = balances.base;
    const auto& quoteBalance = balances.quote;

    decimal_t orderSize{};
    bool instantTrade = false;

    if (payload->direction == OrderDirection::BUY) {
        if (payload->leverage > 0_dec && (balances.m_baseLoan > 0_dec || !balances.m_sellLeverages.empty())){
            return std::unexpected{OrderErrorCode::DUAL_POSITION};
        }
        decimal_t volumeWeightedPrice{};
        if (payload->currency == Currency::BASE){
            decimal_t takerVolume{};
            decimal_t takerTotalPrice{};
            bool done = false;
            for (auto it = book->sellQueue().cbegin(); it != book->sellQueue().cend(); ++it) {
                const auto& level = *it;
                if (sweepCap(book, payload->price, OrderDirection::BUY) < level.price()) break;
                for (const auto tick : level) {
                    if (book->orderToClientInfo().at(tick->id()).agentId == agentId){    // STP
                        if (payload->stpFlag == STPFlag::CO || payload->stpFlag == STPFlag::CN || payload->stpFlag == STPFlag::CB)
                            continue;
                    }
                        
                    decimal_t tickVolume = util::round(tick->totalVolume(), m_params.volumeIncrementDecimals);
                    if (takerVolume + tickVolume >= payloadTotalAmount) {
                        const decimal_t partialVolume = payloadTotalAmount - takerVolume;
                        takerVolume += partialVolume;
                        decimal_t tradeCost = util::round(tick->price() * partialVolume, m_params.quoteIncrementDecimals);
                        takerTotalPrice += tradeCost;
                        m_exchange->simulation()->logDebug(
                            "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} QUOTE ({}*{}) FOR TRADE OF BUY VOLUME-BASED ORDER {}x{}@{} AGAINST {}@{}",
                            m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), tradeCost, 
                            partialVolume, tick->price(), util::dec1p(payload->leverage), payload->volume, payload->price, tickVolume, tick->price());
                        done = true;
                        break;
                    }
                    takerVolume += tickVolume;
                    decimal_t tradeCost = util::round(tick->price() * tickVolume, m_params.quoteIncrementDecimals);
                    takerTotalPrice += tradeCost;
                    m_exchange->simulation()->logDebug(
                        "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} QUOTE ({}*{}) FOR TRADE OF BUY VOLUME-BASED ORDER {}x{}@{} AGAINST {}@{}",
                        m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), tradeCost,
                        tickVolume, tick->price(), util::dec1p(payload->leverage), payload->volume, payload->price, tickVolume, tick->price());
                }
                if (done) break;
            }
            takerTotalPrice = util::round(takerTotalPrice, m_params.quoteIncrementDecimals);
            if (takerVolume > 0_dec) instantTrade = true;
            
            const decimal_t makerVolume = payloadTotalAmount - takerVolume;
            const decimal_t makerTotalPrice = util::round(payload->price * makerVolume, m_params.quoteIncrementDecimals);
            m_exchange->simulation()->logDebug(
                "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} QUOTE ({}*{}) FOR PLACE OF BUY VOLUME-BASED ORDER {}x{}@{}",
                m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), makerTotalPrice,
                makerVolume, payload->price, util::dec1p(payload->leverage), payload->volume, payload->price);
            volumeWeightedPrice = util::round(takerTotalPrice + makerTotalPrice, m_params.quoteIncrementDecimals);
            orderSize = payload->volume;
        } 
        else if (payload->currency == Currency::QUOTE){
            decimal_t takerVolume{};
            decimal_t takerTotalPrice{};
            bool done = false;
            for (auto it = book->sellQueue().cbegin(); it != book->sellQueue().cend(); ++it) {
                const auto& level = *it;
                if (sweepCap(book, payload->price, OrderDirection::BUY) < level.price()) break;
                for (const auto tick : level) {
                    if (book->orderToClientInfo().at(tick->id()).agentId == agentId){    // STP
                        if (payload->stpFlag == STPFlag::CO || payload->stpFlag == STPFlag::CN || payload->stpFlag == STPFlag::CB)
                            continue;
                    }
                    
                    decimal_t tickVolume = util::round(tick->totalVolume(), m_params.volumeIncrementDecimals);
                    if (takerTotalPrice + tickVolume * tick->price() >= payloadTotalAmount) {
                        const decimal_t partialQuote = payloadTotalAmount - takerTotalPrice;
                        takerTotalPrice += partialQuote;
                        takerVolume += util::round(partialQuote / tick->price(), m_params.baseIncrementDecimals);
                        m_exchange->simulation()->logDebug(
                            "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} QUOTE ({}*{}) FOR TRADE OF BUY ORDER {}x{}@{} AGAINST {}@{}",
                            m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), partialQuote,
                            util::round(partialQuote/tick->price(), m_params.baseIncrementDecimals),
                            tick->price(), util::dec1p(payload->leverage), payload->volume, payload->price, tickVolume, tick->price());
                        done = true;
                        break;
                    }
                    takerTotalPrice += tick->price() * tickVolume;
                    takerVolume += tickVolume;
                    m_exchange->simulation()->logDebug(
                            "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} QUOTE ({}*{}) FOR TRADE OF BUY ORDER {}x{}@{} AGAINST {}@{}",
                            m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), 
                            util::round(tick->price() * tickVolume, m_params.quoteIncrementDecimals),
                            util::round(tickVolume, m_params.baseIncrementDecimals),
                            tick->price(), util::dec1p(payload->leverage),
                            payload->volume, payload->price, tickVolume, tick->price());
                }
                if (done) break;
            }
            takerTotalPrice = util::round(takerTotalPrice, m_params.quoteIncrementDecimals);
            if (takerVolume > 0_dec) instantTrade = true;
            const decimal_t makerTotalPrice = util::round(payloadTotalAmount - takerTotalPrice, m_params.quoteIncrementDecimals);
            const decimal_t makerVolume = util::round(makerTotalPrice / payload->price, m_params.baseIncrementDecimals);
            m_exchange->simulation()->logDebug(
                "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} QUOTE ({}*{}) FOR PLACE OF BUY ORDER {}x{}@{}",
                m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()),
                makerTotalPrice, util::round(makerTotalPrice / payload->price, m_params.baseIncrementDecimals), payload->price,
                util::dec1p(payload->leverage), payload->volume, payload->price);
            volumeWeightedPrice = util::round(
                takerTotalPrice + makerTotalPrice, m_params.quoteIncrementDecimals);
            orderSize = util::round(
                (takerVolume + makerVolume) / util::dec1p(payload->leverage), m_params.baseIncrementDecimals);
        }

        if (payload->leverage == 0_dec){
            if (!quoteBalance->canReserve(volumeWeightedPrice)) {
                instantTrade = false;
                return std::unexpected{OrderErrorCode::INSUFFICIENT_QUOTE};
            }
        } else {
            volumeWeightedPrice = util::round(volumeWeightedPrice / util::dec1p(payload->leverage), m_params.quoteIncrementDecimals);
            const decimal_t price = payload->price;
            if (volumeWeightedPrice <= 0_dec){
                return std::unexpected{OrderErrorCode::INVALID_VOLUME};
            }
            if (!balances.canBorrow(volumeWeightedPrice, price, payload->direction)||
                volumeWeightedPrice * payload->leverage + balances.totalLoanInQuote(price) > maxLoan) {
                
                instantTrade = false;
                return std::unexpected{OrderErrorCode::EXCEEDING_LOAN};
            }
        }
        return OrderPlacementValidator::ExpectedResult{
            Result{
                .direction = payload->direction,
                .amount = volumeWeightedPrice,
                .leverage = payload->leverage,
                .orderSize = orderSize,
                .instantTrade = instantTrade
            }
        };

    }
    else {
        if (payload->leverage > 0_dec && (balances.m_quoteLoan > 0_dec || !balances.m_buyLeverages.empty())){
            return std::unexpected{OrderErrorCode::DUAL_POSITION};
        }
        decimal_t volume{};
        if (payload->currency == Currency::BASE){
            for (auto it = book->buyQueue().crbegin(); it != book->buyQueue().crend(); ++it) {
                const auto& level = *it;
                if (level.price() < payload->price) break;
                for (const auto tick : level) {
                    if(book->orderToClientInfo().at(tick->id()).agentId == agentId){    // STP
                        if (payload->stpFlag == STPFlag::CO || payload->stpFlag == STPFlag::CN || payload->stpFlag == STPFlag::CB)
                            continue;
                    }
                    instantTrade = true;
                    break;
                }
            }
            volume = util::round(payload->volume * util::dec1p(payload->leverage), m_params.baseIncrementDecimals);
            orderSize = payload->volume;
            m_exchange->simulation()->logDebug(
                "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} BASE FOR SELL VOLUME-BASED ORDER {}x{}@{}",
                m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), volume,
                util::dec1p(payload->leverage), payload->volume, payload->price);
        } 
        else if (payload->currency == Currency::QUOTE){
            decimal_t takerVolume{};
            decimal_t takerTotalPrice{};
            bool done = false;
            for (auto it = book->buyQueue().crbegin(); it != book->buyQueue().crend(); ++it) {
                const auto& level = *it;
                if (level.price() < payload->price) break;
                for (const auto tick : level) {
                    if(book->orderToClientInfo().at(tick->id()).agentId == agentId){    // STP
                        if (payload->stpFlag == STPFlag::CO || payload->stpFlag == STPFlag::CN || payload->stpFlag == STPFlag::CB)
                            continue;
                    }

                    decimal_t tickVolume = util::round(tick->totalVolume(), m_params.volumeIncrementDecimals);    
                    if (takerTotalPrice + tickVolume * tick->price() >= payloadTotalAmount) {
                        const decimal_t partialQuote = payloadTotalAmount - takerTotalPrice;
                        takerTotalPrice += partialQuote;
                        takerVolume += util::round(partialQuote / tick->price(), m_params.baseIncrementDecimals);
                        m_exchange->simulation()->logDebug(
                            "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} BASE FOR TRADE OF BUY QUOTE-BASED ORDER {}x{}@{} AGAINST {}@{} | pta:{} pq:{}",
                            m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()),
                            util::round(partialQuote / tick->price(), m_params.baseIncrementDecimals),
                            util::dec1p(payload->leverage), payload->volume, payload->price, tickVolume, tick->price(), payloadTotalAmount, partialQuote);
                        done = true;
                        break;
                    }
                    takerTotalPrice += util::round(tick->price() * tickVolume, m_params.quoteIncrementDecimals);
                    takerVolume += util::round(tickVolume, m_params.baseIncrementDecimals);
                    m_exchange->simulation()->logDebug(
                            "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} BASE FOR TRADE OF BUY QUOTE-BASED ORDER {}x{}@{} AGAINST {}@{}",
                            m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), tickVolume,
                            util::dec1p(payload->leverage), payload->volume, payload->price, tickVolume, tick->price());
                }
                if (done) break;
            }
            takerVolume = util::round(takerVolume, m_params.baseIncrementDecimals);
            if (takerVolume > 0_dec) instantTrade = true;
            const decimal_t makerVolume = util::round(
                (payloadTotalAmount - takerTotalPrice) / payload->price, m_params.baseIncrementDecimals) ;
            m_exchange->simulation()->logDebug(
                "{} | AGENT #{} BOOK {} : CALCULATED PRE-RESERVATION OF {} BASE @{} FOR PLACE OF BUY ORDER {}x{}@{}",
                m_exchange->simulation()->currentTimestamp(), agentId, m_exchange->simulation()->bookIdCanon(book->id()), makerVolume, payload->price,
                util::dec1p(payload->leverage), payload->volume, payload->price);
            volume = util::round(takerVolume + makerVolume, m_params.baseIncrementDecimals);
            orderSize = util::round(volume / util::dec1p(payload->leverage), m_params.baseIncrementDecimals);
        }

        if (payload->leverage == 0_dec){
            if (!baseBalance.canReserve(volume)) {
                instantTrade = false;
                return std::unexpected{OrderErrorCode::INSUFFICIENT_BASE};
            }
        } else {
            volume = util::round(volume / util::dec1p(payload->leverage), m_params.baseIncrementDecimals);
            const decimal_t price = payload->price; //book->bestBid();
            if (volume <= 0_dec){
                return std::unexpected{OrderErrorCode::INVALID_VOLUME};
            }
            if (!balances.canBorrow(volume, price, payload->direction) ||
                volume * price * payload->leverage + balances.totalLoanInQuote(price) > maxLoan) {
                
                instantTrade = false;
                return std::unexpected{OrderErrorCode::EXCEEDING_LOAN};
            }
        }
        return OrderPlacementValidator::ExpectedResult{
            Result{
                .direction = payload->direction,
                .amount = volume,
                .leverage = payload->leverage,
                .orderSize = orderSize,
                .instantTrade = instantTrade
            }
        };

    }
}

//-------------------------------------------------------------------------

bool OrderPlacementValidator::checkTimeInForce(
    taosim::book::Book::Ptr book,
    PlaceOrderLimitPayload::Ptr payload,
    AgentId agentId,
    decimal_t takerFeeRate) const noexcept
{
    switch (payload->timeInForce) {
        case TimeInForce::IOC:
            return checkIOC(book, payload, agentId, takerFeeRate);
        case TimeInForce::FOK:
            return checkFOK(book, payload, agentId, takerFeeRate);
        default:
            return true;
    }
}

//-------------------------------------------------------------------------

bool OrderPlacementValidator::checkIOC(
    taosim::book::Book::Ptr book,
    PlaceOrderLimitPayload::Ptr payload,
    AgentId agentId,
    decimal_t takerFeeRate) const noexcept
{
    if (payload->postOnly) [[unlikely]] {
        return false;
    }

    auto nonZeroLevelsView = [](auto&& levels) {
        return levels | ranges::views::filter([](auto&& level) {
            return ranges::any_of(level, [](auto&& o) { return o->volume() != 0_dec; });
        });
    };

    const auto totalVolume = util::round(
        payload->volume * util::dec1p(payload->leverage),
        payload->currency == Currency::BASE
            ? m_params.volumeIncrementDecimals : m_params.quoteIncrementDecimals);

    auto takerVolumeBase = [&] -> decimal_t {
        decimal_t collectedVolume{};
        if (payload->stpFlag == STPFlag::CO) {
            const auto& activeOrders =
                m_exchange->accounts().at(agentId).activeOrders().at(payload->bookId);
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (sweepCap(book, payload->price, OrderDirection::BUY) < level.price()) break;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) continue;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * feeCoeff, m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return totalVolume;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (sweepCap(book, payload->price, OrderDirection::SELL) > level.price()) break;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) continue;
                        const auto tickVolume = util::round(
                            tick->totalVolume(), m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return totalVolume;
                    }
                }
            }
        }
        else if (payload->stpFlag == STPFlag::CN || payload->stpFlag == STPFlag::CB) {
            const auto& activeOrders =
                m_exchange->accounts().at(agentId).activeOrders().at(payload->bookId);
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (sweepCap(book, payload->price, OrderDirection::BUY) < level.price()) break;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) return {};
                        const auto tickVolume = util::round(
                            tick->totalVolume() * feeCoeff, m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return totalVolume;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (sweepCap(book, payload->price, OrderDirection::SELL) > level.price()) break;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) return {};
                        const auto tickVolume = util::round(
                            tick->totalVolume(), m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return totalVolume;
                    }
                }
            }
        }
        else {
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (sweepCap(book, payload->price, OrderDirection::BUY) < level.price()) break;
                    for (const auto& tick : level) {
                        const auto tickVolume = util::round(
                            tick->totalVolume() * feeCoeff, m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return totalVolume;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (sweepCap(book, payload->price, OrderDirection::SELL) > level.price()) break;
                    for (const auto& tick : level) {
                        const auto tickVolume = util::round(
                            tick->totalVolume(), m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return totalVolume;
                    }
                }
            }
        }
        return collectedVolume;
    };

    auto takerVolumeQuote = [&] -> decimal_t {
        decimal_t collectedVolume{};
        if (payload->stpFlag == STPFlag::CO) {
            const auto& activeOrders =
                m_exchange->accounts().at(agentId).activeOrders().at(payload->bookId);
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (sweepCap(book, payload->price, OrderDirection::BUY) < level.price()) break;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) continue;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price() * feeCoeff,
                            m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return totalVolume;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (sweepCap(book, payload->price, OrderDirection::SELL) > level.price()) break;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) continue;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price(), m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return totalVolume;
                    }
                }
            }
        }
        else if (payload->stpFlag == STPFlag::CN || payload->stpFlag == STPFlag::CB) {
            const auto& activeOrders =
                m_exchange->accounts().at(agentId).activeOrders().at(payload->bookId);
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (sweepCap(book, payload->price, OrderDirection::BUY) < level.price()) break;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) return {};
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price() * feeCoeff,
                            m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return totalVolume;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (sweepCap(book, payload->price, OrderDirection::SELL) > level.price()) break;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) return {};
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price(), m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return totalVolume;
                    }
                }
            }
        }
        else {
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (sweepCap(book, payload->price, OrderDirection::BUY) < level.price()) break;
                    for (const auto& tick : level) {
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price() * feeCoeff,
                            m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return totalVolume;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (sweepCap(book, payload->price, OrderDirection::SELL) > level.price()) break;
                    for (const auto& tick : level) {
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price(), m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return totalVolume;
                    }
                }
            }
        }
        return collectedVolume;
    };

    const auto takerVolume =
        payload->currency == Currency::BASE ? takerVolumeBase() : takerVolumeQuote();

    if (takerVolume == 0_dec) {
        return false;
    }

    payload->volume = util::round(
        takerVolume / util::dec1p(payload->leverage),
        payload->currency == Currency::BASE
            ? m_params.volumeIncrementDecimals : m_params.quoteIncrementDecimals);

    return true;
}

//-------------------------------------------------------------------------

bool OrderPlacementValidator::checkFOK(
    taosim::book::Book::Ptr book,
    PlaceOrderLimitPayload::Ptr payload,
    AgentId agentId,
    decimal_t takerFeeRate) const noexcept
{
    if (payload->postOnly) [[unlikely]] {
        return false;
    }
    
    auto nonZeroLevelsView = [](auto&& levels) {
        return levels | ranges::views::filter([](auto&& level) {
            return ranges::any_of(level, [](auto&& o) { return o->volume() != 0_dec; });
        });
    };

    const auto totalVolume = util::round(
        payload->volume * util::dec1p(payload->leverage),
        payload->currency == Currency::BASE
            ? m_params.volumeIncrementDecimals : m_params.quoteIncrementDecimals);
    const auto& activeOrders =
            m_exchange->accounts().at(agentId).activeOrders().at(payload->bookId);

    auto checkBase = [&] -> bool {
        decimal_t collectedVolume{};
        if (payload->stpFlag == STPFlag::CO) {
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (level.price() > payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) continue;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * feeCoeff, m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (level.price() < payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) continue;
                        const auto tickVolume = util::round(
                            tick->totalVolume(), m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            }
        }
        else if (payload->stpFlag == STPFlag::CN) {
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (level.price() > payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) return false;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * feeCoeff, m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (level.price() < payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) return false;
                        const auto tickVolume = util::round(
                            tick->totalVolume(), m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            }
        }
        else if (payload->stpFlag == STPFlag::CB) {
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (level.price() > payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) return true;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * feeCoeff, m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (level.price() < payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) return true;
                        const auto tickVolume = util::round(
                            tick->totalVolume(), m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            }
        }
        else if (payload->stpFlag == STPFlag::DC) {
            decimal_t dynamicTotalVolume = totalVolume;
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (level.price() > payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        const auto tickVolume = util::round(
                            tick->totalVolume() * feeCoeff, m_params.volumeIncrementDecimals);
                        if (it != activeOrders.end()) {
                            dynamicTotalVolume -= tickVolume;
                            if (dynamicTotalVolume <= 0_dec) return true;
                        } else {
                            collectedVolume += tickVolume;
                            if (collectedVolume >= dynamicTotalVolume) return true;
                        }
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (level.price() < payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        const auto tickVolume = util::round(
                            tick->totalVolume(), m_params.volumeIncrementDecimals);
                        if (it != activeOrders.end()) {
                            dynamicTotalVolume -= tickVolume;
                            if (dynamicTotalVolume <= 0_dec) return true;
                        } else {
                            collectedVolume += tickVolume;
                            if (collectedVolume >= dynamicTotalVolume) return true;
                        }
                    }
                }
            }
        }
        else {  // STPFlag::NONE — no self-trade prevention; count every tick
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (level.price() > payload->price) return false;
                    for (const auto& tick : level) {
                        const auto tickVolume = util::round(
                            tick->totalVolume() * feeCoeff, m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (level.price() < payload->price) return false;
                    for (const auto& tick : level) {
                        const auto tickVolume = util::round(
                            tick->totalVolume(), m_params.volumeIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            }
        }
        // Exhausted all crossable levels without reaching the required volume.
        // FOK semantics demand full fill — reject.
        return collectedVolume >= totalVolume;
    };

    auto checkQuote = [&] -> bool {
        decimal_t collectedVolume{};
        if (payload->stpFlag == STPFlag::CO) {
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (level.price() > payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) continue;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price() * feeCoeff,
                            m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (level.price() < payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) continue;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price(), m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            }
        }
        else if (payload->stpFlag == STPFlag::CN) {
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (level.price() > payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) return false;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price() * feeCoeff,
                            m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (level.price() < payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) return false;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price(), m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            }
        }
        else if (payload->stpFlag == STPFlag::CB) {
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (level.price() > payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) return true;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price() * feeCoeff,
                            m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (level.price() < payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it != activeOrders.end()) return true;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price(), m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            }
        }
        else if (payload->stpFlag == STPFlag::DC) {
            decimal_t dynamicTotalVolume = totalVolume;
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (level.price() > payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price() * feeCoeff,
                            m_params.quoteIncrementDecimals);
                        if (it != activeOrders.end()) {
                            dynamicTotalVolume -= tickVolume;
                            if (dynamicTotalVolume <= 0_dec) return true;
                        } else {
                            collectedVolume += tickVolume;
                            if (collectedVolume >= dynamicTotalVolume) return true;
                        }
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (level.price() < payload->price) return false;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price(), m_params.quoteIncrementDecimals);
                        if (it != activeOrders.end()) {
                            dynamicTotalVolume -= tickVolume;
                            if (dynamicTotalVolume <= 0_dec) return true;
                        } else {
                            collectedVolume += tickVolume;
                            if (collectedVolume >= dynamicTotalVolume) return true;
                        }
                    }
                }
            }
        }
        else {  // STPFlag::NONE — no self-trade prevention; count every tick
            if (payload->direction == OrderDirection::BUY) {
                const auto feeCoeff = util::decInv1p(takerFeeRate);
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (level.price() > payload->price) return false;
                    for (const auto& tick : level) {
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price() * feeCoeff,
                            m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (level.price() < payload->price) return false;
                    for (const auto& tick : level) {
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price(), m_params.quoteIncrementDecimals);
                        collectedVolume += tickVolume;
                        if (collectedVolume >= totalVolume) return true;
                    }
                }
            }
        }
        // Exhausted all crossable levels without reaching the required volume.
        return collectedVolume >= totalVolume;
    };

    return payload->currency == Currency::BASE ? checkBase() : checkQuote();
}

//-------------------------------------------------------------------------

bool OrderPlacementValidator::checkPostOnly(
    taosim::book::Book::Ptr book,
    PlaceOrderLimitPayload::Ptr payload,
    AgentId agentId,
    decimal_t takerFeeRate) const noexcept
{
    if (payload->timeInForce == TimeInForce::IOC || payload->timeInForce == TimeInForce::FOK) [[unlikely]] {
        return false;
    }
    if (payload->direction == OrderDirection::BUY && book->sellQueue().empty()
        || payload->direction == OrderDirection::SELL && book->buyQueue().empty()) [[unlikely]] {
        return true;
    }

    auto nonZeroLevelsView = [](auto&& levels) {
        return levels | ranges::views::filter([](auto&& level) {
            return ranges::any_of(level, [](auto&& o) { return o->volume() != 0_dec; });
        });
    };

    if (payload->stpFlag == STPFlag::CO) {
        const auto& activeOrders =
            m_exchange->accounts().at(agentId).activeOrders().at(payload->bookId);
        if (payload->direction == OrderDirection::BUY) {
            for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                if (level.price() > payload->price) break;
                for (const auto& tick : level) {
                    auto it = ranges::find_if(
                        activeOrders, [&](auto order) { return order->id() == tick->id(); });
                    if (it == activeOrders.end()) return false;
                }
            }
        } else {
            for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                if (level.price() < payload->price) break;
                for (const auto& tick : level) {
                    auto it = ranges::find_if(
                        activeOrders, [&](auto order) { return order->id() == tick->id(); });
                    if (it == activeOrders.end()) return false;
                }
            }
        }
        return true;
    }
    else if (payload->stpFlag == STPFlag::CB) {
        const auto& activeOrders =
            m_exchange->accounts().at(agentId).activeOrders().at(payload->bookId);
        if (payload->direction == OrderDirection::BUY) {
            const auto optLevelRef = book->bestSellLevel();
            if (!optLevelRef) { return true; }
            const auto& level = optLevelRef->get();
            if (level.price() > payload->price) { return true; }
            const auto& tick = level.front();
            auto it = ranges::find_if(
                activeOrders, [&](auto order) { return order->id() == tick->id(); });
            return it != activeOrders.end();
        } else {
            const auto optLevelRef = book->bestBuyLevel();
            if (!optLevelRef) { return true; }
            const auto& level = optLevelRef->get();
            if (level.price() < payload->price) return true;
            const auto& tick = level.front();
            auto it = ranges::find_if(
                activeOrders, [&](auto order) { return order->id() == tick->id(); });
            return it != activeOrders.end();
        }
    }
    else if (payload->stpFlag == STPFlag::DC) {
        const auto& activeOrders =
            m_exchange->accounts().at(agentId).activeOrders().at(payload->bookId);
        decimal_t dynamicTotalVolume = util::round(
            payload->volume * util::dec1p(payload->leverage),
            payload->currency == Currency::BASE
                ? m_params.volumeIncrementDecimals : m_params.quoteIncrementDecimals);
        if (payload->currency == Currency::BASE) {
            if (payload->direction == OrderDirection::BUY) {
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (level.price() > payload->price) break;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it == activeOrders.end()) return false;
                        const auto tickVolume = util::round(
                            tick->totalVolume() / util::dec1p(takerFeeRate),
                            m_params.volumeIncrementDecimals);
                        dynamicTotalVolume -= tickVolume;
                        if (dynamicTotalVolume <= 0_dec) return false;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (level.price() < payload->price) break;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it == activeOrders.end()) return false;
                        const auto tickVolume = util::round(
                            tick->totalVolume(), m_params.volumeIncrementDecimals);
                        dynamicTotalVolume -= tickVolume;
                        if (dynamicTotalVolume <= 0_dec) return false;
                    }
                }
            }
        }
        else {
            if (payload->direction == OrderDirection::BUY) {
                for (const auto& level : nonZeroLevelsView(book->sellQueue())) {
                    if (level.price() > payload->price) break;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it == activeOrders.end()) return false;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price() / util::dec1p(takerFeeRate),
                            m_params.quoteIncrementDecimals);
                        dynamicTotalVolume -= tickVolume;
                        if (dynamicTotalVolume <= 0_dec) return false;
                    }
                }
            } else {
                for (const auto& level : nonZeroLevelsView(book->buyQueue()) | ranges::views::reverse) {
                    if (level.price() < payload->price) break;
                    for (const auto& tick : level) {
                        auto it = ranges::find_if(
                            activeOrders, [&](auto order) { return order->id() == tick->id(); });
                        if (it == activeOrders.end()) return false;
                        const auto tickVolume = util::round(
                            tick->totalVolume() * tick->price(), m_params.quoteIncrementDecimals);
                        dynamicTotalVolume -= tickVolume;
                        if (dynamicTotalVolume <= 0_dec) return false;
                    }
                }
            }
        }
        return true;
    }
    else {
        if (payload->direction == OrderDirection::BUY) {
            return payload->price < book->bestAsk();
        } else {
            return payload->price > book->bestBid();
        }
    }
}

//-------------------------------------------------------------------------

bool OrderPlacementValidator::checkMinQuoteOrderSize(
    AgentId agentId, decimal_t quoteNotional) const noexcept
{
    const auto minQuoteOrderSize = m_exchange->config2().minQuoteOrderSize;
    if (minQuoteOrderSize <= 0_dec) return true;
    // A guard, not a live distinction: exchange mode strips every agent but the exchange and the proxy
    //, so every order placed there is already a remote one, and the
    // floor is undeclared in simulation configs. It exists so that a config which one day mixes local
    // background agents with on-chain settlement cannot have a chain-derived minimum thin its book,
    // since a local agent's fills never reach the chain.
    if (agentId < 0) return true;
    return util::round(quoteNotional, m_params.quoteIncrementDecimals) >= minQuoteOrderSize;
}

//-------------------------------------------------------------------------

bool OrderPlacementValidator::checkMinOrderSizeLimit(
    PlaceOrderLimitPayload::Ptr payload) const noexcept
{
    // Contract: Should be called after rounding relevant payload values,
    // and before OrderPlacementValidator::checkTimeInForce.

    const auto totalAmount = util::round(payload->volume * util::dec1p(payload->leverage), 
        payload->currency == Currency::BASE
            ? m_params.volumeIncrementDecimals : m_params.quoteIncrementDecimals);

    if (payload->currency == Currency::BASE) {
        return totalAmount >= m_exchange->config2().minOrderSize;
    } else {
        return util::round(totalAmount / payload->price, m_params.volumeIncrementDecimals)
            >= m_exchange->config2().minOrderSize;
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::matching

//-------------------------------------------------------------------------
