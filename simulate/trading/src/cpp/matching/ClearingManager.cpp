/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/matching/ClearingManager.hpp>

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include <taosim/accounting/margin_utils.hpp>
#include <taosim/matching/FeePolicy.hpp>

#include <range/v3/algorithm/all_of.hpp>

#include <algorithm>
#include <bit>

//-------------------------------------------------------------------------

namespace taosim::matching
{

namespace
{

//-------------------------------------------------------------------------

// What the fill that completes a buy limit moves from the buyer to the seller.
//
// The fill commits the order's remaining reservation, so that nothing reserved for the order
// outlives it. That is the fill's own notional only while the reservation was sized from the
// fills. With a price band configured the placement check reserves limit x size instead, and the
// remainder then also holds what the order saved by filling below its limit. That part is the
// buyer's: the fill moves its notional, and the rest is released when the order is unregistered.
[[nodiscard]] decimal_t completingBuyAmount(
    bool banded, decimal_t reservation, decimal_t notional) noexcept
{
    return banded ? std::min(reservation, notional) : reservation;
}

//-------------------------------------------------------------------------

}  // namespace

//-------------------------------------------------------------------------

namespace
{

// A limit that meets the opposite best price on arrival can take liquidity whatever the placement check predicted.
[[nodiscard]] bool crossesOnArrival(const OrderDesc& orderDesc, const book::Book::Ptr& book)
{
    const auto* limit = std::get_if<LimitOrderDesc>(&orderDesc);
    if (limit == nullptr) return false;
    const auto& payload = limit->payload;
    if (payload->direction == OrderDirection::BUY) {
        return book->sellQueue().hasActiveOrders() && payload->price >= book->bestAsk();
    }
    return book->buyQueue().hasActiveOrders() && payload->price <= book->bestBid();
}

}  // namespace

//-------------------------------------------------------------------------

ClearingManager::ClearingManager(
    MultiBookExchangeAgent* exchange,
    size_t bookCount,
    std::unique_ptr<FeePolicyWrapper> feePolicy,
    OrderPlacementValidator::Parameters validatorParams) noexcept
    : m_exchange{exchange},
      m_feePolicy{std::move(feePolicy)},
      m_orderPlacementValidator{std::move(validatorParams), exchange}
{
    m_marginBuy.resize(bookCount);
    m_marginSell.resize(bookCount);
}

//-------------------------------------------------------------------------

MultiBookExchangeAgent* ClearingManager::exchange() noexcept
{
    return m_exchange;
}

//-------------------------------------------------------------------------

accounting::AccountRegistry& ClearingManager::accounts() noexcept
{
    return m_exchange->accounts();
}

//-------------------------------------------------------------------------

OrderResult ClearingManager::handleOrder(const OrderDesc& orderDesc)
{
    const auto [agentId, bookId, quantity, price, expectedValidationResult] =
        std::visit(
            [&](auto&& desc) {
                using T = std::remove_cvref_t<decltype(desc)>;
                if constexpr (std::same_as<T, MarketOrderDesc>) {
                    return std::make_tuple(
                        desc.agentId,
                        desc.payload->bookId,
                        desc.payload->volume,
                        0_dec,
                        m_orderPlacementValidator.validateMarketOrderPlacement(
                            accounts()[desc.agentId],
                            m_exchange->books()[desc.payload->bookId],
                            desc.payload,
                            *m_feePolicy,
                            m_exchange->getMaxLeverage(),
                            m_exchange->getMaxLoan(),
                            accounts().getAgentId(desc.agentId)));
                }
                else if constexpr (std::same_as<T, LimitOrderDesc>) {
                    return std::make_tuple(
                        desc.agentId,
                        desc.payload->bookId,
                        desc.payload->volume,
                        desc.payload->price,
                        m_orderPlacementValidator.validateLimitOrderPlacement(
                            accounts()[desc.agentId],
                            m_exchange->books()[desc.payload->bookId],
                            desc.payload,
                            *m_feePolicy,
                            m_exchange->getMaxLeverage(),
                            m_exchange->getMaxLoan(),
                            accounts().getAgentId(desc.agentId)));
                }
                else {
                    static_assert(false, "Unrecognized Order descriptor");
                }
            },
            orderDesc);

    if (!expectedValidationResult.has_value()) {
        return {
            .ec =  expectedValidationResult.error(),
            .orderSize = 0_dec
        };
    }
    const auto& validationResult = expectedValidationResult.value();

    auto& account = accounts().at(agentId);
    m_exchange->markDirty(account, bookId);
    auto& balances = account.at(bookId);
    const OrderID orderId = *m_exchange->books()[bookId]->orderIdCounter();

    // THE INSTRUCTION LOG CARRIES EVERY ORDER THAT CAN TAKE LIQUIDITY ON ARRIVAL. instantTrade is the
    // check's reservation prediction, and under an active band a buy predicts no sweep (sweepCap: it reserves its
    // whole volume at its limit), so a banded buy that crossed the asks traded with no instruction line and, filled
    // whole, left no line at all. A limit crossing the opposite best price on arrival is logged too.
    if (validationResult.instantTrade || crossesOnArrival(orderDesc, m_exchange->books()[bookId])) {
        m_exchange->instructionLogCallback(orderDesc, orderId);
    }

    const decimal_t curPrice = validationResult.direction == OrderDirection::BUY ?
        m_exchange->books()[bookId]->bestAsk() : m_exchange->books()[bookId]->bestBid();

    if (validationResult.amount <= 0_dec) {
        fmt::println("Incorrect validation results {} with lev {} at {} price for incoming order #{} in book #{}", 
            validationResult.amount,
            validationResult.leverage,
            curPrice,
            orderId,
            m_exchange->simulation()->bookIdCanon(bookId)
        );
    }

    const auto agentIdCanon = std::holds_alternative<LocalAgentId>(agentId)
        ? m_exchange->accounts().lookupLocalAgentId(std::get<LocalAgentId>(agentId))
        : std::get<AgentId>(agentId);
    const auto bookIdCanon = m_exchange->simulation()->bookIdCanon(bookId);
    const auto sideStr = validationResult.direction == OrderDirection::BUY ? "BUY" : "SELL";
    const auto amountCcyStr = validationResult.direction == OrderDirection::BUY ? "QUOTE" : "BASE";

    m_exchange->simulation()->logDebug(
        "{} | AGENT #{} BOOK {} : MAKING RESERVATION {} {} WITH LEV {} FOR {} ORDER #{}",
        m_exchange->simulation()->currentTimestamp(), agentIdCanon, bookIdCanon,
        validationResult.amount, amountCcyStr, validationResult.leverage, sideStr, orderId);
    auto reserved = balances.makeReservation(orderId, price > 0_dec ? price : curPrice,
        m_exchange->books()[bookId]->bestBid(), m_exchange->books()[bookId]->bestAsk(),
        validationResult.amount, validationResult.leverage, validationResult.direction, m_exchange->simulation()->bookIdCanon(bookId));

    m_exchange->simulation()->logDebug(
        "{} | AGENT #{} BOOK {} : RESERVATION OF {} BASE + {} QUOTE (={} {}) CREATED FOR"
        " {} ORDER #{} ({}x{}@{}) | BEST {} : {} | MAX LEV : {}",
        m_exchange->simulation()->currentTimestamp(), agentIdCanon, bookIdCanon,
        reserved.base, reserved.quote, validationResult.amount, amountCcyStr,
        sideStr, orderId,
        1_dec + validationResult.leverage, quantity,
        price > 0_dec ? fmt::format("{}", price) : "MARKET",
        validationResult.direction == OrderDirection::BUY ? "ASK" : "BID",
        curPrice, m_exchange->getMaxLeverage());

    return {
        .ec = OrderErrorCode::VALID,
        .orderSize = validationResult.orderSize
    };
}

//-------------------------------------------------------------------------

bool ClearingManager::handleClosePosition(const ClosePositionDesc &closeDesc)
{
    const auto& [bookId, agentId, orderId, volumeToClose] = closeDesc;

    accounting::Account& account = accounts()[agentId];
    accounting::Balances& balances = account[bookId];
    const auto& loan = balances.getLoan(orderId);

    if (loan.has_value()){
        const auto settleAmount = closeOrderSize(bookId, agentId, loan->get(), volumeToClose);
        // No active bid means no price to size the closing sell; report the failure instead of
        // dispatching an order for an infinite amount.
        if (!settleAmount) {
            m_exchange->simulation()->logDebug(
                "{} | AGENT #{} BOOK {} : CLOSE POSITION OF ORDER #{} HAS NO BID TO PRICE IT",
                m_exchange->simulation()->currentTimestamp(), agentId,
                m_exchange->simulation()->bookIdCanon(bookId), orderId);
            return false;
        }

        auto orderPayload = MessagePayload::create<PlaceOrderMarketPayload>(
                loan->get().direction() == OrderDirection::BUY ? OrderDirection::SELL : OrderDirection::BUY,
                *settleAmount,
                bookId,
                loan->get().direction() == OrderDirection::BUY ? Currency::QUOTE : Currency::BASE,
                std::nullopt,
                STPFlag::CO,
                orderId
            );
        if (agentId < 0) {
            m_exchange->simulation()->dispatchMessage(
                m_exchange->simulation()->currentTimestamp(),
                0,
                accounts().idBimap().right.at(agentId),
                m_exchange->name(),
                "PLACE_ORDER_MARKET",
                orderPayload
            );
        } else {
            m_exchange->simulation()->dispatchMessage(
                m_exchange->simulation()->currentTimestamp(),
                0,
                "DISTRIBUTED_PROXY_AGENT",
                m_exchange->name(),
                "DISTRIBUTED_PLACE_ORDER_MARKET",
                MessagePayload::create<DistributedAgentResponsePayload>(
                    agentId,
                    orderPayload
                )
            );
        }
        
    } else {
        return false;
    }

    return true;
}

//-------------------------------------------------------------------------

std::optional<decimal_t> ClearingManager::closeOrderSize(
    BookId bookId, AgentId agentId, const accounting::Loan& loan, std::optional<decimal_t> volumeToClose) const
{
    const auto& params = m_exchange->config().parameters();
    const bool closesLong = loan.direction() == OrderDirection::BUY;

    decimal_t size = volumeToClose.value_or(loan.amount());
    if (size >= loan.amount()) {
        const auto takerRate = m_feePolicy->getRates(bookId, agentId).taker;
        size = takerRate > 0_dec
            ? util::roundUp(
                loan.amount() / (1_dec - takerRate),
                closesLong ? params.quoteIncrementDecimals : params.baseIncrementDecimals)
            : loan.amount();
    }
    if (!closesLong) {
        return size;
    }

    const auto bestBid = m_exchange->books()[bookId]->bestBid();
    if (bestBid <= 0_dec) {
        return std::nullopt;
    }
    return util::roundUp(
        bestBid * util::roundUp(size / bestBid, params.baseIncrementDecimals), params.quoteIncrementDecimals);
}

//-------------------------------------------------------------------------

void ClearingManager::handleCancelOrder(const CancelOrderDesc& cancelDesc)
{
    const auto& [bookId, order, volumeToCancel] = cancelDesc;

    const OrderID orderId = order->id();
    const AgentId agentId = m_exchange->books()[bookId]->orderToClientInfo().at(orderId).agentId;

    accounting::Account& account = accounts()[agentId];
    m_exchange->markDirty(account, bookId);
    accounting::Balances& balances = account[bookId];

    const auto freed = [&] -> accounting::ReservationAmounts {
        // A CANCEL MUST ALWAYS BE ABLE TO REMOVE THE ORDER.
        //
        // freeReservation THROWS when there is no reservation to free, and the throw propagates out of
        // handleBatch, which drops the whole CANCEL_ORDERS instruction. The order then stays on the
        // book permanently, uncancellable, and any cancel_all loop spins on it forever.
        //
        // An order CAN legitimately reach this point with nothing backing it. The eager sweep frees the
        // whole reservation predicting a full fill, and a remainder restored by reconciliation is left
        // bare unless restoreReservation succeeds (it is wrapped in a try that only logs). Collapsing
        // per-book quote objects onto the shared one did it too, until that path learned to migrate
        // reservations across.
        //
        // `unregisterLimitOrderCallback` already guards exactly this way; this path did not, which is the
        // whole asymmetry. `freed` is used only for the debug line below, so skipping the free changes
        // nothing else. A stranded reservation is an accounting discrepancy; an uncancellable order is a
        // loss of function, and the discrepancy is the better failure.
        if (!balances.canFree(orderId)) {
            m_exchange->simulation()->logDebug(
                "{} | AGENT #{} BOOK {} : CANCEL of order #{} found no reservation to free; "
                "cancelling anyway",
                m_exchange->simulation()->currentTimestamp(),
                agentId,
                m_exchange->simulation()->bookIdCanon(bookId),
                orderId);
            return {};
        }
        if (order->direction() == OrderDirection::BUY) {
            return balances.freeReservation(
                orderId,
                order->price(),
                m_exchange->books()[bookId]->bestBid(),
                m_exchange->books()[bookId]->bestAsk(),
                order->direction(),
                m_exchange->simulation()->bookIdCanon(bookId),
                // volumeToCancel < order->totalVolume()
                volumeToCancel < order->volume()
                    ? std::make_optional(
                        util::round(
                            util::round(order->price(), m_exchange->config().parameters().priceIncrementDecimals) *
                            util::round(volumeToCancel, m_exchange->config().parameters().volumeIncrementDecimals),
                            m_exchange->config().parameters().quoteIncrementDecimals))
                    : std::nullopt);

        } else {
            return balances.freeReservation(
                orderId,
                order->price(),
                m_exchange->books()[bookId]->bestBid(), 
                m_exchange->books()[bookId]->bestAsk(), 
                order->direction(),
                m_exchange->simulation()->bookIdCanon(bookId),
                // volumeToCancel < order->totalVolume()
                volumeToCancel < order->volume()
                    ? std::make_optional(
                        util::round(volumeToCancel, m_exchange->config().parameters().baseIncrementDecimals))
                    : std::nullopt);
        }
    }();

    if (volumeToCancel >= order->volume()) {
        account.activeOrders()[bookId].erase(order);
        // balances.releaseReservation(order->id());
    }

    m_exchange->simulation()->logDebug(
        "{} | AGENT #{} BOOK {} : CANCELLED {} ORDER #{} ({}@{}) for {} (FREED {} BASE + {} QUOTE)",
        m_exchange->simulation()->currentTimestamp(),
        agentId,
        m_exchange->simulation()->bookIdCanon(bookId),
        order->direction(),
        orderId,
        order->leverage() > 0_dec ? fmt::format("{}x{}",1_dec + order->leverage(),order->volume()) : fmt::format("{}",order->volume()),
        order->price(),
        volumeToCancel,
        freed.base, freed.quote,
        balances.quote->getReserved(),
        balances.base.getReserved());

    if (balances.quote->getReserved() < 0_dec) {
        throw std::runtime_error(fmt::format(
            "{} | AGENT #{} BOOK {} | handleCancelOrder:"
            " Reserved quote balance {} < 0 after cancelling order #{}",
            m_exchange->simulation()->currentTimestamp(),
            agentId,
            m_exchange->simulation()->bookIdCanon(bookId),
            balances.quote->getReserved(), orderId));
    }
    // Residual-reservation diagnostic (debug-gated). The quote balance may be shared
    // across the agent's books (multi-asset portfolio), so it is genuinely idle only when
    // the agent has NO active orders in ANY of its books — a per-book check would flag the
    // reserved amount that legitimately backs sibling-book orders. Base is always per-book.
    const auto bookIdCanon = m_exchange->simulation()->bookIdCanon(bookId);
    const bool quoteIdle = m_exchange->sharedQuoteBalances()
        ? ranges::all_of(account.activeOrders(), [](const auto& orders) { return orders.empty(); })
        : account.activeOrders()[bookId].empty();
    if (quoteIdle && balances.quote->getReserved() > 0_dec) {
        for (const auto& [resOrderId, resAmount] : balances.quote->getReservations()) {
            m_exchange->simulation()->logDebug(
                "handleCancelOrder | residual quote reservation {} for order #{}, no active orders in book #{}",
                resAmount, resOrderId, bookIdCanon);
        }
    }
    if (account.activeOrders()[bookId].empty() && balances.base.getReserved() > 0_dec) {
        for (const auto& [resOrderId, resAmount] : balances.base.getReservations()) {
            m_exchange->simulation()->logDebug(
                "handleCancelOrder | residual base reservation {} for order #{}, no active orders in book #{}",
                resAmount, resOrderId, bookIdCanon);
        }
    }
}

//-------------------------------------------------------------------------

Fees ClearingManager::handleTrade(const TradeDesc& tradeDesc)
{
    const auto [bookId, restingAgentId, aggressingAgentId, trade] = tradeDesc;

    const OrderID restingOrderId = trade->restingOrderID();
    const OrderID aggressingOrderId = trade->aggressingOrderID();

    accounting::Account& restingAccount = accounts()[restingAgentId];
    accounting::Account& aggressingAccount = accounts()[aggressingAgentId];
    m_exchange->markDirty(restingAccount, bookId);
    m_exchange->markDirty(aggressingAccount, bookId);

    const auto& restingAgentActiveOrders = restingAccount.activeOrders()[bookId];
    const auto& aggressingAgentActiveOrders = aggressingAccount.activeOrders()[bookId];

    auto restingOrderIt = std::find_if(
        restingAgentActiveOrders.begin(),
        restingAgentActiveOrders.end(),
        [restingOrderId](const auto o) { return o->id() == restingOrderId; });
    if (restingOrderIt == restingAgentActiveOrders.end()) {
        throw std::runtime_error{fmt::format(
            "{} | AGENT #{} BOOK {} : Resting order #{} not found in active orders while processing {} trade #{} against order #{} from {} for {}@{}.",
            m_exchange->simulation()->currentTimestamp(),
            restingAgentId,
            m_exchange->simulation()->bookIdCanon(bookId),
            restingOrderId,
            trade->direction() == OrderDirection::BUY ? "BUY" : "SELL",
            trade->id(),
            trade->aggressingOrderID(),
            aggressingAgentId,
            trade->volume(),
            trade->price()
        )};
    }
    // Resting orders only ever come off the limit-order book, so the static cast
    // is safe and skips the dynamic_pointer_cast RTTI check (~1 vtable lookup per
    // trade, with thousands of trades per tick).
    LimitOrder::Ptr restingOrder = std::static_pointer_cast<LimitOrder>(*restingOrderIt);

    auto aggressingOrderIt = std::find_if(
        aggressingAgentActiveOrders.begin(),
        aggressingAgentActiveOrders.end(),
        [aggressingOrderId](const auto o) { return o->id() == aggressingOrderId; });
    if (aggressingOrderIt == aggressingAgentActiveOrders.end()) {        
        throw std::runtime_error{fmt::format(
            "{} | AGENT #{} BOOK {} : Aggressing order #{} not found in active orders.",
            m_exchange->simulation()->currentTimestamp(),
            aggressingAgentId,
            m_exchange->simulation()->bookIdCanon(bookId),
            aggressingOrderId)};
    }
    Order::Ptr aggressingOrder = *aggressingOrderIt;

    // A trade by the order that settles a called loan is that call filling.
    if (const auto settleFlag = aggressingOrder->settleFlag(); std::holds_alternative<OrderID>(settleFlag)) {
        forgetMarginCallRefusal(aggressingAgentId, bookId, std::get<OrderID>(settleFlag));
    }

    auto fees = m_feePolicy->calculateFees(tradeDesc);

    // TODO: Rounding should be done inside fees calculation
    fees.maker = util::round(fees.maker, m_exchange->config().parameters().quoteIncrementDecimals);
    fees.taker = util::round(fees.taker, m_exchange->config().parameters().quoteIncrementDecimals);
    
    accounting::Balances& restingBalance = restingAccount[bookId];
    accounting::Balances& aggressingBalance = aggressingAccount[bookId];

    // A side without active quotes reports 0, which is not a price: reservations would be valued
    // at nothing and loan settlement would divide by it. The trade that just printed is the best
    // mark available in that case.
    auto markPrice = [&](decimal_t best) { return best > 0_dec ? best : trade->price(); };
    const decimal_t bestBid = markPrice(m_exchange->books()[bookId]->bestBid());
    const decimal_t bestAsk = markPrice(m_exchange->books()[bookId]->bestAsk());
    const bool banded = m_exchange->config().parameters().maxPriceBand > 0.0;

    // Policy: The direction of the trade is that of the aggressing order.
    if (trade->direction() == OrderDirection::BUY) {
        // Aggressing is BUY, quote reserved; Resting is SELL, base reserved.
        taosim::decimal_t reservation = util::round(aggressingBalance.getReservationInQuote(aggressingOrderId, bestAsk) *
            util::dec1p(aggressingBalance.getLeverage(aggressingOrderId, aggressingOrder->direction())),
            m_exchange->config().parameters().quoteIncrementDecimals
        );

        decimal_t fees_taker_base, fees_taker_quote;
        if (fees.taker < 0_dec){
            fees_taker_base = 0_dec;
            fees_taker_quote = fees.taker;
        } else {
            fees_taker_base = util::roundUp(fees.taker / trade->price(), m_exchange->config().parameters().baseIncrementDecimals);
            fees_taker_quote = util::round(
                fees_taker_base * trade->price() - fees.taker,
                m_exchange->config().parameters().quoteIncrementDecimals);
        }
        

        const auto totalPrice = [&] {
            const auto notional =
                util::round(trade->price(), m_exchange->config().parameters().priceIncrementDecimals) *
                util::round(trade->volume(), m_exchange->config().parameters().volumeIncrementDecimals);
            if (auto limitOrder = std::dynamic_pointer_cast<LimitOrder>(aggressingOrder)) {
                if (reservation == 0_dec) {
                    
                    m_exchange->simulation()->logError(
                        "{} | AGENT #{} BOOK {} | No reservation for aggressing {} order #{} against resting order #{} "
                        "| restVol: {}  aggVol: {}  tradeVol:{}  takerFee:(b:{}|q:-{})  makerFee:{}  reservations(b:{}|q:{})"
                        ,
                        m_exchange->simulation()->currentTimestamp(), 
                        aggressingAgentId, m_exchange->simulation()->bookIdCanon(bookId),
                        aggressingOrder->direction(), aggressingOrderId, restingOrderId,
                        restingOrder->totalVolume(), aggressingOrder->totalVolume(), trade->volume(),
                        fees_taker_base, fees_taker_quote, fees.maker,
                        aggressingBalance.base.getReservation(aggressingOrderId).value_or(0_dec),
                        aggressingBalance.quote->getReservation(aggressingOrderId).value_or(0_dec)
                    );

                    throw std::runtime_error{fmt::format(
                        "{} | AGENT #{} BOOK {} : No reservation for aggressing {} order #{}.",
                        m_exchange->simulation()->currentTimestamp(), 
                        aggressingAgentId, m_exchange->simulation()->bookIdCanon(bookId),
                        aggressingOrder->direction(), aggressingOrderId)};
                }
                if (aggressingOrder->totalVolume() == trade->volume()) {
                    const auto committed = completingBuyAmount(banded, reservation, notional);
                    m_exchange->simulation()->logDebug(
                        "{} | AGENT #{} BOOK {} : Committing reservation amount {} for trade volume {} in {} order #{}.",
                        m_exchange->simulation()->currentTimestamp(),
                        aggressingAgentId, m_exchange->simulation()->bookIdCanon(bookId),
                        util::round(committed / util::dec1p(aggressingBalance.getLeverage(aggressingOrderId, aggressingOrder->direction())), m_exchange->config().parameters().quoteIncrementDecimals),
                        trade->volume(), aggressingOrder->direction(), aggressingOrderId);
                    return committed;
                }
            }
            return notional;
        }();

        decimal_t aggressingMarginCall = {}, restingMarginCall = {};
        // margin call price for margin buying
        if (aggressingOrder->leverage() > 0_dec){
            aggressingMarginCall = accounting::calculateMarginCallPrice(trade->price(), aggressingOrder->leverage(),
                OrderDirection::BUY, m_exchange->getMaintenanceMargin());
            m_marginBuy.at(bookId)[aggressingMarginCall].push_back({
                .orderId = aggressingOrderId, .agentId = aggressingAgentId
            });
        }
        // margin call price for short selling
        if (restingOrder->leverage() > 0_dec){
            restingMarginCall = accounting::calculateMarginCallPrice(trade->price(), restingOrder->leverage(),
                OrderDirection::SELL, m_exchange->getMaintenanceMargin());
            m_marginSell.at(bookId)[restingMarginCall].push_back({
                .orderId = restingOrderId, .agentId = restingAgentId
            });
        }

        const auto aggressingVolume = util::round(totalPrice, m_exchange->config().parameters().quoteIncrementDecimals);
        const auto restingVolume = util::round(trade->volume(), m_exchange->config().parameters().baseIncrementDecimals);
        const auto tradeQuote = util::round(trade->volume() * trade->price(), m_exchange->config().parameters().quoteIncrementDecimals);

        m_exchange->simulation()->logDebug(
            "{} | AGENT #{} BOOK {} : COMMIT {} WITH FEE (b:{},q:{})  FOR AGG BUY ORDER #{} AGAINST {} FOR RESTING SELL ORDER #{} (BEST ASK {} | MARGIN={})",
            m_exchange->simulation()->currentTimestamp(), aggressingAgentId, m_exchange->simulation()->bookIdCanon(bookId),
            aggressingVolume, fees_taker_base, fees_taker_quote, aggressingOrderId, restingVolume, restingOrderId, bestAsk, aggressingMarginCall);
        
        auto removedIdsShortSell = aggressingBalance.commit(
            aggressingOrderId,
            OrderDirection::BUY,
            aggressingVolume,
            restingVolume,
            fees_taker_base,
            fees_taker_quote,
            bestBid,
            bestAsk,
            aggressingMarginCall,
            m_exchange->simulation()->bookIdCanon(bookId),
            aggressingOrder->settleFlag());
        
        if (fees_taker_quote > 0_dec){
            aggressingBalance.quote->deposit(fees_taker_quote, m_exchange->simulation()->bookIdCanon(bookId));
        }

        m_exchange->simulation()->logDebug(
            "{} | AGENT #{} BOOK {} : COMMIT {} WITH FEE {} FOR RESTING SELL ORDER #{} AGAINST {} FOR AGG BUY ORDER #{} (BEST BID {} | MARGIN={})",
            m_exchange->simulation()->currentTimestamp(), restingAgentId, m_exchange->simulation()->bookIdCanon(bookId),
            restingVolume, fees.maker, restingOrderId, aggressingVolume, aggressingOrderId, bestBid, restingMarginCall);
        auto removedIdsMarginBuy = restingBalance.commit(
            restingOrderId, 
            OrderDirection::SELL,
            restingVolume,
            aggressingVolume,
            0_dec,
            fees.maker,
            bestBid,
            bestAsk,
            restingMarginCall,
            m_exchange->simulation()->bookIdCanon(bookId),
            restingOrder->settleFlag());

        m_feePolicy->updateHistory(
            m_exchange->simulation()->currentTimestamp(), bookId, restingAgentId, tradeQuote, false);
        m_feePolicy->updateHistory(
            m_exchange->simulation()->currentTimestamp(), bookId, aggressingAgentId, aggressingVolume, true);

        removeMarginOrders(bookId, OrderDirection::BUY, removedIdsMarginBuy);
        removeMarginOrders(bookId, OrderDirection::SELL, removedIdsShortSell);

        exchange()->simulation()->logDebug("{} | AGENT #{} BOOK {} : AGG BUY ORDER #{} FROM AGENT #{} FOR {} TRADED AGAINST RESTING #{} {}@{} FROM AGENT #{} FOR {} (MAKER {} QUOTE | TAKER b:{}, q:-{} QUOTE)", 
            exchange()->simulation()->currentTimestamp(), aggressingAgentId, m_exchange->simulation()->bookIdCanon(bookId), 
            aggressingOrder->id(), aggressingAgentId, aggressingOrder->leverage() > 0_dec ? fmt::format("{}x{}",1_dec + aggressingOrder->leverage(),aggressingOrder->volume()) : fmt::format("{}",aggressingOrder->volume()), 
            restingOrder->id(), restingOrder->leverage() > 0_dec ? fmt::format("{}x{}",1_dec + restingOrder->leverage(),restingOrder->volume()) : fmt::format("{}",restingOrder->volume()), restingOrder->price(), restingAgentId, trade->volume(), 
            fees.maker, fees_taker_base, fees_taker_quote);

    }
    else {
        // Aggressing is SELL, base reserved; Resting is BUY, quote reserved.
        taosim::decimal_t reservation = util::round(restingBalance.getReservationInQuote(restingOrderId, bestBid) *
            util::dec1p(restingBalance.getLeverage(restingOrderId, restingOrder->direction())),
            m_exchange->config().parameters().quoteIncrementDecimals
        );

        decimal_t fees_maker_base, fees_maker_quote;
        if (fees.maker < 0_dec){
            fees_maker_base = 0_dec;
            fees_maker_quote = fees.maker;
        } else {
            fees_maker_base = util::roundUp(fees.maker / trade->price(), m_exchange->config().parameters().baseIncrementDecimals);
            fees_maker_quote = util::round(
                fees_maker_base * trade->price() - fees.maker,
                m_exchange->config().parameters().quoteIncrementDecimals);
        }

        const auto tradeQuote = util::round(trade->volume() * trade->price(), m_exchange->config().parameters().quoteIncrementDecimals);
        const auto restingVolume = restingOrder->totalVolume() == trade->volume() && reservation > 0_dec
            ? completingBuyAmount(banded, reservation, tradeQuote) : tradeQuote;

        if (reservation == 0_dec) {
            
            m_exchange->simulation()->logError(
                "{} | AGENT #{} BOOK {} | No reservation for resting {} order #{} against aggressing order #{} "
                "| restVol: {}  aggVol: {}  tradeVol:{}  takerFee:{}  makerFee:(b:{}|q:-{})  reservations(b:{}|q:{})  baseReservations: [{}]  quoteReservations: [{}]  baseAddress: {}, quoteAddress: {}"
                ,
                m_exchange->simulation()->currentTimestamp(), 
                aggressingAgentId, m_exchange->simulation()->bookIdCanon(bookId),
                restingOrder->direction(), restingOrderId, aggressingOrderId,
                restingOrder->totalVolume(), aggressingOrder->totalVolume(), trade->volume(),
                fees.taker, fees_maker_base, fees_maker_quote,
                restingBalance.base.getReservation(restingOrderId).value_or(0_dec),
                restingBalance.quote->getReservation(restingOrderId).value_or(0_dec),
                fmt::format("{}", fmt::join(restingBalance.base.getReservations() | ranges::views::transform([](auto&& kv) { return fmt::format("{}: {}", kv.first, kv.second); }), ", ")),
                fmt::format("{}", fmt::join(restingBalance.quote->getReservations() | ranges::views::transform([](auto&& kv) { return fmt::format("{}: {}", kv.first, kv.second); }), ", ")),
                fmt::ptr(&restingBalance.base),
                fmt::ptr(restingBalance.quote.get())
            );

            throw std::runtime_error{fmt::format(
                "{} | AGENT #{} BOOK {} : Trade volume {}, No reservation for resting {} order #{} against aggressing order #{}.",
                m_exchange->simulation()->currentTimestamp(),
                restingAgentId, m_exchange->simulation()->bookIdCanon(bookId), trade->volume(),
                restingOrder->direction(), restingOrderId, aggressingOrderId)};
        } else if (restingOrder->totalVolume() == trade->volume()) {
            m_exchange->simulation()->logDebug(
                "{} | AGENT #{} BOOK {} : Committing reservation amount {} for trade volume {} in {} order #{}.",
                m_exchange->simulation()->currentTimestamp(),
                restingAgentId, m_exchange->simulation()->bookIdCanon(bookId),
                util::round(restingVolume / util::dec1p(restingBalance.getLeverage(restingOrderId, restingOrder->direction())), m_exchange->config().parameters().quoteIncrementDecimals),
                trade->volume(), restingOrder->direction(), restingOrderId);
        }

        taosim::decimal_t aggressingMarginCall = {}, restingMarginCall = {};
        // margin call price for short selling
        if (aggressingOrder->leverage() > 0_dec){
            aggressingMarginCall = accounting::calculateMarginCallPrice(trade->price(), aggressingOrder->leverage(),
                OrderDirection::SELL, m_exchange->getMaintenanceMargin());
            m_marginSell.at(bookId)[aggressingMarginCall].push_back({.orderId = aggressingOrderId, .agentId = aggressingAgentId});
        }
        // margin call price for margin buying
        if (restingOrder->leverage() > 0_dec){
            restingMarginCall = accounting::calculateMarginCallPrice(trade->price(), restingOrder->leverage(),
                OrderDirection::BUY, m_exchange->getMaintenanceMargin());
            (restingOrder->leverage() * trade->price()) / ((1_dec + restingOrder->leverage()) * 
                (1 - m_exchange->getMaintenanceMargin()));
            m_marginBuy.at(bookId)[restingMarginCall].push_back({.orderId = restingOrderId, .agentId = restingAgentId});
        }

        const auto aggressingVolume = util::round(trade->volume(), m_exchange->config().parameters().baseIncrementDecimals);

        
        
        m_exchange->simulation()->logDebug(
            "{} | AGENT #{} BOOK {} : COMMIT {} WITH FEE {} FOR AGG SELL ORDER #{} AGAINST {} FOR RESTING BUY ORDER #{} (BEST ASK {} | MARGIN={})",
            m_exchange->simulation()->currentTimestamp(), aggressingAgentId, m_exchange->simulation()->bookIdCanon(bookId),
            aggressingVolume, fees.taker, aggressingOrderId, restingVolume, restingOrderId, bestAsk, aggressingMarginCall);

        auto removedIdsMarginBuy = aggressingBalance.commit(
            aggressingOrderId,
            OrderDirection::SELL,
            aggressingVolume,
            restingVolume,
            0_dec,
            fees.taker,
            bestBid,
            bestAsk,
            aggressingMarginCall,
            m_exchange->simulation()->bookIdCanon(bookId),
            aggressingOrder->settleFlag());

        m_exchange->simulation()->logDebug(
            "{} | AGENT #{} BOOK {} : COMMIT {} WITH FEE {} FOR RESTING BUY ORDER #{} AGAINST {} FOR AGG SELL ORDER #{} (BEST BID {} | MARGIN={})",
            m_exchange->simulation()->currentTimestamp(), restingAgentId, m_exchange->simulation()->bookIdCanon(bookId),
            restingVolume, fees.maker, restingOrderId, aggressingVolume, aggressingOrderId, bestBid, restingMarginCall);
            
        auto removedIdsShortSell = restingBalance.commit(
            restingOrderId, 
            OrderDirection::BUY,
            restingVolume,
            aggressingVolume,
            fees_maker_base,
            fees_maker_quote,
            bestBid,
            bestAsk,
            restingMarginCall,
            m_exchange->simulation()->bookIdCanon(bookId),
            restingOrder->settleFlag());
        
            if (fees_maker_quote > 0_dec){
            restingBalance.quote->deposit(fees_maker_quote, m_exchange->simulation()->bookIdCanon(bookId));
        }

        m_feePolicy->updateHistory(
            m_exchange->simulation()->currentTimestamp(), bookId, restingAgentId, tradeQuote, false);
        m_feePolicy->updateHistory(
            m_exchange->simulation()->currentTimestamp(), bookId, aggressingAgentId, restingVolume, true);

        removeMarginOrders(bookId, OrderDirection::SELL, removedIdsShortSell);
        removeMarginOrders(bookId, OrderDirection::BUY, removedIdsMarginBuy);

        exchange()->simulation()->logDebug("{} | AGENT #{} BOOK {} : AGG SELL ORDER #{} FROM AGENT #{} FOR {} TRADED AGAINST RESTING #{} {}@{} FROM AGENT #{} FOR {} (MAKER {} QUOTE | TAKER {} QUOTE)", 
            exchange()->simulation()->currentTimestamp(), aggressingAgentId, m_exchange->simulation()->bookIdCanon(bookId), 
            aggressingOrder->id(), aggressingAgentId, aggressingOrder->leverage() > 0_dec ? fmt::format("{}x{}",1_dec + aggressingOrder->leverage(),aggressingOrder->volume()) : fmt::format("{}",aggressingOrder->volume()), 
            restingOrder->id(), restingOrder->leverage() > 0_dec ? fmt::format("{}x{}",1_dec + restingOrder->leverage(),restingOrder->volume()) : fmt::format("{}",restingOrder->volume()), restingOrder->price(), restingAgentId, trade->volume(), 
            fees.maker, fees.taker);
    }

    return fees;
}

//-------------------------------------------------------------------------

void ClearingManager::removeMarginOrders(
    BookId bookId, OrderDirection direction, std::span<std::pair<OrderID, decimal_t>> ids)
{   
    auto& cont = direction == OrderDirection::BUY ? m_marginBuy : m_marginSell;

    auto& marginOrders = cont.at(bookId);

    for (const auto& [orderId, amount] : ids) {
        auto ordersIt = marginOrders.find(amount);
        if (ordersIt == marginOrders.end()) continue;
        auto& orders = ordersIt->second;
        std::erase_if(orders, [&](const auto& order) {
            if (order.orderId != orderId) return false;
            forgetMarginCallRefusal(order.agentId, bookId, orderId);
            return true;
        });
        if (orders.empty()) {
            marginOrders.erase(ordersIt);
        }
    }
}

//-------------------------------------------------------------------------

bool ClearingManager::firstMarginCallRefusal(AgentId agentId, BookId bookId, OrderID loanId)
{
    return m_refusedMarginCalls.emplace(agentId, bookId, loanId).second;
}

//-------------------------------------------------------------------------

void ClearingManager::forgetMarginCallRefusal(AgentId agentId, BookId bookId, OrderID loanId) noexcept
{
    m_refusedMarginCalls.erase({agentId, bookId, loanId});
}

//-------------------------------------------------------------------------

void ClearingManager::forgetMarginCallRefusals(AgentId agentId) noexcept
{
    std::erase_if(m_refusedMarginCalls, [agentId](const auto& key) { return std::get<0>(key) == agentId; });
}

//-------------------------------------------------------------------------

void ClearingManager::updateFeeTiers(Timestamp time) noexcept
{
    m_feePolicy->updateAgentsTiers(time); ////###
}

//-------------------------------------------------------------------------

}  // namespace taosim::matching

//-------------------------------------------------------------------------