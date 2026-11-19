/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/Account.hpp>
#include <taosim/simulation/serialization/LimitOrder.hpp>
#include <common.hpp>

#include <msgpack.hpp>

#include <array>
#include <optional>
#include <string_view>
#include <vector>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::simulation::serialization
{

namespace detail
{

//-------------------------------------------------------------------------

template<typename Packer>
void packBalance(Packer& o, const accounting::Balance& balance, std::string_view currency)
{
    using namespace std::string_literals;

    o.pack_map(5);

    o.pack("c"s);
    o.pack(currency);

    o.pack("t"s);
    o.pack(balance.getTotal());

    o.pack("f"s);
    o.pack(balance.getFree());

    o.pack("r"s);
    o.pack(balance.getReserved());

    o.pack("i"s);
    o.pack(balance.getInitial());
}

//-------------------------------------------------------------------------

template<typename Packer>
void packLoan(Packer& o, const accounting::Loan& loan, OrderID id)
{
    using namespace std::string_literals;

    o.pack_map(5);

    o.pack("i"s);
    o.pack(id);

    o.pack("a"s);
    o.pack(loan.amount());

    o.pack("c"s);
    o.pack(std::to_underlying(
        loan.direction() == OrderDirection::BUY ? Currency::QUOTE : Currency::BASE));

    o.pack("bc"s);
    o.pack(loan.collateral().base());

    o.pack("qc"s);
    o.pack(loan.collateral().quote());
}

//-------------------------------------------------------------------------
// The fields of an account entry that a clean account may omit.

template<typename Packer>
void packHoldings(Packer& o, const accounting::Account& account, const auto& book)
{
    using namespace std::string_literals;

    const auto& balances = account.at(book.id());

    o.pack("bb"s);
    packBalance(o, balances.base, "BASE");

    o.pack("qb"s);
    packBalance(o, *balances.quote, "QUOTE");

    o.pack("bl"s);
    o.pack(balances.m_baseLoan);

    o.pack("ql"s);
    o.pack(balances.m_quoteLoan);

    o.pack("bc"s);
    o.pack(balances.m_baseCollateral);

    o.pack("qc"s);
    o.pack(balances.m_quoteCollateral);

    o.pack("o"s);
    const auto limitOrders = [&] {
        std::vector<LimitOrder> limitOrders;
        for (const auto& order : account.activeOrders().at(book.id())) {
            const auto limitOrder = std::dynamic_pointer_cast<::LimitOrder>(order);
            if (limitOrder == nullptr) continue;
            limitOrders.push_back(LimitOrder{
                .limitOrder = limitOrder,
                .clientOrderId = book.orderToClientInfo().at(order->id()).clientOrderId
            });
        }
        return limitOrders;
    }();
    o.pack_array(limitOrders.size());
    for (const auto& limitOrder : limitOrders) {
        o.pack(limitOrder);
    }

    o.pack("l"s);
    o.pack_map(balances.m_loans.size());
    for (const auto& [id, loan] : balances.m_loans) {
        o.pack(id);
        packLoan(o, loan, id);
    }
}

//-------------------------------------------------------------------------

template<typename Packer>
void packFees(
    Packer& o, const auto& feePolicy, BookId bookId, BookId bookIdCanon, AgentId agentId, auto& bookRates)
{
    using namespace std::string_literals;

    o.pack_map(3);

    o.pack("v"s);
    if (feePolicy.isTiered()) {
        o.pack(feePolicy.agentVolume(bookId, agentId));
    } else {
        o.pack_nil();
    }

    // Tiered rates follow the agent's own volume, and an agent may have a policy of its own; every other
    // agent on the book shares the book's rates, computed once per pack in bookRates.
    const auto rates = [&] {
        if (feePolicy.isTiered() || feePolicy.hasOwnPolicy(agentId)) {
            return feePolicy.getRates(bookId, agentId);
        }
        const auto [it, fresh] = bookRates.try_emplace(bookIdCanon);
        if (fresh) it->second = feePolicy.getRates(bookId, agentId);
        return it->second;
    }();

    o.pack("m"s);
    o.pack(rates.maker);

    o.pack("t"s);
    o.pack(rates.taker);
}

//-------------------------------------------------------------------------

}  // namespace detail

//-------------------------------------------------------------------------
// Whether packAccount writes an agent's entry for a book. Under the exchange's dirtyAccountsOnly an entry
// unchanged on its book since the last delivered publish is left out of the state, unless the exchange
// lists the whole account this time (accountInFull) or the agent's fees are its own (a tiered policy, a
// policy of its own), which only its entry can carry: the validator fills a left-out entry in from its
// last full copy at the book's shared rates (packBookFees).

[[nodiscard]] bool entryPacked(const auto& simulation, AgentId agentId, BookId bookId)
{
    const auto exchange = simulation.exchange();
    if (!exchange->dirtyAccountsOnly()) return true;
    const auto& feePolicy = *exchange->clearingManager().feePolicy();
    return feePolicy.isTiered() || feePolicy.hasOwnPolicy(agentId) || exchange->accountInFull(agentId)
        || exchange->accounts().at(agentId).dirty(bookId);
}

[[nodiscard]] uint32_t entriesPacked(const auto& simulation, AgentId agentId)
{
    uint32_t count{};
    for (const auto& book : simulation.exchange()->books()) {
        count += entryPacked(simulation, agentId, book->id());
    }
    return count;
}

//-------------------------------------------------------------------------
// Each book's shared fee rates, keyed by canonical book id like the books, for the entries the state leaves
// out: an untiered policy prices every agent without a policy of its own the same on a book (the dynamic fee
// follows the book's maker/taker ratio). A realization whose policy is tiered lists none, as it leaves no
// entry out. bookRates is the caller's per-pack cache that packFees fills too.

template<typename Packer>
void packBookFees(Packer& o, const auto& simulations, auto& bookRates)
{
    using namespace std::string_literals;

    std::vector<BookId> listed;
    for (const auto& simulation : simulations) {
        const auto exchange = simulation->exchange();
        const auto& feePolicy = *exchange->clearingManager().feePolicy();
        if (feePolicy.isTiered()) continue;
        std::optional<AgentId> shared;
        for (const auto& [agentId, account] : exchange->accounts()) {
            if (agentId >= 0 && !feePolicy.hasOwnPolicy(agentId)) {
                shared = agentId;
                break;
            }
        }
        if (!shared) continue;
        for (const auto& book : exchange->books()) {
            const BookId bookIdCanon = simulation->bookIdCanon(book->id());
            const auto [it, fresh] = bookRates.try_emplace(bookIdCanon);
            if (fresh) it->second = feePolicy.getRates(book->id(), *shared);
            listed.push_back(bookIdCanon);
        }
    }
    o.pack_map(static_cast<uint32_t>(listed.size()));
    for (const BookId bookIdCanon : listed) {
        const auto& rates = bookRates.at(bookIdCanon);
        o.pack(bookIdCanon);
        o.pack_map(2);
        o.pack("m"s);
        o.pack(rates.maker);
        o.pack("t"s);
        o.pack(rates.taker);
    }
}

//-------------------------------------------------------------------------
// One agent's entries for the books of `simulation` that entryPacked says, keyed by canonical book id, into
// a map the caller opened with entriesPacked's count. An entry written for its fees alone keeps its ids and
// fees and is nil everywhere else. Packing changes nothing: the marks are cleared only once the validator
// has answered (MultiBookExchangeAgent::accountsDelivered), so a state that never arrives loses none.
// bookRates is the caller's per-pack cache of each book's shared fee rates (packFees).

template<typename Packer>
void packAccount(Packer& o, const Simulation& simulation, AgentId agentId, auto& bookRates)
{
    using namespace std::string_literals;

    static constexpr std::array s_omittable{"bb", "qb", "bl", "ql", "bc", "qc", "o", "l"};

    const auto exchange = simulation.exchange();
    const auto& account = exchange->accounts().at(agentId);
    const auto& feePolicy = *exchange->clearingManager().feePolicy();
    const bool allFull = exchange->accountInFull(agentId);

    for (const auto& book : exchange->books()) {
        if (!entryPacked(simulation, agentId, book->id())) continue;
        const BookId bookIdCanon = simulation.bookIdCanon(book->id());

        o.pack(bookIdCanon);

        o.pack_map(11);

        o.pack("i"s);
        o.pack(agentId);

        o.pack("b"s);
        o.pack(bookIdCanon);

        if (allFull || account.dirty(book->id())) {
            detail::packHoldings(o, account, *book);
        } else {
            for (std::string_view key : s_omittable) {
                o.pack(key);
                o.pack_nil();
            }
        }

        o.pack("f"s);
        detail::packFees(o, feePolicy, book->id(), bookIdCanon, agentId, bookRates);
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation::serialization

//-------------------------------------------------------------------------
