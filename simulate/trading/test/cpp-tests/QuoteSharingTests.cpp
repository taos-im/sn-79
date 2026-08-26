// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// TAO is ONE balance per agent, shared across every book. Restoring must not break that.
//
// THE DEFECT. `helpers::makeHoldings` builds a fresh account by creating a single Balance and handing
// the SAME shared_ptr to every book, which is correct: a miner has one TAO balance, not one per subnet.
//
// `Balances::fromJson` cannot know that. It allocates a new quote per entry:
//
//     std::make_shared<Balance>(Balance::fromJson(json["quote"]))
//
// so every checkpoint restore gave each book its own independent TAO balance.
// `reconcileSimulationBalances` then re-links only `holdings().front().quote` to the chain figure, so
// book 0 tracked reality and every other book kept whatever the checkpoint happened to hold.
//
// Those per-book figures are what balance consumers read, so a miner sizing an order against any book
// but the first was reading a balance the engine did not believe in, and would be refused by an engine
// behaving perfectly correctly.
//
// The engine's ENFORCEMENT was never wrong: reservations still went through whichever quote object the
// book pointed at, and a controlled experiment showed a second order exceeding the shared balance being
// refused. This was a reporting defect with a real operational bite, not an accounting one.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <taosim/accounting/Balance.hpp>
#include <taosim/accounting/Balances.hpp>

using taosim::accounting::Balance;
using taosim::accounting::Balances;

namespace
{

// The shape both restore paths now produce: the first book donates the quote, the rest point at it.
std::vector<Balances> restoreLikeCheckpoint(int bookCount, double freeTao)
{
    std::vector<Balances> holdings;
    std::shared_ptr<Balance> sharedQuote;
    for (int i = 0; i < bookCount; ++i) {
        auto quote = std::make_shared<Balance>(taosim::decimal_t{freeTao}, "", 9u);
        Balances bal{};
        bal.quote = sharedQuote ? sharedQuote : (sharedQuote = quote);
        holdings.push_back(std::move(bal));
    }
    return holdings;
}

}  // namespace

TEST(QuoteSharing, EveryBookPointsAtTheSameQuoteObject)
{
    const auto holdings = restoreLikeCheckpoint(129, 129.23309169);
    ASSERT_EQ(holdings.size(), 129u);
    for (const auto& bal : holdings) {
        EXPECT_EQ(bal.quote.get(), holdings.front().quote.get())
            << "a book has its own quote object, so a reservation on one book will be invisible to it";
    }
}

TEST(QuoteSharing, AChangeThroughOneBookIsVisibleFromEveryOther)
{
    // The property that actually matters: reconcile writes through holdings().front().quote, and every
    // other book must see it. Before the fix only book 0 tracked the chain figure.
    auto holdings = restoreLikeCheckpoint(129, 100.0);
    holdings.front().quote->getFree() = taosim::decimal_t{42.5};

    for (size_t i = 0; i < holdings.size(); ++i) {
        EXPECT_EQ(holdings[i].quote->getFree(), taosim::decimal_t{42.5})
            << "book " << i << " did not see a write made through book 0";
    }
}

TEST(QuoteSharing, AReservationOnOneBookReducesFreeEverywhere)
{
    // What a miner reads. Reserving against one book must reduce the spendable figure reported by all of
    // them, because there is only one TAO balance to spend.
    auto holdings = restoreLikeCheckpoint(4, 129.23309169);
    holdings[1].quote->getFree() = taosim::decimal_t{56.68295537};

    for (size_t i = 0; i < holdings.size(); ++i) {
        EXPECT_EQ(holdings[i].quote->getFree(), taosim::decimal_t{56.68295537})
            << "book " << i << " still reports the pre-reservation balance, which is what misled a miner";
    }
}

TEST(QuoteSharing, IndependentQuotesAreWhatTheDefectLookedLike)
{
    // Negative control: this is the OLD behaviour, kept so the test above cannot pass vacuously. If
    // someone reinstates a per-book allocation, the assertions above start failing and this one explains
    // what they will see.
    std::vector<Balances> holdings;
    for (int i = 0; i < 4; ++i) {
        Balances bal{};
        bal.quote = std::make_shared<Balance>(taosim::decimal_t{129.0}, "", 9u);
        holdings.push_back(std::move(bal));
    }
    holdings.front().quote->getFree() = taosim::decimal_t{56.0};

    EXPECT_EQ(holdings[0].quote->getFree(), taosim::decimal_t{56.0});
    EXPECT_EQ(holdings[1].quote->getFree(), taosim::decimal_t{129.0})
        << "with per-book quotes, the other books keep a stale figure: exactly the reported defect";
}
