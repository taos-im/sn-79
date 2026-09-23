/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "taosim/statshub/StatsHub.hpp"

#include <taosim/book/Book.hpp>

#include <vector>

//-------------------------------------------------------------------------

namespace taosim::stats
{

//-------------------------------------------------------------------------

struct Impl
{
    struct Slot
    {
        book::Book* book{};
        bs2::scoped_connection quoteFeed;
        bs2::scoped_connection tradeFeed;
    };

    std::vector<Slot> slots;
    // Contiguous and separate from the slots so the whole series can be handed out by
    // reference, which is what checkpoint restore converts into.
    std::vector<book::BookTradeStats> tradeStats;
    std::vector<BarSeries> bars;
    std::function<Timestamp()> now;

    // Every entry point advances the clock before it reads or writes, which is what keeps a
    // bar's contents independent of who touched the series and when.
    void rollTo(size_t bookId, Timestamp at) { bars[bookId].roll(at, tradeStats[bookId]); }
};

//-------------------------------------------------------------------------

StatsHub::StatsHub(const Desc& desc)
    : m_impl{std::make_unique<Impl>()}
{
    m_impl->slots.reserve(desc.books.size());
    for (const auto& book : desc.books) {
        m_impl->slots.push_back({.book = book.get()});
    }
    m_impl->tradeStats.resize(desc.books.size());
    m_impl->now = desc.now;
    m_impl->bars.reserve(desc.books.size());
    for (auto i = 0z; i < std::ssize(desc.books); ++i) {
        m_impl->bars.emplace_back(desc.barPeriod, desc.barCapacity);
    }
    // Connected only once every slot and both series have their final addresses.
    for (auto i = 0z; i < std::ssize(m_impl->slots); ++i) {
        auto& slot = m_impl->slots[i];
        auto* impl = m_impl.get();
        const auto bookId = static_cast<size_t>(i);
        slot.quoteFeed = slot.book->signals().L2.connect(
            [impl, bookId](const book::Book* book) {
                const auto currentSpread = book->spread();
                const auto at = impl->now ? impl->now() : Timestamp{};
                impl->rollTo(bookId, at);
                // A one-sided book has no mid and no spread. Passing 0 tells the series to
                // carry what it had rather than to move.
                impl->bars[bookId].observeQuote(
                    util::decimal2double(book->midPrice()),
                    currentSpread ? util::decimal2double(*currentSpread) : 0.0,
                    at);
            });
        // The trade carries the timestamp the book stamped on it, which is the simulation
        // time of the match. Rolling BEFORE recording is what puts a trade landing exactly
        // on a boundary into the bar that is opening rather than the one that just closed.
        slot.tradeFeed = slot.book->signals().trade.connect(
            [impl, bookId](Trade::Ptr trade, BookId) {
                impl->rollTo(bookId, trade->timestamp());
                impl->bars[bookId].recordTrade(
                    util::decimal2double(trade->price()), trade->timestamp());
                impl->tradeStats[bookId].record(
                    trade->price(), trade->volume(), trade->timestamp());
            });
    }
}

//-------------------------------------------------------------------------

StatsHub::~StatsHub() noexcept = default;

//-------------------------------------------------------------------------

L1Snapshot StatsHub::l1(BookId bookId) const
{
    const auto& book = *m_impl->slots.at(bookId).book;

    L1Snapshot snapshot{};
    if (!book.sellQueue().empty()) {
        const auto& bestSellLevel = book.sellQueue().front();
        snapshot.bestAskPrice = bestSellLevel.price();
        snapshot.bestAskVolume = bestSellLevel.volume();
        snapshot.askTotalVolume = book.sellQueue().volume();
    }
    if (!book.buyQueue().empty()) {
        const auto& bestBuyLevel = book.buyQueue().back();
        snapshot.bestBidPrice = bestBuyLevel.price();
        snapshot.bestBidVolume = bestBuyLevel.volume();
        snapshot.bidTotalVolume = book.buyQueue().volume();
    }
    return snapshot;
}

//-------------------------------------------------------------------------

L2Snapshot StatsHub::l2(BookId bookId, size_t depth) const
{
    const auto& book = *m_impl->slots.at(bookId).book;

    auto toLevel = [](const auto& level) -> L2Level {
        return {.price = level.price(), .quantity = level.volume()};
    };
    return {
        .bids = book.buyQueue() | views::reverse | views::take(depth)
            | views::transform(toLevel) | ranges::to<std::vector>,
        .asks = book.sellQueue() | views::take(depth)
            | views::transform(toLevel) | ranges::to<std::vector>
    };
}

//-------------------------------------------------------------------------

const book::BookTradeStats& StatsHub::tradeStats(BookId bookId) const
{
    return m_impl->tradeStats.at(bookId);
}

//-------------------------------------------------------------------------

std::vector<book::BookTradeStats>& StatsHub::tradeStats() noexcept
{
    return m_impl->tradeStats;
}

//-------------------------------------------------------------------------

BarWindow StatsHub::window(BookId bookId, uint32_t bars) const
{
    m_impl->rollTo(bookId, m_impl->now ? m_impl->now() : Timestamp{});
    return m_impl->bars.at(bookId).window(bars);
}

//-------------------------------------------------------------------------

uint32_t StatsHub::barsAvailable(BookId bookId) const
{
    m_impl->rollTo(bookId, m_impl->now ? m_impl->now() : Timestamp{});
    return m_impl->bars.at(bookId).barsAvailable();
}

//-------------------------------------------------------------------------

uint64_t StatsHub::barsClosed(BookId bookId) const
{
    m_impl->rollTo(bookId, m_impl->now ? m_impl->now() : Timestamp{});
    return m_impl->bars.at(bookId).barsClosed();
}

//-------------------------------------------------------------------------

std::vector<double> StatsHub::returns(BookId bookId, uint32_t n) const
{
    m_impl->rollTo(bookId, m_impl->now ? m_impl->now() : Timestamp{});
    return m_impl->bars.at(bookId).returns(n);
}

//-------------------------------------------------------------------------

Timestamp StatsHub::barPeriod() const noexcept
{
    return m_impl->bars.front().period();
}

//-------------------------------------------------------------------------

//-------------------------------------------------------------------------

}  // namespace taosim::stats

//-------------------------------------------------------------------------
