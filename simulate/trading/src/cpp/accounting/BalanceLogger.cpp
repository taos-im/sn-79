/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/accounting/BalanceLogger.hpp>

#include <fmt/ranges.h>
#include <spdlog/sinks/basic_file_sink.h>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

//-------------------------------------------------------------------------

BalanceLogger::BalanceLogger(
    const fs::path& filepath,
    decltype(ExchangeSignals::L3)& signal,
    AccountRegistry* registry) noexcept
    : m_filepath{filepath}, m_registry{registry}
{
    const bool fileExisted = fs::exists(m_filepath);

    m_logger = std::make_unique<spdlog::logger>(
        "BalanceLogger", std::make_unique<spdlog::sinks::basic_file_sink_st>(m_filepath));
    m_logger->set_level(spdlog::level::trace);
    m_logger->set_pattern("%v");

    m_feed = signal.connect([this](L3LogEvent event) { log(event); });

    if (fileExisted) return;

    const auto header = fmt::format(
        "time,eventId,{}",
        fmt::join(
            *m_registry
            | views::keys
            | views::transform([](AgentId agentId) {
                return fmt::format("br{0},bf{0},bt{0},qr{0},qf{0},qt{0}", agentId);
            }),
            ","));

    m_logger->trace(header);
    m_logger->flush();
}     

//-------------------------------------------------------------------------

void BalanceLogger::log(L3LogEvent event) const
{
    const auto [bookId, timestamp] = std::visit(
        [](auto&& item) -> std::pair<BookId, Timestamp> {
            using T = std::remove_cvref_t<decltype(item)>;
            if constexpr (std::same_as<T, InstructionLogContext>) {
                return {};
            } else if constexpr (std::same_as<T, OrderWithLogContext>) {
                return {item.logContext->bookId, item.order->timestamp()};
            } else if constexpr (std::same_as<T, TradeWithLogContext>) {
                return {item.logContext->bookId, item.trade->timestamp()};
            } else if constexpr (std::same_as<T, CancellationWithLogContext>) {
                return {item.logContext->bookId, item.logContext->timestamp};
            } else if constexpr (std::same_as<T, AgentResetLogContext>) {
                return {item.bookId, item.timestamp};
            } else {
                static_assert(false, "Unknown L3LogEvent::item type");
            }
        },
        event.item);

    const auto entry = fmt::format(
        "{},{},{}",
        timestamp,
        event.id,
        fmt::join(
            *m_registry
            | views::values
            | views::transform([bookId](const auto& account) {
                const auto& balances = account.at(bookId);
                return fmt::format(
                    "{},{},{},{},{},{}",
                    balances.base.getReserved(),
                    balances.base.getFree(),
                    balances.base.getTotal(),
                    balances.quote->getReserved(),
                    balances.quote->getFree(),
                    balances.quote->getTotal());
            }),
            ","));

    m_logger->trace(entry);
    m_logger->flush();
}

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------
