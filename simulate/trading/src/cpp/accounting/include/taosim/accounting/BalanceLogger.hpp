/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "AccountRegistry.hpp"
#include <taosim/matching/ExchangeSignals.hpp>
#include "common.hpp"

#include <spdlog/spdlog.h>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

//-------------------------------------------------------------------------

class BalanceLogger
{
public:
    BalanceLogger(
        const fs::path& filepath,
        decltype(matching::ExchangeSignals::L3)& signal,
        AccountRegistry* registry) noexcept;

    [[nodiscard]] const fs::path& filepath() const noexcept { return m_filepath; }

    void log(L3LogEvent event) const;

private:
    std::unique_ptr<spdlog::logger> m_logger;
    fs::path m_filepath;
    bs2::scoped_connection m_feed;
    AccountRegistry* m_registry;
};

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------