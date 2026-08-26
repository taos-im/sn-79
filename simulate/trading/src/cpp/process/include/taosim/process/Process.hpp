/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "common.hpp"

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

struct ProcessDesc
{
    Timestamp updatePeriod{1};
};

//-------------------------------------------------------------------------

class Process
{
public:
    using ValueSignal = UnsyncSignal<void(double)>;

    virtual ~Process() noexcept = default;

    virtual void update(Timestamp timestamp) = 0;
    [[nodiscard]] virtual double value() const = 0;
    // Value recorded to the per-process CSV (the replay source). Defaults to value();
    // processes whose value() is a read-time transform of latent state (e.g. the
    // interpolating FundamentalPrice) override this to record the latent path, so the
    // CSV's meaning is independent of presentation options.
    [[nodiscard]] virtual double loggedValue() const { return value(); }
    [[nodiscard]] virtual uint64_t count() const { return 0; }

    [[nodiscard]] auto&& valueSignal(this auto&& self) noexcept { return self.m_valueSignal; }
    [[nodiscard]] auto&& updatePeriod(this auto&& self) noexcept { return self.m_updatePeriod; }
    [[nodiscard]] auto&& values(this auto&& self) noexcept { return self.m_values; }

protected:
    Process() noexcept = default;

    ValueSignal m_valueSignal;
    Timestamp m_updatePeriod{1};
    std::vector<double> m_values;
    size_t m_valueIdx{};
};

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------
