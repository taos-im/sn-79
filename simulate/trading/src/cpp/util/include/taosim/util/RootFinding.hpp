/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cmath>

//-------------------------------------------------------------------------

namespace taosim::util
{

//-------------------------------------------------------------------------

// Why a solve did not produce a root. SameSign* says where the root actually is:
// for a decreasing residual, both-positive puts it above the upper bound and
// both-negative below the lower bound, which is usually a statement about the
// agent's balances rather than about arithmetic.
enum class RootStatus
{
    Converged,
    BadBracket,
    NonFiniteBound,
    SameSignPositive,
    SameSignNegative,
    NonFiniteMid,
};

//-------------------------------------------------------------------------

[[nodiscard]] constexpr const char* rootStatusName(RootStatus s) noexcept
{
    switch (s) {
        case RootStatus::Converged: return "converged";
        case RootStatus::BadBracket: return "badBracket";
        case RootStatus::NonFiniteBound: return "nonFiniteBound";
        case RootStatus::SameSignPositive: return "sameSignPositive";
        case RootStatus::SameSignNegative: return "sameSignNegative";
        case RootStatus::NonFiniteMid: return "nonFiniteMid";
    }
    return "unknown";
}

//-------------------------------------------------------------------------

// The original bracket and its residuals are carried out so a caller can report
// what it actually handed the solver, rather than inferring it afterwards.
struct BracketedRoot
{
    double value;
    bool converged;
    RootStatus status;
    double lo;
    double hi;
    double fLo;
    double fHi;
};

//-------------------------------------------------------------------------

// Bisection over a sign-changing bracket. The mean-variance price residuals
// carry a 1/price singularity, so an unbracketed iteration walks into it from
// any start that is not already near the root.
template<typename F>
[[nodiscard]] BracketedRoot solveScalarBracketed(
    F&& residual, double lo, double hi, double rtol = 1e-10, int maxIter = 200)
{
    const double lo0 = lo;
    const double hi0 = hi;
    double flo0 = std::numeric_limits<double>::quiet_NaN();
    double fhi0 = flo0;

    const auto out = [&](double v, bool conv, RootStatus s) -> BracketedRoot {
        return {.value = v, .converged = conv, .status = s,
                .lo = lo0, .hi = hi0, .fLo = flo0, .fHi = fhi0};
    };
    const auto fail = [&](double v, RootStatus s) { return out(v, false, s); };
    const auto ok = [&](double v) { return out(v, true, RootStatus::Converged); };

    if (!(lo > 0.0) || !(lo < hi)) return fail(hi, RootStatus::BadBracket);

    const double flo = flo0 = residual(lo);
    const double fhi = fhi0 = residual(hi);
    if (!std::isfinite(flo) || !std::isfinite(fhi)) return fail(hi, RootStatus::NonFiniteBound);
    if (flo == 0.0) return ok(lo);
    if (fhi == 0.0) return ok(hi);
    if ((flo > 0.0) == (fhi > 0.0)) {
        return fail(hi, flo > 0.0 ? RootStatus::SameSignPositive : RootStatus::SameSignNegative);
    }

    const bool loPositive = flo > 0.0;
    for (int i = 0; i < maxIter && hi - lo > rtol * hi; ++i) {
        const double mid = lo + 0.5 * (hi - lo);
        const double fmid = residual(mid);
        if (!std::isfinite(fmid)) return fail(mid, RootStatus::NonFiniteMid);
        if (fmid == 0.0) return ok(mid);
        if ((fmid > 0.0) == loPositive) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return ok(lo + 0.5 * (hi - lo));
}

//-------------------------------------------------------------------------

}  // namespace taosim::util

//-------------------------------------------------------------------------
