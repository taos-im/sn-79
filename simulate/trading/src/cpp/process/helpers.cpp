/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/process/helpers.hpp>

#include <taosim/simulation/SharedResources.hpp>

#include <cmath>
#include <cstdint>

//-------------------------------------------------------------------------

namespace taosim::process::helpers
{

//-------------------------------------------------------------------------

namespace
{

double gamma_fn(int64_t k, double H) noexcept
{
    return 0.5 * (std::pow(std::abs(k - 1), 2.0 * H)
                - 2.0 * std::pow(std::abs(k), 2.0 * H)
                + std::pow(std::abs(k + 1), 2.0 * H));
}

}  // namespace

//-------------------------------------------------------------------------

void precomputeFundamentalPriceL(Eigen::MatrixXf& L, double hurst)
{
    const Eigen::Index n = L.rows();
    if (n == 0) return;

    // Symmetric Toeplitz fractional-Gaussian-noise covariance: Gamma_{ij}
    // depends only on |i-j|, so cache the n distinct values and fan out.
    Eigen::VectorXd gammas(n);
    for (Eigen::Index k = 0; k < n; ++k) {
        gammas(k) = (k == 0) ? 1.0 : gamma_fn(k, hurst);
    }

    Eigen::MatrixXd cov(n, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j <= i; ++j) {
            cov(i, j) = gammas(i - j);
        }
    }

    // Factorise in double (Eigen's blocked LLT: BLAS-3 panel + SIMD) for accuracy,
    // then store the factor as float to halve this n x n matrix's footprint. The sim
    // uses L * z for fBm increments, so the narrower mantissa shifts trajectories by
    // a small amount but they remain statistically equivalent.
    const Eigen::MatrixXd factor = cov.selfadjointView<Eigen::Lower>().llt().matrixL();
    L = factor.cast<float>();
}

//-------------------------------------------------------------------------

void initSharedResources(
    taosim::simulation::SharedResources& shared, pugi::xml_node simuNode)
{
    auto fpNode = simuNode
        .child("Agents")
        .child("MultiBookExchangeAgent")
        .child("Books")
        .child("Processes")
        .child("FundamentalPrice");
    if (!fpNode) return;

    const auto duration = simuNode.attribute("duration").as_ullong();
    const auto updatePeriod = fpNode.attribute("updatePeriod").as_ullong(1);
    const double hurst = fpNode.attribute("Hurst").as_double(0.5);
    const auto n = duration / updatePeriod + 2;

    shared.fundamentalPriceL = Eigen::MatrixXf::Zero(n, n);
    precomputeFundamentalPriceL(shared.fundamentalPriceL, hurst);
}

//-------------------------------------------------------------------------

}  // namespace taosim::process::helpers

//-------------------------------------------------------------------------
