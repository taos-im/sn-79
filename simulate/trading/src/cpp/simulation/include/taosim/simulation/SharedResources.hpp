/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <Eigen/Dense>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

struct SharedResources
{
    // Float (not double): this is n x n with n ~ duration/updatePeriod (thousands),
    // so storing it as float halves a dominant chunk of per-background memory. It is
    // factorised in double and only stored as float (see precomputeFundamentalPriceL).
    Eigen::MatrixXf fundamentalPriceL;
};

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
