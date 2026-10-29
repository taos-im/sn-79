/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// A resumed GBM or JumpDiffusion continues exactly as the uninterrupted process would.
//
// A book's process manager packed both as nil, so a resume restarted them at X0 with a fresh generator and the path
// jumped at every resume; the last test holds the book's checkpoint, the others the processes' own adaptors.
//
// Their normal distributions draw in pairs and keep the second value for the next call, and the Poisson distribution
// holds a normal distribution of its own. The checkpoint carried the generator, whose position was right, but not the
// distributions, so a process checkpointed with a normal held back drew a new pair after the resume and its path left
// the uninterrupted one at the first step. Each is stepped an odd number of times so a value is held, checkpointed and
// restored into a fresh process, and both are stepped on: the values must be identical.

#include <taosim/book/BookProcessManager.hpp>
#include <taosim/checkpoint/serialization/book/BookProcessManager.hpp>
#include <taosim/checkpoint/serialization/process/GBM.hpp>
#include <taosim/checkpoint/serialization/process/JumpDiffusion.hpp>
#include <taosim/process/GBM.hpp>
#include <taosim/process/JumpDiffusion.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

using taosim::process::GBM;
using taosim::process::GBMDesc;
using taosim::process::JumpDiffusion;
using taosim::process::JumpDiffusionDesc;

namespace
{

constexpr uint64_t kSeed = 20261022;

template<typename T>
void restoreInto(const T& value, T& restored)
{
    taosim::serialization::BinaryStream stream;
    msgpack::packer<taosim::serialization::BinaryStream> packer{stream};
    packer.pack(value);
    auto handle = msgpack::unpack(stream.data(), stream.size());
    handle.get().convert(restored);
}

template<typename P>
std::vector<double> stepOn(P& process, int steps)
{
    std::vector<double> out;
    for (int i = 0; i < steps; ++i) {
        process.update(Timestamp{});
        out.push_back(process.value());
    }
    return out;
}

template<typename P, typename Desc>
void expectAResumeContinuesThePath(const Desc& desc)
{
    P uninterrupted{desc};
    (void)stepOn(uninterrupted, 3);
    P resumed{desc};
    restoreInto(uninterrupted, resumed);
    EXPECT_EQ(stepOn(resumed, 6), stepOn(uninterrupted, 6))
        << "the resumed path left the uninterrupted one: a distribution's held value was not in the checkpoint";
}

GBMDesc gbmDesc()
{
    return GBMDesc{.X0 = 100.0, .mu = 0.0, .sigma = 0.2, .dt = 1.0 / 86400.0, .seed = kSeed, .proc = {}};
}

JumpDiffusionDesc jumpDesc()
{
    return JumpDiffusionDesc{
        .X0 = 100.0, .mu = 0.0, .sigma = 0.2, .dt = 1.0 / 86400.0,
        .lambda = 30.0, .muJump = 0.0, .sigmaJump = 0.01, .seed = kSeed, .proc = {}};
}

std::vector<double> stepBook(taosim::book::BookProcessManager& manager, int steps)
{
    std::vector<double> out;
    for (int i = 0; i < steps; ++i) {
        for (auto& [name, processes] : manager.container()) {
            for (auto& process : processes) {
                process->update(Timestamp{});
                out.push_back(process->value());
            }
        }
    }
    return out;
}

}  // namespace

//-------------------------------------------------------------------------

TEST(ProcessDistributionResume, AResumedGBMContinuesTheUninterruptedPath)
{
    expectAResumeContinuesThePath<GBM>(gbmDesc());
}

TEST(ProcessDistributionResume, AResumedJumpDiffusionContinuesTheUninterruptedPath)
{
    expectAResumeContinuesThePath<JumpDiffusion>(jumpDesc());
}

TEST(ProcessDistributionResume, ABookCheckpointCarriesItsGBMAndJumpDiffusion)
{
    taosim::book::BookProcessManager uninterrupted, resumed;
    for (auto* manager : {&uninterrupted, &resumed}) {
        manager->container()["gbm"].push_back(std::make_unique<GBM>(gbmDesc()));
        manager->container()["jump"].push_back(std::make_unique<JumpDiffusion>(jumpDesc()));
    }
    (void)stepBook(uninterrupted, 3);
    restoreInto(uninterrupted, resumed);
    EXPECT_EQ(stepBook(resumed, 6), stepBook(uninterrupted, 6))
        << "a book's GBM or JumpDiffusion restarted at the resume instead of continuing its path";
}

//-------------------------------------------------------------------------
