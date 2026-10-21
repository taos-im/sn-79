/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// A checkpoint written by the 0.6.3 engine, restored by the current one. Two parts of the state changed
// form: the dynamic fee policy kept one last-update time for the whole policy where it now keeps one per
// book, and the random generator's draw count was a 32-bit field where it is now 64-bit. Each is packed
// here exactly as the 0.6.3 packer wrote it and restored through the reader a resume uses.

#include <taosim/checkpoint/serialization/matching/FeePolicyWrapper.hpp>
#include <taosim/checkpoint/serialization/process/RNG.hpp>
#include <taosim/matching/DynamicFeePolicy.hpp>
#include <taosim/matching/FeePolicyWrapper.hpp>
#include <taosim/process/RNG.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string_view>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::matching;
using namespace taosim::literals;
using taosim::process::RNG;

//-------------------------------------------------------------------------

namespace
{

using Packer = msgpack::packer<taosim::serialization::BinaryStream>;

constexpr int kHistorySlots = 3;
constexpr uint64_t kSeed = 1234567;
const std::string kAgentBaseName = "STYLIZED_TRADER_AGENT";

std::unique_ptr<DynamicFeePolicy> makeDynamicPolicy()
{
    return std::make_unique<DynamicFeePolicy>(DFPolicyDesc{
        .simulation = nullptr,
        .historySlots = kHistorySlots,
        .makerFee = 0_dec,
        .takerFee = 0_dec,
        .maxMakerRate = DEC(0.015),
        .maxTakerRate = DEC(0.015),
        .targetMTR = DEC(0.4),
        .shapeMakerFee = 2_dec,
        .shapeMakerRebate = 2_dec});
}

// The 0.6.3 packer: five keys, and lastUpdate is a single time for the whole policy.
void packDynamicPolicy063(Packer& pk, const DynamicFeePolicy& policy, Timestamp policyWideLastUpdate)
{
    pk.pack_map(5);
    pk.pack("type");
    pk.pack("dynamic");
    pk.pack("lastUpdate");
    pk.pack(policyWideLastUpdate);
    pk.pack("totalVolumes");
    pk.pack(policy.totalVolumes());
    pk.pack("totalVolumesPrev");
    pk.pack(policy.totalVolumesPrev());
    pk.pack("volumes");
    pk.pack(policy.volumes());
}

void expectSameHistory(const DynamicFeePolicy& restored, const DynamicFeePolicy& written)
{
    ASSERT_EQ(restored.volumes().size(), written.volumes().size());
    for (const auto& [bookId, slots] : written.volumes()) {
        ASSERT_TRUE(restored.volumes().contains(bookId)) << "book " << bookId;
        const auto& got = restored.volumes().at(bookId);
        EXPECT_EQ(got.capacity(), slots.capacity()) << "book " << bookId;
        ASSERT_EQ(got.size(), slots.size()) << "book " << bookId;
        for (size_t i = 0; i < slots.size(); ++i) {
            EXPECT_EQ(got[i].aggressive, slots[i].aggressive) << "book " << bookId << " slot " << i;
            EXPECT_EQ(got[i].passive, slots[i].passive) << "book " << bookId << " slot " << i;
        }
    }
    ASSERT_EQ(restored.totalVolumes().size(), written.totalVolumes().size());
    for (const auto& [bookId, total] : written.totalVolumes()) {
        EXPECT_EQ(restored.totalVolumes().at(bookId).aggressive, total.aggressive) << "book " << bookId;
        EXPECT_EQ(restored.totalVolumes().at(bookId).passive, total.passive) << "book " << bookId;
    }
    ASSERT_EQ(restored.totalVolumesPrev().size(), written.totalVolumesPrev().size());
    for (const auto& [bookId, prev] : written.totalVolumesPrev()) {
        EXPECT_EQ(restored.totalVolumesPrev().at(bookId).aggressive, prev.aggressive) << "book " << bookId;
        EXPECT_EQ(restored.totalVolumesPrev().at(bookId).passive, prev.passive) << "book " << bookId;
    }
}

}  // namespace

//-------------------------------------------------------------------------

TEST(CheckpointFormat063, AFeePolicyWithOnlyThePolicyWideTimeRestoresThroughTheWrapper)
{
    auto defaultWritten = makeDynamicPolicy();
    defaultWritten->updateHistory(10, 0, 1, 5_dec, true);
    defaultWritten->updateHistory(10, 1, 1, 7_dec, false);
    defaultWritten->updateHistory(20, 0, 1, 1_dec, false);
    defaultWritten->updateHistory(30, 2, 1, 2_dec, true);
    auto agentWritten = makeDynamicPolicy();
    agentWritten->updateHistory(15, 0, 2, 3_dec, false);
    agentWritten->updateHistory(25, 0, 2, 4_dec, true);
    constexpr Timestamp kDefaultLastUpdate = 30, kAgentLastUpdate = 25;

    taosim::serialization::BinaryStream stream;
    Packer pk{stream};
    pk.pack_map(2);
    pk.pack("defaultFeePolicy");
    packDynamicPolicy063(pk, *defaultWritten, kDefaultLastUpdate);
    pk.pack("agentBaseNameFeePolicies");
    pk.pack_map(1);
    pk.pack(kAgentBaseName);
    packDynamicPolicy063(pk, *agentWritten, kAgentLastUpdate);

    FeePolicyWrapper wrapper{makeDynamicPolicy(), nullptr};
    wrapper[kAgentBaseName] = makeDynamicPolicy();
    msgpack::unpack(stream.data(), stream.size()).get().convert(wrapper);

    const auto& defaultRestored = dynamic_cast<const DynamicFeePolicy&>(*wrapper.feePolicy());
    const auto& agentRestored =
        dynamic_cast<const DynamicFeePolicy&>(*wrapper.agentBaseNameFeePolicies().at(kAgentBaseName));

    expectSameHistory(defaultRestored, *defaultWritten);
    expectSameHistory(agentRestored, *agentWritten);

    // Every book with a history takes the policy-wide time, which is where 0.6.3 would have continued.
    EXPECT_EQ(defaultRestored.lastUpdate(),
        (std::map<BookId, Timestamp>{{0, kDefaultLastUpdate}, {1, kDefaultLastUpdate}, {2, kDefaultLastUpdate}}));
    EXPECT_EQ(agentRestored.lastUpdate(), (std::map<BookId, Timestamp>{{0, kAgentLastUpdate}}));

    auto& resumed = dynamic_cast<DynamicFeePolicy&>(*wrapper.feePolicy());
    const auto slotsBefore = resumed.volumes().at(1).size();
    resumed.updateHistory(kDefaultLastUpdate, 1, 1, 1_dec, true);
    EXPECT_EQ(resumed.volumes().at(1).size(), slotsBefore) << "a trade at the restored time opened a new slot";
    resumed.updateHistory(kDefaultLastUpdate + 10, 1, 1, 1_dec, true);
    EXPECT_EQ(resumed.volumes().at(1).size(), slotsBefore + 1) << "a trade at a later time joined the old slot";
}

//-------------------------------------------------------------------------

TEST(CheckpointFormat063, AGeneratorWithAThirtyTwoBitCountRestoresItsPositionAndCount)
{
    // Above 16 bits, so msgpack writes the 0.6.3 uint32_t count in its four-byte form.
    constexpr uint32_t kCount = 70'000;

    taosim::serialization::BinaryStream stream;
    Packer pk{stream};
    pk.pack_map(2);
    pk.pack("callCount");
    pk.pack(kCount);
    pk.pack("seed");
    pk.pack(kSeed);

    const std::string_view bytes{stream.data(), stream.size()};
    const auto key = bytes.find("callCount");
    ASSERT_NE(key, std::string_view::npos);
    ASSERT_EQ(static_cast<uint8_t>(bytes[key + std::string_view{"callCount"}.size()]), 0xce)
        << "the count was not written as a 32-bit integer";

    RNG restored;
    msgpack::unpack(stream.data(), stream.size()).get().convert(restored);

    EXPECT_EQ(restored.seedValue(), kSeed);
    EXPECT_EQ(restored.callCount(), uint64_t{kCount}) << "the next checkpoint would record the wrong position";

    RNG reference{kSeed};
    for (uint32_t i = 0; i < kCount; ++i) reference();
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(restored(), reference()) << "draw " << i << " after the restore";
    }
    EXPECT_EQ(restored.callCount(), reference.callCount());
}

//-------------------------------------------------------------------------
