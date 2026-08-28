// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// The duplicate-delivery guard's discriminator.
//
// A response timeout on the validator side once caused the same request to be delivered twice. The engine
// had no idea and executed it twice: one order of 1.0 alpha minted trade ids 683 AND 684 with the same
// engine timestamp, and the second never settled and became a phantom with no tape row. The validator-side
// fix stops provoking it; this guard makes it impossible.
//
// `Batch.step` was the obvious key and is wrong: the validator sends a reconciliation AND an instruction
// batch under the same step, so guarding on it would silently drop every second request and stop the
// exchange filling anything. The request BYTES are unique per request, because a re-send transmits the
// identical buffer while any genuinely new request differs in at least its instruction sequence numbers.

#include <gtest/gtest.h>

#include <taosim/ipc/MsgPackChannel.hpp>

#include <cstdint>
#include <string>
#include <vector>

using taosim::ipc::detail::hashRequestBytes;

TEST(RequestFingerprint, TheSameBytesHashTheSame)
{
    const std::string buf = "\x82\xa4step\x01\xacinstructions\x90";
    EXPECT_EQ(hashRequestBytes(buf.data(), buf.size()),
              hashRequestBytes(buf.data(), buf.size()));
}

TEST(RequestFingerprint, ADifferentByteChangesTheHash)
{
    std::string a = "\x82\xa4step\x01\xacinstructions\x90";
    std::string b = a;
    b[6] = '\x02';  // step 1 -> 2
    EXPECT_NE(hashRequestBytes(a.data(), a.size()), hashRequestBytes(b.data(), b.size()));
}

TEST(RequestFingerprint, ADifferentLengthChangesTheHash)
{
    // The case that matters most: the same prefix with one more instruction appended must not collide,
    // or a growing batch would be mistaken for a re-send of its own prefix.
    const std::string a = "\x82\xa4step\x01";
    const std::string b = a + "\xa1x";
    EXPECT_NE(hashRequestBytes(a.data(), a.size()), hashRequestBytes(b.data(), b.size()));
}

TEST(RequestFingerprint, EmptyAndNullAreZeroSoTheyNeverMatchARealRequest)
{
    // Zero is the "nothing served yet" sentinel, so it must be unreachable for real input.
    EXPECT_EQ(hashRequestBytes(nullptr, 0), 0u);
    EXPECT_EQ(hashRequestBytes(nullptr, 16), 0u);
    const char c = '\0';
    EXPECT_EQ(hashRequestBytes(&c, 0), 0u);
}

TEST(RequestFingerprint, ANonEmptyRequestNeverHashesToTheSentinel)
{
    // A single zero byte is the input most likely to mix down to zero; it must not.
    const char zeros[8] = {};
    for (std::size_t n = 1; n <= sizeof(zeros); ++n) {
        EXPECT_NE(hashRequestBytes(zeros, n), 0u) << "size " << n;
    }
}

TEST(RequestFingerprint, TwoBatchesDifferingOnlyInASequenceNumberDiffer)
{
    // The validator stamps every instruction with a client order id that embeds a per-instruction
    // sequence, so two legitimate batches that are otherwise identical still differ in their bytes. This
    // is what makes a bytes-based guard safe rather than over-eager.
    auto batch = [](std::uint32_t coid) {
        std::string s = "\x82\xa4step\x01\xacinstructions\x91\x81\xa1\x63";
        s.append(reinterpret_cast<const char*>(&coid), sizeof(coid));
        return s;
    };
    const auto a = batch(0x11223344u);
    const auto b = batch(0x11223345u);
    EXPECT_NE(hashRequestBytes(a.data(), a.size()), hashRequestBytes(b.data(), b.size()));
}
