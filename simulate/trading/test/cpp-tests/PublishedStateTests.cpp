// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// What a multi-asset run publishes to its validator, read as the validator reads it.
//
// The engine packs every realization's part of the published state; the workers do that packing, and the
// clearing of filled orders before it, each on its own realizations rather than in the barrier's single-threaded
// completion. These tests run a publishing run in a child process on private IPC names, answer each state as a
// validator does (an empty response list) and keep the bytes, so what was published can be checked. A realization
// that fails in continue mode no longer steps on its worker; its part must still be its state, frozen at the failure,
// and the other realizations must go on publishing.

#include "MultiAssetRunFixture.hpp"

#include <taosim/ipc/PosixMessageQueue.hpp>
#include <taosim/serialization/msgpack/utils.hpp>
#include <taosim/simulation/SimulationManager.hpp>
#include <taosim/simulation/multiasset/serialization/MultiAssetValidatorRequest.hpp>
#include <taosim/simulation/serialization/ValidatorRequest.hpp>

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim::test::multiasset_run;

namespace bipc = boost::interprocess;

namespace
{

constexpr std::string_view kChildTest = "PublishedStateChild.DISABLED_RunsAndPublishes";
constexpr std::string_view kIpcSuffixVar = "TAOSIM_IPC_SUFFIX";
constexpr std::string_view kPortVar = "TAOSIM_TEST_PUB_PORT";
constexpr std::string_view kFailVar = "TAOSIM_TEST_PUB_FAIL";
constexpr std::string_view kOutVar = "TAOSIM_TEST_PUB_OUT";
constexpr std::string_view kContinue = R"(stopOnRealizationFailure="0")";
constexpr uint32_t kRealizations = 3;
constexpr uint64_t kSteps = 6;
constexpr uint64_t kFailTime = 2 * kStep;

// Answers the run's lifecycle posts with 200, as the validator's notice route does.
class LifecycleStub
{
public:
    LifecycleStub()
    {
        accept();
        m_thread = std::jthread{[this] { m_ctx.run(); }};
    }

    ~LifecycleStub() { m_ctx.stop(); }

    [[nodiscard]] uint16_t port() const { return m_acceptor.local_endpoint().port(); }

private:
    void accept()
    {
        m_acceptor.async_accept([this](boost::beast::error_code ec, boost::asio::ip::tcp::socket socket) {
            if (ec) return;
            namespace http = boost::beast::http;
            boost::beast::flat_buffer buffer;
            http::request<http::string_body> req;
            http::read(socket, buffer, req, ec);
            if (!ec) {
                http::response<http::string_body> res{http::status::ok, req.version()};
                res.set(http::field::content_type, "application/json");
                res.body() = "null";
                res.prepare_payload();
                http::write(socket, res, ec);
            }
            socket.shutdown(boost::asio::ip::tcp::socket::shutdown_send, ec);
            accept();
        });
    }

    boost::asio::io_context m_ctx;
    boost::asio::ip::tcp::acceptor m_acceptor{
        m_ctx, boost::asio::ip::tcp::endpoint{boost::asio::ip::make_address("127.0.0.1"), 0}};
    std::jthread m_thread;
};

// Reads each published state from the state segment as the validator does, keeps its bytes, and answers with an
// empty response list.
class StandInValidator
{
public:
    explicit StandInValidator(std::string suffix)
        : m_suffix{std::move(suffix)},
          m_req{desc("taosim-req")},
          m_res{desc("taosim-res")}
    {
        m_thread = std::jthread{[this](std::stop_token stop) { serve(stop); }};
    }

    ~StandInValidator()
    {
        m_thread.request_stop();
        m_thread.join();
        for (const auto* base : {"taosim-req", "taosim-res"}) {
            taosim::ipc::PosixMessageQueue::remove(fmt::format("/{}{}", base, m_suffix));
        }
        bipc::shared_memory_object::remove(fmt::format("state{}", m_suffix).c_str());
        bipc::shared_memory_object::remove(fmt::format("responses{}", m_suffix).c_str());
    }

    [[nodiscard]] std::vector<std::string> states() const
    {
        const std::scoped_lock lock{m_mutex};
        return m_states;
    }

private:
    taosim::ipc::PosixMessageQueueDesc desc(std::string_view base) const
    {
        taosim::ipc::PosixMessageQueueDesc d{.name = fmt::format("{}{}", base, m_suffix)};
        d.timeout = std::optional<size_t>{static_cast<size_t>(std::chrono::nanoseconds{std::chrono::milliseconds{200}}.count())};
        return d;
    }

    void serve(std::stop_token stop)
    {
        taosim::serialization::BinaryStream reply;
        msgpack::packer pk{reply};
        pk.pack_map(1);
        pk.pack("responses");
        pk.pack_array(0);
        while (!stop.stop_requested()) {
            size_t size{};
            if (m_req.receive(std::span{std::bit_cast<char*>(&size), sizeof(size)}) == -1) continue;
            {
                bipc::shared_memory_object shm{bipc::open_only, fmt::format("state{}", m_suffix).c_str(), bipc::read_only};
                bipc::mapped_region region{shm, bipc::read_only};
                const std::scoped_lock lock{m_mutex};
                m_states.emplace_back(static_cast<const char*>(region.get_address()), size);
            }
            bipc::shared_memory_object out{
                bipc::open_or_create, fmt::format("responses{}", m_suffix).c_str(), bipc::read_write};
            out.truncate(static_cast<bipc::offset_t>(reply.size()));
            bipc::mapped_region region{out, bipc::read_write};
            std::memcpy(region.get_address(), reply.data(), reply.size());
            const size_t replySize = reply.size();
            m_res.send(std::span{std::bit_cast<const char*>(&replySize), sizeof(replySize)});
        }
    }

    std::string m_suffix;
    taosim::ipc::PosixMessageQueue m_req;
    taosim::ipc::PosixMessageQueue m_res;
    mutable std::mutex m_mutex;
    std::vector<std::string> m_states;
    std::jthread m_thread;
};

struct Published
{
    uint64_t timestamp{};
    std::vector<std::string> instances;  // each realization's part, re-packed, in flat order
    std::vector<std::string> comparable;  // the same parts with the child's process id replaced
};

[[nodiscard]] std::string repack(const msgpack::object& o)
{
    taosim::serialization::BinaryStream stream;
    msgpack::packer pk{stream};
    pk.pack(o);
    return {stream.data(), stream.size()};
}

// A part with the child's temporary directory, which carries its process id (the logDir, and the start notice's run
// directory), replaced by one placeholder, so two runs compare byte for byte.
[[nodiscard]] std::string comparable(std::string bytes)
{
    static const std::regex pid{R"(taosim-test-ma-[0-9]+-child)"};
    return std::regex_replace(bytes, pid, "taosim-test-ma-PID-child");
}

[[nodiscard]] uint64_t fnv1a(std::string_view bytes)
{
    uint64_t h = 1469598103934665603ull;
    for (const unsigned char c : bytes) h = (h ^ c) * 1099511628211ull;
    return h;
}

[[nodiscard]] Published parse(const std::string& bytes)
{
    Published out;
    const auto oh = msgpack::unpack(bytes.data(), bytes.size());
    const auto& top = oh.get();
    EXPECT_EQ(top.type, msgpack::type::MAP);
    if (top.type != msgpack::type::MAP) return out;
    for (const auto& [k, v] : top.via.map) {
        const auto key = k.as<std::string_view>();
        if (key == "timestamp") out.timestamp = v.as<uint64_t>();
        if (key != "backgrounds") continue;
        for (const auto& bg : v.via.array) {
            for (const auto& [bk, bv] : bg.via.map) {
                if (bk.as<std::string_view>() != "instances") continue;
                for (const auto& instance : bv.via.array) {
                    EXPECT_EQ(instance.type, msgpack::type::MAP) << "a published part is not a validator request";
                    out.instances.push_back(repack(instance));
                    out.comparable.push_back(comparable(out.instances.back()));
                }
            }
        }
    }
    return out;
}

struct PublishingRun
{
    std::vector<Published> published;
    std::vector<std::string> finalParts;  // each realization packed after the run, re-packed
    ChildExit exit;
};

[[nodiscard]] PublishingRun runPublishingInChild(std::string_view name, std::optional<uint32_t> fail)
{
    const auto dir = fs::temp_directory_path() / fmt::format("taosim-test-pub-{}-{}", ::getpid(), name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    const auto suffix = fmt::format("-test-pub-{}-{}", ::getpid(), name);
    LifecycleStub lifecycle;
    PublishingRun out;
    {
        StandInValidator validator{suffix};
        out.exit = runTestInChild(
            kChildTest,
            {{std::string{kIpcSuffixVar}, suffix},
             {std::string{kPortVar}, std::to_string(lifecycle.port())},
             {std::string{kFailVar}, fail ? std::to_string(*fail) : std::string{}},
             {std::string{kOutVar}, (dir / "parts").string()}},
            dir / "child.log");
        for (const auto& state : validator.states()) out.published.push_back(parse(state));
    }
    if (out.exit.clean()) {
        const auto bytes = readBytes(dir / "parts");
        std::vector<std::string> raw;
        msgpack::unpack(bytes.data(), bytes.size()).get().convert(raw);
        for (const auto& part : raw) out.finalParts.push_back(repack(msgpack::unpack(part.data(), part.size()).get()));
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    return out;
}

}  // namespace

//-------------------------------------------------------------------------

TEST(PublishedState, EveryStepPublishesEveryRealizationsPart)
{
    const auto run = runPublishingInChild("all", std::nullopt);
    ASSERT_TRUE(run.exit.clean()) << run.exit.log;
    ASSERT_EQ(run.published.size(), kSteps) << "one state per step, past a grace period of zero";
    for (const auto& state : run.published) {
        ASSERT_EQ(state.instances.size(), kRealizations);
        for (uint32_t i = 0; i < kRealizations; ++i) {
            EXPECT_TRUE(state.instances[i].contains(fmt::format("/bg-{}", i)))
                << "at " << state.timestamp << " part " << i << " is not realization " << i << "'s";
        }
    }
    for (uint32_t i = 0; i < kRealizations; ++i) {
        EXPECT_NE(run.published.front().instances[i], run.published.back().instances[i])
            << "realization " << i << "'s part did not move over the run";
    }
}

TEST(PublishedState, AFailedRealizationPublishesItsFrozenStateAndTheOthersGoOn)
{
    const auto run = runPublishingInChild("failed", 1);
    ASSERT_TRUE(run.exit.clean()) << run.exit.log;
    ASSERT_EQ(run.finalParts.size(), kRealizations);
    ASSERT_EQ(run.published.size(), kSteps) << "the other realizations stopped publishing after the failure";
    std::optional<size_t> firstAfter;
    for (size_t i = 0; i < run.published.size(); ++i) {
        const auto& state = run.published[i];
        ASSERT_EQ(state.instances.size(), kRealizations);
        if (state.timestamp <= kFailTime) continue;
        if (!firstAfter) firstAfter = i;
        EXPECT_EQ(state.instances[1], run.finalParts[1])
            << "at " << state.timestamp << " the failed realization published something other than its frozen state";
    }
    ASSERT_TRUE(firstAfter.has_value() && *firstAfter + 1 < run.published.size());
    for (const uint32_t i : {0u, 2u}) {
        EXPECT_NE(run.published.back().instances[i], run.published[*firstAfter].instances[i])
            << "realization " << i << " did not step on beside the failure";
    }
}

// The seeded run publishes the same parts every time, and their digests, printed, compare one build with another:
// the parity of a change to how the state is packed (onto the workers, for one) is that the digests of a
// build before it and a build after it agree, step for step.
TEST(PublishedState, TheSeededRunPublishesTheSamePartsEveryTime)
{
    const auto a = runPublishingInChild("parity-a", std::nullopt);
    const auto b = runPublishingInChild("parity-b", std::nullopt);
    ASSERT_TRUE(a.exit.clean() && b.exit.clean()) << a.exit.log << b.exit.log;
    ASSERT_EQ(a.published.size(), b.published.size());
    for (size_t step = 0; step < a.published.size(); ++step) {
        EXPECT_EQ(a.published[step].comparable, b.published[step].comparable) << "step " << step << " differs";
        std::string line = fmt::format("published-digest step={} t={}", step, a.published[step].timestamp);
        for (const auto& part : a.published[step].comparable) line += fmt::format(" {:016x}", fnv1a(part));
        fmt::println("{}", line);
    }
}

// A caller that packs the request whole, without the parts its workers pack, still gets every realization, in order:
// the request then packs each one itself. (Packing a realization consumes its proxy's notices, so the two forms
// cannot be compared on one state; the digests above compare the published bytes of a build before the change.)
TEST(PublishedState, ARequestPackedWholeStillCarriesEveryRealization)
{
    const MultiAssetRun run{
        "whole", {.realizations = kRealizations, .threads = 1, .steps = kSteps, .ckptIntervalInSteps = 0}};
    const auto orchestrator = run.build();
    orchestrator->run();
    taosim::serialization::HumanReadableStream whole;
    msgpack::pack(whole, taosim::simulation::multiasset::serialization::MultiAssetValidatorRequest{
        .simulations = orchestrator->checkpointBlocks(),
        .config = &orchestrator->config(),
        .logDir = orchestrator->config().baseDir});
    const auto state = parse(std::string{whole.data(), whole.size()});
    ASSERT_EQ(state.instances.size(), kRealizations);
    for (uint32_t i = 0; i < kRealizations; ++i) {
        EXPECT_TRUE(state.instances[i].contains(fmt::format("/bg-{}", i))) << "part " << i << " is not realization " << i << "'s";
    }
}

// The child side of the tests above, run only when re-executed by them.
TEST(PublishedStateChild, DISABLED_RunsAndPublishes)
{
    const char* suffix = std::getenv(kIpcSuffixVar.data());
    const char* port = std::getenv(kPortVar.data());
    const char* fail = std::getenv(kFailVar.data());
    const char* out = std::getenv(kOutVar.data());
    if (suffix == nullptr || *suffix == '\0' || port == nullptr || fail == nullptr || out == nullptr) {
        GTEST_SKIP() << "runs only as the child of a PublishedState test";
    }
    ASSERT_TRUE(taosim::simulation::SimulationManager::s_statePublishShmName.ends_with(suffix));
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    const MultiAssetRun run{
        "child", {.realizations = kRealizations, .threads = 1, .steps = kSteps, .ckptIntervalInSteps = 0,
                  .extraAttrs = fmt::format(
                      R"(useMessagePack="1" host="127.0.0.1" port="{}" generalMsgEndpoint="/account" {})", port,
                      kContinue)}};
    const auto orchestrator = run.build();
    if (*fail != '\0') {
        const auto* sim = orchestrator->checkpointBlocks()[std::stoul(fail)].get();
        sim->signals().step.connect([sim] {
            if (sim->currentTimestamp() >= kFailTime) throw std::runtime_error{"injected failure"};
        });
    }
    orchestrator->run();

    const auto& blocks = orchestrator->checkpointBlocks();
    std::vector<std::string> parts;
    for (size_t i = 0; i < blocks.size(); ++i) {
        // The stream the engine publishes through: decimals as doubles, as the validator reads them.
        taosim::serialization::HumanReadableStream stream{1uz << 22};
        msgpack::pack(stream, taosim::simulation::serialization::ValidatorRequest{
            .simulations = std::span{blocks}.subspan(i, 1), .logDir = blocks[i]->logDir()});
        parts.emplace_back(stream.data(), stream.size());
    }
    taosim::serialization::BinaryStream stream;
    msgpack::packer pk{stream};
    pk.pack(parts);
    std::ofstream{out, std::ios::binary}.write(stream.data(), stream.size());
}

//-------------------------------------------------------------------------
