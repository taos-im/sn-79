// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// A realization of a multi-asset run that throws is logged at the throw with its index, once. By
// default it stops the run at the next barrier: nothing is published or checkpointed for that step,
// the validator gets one failure notice and no end notice, the last checkpoint before the failure is
// kept past the retention bound, and the run exits with the failure. A run that sets
// stopOnRealizationFailure="0" goes on without the realization instead: the others keep stepping
// exactly as they would have, the run's clock (the grace gate, the checkpoint name) follows them,
// checkpoints mark the realization, and a resume keeps it stopped. Either way a run with nothing left
// stepping stops.
//
// Failures are injected from a realization's own step signal. The in-process runs are offline, so they
// open no queue; a run that posts notices goes to a child process with private queue names.

#include "MultiAssetRunFixture.hpp"

#include <taosim/checkpoint/CheckpointFormat.hpp>
#include <taosim/ipc/PosixMessageQueue.hpp>
#include <taosim/serialization/msgpack/utils.hpp>
#include <taosim/simulation/SimulationManager.hpp>

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <gtest/gtest.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim::test::multiasset_run;

//-------------------------------------------------------------------------

namespace
{

constexpr std::string_view kInjected = "injected failure";
constexpr std::string_view kContinue = R"(stopOnRealizationFailure="0")";

// Makes realization `idx` throw at the end of its first step that reaches `time`.
void failAt(const Orchestrator& orchestrator, uint32_t idx, uint64_t time)
{
    const auto* sim = orchestrator.checkpointBlocks()[idx].get();
    sim->signals().step.connect([sim, time] {
        if (sim->currentTimestamp() >= time) throw std::runtime_error{std::string{kInjected}};
    });
}

[[nodiscard]] std::vector<uint64_t> clocks(const Orchestrator& orchestrator)
{
    std::vector<uint64_t> out;
    for (const auto& sim : orchestrator.checkpointBlocks()) out.push_back(sim->currentTimestamp());
    return out;
}

[[nodiscard]] std::vector<uint64_t> checkpointTimes(const MultiAssetRun& run)
{
    std::vector<uint64_t> out;
    for (const auto& dir : run.checkpoints()) out.push_back(std::stoull(dir.stem().string()));
    return out;
}

[[nodiscard]] size_t occurrences(std::string_view haystack, std::string_view needle)
{
    size_t count{};
    for (auto pos = haystack.find(needle); pos != std::string_view::npos; pos = haystack.find(needle, pos + 1)) {
        ++count;
    }
    return count;
}

constexpr std::string_view kChildTest = "RealizationFailureChild.DISABLED_RunsAndPacksEveryRealization";
constexpr std::string_view kFailVar = "TAOSIM_TEST_MA_FAIL";
constexpr std::string_view kOutVar = "TAOSIM_TEST_MA_OUT";

// Every realization's packed state at the end of a run in a child process, realization `fail` throwing at
// the end of its step to 2 s; nothing when the child did not finish.
[[nodiscard]] std::vector<std::string> runInChild(std::string_view name, std::optional<uint32_t> fail)
{
    const auto dir = fs::temp_directory_path() / fmt::format("taosim-test-ma-child-{}-{}", ::getpid(), name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    const auto exit = runTestInChild(
        kChildTest,
        {{std::string{kFailVar}, fail ? std::to_string(*fail) : std::string{}},
         {std::string{kOutVar}, (dir / "blocks").string()}},
        dir / "child.log");
    std::vector<std::string> blocks;
    if (exit.clean()) {
        const auto bytes = readBytes(dir / "blocks");
        msgpack::unpack(bytes.data(), bytes.size()).get().convert(blocks);
    } else {
        ADD_FAILURE() << exit.log;
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    return blocks;
}

// Answers each lifecycle post as the validator's notice route does and keeps every message in it.
class NoticeRecorder
{
public:
    NoticeRecorder()
    {
        accept();
        m_thread = std::jthread{[this] { m_ctx.run(); }};
    }

    ~NoticeRecorder() { m_ctx.stop(); }

    [[nodiscard]] uint16_t port() const { return m_acceptor.local_endpoint().port(); }

    // Every message of the given type, each as its own JSON document.
    [[nodiscard]] std::vector<std::string> messages(std::string_view type) const
    {
        const std::scoped_lock lock{m_mutex};
        std::vector<std::string> out;
        for (const auto& [t, json] : m_messages) {
            if (t == type) out.push_back(json);
        }
        return out;
    }

private:
    void accept()
    {
        m_acceptor.async_accept([this](boost::beast::error_code ec, boost::asio::ip::tcp::socket socket) {
            if (ec) return;
            serve(std::move(socket));
            accept();
        });
    }

    void serve(boost::asio::ip::tcp::socket socket)
    {
        namespace http = boost::beast::http;
        boost::beast::flat_buffer buffer;
        http::request<http::string_body> req;
        boost::beast::error_code ec;
        http::read(socket, buffer, req, ec);
        if (ec) return;
        rapidjson::Document body;
        body.Parse(req.body().c_str());
        if (body.IsObject() && body.HasMember("messages") && body["messages"].IsArray()) {
            const std::scoped_lock lock{m_mutex};
            for (const auto& message : body["messages"].GetArray()) {
                if (!message.IsObject() || !message.HasMember("type") || !message["type"].IsString()) continue;
                rapidjson::StringBuffer text;
                rapidjson::Writer<rapidjson::StringBuffer> writer{text};
                message.Accept(writer);
                m_messages.emplace_back(message["type"].GetString(), text.GetString());
            }
        }
        http::response<http::string_body> res{http::status::ok, req.version()};
        res.set(http::field::content_type, "application/json");
        res.body() = "null";
        res.prepare_payload();
        http::write(socket, res, ec);
        socket.shutdown(boost::asio::ip::tcp::socket::shutdown_send, ec);
    }

    boost::asio::io_context m_ctx;
    boost::asio::ip::tcp::acceptor m_acceptor{
        m_ctx, boost::asio::ip::tcp::endpoint{boost::asio::ip::make_address("127.0.0.1"), 0}};
    mutable std::mutex m_mutex;
    std::vector<std::pair<std::string, std::string>> m_messages;
    std::jthread m_thread;
};

constexpr std::string_view kNoticeChildTest = "RealizationFailureChild.DISABLED_RunsAndPostsItsNotices";
constexpr std::string_view kPortVar = "TAOSIM_TEST_MA_PORT";
constexpr std::string_view kModeVar = "TAOSIM_TEST_MA_MODE";
constexpr std::string_view kIpcSuffixVar = "TAOSIM_IPC_SUFFIX";
// Longer than a notice carries, so the clipping shows.
const std::string kLongError = std::string{kInjected} + ": " + std::string(400, 'x');

// A run that posts its lifecycle notices to `recorder`, in a child process with private queue names,
// realization 1 throwing kLongError at the end of its step to 2 s. Its grace period outlasts the run,
// so it publishes no state and needs no validator beyond the recorder.
[[nodiscard]] ChildExit runPostingInChild(std::string_view name, std::string_view mode, const NoticeRecorder& recorder)
{
    const auto dir = fs::temp_directory_path() / fmt::format("taosim-test-ma-notice-{}-{}", ::getpid(), name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    const auto suffix = fmt::format("-test-ma-notice-{}-{}", ::getpid(), name);
    auto exit = runTestInChild(
        kNoticeChildTest,
        {{std::string{kIpcSuffixVar}, suffix},
         {std::string{kPortVar}, std::to_string(recorder.port())},
         {std::string{kModeVar}, std::string{mode}}},
        dir / "child.log");
    for (const auto* base : {"taosim-req", "taosim-res"}) {
        taosim::ipc::PosixMessageQueue::remove(fmt::format("/{}{}", base, suffix));
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    return exit;
}

[[nodiscard]] std::vector<uint64_t> failedIn(const fs::path& ckpt)
{
    const auto common = readCommon(ckpt);
    const auto list = taosim::serialization::msgpackFindMapObj(common.get(), "failedRealizations");
    if (!list) return {};
    return list->get().as<std::vector<uint64_t>>();
}

}  // namespace

//-------------------------------------------------------------------------

TEST(RealizationFailure, InContinueModeAFailedRealizationDoesNotStopTheOthersInItsCohort)
{
    const MultiAssetRun run{"cohort", {.realizations = 3, .threads = 1, .extraAttrs = std::string{kContinue}}};
    const auto orchestrator = run.build();
    failAt(*orchestrator, 1, 2 * kStep);

    EXPECT_NO_THROW(orchestrator->run());

    const auto t = clocks(*orchestrator);
    EXPECT_EQ(t[0], run.endTime());
    EXPECT_EQ(t[1], 2 * kStep) << "the failed realization stepped on";
    EXPECT_EQ(t[2], run.endTime());
}

TEST(RealizationFailure, TheFailureIsLoggedOnceAtTheThrowWithTheRealizationsIndex)
{
    const MultiAssetRun run{"logged", {.realizations = 3, .threads = 1}};
    const auto orchestrator = run.build();
    failAt(*orchestrator, 1, 2 * kStep);

    testing::internal::CaptureStdout();
    try {
        orchestrator->run();
    } catch (...) {
    }
    const auto out = testing::internal::GetCapturedStdout();

    EXPECT_EQ(occurrences(out, "REALIZATION FAILED"), 1u) << out;
    const auto at = out.find("REALIZATION FAILED");
    ASSERT_NE(at, std::string::npos) << out;
    const auto line = out.substr(at, out.find('\n', at) - at);
    EXPECT_TRUE(line.contains("realization 1 (bg-1)")) << line;
    EXPECT_TRUE(line.contains(kInjected)) << line;
}

// Each run here is the first of its own process, so that the runs differ only in the failure.
TEST(RealizationFailure, InContinueModeTheOthersStepExactlyAsInARunWithoutTheFailure)
{
    const auto reference = runInChild("reference", std::nullopt);
    ASSERT_EQ(reference.size(), 3u);
    ASSERT_TRUE(runInChild("reference-again", std::nullopt) == reference)
        << "two runs of one config differ, so a difference below would mean nothing";

    const auto secondFails = runInChild("second-fails", 1);
    ASSERT_EQ(secondFails.size(), 3u);
    EXPECT_TRUE(secondFails[0] == reference[0]) << "realization 0 stepped differently beside a failure";
    EXPECT_FALSE(secondFails[1] == reference[1]);
    EXPECT_TRUE(secondFails[2] == reference[2]) << "realization 2 stepped differently beside a failure";

    const auto firstFails = runInChild("first-fails", 0);
    ASSERT_EQ(firstFails.size(), 3u);
    EXPECT_FALSE(firstFails[0] == reference[0]);
    EXPECT_TRUE(firstFails[1] == reference[1]) << "realization 1 stepped differently beside a failure";
    EXPECT_TRUE(firstFails[2] == reference[2]) << "realization 2 stepped differently beside a failure";
}

// The child side of the test above, run only when re-executed by it.
TEST(RealizationFailureChild, DISABLED_RunsAndPacksEveryRealization)
{
    const char* fail = std::getenv(kFailVar.data());
    const char* out = std::getenv(kOutVar.data());
    if (fail == nullptr || out == nullptr) {
        GTEST_SKIP() << "runs only as the child of a RealizationFailure test";
    }
    // One name for every child, so that the runs differ only in the failure each one is given.
    const MultiAssetRun run{
        "child", {.realizations = 3, .threads = 1, .ckptIntervalInSteps = 0, .extraAttrs = std::string{kContinue}}};
    const auto orchestrator = run.build();
    if (*fail != '\0') failAt(*orchestrator, static_cast<uint32_t>(std::stoul(fail)), 2 * kStep);
    try {
        orchestrator->run();
    } catch (...) {
    }
    taosim::serialization::BinaryStream stream;
    msgpack::packer pk{stream};
    pk.pack(packedBlocks(*orchestrator));
    std::ofstream{out, std::ios::binary}.write(stream.data(), stream.size());
}

TEST(RealizationFailure, CheckpointsAreNamedByTheRealizationsStillSteppingWhenTheFirstFails)
{
    const MultiAssetRun run{
        "named", {.realizations = 3, .threads = 2, .ckptIntervalInSteps = 1, .extraAttrs = std::string{kContinue}}};
    const auto orchestrator = run.build();
    failAt(*orchestrator, 0, 1 * kStep);

    try {
        orchestrator->run();
    } catch (...) {
    }

    const auto times = checkpointTimes(run);
    ASSERT_FALSE(times.empty());
    EXPECT_EQ(times.back(), run.endTime()) << "the checkpoint name stopped with realization 0";
    const auto common = readCommon(run.latestCheckpoint());
    EXPECT_EQ(taosim::serialization::msgpackFindMap<uint64_t>(common.get(), "timestamp"), run.endTime());
}

TEST(RealizationFailure, TheGracePeriodEndsOnTheClockOfTheRealizationsStillStepping)
{
    const MultiAssetRun run{
        "grace", {.realizations = 3, .threads = 1, .ckptIntervalInSteps = 1, .gracePeriod = 3 * kStep,
                  .extraAttrs = std::string{kContinue}}};
    const auto orchestrator = run.build();
    failAt(*orchestrator, 0, 1 * kStep);

    try {
        orchestrator->run();
    } catch (...) {
    }

    EXPECT_FALSE(orchestrator->warmingUp()) << "the run is still in its grace period at its end";
    EXPECT_FALSE(checkpointTimes(run).empty()) << "no checkpoint was written after the grace period";
}

TEST(RealizationFailure, ACheckpointAfterTheFailureMarksTheRealizationAndDeclaresIt)
{
    const MultiAssetRun run{
        "marked", {.realizations = 3, .threads = 1, .ckptIntervalInSteps = 1, .extraAttrs = std::string{kContinue}}};
    const auto orchestrator = run.build();
    failAt(*orchestrator, 1, 3 * kStep);

    try {
        orchestrator->run();
    } catch (...) {
    }

    const auto latest = run.latestCheckpoint();
    EXPECT_EQ(failedIn(latest), std::vector<uint64_t>{1});
    const auto format = taosim::checkpoint::readFormat(readCommon(latest).get());
    EXPECT_EQ(format.capabilities, std::vector<std::string>{"failedRealizations"});
    const auto before = run.checkpoints().front();
    EXPECT_EQ(std::stoull(before.stem().string()), 2 * kStep);
    EXPECT_TRUE(failedIn(before).empty()) << "a checkpoint before the failure marks it";
    EXPECT_TRUE(taosim::checkpoint::readFormat(readCommon(before).get()).capabilities.empty());
}

TEST(RealizationFailure, AResumeOfAMarkedCheckpointKeepsTheRealizationStopped)
{
    const MultiAssetRun run{
        "resumed", {.realizations = 3, .threads = 1, .steps = 8, .ckptIntervalInSteps = 2,
                    .extraAttrs = std::string{kContinue}}};
    {
        const auto orchestrator = run.build();
        failAt(*orchestrator, 1, 3 * kStep);
        try {
            orchestrator->run();
        } catch (...) {
        }
    }
    fs::path marked;
    for (const auto& dir : run.checkpoints()) {
        if (std::stoull(dir.stem().string()) == 5 * kStep) marked = dir;
    }
    ASSERT_FALSE(marked.empty());
    ASSERT_EQ(failedIn(marked), std::vector<uint64_t>{1});

    const auto resumed = Orchestrator::fromCheckpoint(marked.string());
    const auto frozen = packedBlocks(*resumed)[1];
    EXPECT_NO_THROW(resumed->run());

    const auto t = clocks(*resumed);
    EXPECT_EQ(t[0], run.endTime());
    EXPECT_EQ(t[2], run.endTime());
    EXPECT_EQ(t[1], 5 * kStep) << "the failed realization stepped after the resume";
    EXPECT_TRUE(packedBlocks(*resumed)[1] == frozen) << "the failed realization's state changed after the resume";
    EXPECT_EQ(failedIn(run.latestCheckpoint()), std::vector<uint64_t>{1});
}

TEST(RealizationFailure, TheLastCheckpointBeforeTheFailureIsKeptPastTheRetentionBound)
{
    const MultiAssetRun run{
        "kept", {.realizations = 3, .threads = 1, .steps = 9, .ckptIntervalInSteps = 1, .ckptNumLastFilesToKeep = 2,
                 .extraAttrs = std::string{kContinue}}};
    const auto orchestrator = run.build();
    failAt(*orchestrator, 1, 4 * kStep);

    try {
        orchestrator->run();
    } catch (...) {
    }

    const auto times = checkpointTimes(run);
    ASSERT_EQ(times.size(), 3u) << fmt::format("{}", fmt::join(times, ", "));
    EXPECT_EQ(times[0], 3 * kStep) << "the last checkpoint before the failure was cleaned up";
    EXPECT_EQ(times[2], run.endTime());
    EXPECT_TRUE(failedIn(run.checkpoints().front()).empty());
}

TEST(RealizationFailure, AFailedRealizationStopsTheRunAtTheNextBarrierByDefault)
{
    const MultiAssetRun run{"default-stop", {.realizations = 3, .threads = 1, .ckptIntervalInSteps = 1}};
    const auto orchestrator = run.build();
    failAt(*orchestrator, 1, 3 * kStep);

    EXPECT_THROW(orchestrator->run(), std::runtime_error);

    EXPECT_EQ(clocks(*orchestrator), (std::vector<uint64_t>{3 * kStep, 3 * kStep, 3 * kStep}));
    EXPECT_EQ(checkpointTimes(run), std::vector<uint64_t>{2 * kStep}) << "a checkpoint was taken after the failure";
}

TEST(RealizationFailure, AStoppedRunKeepsItsLastCheckpointThroughTheCleanupOfARelaunch)
{
    const MultiAssetRun run{
        "default-kept", {.realizations = 3, .threads = 1, .steps = 9, .ckptIntervalInSteps = 1,
                         .ckptNumLastFilesToKeep = 2}};
    {
        const auto orchestrator = run.build();
        failAt(*orchestrator, 1, 4 * kStep);
        EXPECT_THROW(orchestrator->run(), std::runtime_error);
    }
    const auto kept = run.latestCheckpoint();
    ASSERT_EQ(std::stoull(kept.stem().string()), 3 * kStep);
    EXPECT_TRUE(fs::exists(kept / taosim::checkpoint::CheckpointManager::s_keepMarkerName))
        << "the last checkpoint before the failure carries no keep marker";

    const auto relaunched = Orchestrator::fromCheckpoint(kept.string());
    EXPECT_NO_THROW(relaunched->run());

    EXPECT_TRUE(fs::exists(kept)) << "the relaunch's cleanup removed the checkpoint it was relaunched from";
    const auto times = checkpointTimes(run);
    ASSERT_EQ(times.size(), 3u) << fmt::format("{}", fmt::join(times, ", "));
    EXPECT_EQ(times.front(), 3 * kStep);
    EXPECT_EQ(times.back(), run.endTime());
    for (const auto t : clocks(*relaunched)) EXPECT_EQ(t, run.endTime());
}

TEST(RealizationFailure, AStoppedRunPostsOneFailureNoticeAndNoEnd)
{
    const NoticeRecorder recorder;

    const auto exit = runPostingInChild("stop", "stop", recorder);

    EXPECT_FALSE(exit.clean()) << "a run stopped by a failure exited cleanly\n" << exit.log;
    EXPECT_FALSE(recorder.messages("EVENT_SIMULATION_START").empty()) << exit.log;
    EXPECT_TRUE(recorder.messages("EVENT_SIMULATION_END").empty())
        << "a failed run announced its end, so a validator would open the next run\n" << exit.log;
    const auto failures = recorder.messages("EVENT_SIMULATION_FAILED");
    ASSERT_EQ(failures.size(), 1u) << exit.log;

    rapidjson::Document notice;
    notice.Parse(failures.front().c_str());
    ASSERT_TRUE(notice.HasMember("payload") && notice["payload"].IsObject()) << failures.front();
    const auto& payload = notice["payload"];
    EXPECT_EQ(payload["realization"].GetUint(), 1u);
    EXPECT_STREQ(payload["background"].GetString(), "bg");
    EXPECT_EQ(payload["time"].GetUint64(), 2 * kStep);
    EXPECT_EQ(notice["timestamp"].GetUint64(), 2 * kStep);
    const std::string error = payload["error"].GetString();
    EXPECT_TRUE(error.starts_with(kInjected)) << error;
    EXPECT_LE(error.size(), 200u) << "the error text was not clipped";
    EXPECT_GE(error.size(), 150u) << error;
}

TEST(RealizationFailure, InContinueModeARunPostsItsEndAndNoFailureNotice)
{
    const NoticeRecorder recorder;

    const auto exit = runPostingInChild("continue", "continue", recorder);

    EXPECT_TRUE(exit.clean()) << exit.log;
    EXPECT_TRUE(recorder.messages("EVENT_SIMULATION_FAILED").empty()) << exit.log;
    EXPECT_EQ(recorder.messages("EVENT_SIMULATION_END").size(), 1u) << exit.log;
}

// The child side of the two tests above. It refuses to run without a private queue suffix, so it can
// never open another engine's queues.
TEST(RealizationFailureChild, DISABLED_RunsAndPostsItsNotices)
{
    const char* port = std::getenv(kPortVar.data());
    const char* mode = std::getenv(kModeVar.data());
    const char* suffix = std::getenv(kIpcSuffixVar.data());
    if (port == nullptr || mode == nullptr || suffix == nullptr || *suffix == '\0') {
        GTEST_SKIP() << "runs only as the child of a RealizationFailure notice test";
    }
    ASSERT_TRUE(taosim::simulation::SimulationManager::s_validatorReqMessageQueueName.ends_with(suffix));
    ASSERT_TRUE(taosim::simulation::SimulationManager::s_validatorResMessageQueueName.ends_with(suffix));
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    const MultiAssetRun run{
        "notice", {.realizations = 3, .threads = 1, .ckptIntervalInSteps = 0, .gracePeriod = UINT64_MAX,
                   .extraAttrs = fmt::format(
                       R"(useMessagePack="1" host="127.0.0.1" port="{}" generalMsgEndpoint="/account" {})", port,
                       std::string_view{mode} == "continue" ? kContinue : std::string_view{})}};
    const auto orchestrator = run.build();
    const auto* sim = orchestrator->checkpointBlocks()[1].get();
    sim->signals().step.connect([sim] {
        if (sim->currentTimestamp() >= 2 * kStep) throw std::runtime_error{kLongError};
    });
    orchestrator->run();
}

TEST(RealizationFailure, InContinueModeARunWithNoRealizationLeftStopsAtTheNextBarrier)
{
    const MultiAssetRun run{
        "none-left", {.realizations = 2, .threads = 1, .ckptIntervalInSteps = 1, .extraAttrs = std::string{kContinue}}};
    const auto orchestrator = run.build();
    failAt(*orchestrator, 0, 3 * kStep);
    failAt(*orchestrator, 1, 3 * kStep);

    EXPECT_THROW(orchestrator->run(), std::runtime_error);

    EXPECT_EQ(checkpointTimes(run), std::vector<uint64_t>{2 * kStep});
}

//-------------------------------------------------------------------------
