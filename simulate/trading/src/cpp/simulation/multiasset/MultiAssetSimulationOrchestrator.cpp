/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/simulation/multiasset/MultiAssetSimulationOrchestrator.hpp>

#include <MultiBookExchangeAgent.hpp>
#include <Simulation.hpp>
#include <taosim/checkpoint/CheckpointFormat.hpp>
#include <taosim/checkpoint/CheckpointManager.hpp>
#include <taosim/checkpoint/helpers.hpp>
#include <taosim/filesystem/utils.hpp>
#include <taosim/ipc/PosixMessageQueue.hpp>
#include <taosim/ipc/ipc.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>
#include <taosim/message/PayloadFactory.hpp>
#include <taosim/process/helpers.hpp>
#include <taosim/serialization/msgpack/common.hpp>
#include <taosim/serialization/msgpack/utils.hpp>
#include <taosim/simulation/LifecycleNotice.hpp>
#include <taosim/simulation/SharedResources.hpp>
#include <taosim/simulation/SimulationManager.hpp>
#include <taosim/simulation/multiasset/MultiAssetError.hpp>
#include <taosim/simulation/multiasset/serialization/MultiAssetValidatorRequest.hpp>
#include <taosim/simulation/util.hpp>
#include <taosim/util/SLTPDebug.hpp>
#include <taosim/xml/helpers.hpp>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <date/date.h>
#include <date/tz.h>
#include <chrono>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <msgpack.hpp>
#include <pugixml.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <bit>
#include <csignal>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// The stop signal only records the request (an atomic store; nothing else is async-signal-safe
// here). Until 30 September 2026 this orchestrator registered no handler at all, so SIGTERM ended
// the process on the default disposition with no checkpoint, and every restart of a multi-asset
// run was a crash-resume from the last periodic checkpoint. SimulationManager's handler sits in an
// anonymous namespace of its own translation unit; this one feeds the same flag through
// requestStop(), and run()'s barrier completion is the one place that acts on it.
extern "C" void onMultiAssetStopSignal(int) noexcept
{
    taosim::simulation::requestStop();
}

//-------------------------------------------------------------------------

namespace taosim::simulation::multiasset
{

//-------------------------------------------------------------------------

namespace
{

//-------------------------------------------------------------------------
// Resolve every referenced background config: each must exist and be a
// <Simulation> document. Returned index-aligned with `backgrounds`.

std::vector<pugi::xml_document> loadBackgroundDocs(const std::vector<BackgroundDesc>& backgrounds)
{
    auto loadAndValidate = [](const BackgroundDesc& bg) {
        std::string_view label = bg.name.empty() ? "(unnamed)" : bg.name.c_str();
        if (!std::filesystem::exists(bg.path)) {
            throw MultiAssetError{fmt::format(
                "background '{}' config not found: {}", label, bg.path.string())};
        }
        pugi::xml_document doc = xml::loadDocument(bg.path);
        if (!doc.child("Simulation")) {
            throw MultiAssetError{fmt::format(
                "background '{}' is not a <Simulation> config: {}", label, bg.path.string())};
        }
        // A BACKGROUND'S RUN-LEVEL SETTINGS DO NOT APPLY, and used to vanish without a word: these are
        // read from a single-market root, never from a background under the wrapper.
        const auto root = doc.child("Simulation");
        for (const auto* attr : {"traceTime", "measureStepWallClockTime", "sltpDebug", "bookStateEndpoint"}) {
            if (root.attribute(attr)) {
                fmt::println(" - background '{}' sets {}, which a multi-asset run reads from the wrapper only",
                    label, attr);
            }
        }
        return doc;
    };

    return backgrounds | ranges::views::transform(loadAndValidate) | ranges::to<std::vector>;
}

}  // namespace

//-------------------------------------------------------------------------
// Owned run state, hidden behind the PIMPL. The run machinery itself is filled
// in by the next step (see run()).

struct Impl
{
    MultiAssetConfig config;
    ResourceAllocation allocation;
    std::vector<pugi::xml_document> backgroundDocs;            // index-aligned with backgrounds
    std::vector<simulation::SharedResources> sharedResources;  // one per background
    std::vector<std::unique_ptr<Simulation>> simulations;      // the N realizations, flat-indexed
    std::vector<uint32_t> bookIdBases;                         // per realization: its first canonical book id
    std::vector<uint32_t> bookCounts;                          // per realization: its book count
    std::vector<std::string> backgroundNames;                  // per realization: its background's name
    // Validator IPC, populated only when useMessagePack: the request/response message
    // queues that pair with the state/responses shared-memory segments. Channel names
    // are shared with SimulationManager, so one validator serves either run mode.
    std::unique_ptr<ipc::PosixMessageQueue> validatorReqQueue;
    std::unique_ptr<ipc::PosixMessageQueue> validatorResQueue;
    // Checkpointing, populated only when ckptIntervalInSteps > 0. The pool is dedicated
    // (not the run pool, which is parked on the per-step barrier when a save is taken).
    std::unique_ptr<boost::asio::thread_pool> ckptPool;
    std::unique_ptr<checkpoint::CheckpointManager> checkpointManager;
    // Per realization, set when it throws: it is stepped no more and the run goes on without it. A
    // worker writes only its own cohort's entries; the others are read once every worker has met at
    // the barrier.
    std::vector<uint8_t> failed;
    // Whether the last checkpoint before the run's first failure has been kept (or the failures came
    // with the checkpoint this run resumed).
    bool failureHandled{};
};

//-------------------------------------------------------------------------

namespace
{

constexpr std::string_view kFailedRealizationsKey{"failedRealizations"};

// THE RUN'S CLOCK IS THE FIRST REALIZATION STILL STEPPING. All realizations share one grid, so while any
// is healthy this is the grid time; a failed realization's own clock stops where it failed.
const Simulation& clockOf(const Impl& impl)
{
    for (size_t idx = 0; idx < impl.simulations.size(); ++idx) {
        if (!impl.failed[idx]) return *impl.simulations[idx];
    }
    return *impl.simulations.front();
}

std::vector<uint32_t> failedRealizations(const Impl& impl)
{
    std::vector<uint32_t> out;
    for (uint32_t idx = 0; idx < impl.failed.size(); ++idx) {
        if (impl.failed[idx]) out.push_back(idx);
    }
    return out;
}

std::string describe(const std::exception_ptr& error)
{
    try {
        std::rethrow_exception(error);
    } catch (const std::exception& e) {
        return e.what();
    } catch (...) {
        return "an exception that is not a std::exception";
    }
}

}  // namespace

//-------------------------------------------------------------------------
// Single-threaded validator round-trip, driven from the per-step barrier completion
// (after every realization has stepped, before the per-step logs are cleared):
// publish the collective state, block on the validator's instructions, then route
// each one back to its realization. Mirrors SimulationManager's message-pack path;
// the realizations' canonical bases let decanonize() recover (realization, book) even
// though realizations hold different numbers of books.

// THE PER-STEP WALL-CLOCK BREAKDOWN, in SimulationManager's PROCESSED format and meaning: STATE joins the
// realizations' packed parts into the request (the workers packed them), PARSE waits for the validator and unpacks its answer, PROC dispatches it. Printed on every way
// out of publishState, with a phase the step never reached counted as nothing.
struct StepWallClock
{
    using clock = std::chrono::high_resolution_clock;
    bool on{};
    Timestamp stepNs{}, nowNs{};
    clock::time_point t0{clock::now()}, tState{}, tParse{};

    ~StepWallClock()
    {
        if (!on) return;
        using namespace std::chrono;
        const auto t1 = clock::now();
        if (tState == clock::time_point{}) tState = t1;
        if (tParse == clock::time_point{}) tParse = t1;
        fmt::println(
            "PROCESSED {:%T} - {:%T} ({:.4f}s | PARSE {:.4f}s | PROC {:.4f}s | STATE {:.4f}s)",
            duration_cast<seconds>(duration<Timestamp, std::nano>{nowNs - stepNs}),
            duration_cast<seconds>(duration<Timestamp, std::nano>{nowNs}),
            duration<double>(t1 - t0).count(), duration<double>(tParse - tState).count(),
            duration<double>(t1 - tParse).count(), duration<double>(tState - t0).count());
    }
};

void publishState(Impl& impl, std::span<const taosim::serialization::HumanReadableStream> packed)
{
    const auto& sims = impl.simulations;
    const auto& clock = clockOf(impl);
    const auto now = clock.currentTimestamp();
    StepWallClock wall{.on = impl.config.measureStepWallClockTime,
                       .stepNs = clock.time().step, .nowNs = clock.time().current};

    taosim::serialization::HumanReadableStream stream{1uz << 27};
    msgpack::pack(stream, serialization::MultiAssetValidatorRequest{
        .simulations = sims,
        .config = &impl.config,
        .logDir = impl.config.baseDir,
        .instances = packed
    });

    bipc::shared_memory_object shmReq{
        bipc::open_or_create, SimulationManager::s_statePublishShmName.data(), bipc::read_write};
    shmReq.truncate(stream.size());
    bipc::mapped_region reqRegion{shmReq, bipc::read_write};
    std::memcpy(reqRegion.get_address(), stream.data(), stream.size());
    wall.tState = StepWallClock::clock::now();

    // Both queues are drained before each send: a reply left in the response queue (by a validator
    // that answered a previous engine, or a late answer to a resent state) would otherwise be read
    // as the answer to this state, and every later reply would pair one state late.
    //
    // A STOP IS HONOURED WHILE NO VALIDATOR ANSWERS. A stop is acted on at the step's barrier, after
    // this wait, and a validator that has stopped never answers, so without this the engine waits
    // here until it is killed and writes no shutdown checkpoint. The step then goes unanswered.
    const auto resByteSize = ipc::sendAndAwaitReply(
        *impl.validatorReqQueue, *impl.validatorResQueue, stream.size(),
        [] { return stopRequested(); },
        [](std::string_view what) {
            fmt::println("{} /{} timed out, retrying...",
                what == "send" ? "Sending to" : "Receive from",
                what == "send" ? SimulationManager::s_validatorReqMessageQueueName
                               : SimulationManager::s_validatorResMessageQueueName);
        });
    if (!resByteSize) {
        fmt::println("Stop requested while waiting for the validator; this step goes unanswered");
        return;
    }

    bipc::shared_memory_object shmRes{
        bipc::open_only, SimulationManager::s_remoteResponsesShmName.data(), bipc::read_write};
    bipc::mapped_region resRegion{shmRes, bipc::read_write};

    msgpack::object_handle oh;
    try {
        oh = msgpack::unpack(std::bit_cast<const char*>(resRegion.get_address()), *resByteSize);
    } catch (const std::exception& e) {
        fmt::println("Error unpacking validator responses: {}", e.what());
        return;
    }
    const msgpack::object obj = oh.get();
    wall.tParse = StepWallClock::clock::now();

    // Source/target names are taken from the leading realization; the proxy and
    // exchange are named identically across realizations, so the routed realization
    // resolves them all the same.
    const auto& reprSimu = sims.front();
    const Timestamp maxInstructionDelay = reprSimu->config().maxInstructionDelay();
    auto unpackResponse = [&](const msgpack::object& o) -> Message::Ptr {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }
        std::optional<AgentId> agentId;
        std::optional<Timestamp> delay;
        std::string type;
        for (const auto& [k, val] : o.via.map) {
            const auto key = k.as<std::string_view>();
            if (key == "agentId") agentId = val.as<AgentId>();
            else if (key == "delay") delay = val.as<Timestamp>();
            else if (key == "type") type = val.as<std::string>();
        }
        if (!agentId || !delay || type.empty()) {
            throw taosim::serialization::MsgPackError{};
        }
        taosim::simulation::checkInstructionDelay(*delay, maxInstructionDelay);
        MessagePayload::Ptr payload;
        for (const auto& [k, val] : o.via.map) {
            if (k.as<std::string_view>() == "payload") {
                payload = PayloadFactory::createFromMessagePack(val, type);
                break;
            }
        }
        if (payload == nullptr) {
            throw taosim::serialization::MsgPackError{};
        }
        auto msg = std::make_shared<Message>();
        msg->occurrence = now;
        msg->arrival = now + *delay;
        // The leading realization may be a proxy-less (purely local) background;
        // any realization's proxy name works as the source label — they are named
        // identically — so take the first one that has one.
        msg->source = [&]() -> std::string {
            for (const auto& sim : sims) {
                if (sim->proxy() != nullptr) return sim->proxy()->name();
            }
            return "DISTRIBUTED_PROXY_AGENT";
        }();
        msg->targets = {reprSimu->exchange()->name()};
        msg->type = fmt::format("{}_{}", "DISTRIBUTED", type);
        msg->payload = MessagePayload::create<DistributedAgentResponsePayload>(*agentId, payload);
        return msg;
    };

    if (obj.type != msgpack::type::MAP) return;
    // Before the responses, so a round with none still updates the set.
    if (const auto capped = taosim::simulation::readCappedPairs(obj)) {
        const auto split = taosim::simulation::splitCappedPairs(*capped, impl.bookIdBases, impl.bookCounts);
        for (const auto& [simIdx, sim] : views::enumerate(sims)) {
            sim->exchange()->setCappedPairs(split[simIdx]);
        }
    }
    // The member named "responses", or the only member, as SimulationManager reads it.
    const msgpack::object* responsesObj{};
    for (const auto& [k, v] : obj.via.map) {
        if (obj.via.map.size == 1 || (k.type == msgpack::type::STR && k.as<std::string_view>() == "responses")) {
            responsesObj = &v;
        }
    }
    if (responsesObj == nullptr) return;
    if (responsesObj->type != msgpack::type::ARRAY || responsesObj->via.array.size == 0) return;

    std::vector<Message::Ptr> responses;
    size_t refusedForDelay{};
    for (const auto& response : responsesObj->via.array) {
        try {
            responses.push_back(unpackResponse(response));
        } catch (const taosim::simulation::InstructionDelayRefused&) {
            ++refusedForDelay;
        } catch (const std::exception& e) {
            fmt::println("Error unpacking a validator response: {}", e.what());
        }
    }
    if (refusedForDelay > 0) {
        fmt::println(
            "Refused {} validator instruction(s) delayed beyond maxInstructionDelay {}",
            refusedForDelay, maxInstructionDelay);
    }
    for (const auto& response : responses) {
        const auto [msg, simIdx, unrouted] = decanonize(response, impl.bookIdBases, impl.bookCounts);
        if (unrouted) {
            fmt::println("Dropping a validator instruction that names a book no realization owns");
            continue;
        }
        if (!simIdx) {
            for (const auto& sim : sims) sim->queueMessage(msg);
            continue;
        }
        sims.at(*simIdx)->queueMessage(msg);
    }
}

//-------------------------------------------------------------------------

MultiAssetSimulationOrchestrator::MultiAssetSimulationOrchestrator()
    : m_impl{std::make_unique<Impl>()}
{}

//-------------------------------------------------------------------------
// Out-of-line so Impl is complete at the point of destruction (PIMPL).

MultiAssetSimulationOrchestrator::~MultiAssetSimulationOrchestrator() = default;

//-------------------------------------------------------------------------

const MultiAssetConfig& MultiAssetSimulationOrchestrator::config() const noexcept
{
    return m_impl->config;
}

//-------------------------------------------------------------------------

const ResourceAllocation& MultiAssetSimulationOrchestrator::allocation() const noexcept
{
    return m_impl->allocation;
}

//-------------------------------------------------------------------------
// CheckpointSource. All realizations share one (wrapper-stamped) grid, so the run's clock
// dates a checkpoint and gates warmup.

bool MultiAssetSimulationOrchestrator::warmingUp() const
{
    return clockOf(*m_impl).currentTimestamp() < m_impl->config.gracePeriod;
}

//-------------------------------------------------------------------------

uint64_t MultiAssetSimulationOrchestrator::checkpointTimestamp() const
{
    return clockOf(*m_impl).currentTimestamp();
}

//-------------------------------------------------------------------------

std::span<const std::unique_ptr<Simulation>>
MultiAssetSimulationOrchestrator::checkpointBlocks() const
{
    return m_impl->simulations;
}

//-------------------------------------------------------------------------

boost::asio::thread_pool& MultiAssetSimulationOrchestrator::checkpointExecutor() const
{
    return *m_impl->ckptPool;
}

//-------------------------------------------------------------------------
// The "common" state: the shared grid timestamp plus, per realization, the sizes of
// its log files (each realization keeps its own subdirectory of the run root), and the
// realizations that have failed, when any has. The per-book block state is written
// generically by the CheckpointManager.

void MultiAssetSimulationOrchestrator::writeCommonCheckpoint(
    const std::filesystem::path& commonFile) const
{
    const auto& sims = m_impl->simulations;
    const auto failed = failedRealizations(*m_impl);

    taosim::serialization::BinaryStream stream;
    msgpack::packer packer{stream};

    packer.pack_map(2 + checkpoint::kFormatKeyCount + (failed.empty() ? 0 : 1));

    packer.pack(std::string{"timestamp"});
    packer.pack(clockOf(*m_impl).currentTimestamp());

    packer.pack(std::string{"logFileSizes"});
    packer.pack_array(static_cast<uint32_t>(sims.size()));
    for (const auto& sim : sims) {
        const auto files = filesystem::collectMatchingPaths(
            sim->logDir(),
            [](auto&& p) {
                return std::filesystem::is_regular_file(p)
                    && std::regex_match(
                        p.filename().string(),
                        checkpoint::CheckpointManager::s_relevantLogFilePattern);
            });
        packer.pack_map(static_cast<uint32_t>(files.size()));
        for (const auto& file : files) {
            packer.pack(file.filename().c_str());
            packer.pack(std::filesystem::file_size(file));
        }
    }

    static constexpr std::array<std::string_view, 1> markedFailures{checkpoint::kFailedRealizationsCapability};
    if (!failed.empty()) {
        packer.pack(kFailedRealizationsKey);
        packer.pack(failed);
    }
    checkpoint::packFormat(
        packer, failed.empty() ? std::span<const std::string_view>{} : std::span{markedFailures});

    checkpoint::atomicWrite(commonFile, {stream.data(), stream.size()});
}

//-------------------------------------------------------------------------

std::unique_ptr<MultiAssetSimulationOrchestrator> MultiAssetSimulationOrchestrator::fromConfig(
    const std::filesystem::path& configPath,
    const std::filesystem::path& baseDir,
    std::optional<uint64_t> resumeTimestamp)
{
    pugi::xml_document doc = xml::loadDocument(configPath);

    pugi::xml_node root = doc.child(kRootElement.data());
    if (!root) {
        throw MultiAssetError{fmt::format(
            "'{}' missing root element <{}>", configPath.c_str(), kRootElement)};
    }

    fmt::println(" - '{}' loaded successfully (multi-asset)", configPath.c_str());

    // Prototype pattern: build from XML, then inject the runtime-derived fields.
    // Private ctor (PIMPL factory) — make_unique can't reach it, so new directly.
    auto orchestrator =
        std::unique_ptr<MultiAssetSimulationOrchestrator>(new MultiAssetSimulationOrchestrator{});
    auto& impl = *orchestrator->m_impl;

    impl.config = MultiAssetConfig::fromXML(root, configPath.parent_path());
    impl.config.sourcePath = configPath;
    // SimulationManager sets this from its own root, a path a multi-asset run never takes.
    ::taosim::util::SLTPDebugger::enabled = impl.config.sltpDebug;

    // Each run gets its own directory under baseDir (baseDir/<id>), like the single-config
    // scheme: <id> is the wrapper's 'id' attribute or a generated timestamp (stamped back
    // and persisted), so a restore recreates the same directory. Every realization, the
    // checkpoint store, and the persisted config live under this run root.
    const std::string runId = [&] {
        const std::string specifiedId = root.attribute("id").as_string();
        if (!specifiedId.empty()) return specifiedId;
        using namespace std::chrono;
        const auto dateTimeId = date::format(
            "%Y%m%d_%H%M%S",
            date::make_zoned(date::current_zone(), time_point_cast<seconds>(system_clock::now())));
        xml::setAttribute(root, "id", dateTimeId.c_str());
        return dateTimeId;
    }();
    const auto runRoot = baseDir / runId;
    std::filesystem::create_directories(runRoot);
    impl.config.baseDir = runRoot;

    // Persist the wrapper config into the run root with the run id and ABSOLUTE background
    // paths, so the run is reproducible and restorable from its own directory alone.
    {
        uint32_t persistBgIdx = 0;
        for (pugi::xml_node bgNode : root.children("Background")) {
            bgNode.attribute("path").set_value(
                impl.config.backgrounds[persistBgIdx].path.string().c_str());
            ++persistBgIdx;
        }
    }
    doc.save_file((runRoot / "config.xml").c_str());

    // The orchestrator owns one global time grid; every realization is stamped with
    // it so all Simulations advance in lockstep (required for the per-step barrier).
    if (impl.config.step == 0 || impl.config.duration == 0) {
        throw MultiAssetError{fmt::format(
            "'{}' requires positive 'step' and 'duration' (the shared global time grid)",
            kRootElement)};
    }

    impl.backgroundDocs = loadBackgroundDocs(impl.config.backgrounds);

    const auto simulationCount = ranges::accumulate(
        impl.config.backgrounds | ranges::views::transform(&BackgroundDesc::instanceCount), 0u);

    impl.allocation = computeResourceAllocation(ResourceAllocationDesc{
        .simulationCount = simulationCount,
        .requestedWorkers = impl.config.requestedWorkers,
        .hardwareConcurrency = std::thread::hardware_concurrency(),
        .checkpointWorkers = 0u  // TODO: surface from config alongside the other ckpt knobs.
    });

    // The wrapper governs the time grid: stamp it onto every background before any
    // duration is read, so the covariance factor L is sized for the actual run
    // length (n ~ duration/updatePeriod) — not the background's own duration — and
    // each realization (an adjusted copy) inherits the grid. L is shared across a
    // background's instances (it depends only on its FundamentalPrice params).
    impl.sharedResources.resize(impl.config.backgrounds.size());
    for (auto&& [bgIdx, bgDoc] : ranges::views::enumerate(impl.backgroundDocs)) {
        const auto simu = bgDoc.child("Simulation");
        xml::setAttribute(simu, "start", impl.config.start);
        xml::setAttribute(simu, "step", impl.config.step);
        xml::setAttribute(simu, "duration", impl.config.duration);
        // On resume, continue each realization from the checkpoint instant.
        if (resumeTimestamp) {
            xml::setAttribute(simu, "current", *resumeTimestamp);
        }
        process::helpers::initSharedResources(impl.sharedResources[bgIdx], simu);
    }

    // One Simulation per realization: the background config adjusted for this
    // instance (id, rngSeed when the wrapper declares a seed), flat-
    // indexed across all backgrounds. Each realization keeps its artifacts in its own
    // subdirectory of the run root (<bgName>-<instance>), beside the run-wide ckpt
    // store and persisted config.
    // Book ids run consecutively across realizations (each numbered from the sum of the
    // book counts before it), so the canonical ids in file names are unique run-wide.
    impl.simulations.reserve(impl.allocation.simulationCount);
    uint32_t flatIdx = 0;
    for (const auto& [bgIdx, bg] : ranges::views::enumerate(impl.config.backgrounds)) {
        const auto simuNode = impl.backgroundDocs[bgIdx].child("Simulation");
        for (uint32_t instance = 0; instance < bg.instanceCount; ++instance) {
            // An unnamed background is `bg<index>`, the name the validator model and the
            // acceptance layer already derive (models.py from_multiasset_xml, ma_layer.py);
            // plain "bg" made two unnamed backgrounds write into one directory.
            const auto id = fmt::format(
                "{}-{}", bg.name.empty() ? fmt::format("bg{}", bgIdx) : bg.name, instance);
            // Derived only under a wrapper seed; otherwise the background's own rngSeed
            // (or device seeding) stands, as in a single-config run.
            const auto seed = impl.config.masterSeed.transform([&](uint64_t master) {
                return realizationSeed(master, static_cast<uint32_t>(bgIdx), instance);
            });
            // simuNode already carries the wrapper grid (stamped above); the adjusted
            // copy inherits it and adds only the per-realization id/rngSeed.
            const auto adjusted = adjustedBackgroundDoc(simuNode, seed, id);
            const auto logDir = runRoot / id;
            std::filesystem::create_directories(logDir);
            const auto bookCount = simuNode.child("Agents").child("MultiBookExchangeAgent")
                .child("Books").attribute("instanceCount").as_uint(1);
            const uint32_t base = impl.bookIdBases.empty()
                ? 0u : impl.bookIdBases.back() + impl.bookCounts.back();
            impl.bookIdBases.push_back(base);
            impl.bookCounts.push_back(bookCount);
            impl.backgroundNames.push_back(bg.name.empty() ? fmt::format("bg{}", bgIdx) : bg.name);
            auto sim = std::make_unique<Simulation>(
                flatIdx, bookCount, logDir, &impl.sharedResources[bgIdx]);
            sim->setBookIdBase(base);
            sim->setRunDir(runRoot);
            sim->configure(adjusted.child("Simulation"));
            impl.simulations.push_back(std::move(sim));
            ++flatIdx;
        }
    }
    impl.failed.assign(impl.simulations.size(), 0);

    // Validator IPC: only when publishing is requested. Channel names are shared with
    // SimulationManager so one validator process serves either run mode (one at a time).
    if (impl.config.useMessagePack) {
        impl.validatorReqQueue = std::make_unique<ipc::PosixMessageQueue>(
            ipc::PosixMessageQueueDesc{
                .name = SimulationManager::s_validatorReqMessageQueueName.data()});
        impl.validatorResQueue = std::make_unique<ipc::PosixMessageQueue>(
            ipc::PosixMessageQueueDesc{
                .name = SimulationManager::s_validatorResMessageQueueName.data()});
        // The queues outlive a killed engine: drop any state or reply it left behind.
        impl.validatorReqQueue->flush();
        impl.validatorResQueue->flush();
    }

    // Checkpointing: a dedicated write pool (the run pool is parked on the barrier while a
    // save runs) and the store manager, writing into the run root (-> runRoot/ckpt).
    if (impl.config.ckptIntervalInSteps > 0) {
        impl.ckptPool = std::make_unique<boost::asio::thread_pool>(impl.config.ckptNumWorkers);
        impl.checkpointManager = std::make_unique<checkpoint::CheckpointManager>(
            checkpoint::CheckpointingDesc{
                .source = orchestrator.get(),
                .runDir = runRoot,
                .intervalInSteps = impl.config.ckptIntervalInSteps,
                .numLastFilesToKeep = static_cast<ssize_t>(impl.config.ckptNumLastFilesToKeep),
                .measureWallClockTime = impl.config.ckptMeasureWallClockTime
            });
    }

    return orchestrator;
}

//-------------------------------------------------------------------------
// Restore: rebuild every realization from the persisted wrapper config (stamped to
// resume at the checkpoint instant), then apply the per-realization block state. The
// block restore re-establishes a shared-quote topology only where a checkpoint written under the
// old stamp has one (reestablishSharedTopology); a per-book run has none.

std::unique_ptr<MultiAssetSimulationOrchestrator> MultiAssetSimulationOrchestrator::fromCheckpoint(
    const checkpoint::CheckpointToken& ckptToken)
{
    const std::filesystem::path runDir = checkpoint::runDirFromToken(ckptToken);
    const std::filesystem::path ckptDir = checkpoint::ckptDirFromToken(ckptToken);
    const std::filesystem::path configPath = runDir / "config.xml";

    fmt::println("Loading multi-asset checkpoint {}...", ckptDir.c_str());

    auto loaded = checkpoint::loadCheckpointObjects(ckptDir);
    const msgpack::object commonObj = loaded.common.get();

    const auto ckptTimestamp =
        taosim::serialization::msgpackFindMap<uint64_t>(commonObj, "timestamp");
    if (!ckptTimestamp) {
        throw MultiAssetError{"multi-asset common checkpoint missing 'timestamp'"};
    }

    fmt::println("Checkpoint loaded successfully; initializing realizations...");

    // Before fromConfig: the realizations' loggers open their current files as they are built.
    checkpoint::setAsideLogsWrittenAfter(runDir, commonObj, *ckptTimestamp);

    // runDir is the run root (baseDir/<id>); pass its parent so fromConfig rebuilds the
    // SAME run root from the persisted 'id' (matching the single-config restore).
    auto orchestrator = fromConfig(configPath, runDir.parent_path(), *ckptTimestamp);

    fmt::println("Setting up realization state according to the checkpoint...");

    auto& impl = *orchestrator->m_impl;
    checkpoint::setupUsingCkptDataMultiAsset(
        impl.simulations, impl.checkpointManager.get(), commonObj, loaded.blocks);

    // A realization that failed before the checkpoint stays stopped: its state is what it held when it
    // threw, published as it was, never stepped again.
    if (const auto failedObj = taosim::serialization::msgpackFindMapObj(commonObj, kFailedRealizationsKey)) {
        for (const auto idx : failedObj->get().as<std::vector<uint32_t>>()) {
            if (idx >= impl.simulations.size()) {
                throw MultiAssetError{fmt::format(
                    "the checkpoint marks realization {} failed, but the run has {}", idx, impl.simulations.size())};
            }
            impl.failed[idx] = 1;
            fmt::println("Realization {} ({}) failed before this checkpoint and stays stopped",
                idx, impl.simulations[idx]->logDir().filename().c_str());
        }
        impl.failureHandled = true;
    }

    fmt::println("Load from checkpoint successful.");

    return orchestrator;
}

//-------------------------------------------------------------------------

namespace
{

//-------------------------------------------------------------------------
// Validator lifecycle notices, mirroring SimulationManager::publishStartInfo /
// publishEndInfo with the one difference the validator asks for: ONE
// EVENT_SIMULATION_START PER REALIZATION, each naming that realization's log
// directory, all at the shared grid start, so the validator's on_start absorbs them
// as one episode and normalizes the directory to the run root. Sent only when the
// run publishes (useMessagePack) AND the wrapper names a validator (host/port): the
// orchestrator has no JSON state path, so a START without ticks would leave the
// validator waiting for an episode that never arrives. Offline runs send nothing.
// START is suppressed on a resumed run (state != INACTIVE), as in the single-config
// path; END is sent only by a run that ran its grid to the end without a stop, so
// neither a failed run nor a stopped one announces itself as finished.

[[nodiscard]] bool publishesLifecycle(const Impl& impl) noexcept
{
    const auto& net = impl.config.netInfo;
    return impl.config.useMessagePack && !net.host.empty() && !net.port.empty();
}

void publishStartInfo(const Impl& impl)
{
    const auto& sims = impl.simulations;
    if (!publishesLifecycle(impl) || sims.front()->state() != SimulationState::INACTIVE) {
        return;
    }
    const auto notices = sims
        | ranges::views::transform([](const auto& sim) {
            return makeStartNotice(sim->time().start, sim->logDir());
        })
        | ranges::to<std::vector>;
    postGeneralMessage(
        *sims.front(), impl.config.netInfo, impl.config.generalMsgEndpoint,
        makeNoticeBatch(notices));
}

// Kept once, at the first failure: the newest checkpoint then is the last that holds every
// realization healthy, so a resume from it brings the failed one back.
void keepTheLastCheckpointBeforeTheFailure(Impl& impl) noexcept
{
    if (!impl.checkpointManager) return;
    try {
        if (const auto kept = impl.checkpointManager->keepLatest()) {
            fmt::println("Keeping checkpoint {}, the last before the failure, past the retention bound",
                kept->c_str());
        }
    } catch (const std::exception& e) {
        fmt::println("Could not keep the last checkpoint before the failure: {}", e.what());
    }
}

// The one notice a run stopped by a failed realization sends, before it exits. Sent once and not
// retried: an engine waiting on a validator that has gone would never exit.
struct RealizationFailure
{
    uint32_t realization{};
    uint64_t time{};
    std::string error;
};

void publishFailureInfo(const Impl& impl, const RealizationFailure& failure) noexcept
{
    if (!publishesLifecycle(impl)) return;
    try {
        const std::vector notices{makeFailureNotice(
            failure.realization, impl.backgroundNames.at(failure.realization), failure.time, failure.error)};
        postGeneralMessage(
            *impl.simulations.front(), impl.config.netInfo, impl.config.generalMsgEndpoint,
            makeNoticeBatch(notices), true);
    } catch (const std::exception& e) {
        fmt::println("Could not send the failure notice: {}", e.what());
    }
}

void publishEndInfo(const Impl& impl)
{
    if (!publishesLifecycle(impl)) return;
    const auto& sims = impl.simulations;
    const std::vector notices{makeEndNotice(sims.front()->time().start)};
    postGeneralMessage(
        *sims.front(), impl.config.netInfo, impl.config.generalMsgEndpoint,
        makeNoticeBatch(notices));
}

}  // namespace

//-------------------------------------------------------------------------

void MultiAssetSimulationOrchestrator::run()
{
    const MultiAssetConfig& cfg = m_impl->config;
    const ResourceAllocation& alloc = m_impl->allocation;

    fmt::println("Multi-asset run plan:");
    fmt::println("  source        : {}", cfg.sourcePath.c_str());
    fmt::println("  artifacts dir : {}", cfg.baseDir.c_str());
    fmt::println("  backgrounds   : {}", cfg.backgrounds.size());
    for (const auto& bg : cfg.backgrounds) {
        fmt::println("    - {:<16} x{:<4} {}",
            bg.name.empty() ? "(unnamed)" : bg.name, bg.instanceCount, bg.path.c_str());
    }
    fmt::println("  simulations   : {} (sum of instanceCount)", alloc.simulationCount);
    fmt::println("  workers       : {} (thread pool {})", alloc.workerCount, alloc.threadPoolSize);
    fmt::println("  cohort sizes  : [{}]", fmt::join(alloc.cohortSizes, ", "));
    if (cfg.masterSeed) {
        fmt::println("  master seed   : {}", *cfg.masterSeed);
    } else {
        fmt::println("  master seed   : unset (backgrounds keep their own rngSeed)");
    }
    fmt::println("  constructed   : {} Simulation(s)", m_impl->simulations.size());

    auto& sims = m_impl->simulations;
    if (sims.empty()) return;

    // All realizations share one (wrapper-stamped) time grid, so the number of global
    // steps is derived once here (every worker must arrive at the barrier the same number
    // of times). It's the steps REMAINING to the grid end (start + duration): on a fresh
    // run the realizations sit at `start`, giving duration/step; on a restored run they
    // sit at the checkpoint time, so we run only what's left. Running a full duration/step
    // from a resumed offset would overrun the grid and index the fundamental-price L
    // matrix (sized for the grid) out of bounds.
    const auto endTime = cfg.start + cfg.duration;
    const auto currentTime = sims.front()->currentTimestamp();
    const auto stepCount = endTime > currentTime
        ? (endTime - currentTime + cfg.step - 1) / cfg.step
        : uint64_t{};

    // Each worker thread drives a contiguous cohort of Simulations: per global step
    // it steps every Simulation in its cohort in order and purges its filled orders, then
    // meets the others at the barrier (sized to the worker count). The completion runs
    // single-threaded once all have stepped, in the single-config manager's order: publish
    // the collective state to the validator (if enabled, past the grace period), clear
    // the per-step L3 log that publishing just consumed, then take a checkpoint (last, so
    // it captures the cleared step boundary — its writes run on the dedicated ckpt pool
    // while the workers are parked here). publish/checkpoint failures are logged, never
    // thrown — the completion must stay noexcept.
    auto& impl = *m_impl;
    // The start notices go out before any realization starts, as in the single-config
    // path; the validator then sees START, the grace period of silence, and the ticks.
    publishStartInfo(impl);
    const bool publish = cfg.useMessagePack;
    std::signal(SIGINT, onMultiAssetStopSignal);
    std::signal(SIGTERM, onMultiAssetStopSignal);
    std::atomic_bool leaving{false};
    bool stoppedOnFailure{false};
    // Each realization's part of the published state, packed by its own worker once it has stepped, so the
    // completion only joins them instead of packing every realization on one thread while the others wait.
    std::vector<taosim::serialization::HumanReadableStream> packed(sims.size());
    // A part that failed to pack skips that step's publish, as a failure inside publishState always has.
    std::atomic_bool packFailed{false};
    boost::asio::thread_pool pool{alloc.workerCount};
    std::barrier barrier{
        static_cast<std::ptrdiff_t>(alloc.workerCount),
        [&sims, &impl, &leaving, &stoppedOnFailure, &packed, &packFailed, publish]() noexcept {
            // A failed realization stops the run here, with nothing published or checkpointed for this
            // step, unless the run asked to go on without it and some realization is still stepping.
            if (const auto failures = static_cast<size_t>(std::ranges::count(impl.failed, uint8_t{1}))) {
                if (!impl.failureHandled) {
                    impl.failureHandled = true;
                    keepTheLastCheckpointBeforeTheFailure(impl);
                }
                if (impl.config.stopOnRealizationFailure || failures == sims.size()) {
                    fmt::println("Stopping the run at this step: {} of {} realization(s) failed",
                        failures, sims.size());
                    stoppedOnFailure = true;
                    leaving.store(true, std::memory_order_release);
                    return;
                }
            }
            // Ghosts BEFORE publishing. Filled orders stay in the book as zero-volume
            // ghosts for the exchange service's on-chain reconciliation; nothing in a
            // simulation run reads them, and publishing them left a consumed best level
            // at the head of the book, so the validator's midquote (and every miner's view
            // of the top levels) differed from the single-config run on the same events.
            // The L3 record IS what publishing consumes, so that is cleared after. Each worker clears its own
            // realizations before it arrives (below), in parallel, instead of this completion clearing all of
            // them on one thread while the others wait.
            if (impl.config.traceTime) {
                const auto total = clockOf(impl).time().current;
                const auto secs = total / 1'000'000'000;
                fmt::println("TIME : {:02d}:{:02d}:{:02d}.{:09d}",
                    secs / 3600, (secs / 60) % 60, secs % 60, total % 1'000'000'000);
            }
            if (publish && clockOf(impl).currentTimestamp() >= impl.config.gracePeriod) {
                // A failed realization no longer steps, so its worker packs no part for it: pack it here, its
                // ghosts cleared first, as this completion packed and cleared every realization before the
                // workers took that over.
                for (size_t i = 0; i < sims.size(); ++i) {
                    if (!impl.failed[i]) continue;
                    try {
                        sims[i]->clearFilledOrders();
                        packed[i] = taosim::serialization::HumanReadableStream{1uz << 22};
                        msgpack::pack(packed[i], taosim::simulation::serialization::ValidatorRequest{
                            .simulations = std::span{sims}.subspan(i, 1),
                            .logDir = sims[i]->logDir()
                        });
                    } catch (const std::exception& e) {
                        fmt::println("publishState error (continuing): {}", e.what());
                        packFailed.store(true, std::memory_order_relaxed);
                    }
                }
            }
            if (publish && clockOf(impl).currentTimestamp() >= impl.config.gracePeriod
                && !packFailed.exchange(false)) {
                try {
                    publishState(impl, packed);
                } catch (const std::exception& e) {
                    fmt::println("publishState error (continuing): {}", e.what());
                }
            }
            for (const auto& sim : sims) {
                sim->exchange()->L3Record().clear();
            }
            if (impl.checkpointManager) {
                impl.checkpointManager->saveCheckpoint();
            }
            // THE ONE POINT WHERE A STOP IS CONSISTENT, as in SimulationManager::runSimulations:
            // every realization has stepped and none has resumed. Deciding and checkpointing here
            // makes "write the checkpoint" and "leave" a single observation of the request; the
            // workers read the flag once released and leave together.
            if (stopRequested()) {
                leaving.store(true, std::memory_order_release);
                if (impl.checkpointManager) {
                    impl.checkpointManager->saveCheckpointOnShutdown();
                }
            }
        }};

    // A REALIZATION THAT THROWS IS STOPPED ON ITS OWN, logged here, once, with its index. Its worker
    // finishes the step for the rest of its cohort and the barrier completion decides whether the run
    // goes on. The first such failure is the one a stopped run reports and rethrows after the join.
    std::mutex errorMutex;
    std::exception_ptr firstError;
    std::optional<RealizationFailure> firstFailure;
    auto isolate = [&](uint32_t idx, auto&& action) {
        if (impl.failed[idx]) return;
        try {
            action(*sims[idx]);
        } catch (...) {
            impl.failed[idx] = 1;
            const auto error = std::current_exception();
            const auto what = describe(error);
            const std::scoped_lock lock{errorMutex};
            if (!firstError) {
                firstError = error;
                firstFailure = RealizationFailure{
                    .realization = idx, .time = sims[idx]->currentTimestamp(), .error = what};
            }
            fmt::println(
                "REALIZATION FAILED: realization {} ({}) threw at {}: {}; {}",
                idx, sims[idx]->logDir().filename().c_str(), sims[idx]->currentTimestamp(), what,
                cfg.stopOnRealizationFailure ? "the run stops at the end of this step"
                                             : "it stops here and the other realizations keep stepping");
        }
    };

    // An exception outside any realization's step still drops its worker out of the barrier, so the
    // survivors are not stranded, and is rethrown after a clean join.
    std::exception_ptr workerError;

    uint32_t cohortBegin = 0;
    for (uint32_t w = 0; w < alloc.workerCount; ++w) {
        const auto begin = cohortBegin;
        const auto size = alloc.cohortSizes[w];
        cohortBegin += size;
        // begin/size are loop-locals: they MUST be captured by value (each worker
        // snapshots its own cohort range) — a by-reference '[&]' capture dangles.
        boost::asio::post(
            pool,
            [&, begin, size, stepCount] {
                try {
                    const auto end = begin + size;
                    for (uint32_t idx = begin; idx < end; ++idx) {
                        isolate(idx, [](Simulation& sim) {
                            if (sim.state() == taosim::simulation::SimulationState::INACTIVE) {
                                sim.start();
                            }
                        });
                    }
                    for (uint64_t s = 0; s < stepCount; ++s) {
                        for (uint32_t idx = begin; idx < end; ++idx) {
                            isolate(idx, [](Simulation& sim) {
                                sim.step();
                                sim.clearFilledOrders();  // before the barrier, so before publishing
                            });
                            // The completion publishes on the same condition; the grid is shared. A failed
                            // realization's part is packed by the completion.
                            const auto& sim = sims[idx];
                            if (!impl.failed[idx] && publish && sim->currentTimestamp() >= impl.config.gracePeriod) {
                                // Caught here, not by isolate: a part that cannot be packed skips this step's
                                // publish and does not fail the realization.
                                try {
                                    packed[idx] = taosim::serialization::HumanReadableStream{1uz << 22};
                                    msgpack::pack(packed[idx], taosim::simulation::serialization::ValidatorRequest{
                                        .simulations = std::span{sims}.subspan(idx, 1),
                                        .logDir = sim->logDir()
                                    });
                                } catch (const std::exception& e) {
                                    fmt::println("publishState error (continuing): {}", e.what());
                                    packFailed.store(true, std::memory_order_relaxed);
                                }
                            }
                        }
                        barrier.arrive_and_wait();
                        if (leaving.load(std::memory_order_acquire)) break;
                    }
                    for (uint32_t idx = begin; idx < end; ++idx) {
                        isolate(idx, [](Simulation& sim) { sim.stop(); });
                    }
                } catch (...) {
                    {
                        const std::scoped_lock lock{errorMutex};
                        if (!workerError) workerError = std::current_exception();
                    }
                    barrier.arrive_and_drop();
                }
            });
    }
    pool.join();

    if (workerError) std::rethrow_exception(workerError);

    if (stoppedOnFailure) {
        if (firstFailure) publishFailureInfo(impl, *firstFailure);
        fmt::println("Stopped after a realization failed; no end notice published");
        if (firstError) std::rethrow_exception(firstError);
        throw MultiAssetError{"no realization is left stepping"};
    }
    if (leaving.load(std::memory_order_acquire)) {
        fmt::println("Stopped on request; no end notice published");
        return;
    }
    if (const auto failed = failedRealizations(impl); !failed.empty()) {
        fmt::println("The run reached its end with {} failed realization(s): {}", failed.size(), fmt::join(failed, ", "));
    }
    publishEndInfo(impl);
    fmt::println("Multi-asset run complete.");
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation::multiasset

//-------------------------------------------------------------------------
