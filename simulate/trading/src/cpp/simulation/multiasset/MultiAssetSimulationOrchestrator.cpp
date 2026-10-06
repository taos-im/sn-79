/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/simulation/multiasset/MultiAssetSimulationOrchestrator.hpp>

#include <MultiBookExchangeAgent.hpp>
#include <Simulation.hpp>
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
#include <taosim/xml/helpers.hpp>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <date/date.h>
#include <date/tz.h>
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
    // Validator IPC, populated only when useMessagePack: the request/response message
    // queues that pair with the state/responses shared-memory segments. Channel names
    // are shared with SimulationManager, so one validator serves either run mode.
    std::unique_ptr<ipc::PosixMessageQueue> validatorReqQueue;
    std::unique_ptr<ipc::PosixMessageQueue> validatorResQueue;
    // Checkpointing, populated only when ckptIntervalInSteps > 0. The pool is dedicated
    // (not the run pool, which is parked on the per-step barrier when a save is taken).
    std::unique_ptr<boost::asio::thread_pool> ckptPool;
    std::unique_ptr<checkpoint::CheckpointManager> checkpointManager;
};

//-------------------------------------------------------------------------
// Single-threaded validator round-trip, driven from the per-step barrier completion
// (after every realization has stepped, before the per-step logs are cleared):
// publish the collective state, block on the validator's instructions, then route
// each one back to its realization. Mirrors SimulationManager's message-pack path;
// the realizations' canonical bases let decanonize() recover (realization, book) even
// though realizations hold different numbers of books.

void publishState(Impl& impl)
{
    const auto& sims = impl.simulations;
    const auto now = sims.front()->currentTimestamp();

    taosim::serialization::HumanReadableStream stream{1uz << 27};
    msgpack::pack(stream, serialization::MultiAssetValidatorRequest{
        .simulations = sims,
        .config = &impl.config,
        .logDir = impl.config.baseDir
    });

    bipc::shared_memory_object shmReq{
        bipc::open_or_create, SimulationManager::s_statePublishShmName.data(), bipc::read_write};
    shmReq.truncate(stream.size());
    bipc::mapped_region reqRegion{shmReq, bipc::read_write};
    std::memcpy(reqRegion.get_address(), stream.data(), stream.size());

    retry:
    const size_t packedSize = stream.size();
    impl.validatorReqQueue->flush();
    if (!impl.validatorReqQueue->send(
            std::span<const char>{std::bit_cast<const char*>(&packedSize), sizeof(packedSize)})) {
        fmt::println("Sending to /{} timed out, retrying...",
            SimulationManager::s_validatorReqMessageQueueName);
        goto retry;
    }
    size_t resByteSize;
    if (impl.validatorResQueue->receive(
            std::span<char>{std::bit_cast<char*>(&resByteSize), sizeof(resByteSize)}) == -1) {
        fmt::println("Receive from /{} timed out, retrying...",
            SimulationManager::s_validatorResMessageQueueName);
        goto retry;
    }

    bipc::shared_memory_object shmRes{
        bipc::open_only, SimulationManager::s_remoteResponsesShmName.data(), bipc::read_write};
    bipc::mapped_region resRegion{shmRes, bipc::read_write};

    msgpack::object_handle oh;
    try {
        oh = msgpack::unpack(std::bit_cast<const char*>(resRegion.get_address()), resByteSize);
    } catch (const std::exception& e) {
        fmt::println("Error unpacking validator responses: {}", e.what());
        return;
    }
    const msgpack::object obj = oh.get();

    // Source/target names are taken from the leading realization; the proxy and
    // exchange are named identically across realizations, so the routed realization
    // resolves them all the same.
    const auto& reprSimu = sims.front();
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

    if (obj.type != msgpack::type::MAP || obj.via.map.size != 1) return;
    const auto& responsesObj = obj.via.map.ptr[0].val;
    if (responsesObj.type != msgpack::type::ARRAY || responsesObj.via.array.size == 0) return;

    std::vector<Message::Ptr> responses;
    for (const auto& response : responsesObj.via.array) {
        try {
            responses.push_back(unpackResponse(response));
        } catch (const std::exception& e) {
            fmt::println("Error unpacking a validator response: {}", e.what());
        }
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
// CheckpointSource. All realizations share one (wrapper-stamped) grid, so the first
// dates a checkpoint and gates warmup.

bool MultiAssetSimulationOrchestrator::warmingUp() const
{
    return m_impl->simulations.front()->currentTimestamp() < m_impl->config.gracePeriod;
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
// its log files (each realization keeps its own subdirectory of the run root). The
// per-book block state is written generically by the CheckpointManager.

void MultiAssetSimulationOrchestrator::writeCommonCheckpoint(
    const std::filesystem::path& commonFile) const
{
    const auto& sims = m_impl->simulations;

    taosim::serialization::BinaryStream stream;
    msgpack::packer packer{stream};

    packer.pack_map(2);

    packer.pack(std::string{"timestamp"});
    packer.pack(sims.front()->currentTimestamp());

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
            auto sim = std::make_unique<Simulation>(
                flatIdx, bookCount, logDir, &impl.sharedResources[bgIdx]);
            sim->setBookIdBase(base);
            sim->configure(adjusted.child("Simulation"));
            impl.simulations.push_back(std::move(sim));
            ++flatIdx;
        }
    }

    // Validator IPC: only when publishing is requested. Channel names are shared with
    // SimulationManager so one validator process serves either run mode (one at a time).
    if (impl.config.useMessagePack) {
        impl.validatorReqQueue = std::make_unique<ipc::PosixMessageQueue>(
            ipc::PosixMessageQueueDesc{
                .name = SimulationManager::s_validatorReqMessageQueueName.data()});
        impl.validatorResQueue = std::make_unique<ipc::PosixMessageQueue>(
            ipc::PosixMessageQueueDesc{
                .name = SimulationManager::s_validatorResMessageQueueName.data()});
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

    // runDir is the run root (baseDir/<id>); pass its parent so fromConfig rebuilds the
    // SAME run root from the persisted 'id' (matching the single-config restore).
    auto orchestrator = fromConfig(configPath, runDir.parent_path(), *ckptTimestamp);

    fmt::println("Setting up realization state according to the checkpoint...");

    auto& impl = *orchestrator->m_impl;
    checkpoint::setupUsingCkptDataMultiAsset(
        impl.simulations, impl.checkpointManager.get(), commonObj, loaded.blocks);

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
// path; END is sent only after a clean run, so a failed one never announces itself
// as finished (the single-config manager sends it unconditionally).

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
    // it steps every Simulation in its cohort in order, then meets the others at the
    // barrier (sized to the worker count). The completion runs single-threaded once all
    // have stepped, in the single-config manager's order: purge the filled orders, publish
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
    boost::asio::thread_pool pool{alloc.workerCount};
    std::barrier barrier{
        static_cast<std::ptrdiff_t>(alloc.workerCount),
        [&sims, &impl, &leaving, publish]() noexcept {
            // Ghosts BEFORE publishing. Filled orders stay in the book as zero-volume
            // ghosts for the exchange service's on-chain reconciliation; nothing in a
            // simulation run reads them, and publishing them left a consumed best level
            // at the head of the book, so the validator's midquote (and every miner's view
            // of the top levels) differed from the single-config run on the same events.
            // The L3 record IS what publishing consumes, so that is cleared after.
            for (const auto& sim : sims) {
                sim->clearFilledOrders();
            }
            if (publish
                && impl.simulations.front()->currentTimestamp() >= impl.config.gracePeriod) {
                try {
                    publishState(impl);
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

    // First worker exception wins; the failing worker drops out of the barrier so
    // the survivors aren't stranded, and it's rethrown after a clean join.
    std::mutex errorMutex;
    std::exception_ptr firstError;

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
                    const auto cohort = std::span{sims}.subspan(begin, size);
                    for (const auto& sim : cohort) {
                        if (sim->state() == taosim::simulation::SimulationState::INACTIVE) {
                            sim->start();
                        }
                    }
                    for (uint64_t s = 0; s < stepCount; ++s) {
                        for (const auto& sim : cohort) {
                            sim->step();
                        }
                        barrier.arrive_and_wait();
                        if (leaving.load(std::memory_order_acquire)) break;
                    }
                    for (const auto& sim : cohort) {
                        sim->stop();
                    }
                } catch (...) {
                    {
                        const std::scoped_lock lock{errorMutex};
                        if (!firstError) firstError = std::current_exception();
                    }
                    barrier.arrive_and_drop();
                }
            });
    }
    pool.join();

    if (firstError) std::rethrow_exception(firstError);

    publishEndInfo(impl);
    fmt::println("Multi-asset run complete.");
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation::multiasset

//-------------------------------------------------------------------------
