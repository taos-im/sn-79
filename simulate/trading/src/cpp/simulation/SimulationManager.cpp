/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/checkpoint/CheckpointError.hpp>
#include <taosim/checkpoint/helpers.hpp>
#include <taosim/filesystem/TempPath.hpp>
#include <taosim/filesystem/utils.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>
#include <taosim/message/PayloadFactory.hpp>
#include <taosim/process/helpers.hpp>
#include <taosim/replay/helpers.hpp>
#include <taosim/serialization/msgpack/common.hpp>
#include <taosim/serialization/msgpack/utils.hpp>
#include <taosim/simulation/SimulationError.hpp>
#include <taosim/simulation/SimulationManager.hpp>
#include <taosim/simulation/helpers.hpp>
#include <taosim/simulation/serialization/ValidatorRequest.hpp>
#include <taosim/simulation/util.hpp>
#include <taosim/util/SLTPDebug.hpp>
#include <taosim/xml/helpers.hpp>


#include <boost/algorithm/string.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <date/date.h>
#include <date/tz.h>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <msgpack.hpp>

#include <barrier>
#include <latch>
#include <ranges>
#include <source_location>
#include <thread>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------


//-------------------------------------------------------------------------

void SimulationManager::runSimulations()
{
    std::barrier barrier{
        m_blockInfo.count,
        [&] {
            for (auto& simulation : m_simulations) {
                simulation->clearFilledOrders();
            }
            publishState();
            m_stepSignal();
        }};
    std::latch latch{m_blockInfo.count};

    publishStartInfo();

    for (auto& simulation : m_simulations) {
        boost::asio::post(
            *m_threadPool,
            [&] {
                simulation->simulate(barrier);
                latch.count_down();
            });
    }
    latch.wait();

    publishEndInfo();
}

//-------------------------------------------------------------------------

void SimulationManager::runReplay()
{
    const auto& replayDesc = m_replayManager->desc();

    const auto& simulation = m_simulations.at(*replayDesc.bookId / m_blockInfo.dimension);

    for (const auto& [simulation, path] : views::zip(m_simulations, m_replayManager->initialBalancesPaths())) {
        rapidjson::Document balancesJson = json::loadJson(path);
        for (const auto& member : balancesJson.GetObject()) {
            const auto name = member.name.GetString();
            const AgentId agentId = std::stoi(name);
            BookId bookId{};
            for (const auto& balsJson : balancesJson[name].GetArray()) {
                auto& bals = simulation->exchange()->accounts().at(agentId).at(bookId);
                bals.base = taosim::accounting::Balance(
                    taosim::json::getDecimal(balsJson["base"]),
                    "",
                    bals.m_roundParams.baseDecimals);
                bals.quote = std::make_shared<taosim::accounting::Balance>(
                    taosim::json::getDecimal(balsJson["quote"]),
                    "",
                    bals.m_roundParams.quoteDecimals);
                ++bookId;
            }
        }
    }

    for (auto&& pathGroup : m_replayManager->runtimePathGroups()[*replayDesc.bookId]) {
        auto&& [replayLogFile, L2LogFile] = pathGroup;
        std::ifstream ifs{replayLogFile, std::ios::in};
        std::vector<std::string> lines;
        std::string buf;
        std::getline(ifs, buf);
        size_t lineCounter{1};
        simulation->timestampToMidPrice() =
            replay::helpers::makeTimestampToMidPriceMapping(L2LogFile);
        while (true) {
            lines.clear();
            while (std::getline(ifs, buf)) {
                lines.push_back(buf);
                ++lineCounter;
            }
            if (lines.empty()) break;
            for (const auto& line : lines) {
                Message::Ptr msg = replay::helpers::createMessageFromLogFileEntry(line, lineCounter);
                if (simulation->isReplacedAgent(msg->source)) continue;
                simulation->queueMessage(msg);
                simulation->time().duration = msg->arrival;
            }
        }
    }

    simulation->simulate();
}

//-------------------------------------------------------------------------

void SimulationManager::runReplayAdvanced()
{
    for (const auto& [simulation, path] : views::zip(m_simulations, m_replayManager->initialBalancesPaths())) {
        rapidjson::Document balancesJson = json::loadJson(path);
        for (const auto& member : balancesJson.GetObject()) {
            const auto name = member.name.GetString();
            const AgentId agentId = std::stoi(name);
            BookId bookId{};
            for (const auto& balsJson : balancesJson[name].GetArray()) {
                auto& bals = simulation->exchange()->accounts().at(agentId).at(bookId);
                bals.base = taosim::accounting::Balance(
                    taosim::json::getDecimal(balsJson["base"]),
                    "",
                    bals.m_roundParams.baseDecimals);
                bals.quote = std::make_shared<taosim::accounting::Balance>(
                    taosim::json::getDecimal(balsJson["quote"]),
                    "",
                    bals.m_roundParams.quoteDecimals);
                ++bookId;
            }
        }
    }

    struct BookReplayFilesState
    {
        std::vector<std::ifstream> fileStreams;
        std::vector<size_t> lineCounters;
        size_t currentFileIdx{};

        [[nodiscard]] auto& currentFile()
        {
            return fileStreams.at(std::min(currentFileIdx, fileStreams.size() - 1));
        }
    
        [[nodiscard]] size_t currentLineCounter() const noexcept
        {
            return lineCounters.at(std::min(currentFileIdx, lineCounters.size() - 1));
        }
    
        [[nodiscard]] bool done() const noexcept { return currentFileIdx >= fileStreams.size(); }
    
        bool getLine(std::string& buf)
        {
            if (done()) return false;
            if (!std::getline(currentFile(), buf)) {
                ++currentFileIdx;
                if (done()) return false;
                std::getline(currentFile(), buf);
            }
            ++lineCounters.at(currentFileIdx);
            return true;
        }
    };

    /*auto bookIdToReplayFilesState = bookIdToReplayLogPaths
        | views::transform([](auto&& replayLogPaths) {
            return BookReplayFilesState{
                .fileStreams = replayLogPaths
                    | views::transform([](auto&& path) {
                        std::ifstream ifs{path, std::ios::in};
                        std::string sink;
                        std::getline(ifs, sink);  // Discard header.
                        return ifs;
                    })
                    | ranges::to<std::vector>,
                .lineCounters = std::vector<size_t>(replayLogPaths.size(), 1)
            };
        })
        | ranges::to<std::vector>;

    m_stepSignal.connect([&] {
        const auto& reprSimu = m_simulations.front();
        const auto& time = reprSimu->time();
        const auto cutoff = time.current + time.step;
        for (BookId bookId{}; bookId < bookIdToReplayFilesState.size(); ++bookId) {
            const auto& simulation = m_simulations.at(bookId / m_blockInfo.dimension);
            auto& state = bookIdToReplayFilesState.at(bookId);
            if (state.done()) continue;
            std::string lineBuf;
            while (true) {
                if (!state.getLine(lineBuf)) break;
                const auto msg = replay::helpers::createMessageFromLogFileEntry(
                    lineBuf, state.currentLineCounter() - 1);
                if (simulation->isReplacedAgent(msg->source)) continue;
                simulation->queueMessage(msg);
                simulation->time().duration = msg->arrival;
                if (msg->arrival >= cutoff) break;
            }
        }
    });*/

    runSimulations();
}

//-------------------------------------------------------------------------

void SimulationManager::publishStartInfo()
{
    if (!online() || m_simulations.front()->state() != SimulationState::INACTIVE) return;

    rapidjson::Document json = [this] {
        const auto& reprSimu = m_simulations.front();
        const auto msg = Message::create(
            reprSimu->time().start,
            0,
            "SIMULATION",
            "*",
            "EVENT_SIMULATION_START",
            MessagePayload::create<StartSimulationPayload>(m_logDir.generic_string()));
        rapidjson::Document json{rapidjson::kObjectType};
        auto& allocator = json.GetAllocator();
        json.AddMember(
            "messages",
            [&] {
                rapidjson::Document messagesJson{rapidjson::kArrayType, &allocator};
                rapidjson::Document msgJson{&allocator};
                msg->jsonSerialize(msgJson);
                messagesJson.PushBack(msgJson, allocator);
                return messagesJson;
            }().Move(),
            allocator);
        return json;
    }();
    rapidjson::Document res;

    net::asio::io_context ctx;
    net::asio::co_spawn(
        ctx,
        net::asyncSendOverNetwork(net::AsyncSendContext{
            .logger = *m_simulations.front(),
            .netInfo = m_netInfo,
            .reqBody = json,
            .endpoint = m_generalMsgEndpoint,
            .resJson = res
        }),
        net::asio::detached);
    ctx.run();
}

//-------------------------------------------------------------------------

void SimulationManager::publishEndInfo()
{
    if (!online()) return;

    rapidjson::Document json = [this] {
        const auto& reprSimu = m_simulations.front();
        const auto msg = Message::create(
            reprSimu->time().start,
            0,
            "SIMULATION",
            "*",
            "EVENT_SIMULATION_END",
            MessagePayload::create<EmptyPayload>());
        rapidjson::Document json{rapidjson::kObjectType};
        auto& allocator = json.GetAllocator();
        json.AddMember(
            "messages",
            [&] {
                rapidjson::Document messagesJson{rapidjson::kArrayType, &allocator};
                rapidjson::Document msgJson{&allocator};
                msg->jsonSerialize(msgJson);
                messagesJson.PushBack(msgJson, allocator);
                return messagesJson;
            }().Move(),
            allocator);
        return json;
    }();
    rapidjson::Document res;

    net::asio::io_context ctx;
    net::asio::co_spawn(
        ctx,
        net::asyncSendOverNetwork(net::AsyncSendContext{
            .logger = *m_simulations.front(),
            .netInfo = m_netInfo,
            .reqBody = json,
            .endpoint = m_generalMsgEndpoint,
            .resJson = res
        }),
        net::asio::detached);
    ctx.run();
}

//-------------------------------------------------------------------------

void SimulationManager::publishState()
{
    if (warmingUp() || !online()) return;

    if (m_useMessagePack) {
        publishStateMessagePack();
    } else {
        publishStateJson();
    }
}

//-------------------------------------------------------------------------

void SimulationManager::publishStateJson()
{
    rapidjson::Document stateJson = makeStateJson();
    rapidjson::Document resJson;

    net::asio::io_context ctx;
    net::asio::co_spawn(
        ctx,
        net::asyncSendOverNetwork(net::AsyncSendContext{
            .logger = *m_simulations.front(),
            .netInfo = m_netInfo,
            .reqBody = stateJson,
            .endpoint = m_bookStateEndpoint,
            .resJson = resJson
        }),
        net::asio::detached);
    ctx.run();

    const auto& reprSimu = m_simulations.front();
    const Timestamp now = reprSimu->currentTimestamp();

    for (const auto& response : resJson["responses"].GetArray()) {
        const auto [msg, blockIdx] = decanonize(
            Message::fromJsonResponse(response, now, reprSimu->proxy()->name()),
            m_blockInfo.dimension);
        if (!blockIdx) {
            for (const auto& simulation : m_simulations) {
                simulation->queueMessage(msg);
            }
            continue;
        }
        m_simulations.at(*blockIdx)->queueMessage(msg);
    }
}

//-------------------------------------------------------------------------

void SimulationManager::publishStateMessagePack()
{
    static constexpr auto ctx = std::source_location::current().function_name();

    const auto& reprSimu = m_simulations.front();
    const auto now = reprSimu->currentTimestamp();

    auto getTime = [] {
        return std::make_optional(std::chrono::high_resolution_clock::now());
    };

    if (m_measureStepWallClockTime && m_measurements.t0parse) {
        const auto t = getTime();
        m_measurements.t1proc = t;
        m_measurements.t0state = t;
    }

    taosim::serialization::HumanReadableStream stream{1uz << 27};
    const serialization::ValidatorRequest req{.mngr = this};
    msgpack::pack(stream, req);

    bipc::shared_memory_object shmReq{
        bipc::open_or_create,
        s_statePublishShmName.data(),
        bipc::read_write
    };
    shmReq.truncate(stream.size());
    bipc::mapped_region reqRegion{shmReq, bipc::read_write};
    std::memcpy(reqRegion.get_address(), stream.data(), stream.size());

    if (m_measureStepWallClockTime && m_measurements.t0parse) {
        const auto simuTimeDetails = reprSimu->time();
        using namespace std::chrono;
        const std::pair stepTimeSpan{
            duration<Timestamp, std::nano>{simuTimeDetails.current - simuTimeDetails.step},
            duration<Timestamp, std::nano>{simuTimeDetails.current}
        };
        const auto t = getTime();
        m_measurements.t1state = t;
        const auto total = duration<double>(
            *m_measurements.t1state - *m_measurements.t0parse);
        const auto parse = duration<double>(
            *m_measurements.t1parse - *m_measurements.t0parse);
        const auto proc = duration<double>(
            *m_measurements.t1proc - *m_measurements.t0proc);
        const auto state = duration<double>(
            *m_measurements.t1state - *m_measurements.t0state);
        fmt::println(
            "PROCESSED {:%T} - {:%T} ({:.4f}s | PARSE {:.4f}s | PROC {:.4f}s | STATE {:.4f}s)",
            duration_cast<seconds>(stepTimeSpan.first),
            duration_cast<seconds>(stepTimeSpan.second),
            total.count(), parse.count(), proc.count(), state.count());
    }

    retryMessagePack:

    const size_t packedSize = stream.size();
    m_validatorReqMessageQueue->flush();
    const bool mqSendSuccess = m_validatorReqMessageQueue->send(
        std::span<const char>{std::bit_cast<const char*>(&packedSize), sizeof(packedSize)});
    if (!mqSendSuccess) {
        fmt::println("Sending to /{} timed out, flushing and retrying...", s_validatorReqMessageQueueName);
        goto retryMessagePack;
    }

    size_t resByteSize;
    const bool mqRecvSuccess = m_validatorResMessageQueue->receive(
        std::span<char>{std::bit_cast<char*>(&resByteSize), sizeof(resByteSize)}) != -1;
    if (!mqRecvSuccess) {
        fmt::println("Receive from /{} timed out, flushing and retrying...", s_validatorResMessageQueueName);
        goto retryMessagePack;
    }

    if (m_measureStepWallClockTime) {
        const auto t = getTime();
        m_measurements.t0parse = t;
    }

    bipc::shared_memory_object shmRes{
        bipc::open_only,
        s_remoteResponsesShmName.data(),
        bipc::read_write
    };
    bipc::mapped_region resRegion{shmRes, bipc::read_write};

    msgpack::object_handle oh;
    try {
        oh = msgpack::unpack(std::bit_cast<const char*>(resRegion.get_address()), resByteSize);
    }
    catch (const std::exception& e) {
        fmt::println("Error unpacking responses: {}", e.what());
        if (m_measureStepWallClockTime) {
            const auto t = getTime();
            m_measurements.t1parse = t;
            m_measurements.t0proc = t;
        }
        return;
    }
    msgpack::object obj = oh.get();

    auto unpackResponse = [&](const msgpack::object& o) {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }
        struct Response
        {
            std::optional<AgentId> agentId{};
            std::optional<Timestamp> delay{};
            std::string type{};
            MessagePayload::Ptr payload{};
        };
        Response res;
        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();
            if (key == "agentId") {
                res.agentId = std::make_optional(val.as<AgentId>());
            }
            else if (key == "delay") {
                res.delay = std::make_optional(val.as<Timestamp>());
            }
            else if (key == "type") {
                res.type = val.as<std::string>();
            }
        }
        if (!res.agentId) {
            throw taosim::serialization::MsgPackError{};
        }
        if (!res.delay) {
            throw taosim::serialization::MsgPackError{};
        }
        if (res.type.empty()) {
            throw taosim::serialization::MsgPackError{};
        }
        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();
            if (key == "payload") {
                res.payload = PayloadFactory::createFromMessagePack(val, res.type);
                break;
            }
        }
        if (res.payload == nullptr) {
            throw taosim::serialization::MsgPackError{};
        }
        auto msg = std::make_shared<Message>();
        msg->occurrence = now;
        msg->arrival = now + *res.delay;
        msg->source = reprSimu->proxy()->name();
        msg->targets = {reprSimu->exchange()->name()};
        msg->type = fmt::format("{}_{}", "DISTRIBUTED", res.type);
        msg->payload =
            MessagePayload::create<DistributedAgentResponsePayload>(*res.agentId, res.payload);
        return msg;
    };

    if (m_measureStepWallClockTime) {
        const auto t = getTime();
        m_measurements.t1parse = t;
        m_measurements.t0proc = t;
    }
    if (obj.type != msgpack::type::MAP) {
        fmt::println("MAP type check failed for responses");
        return;
    }
    if (obj.via.map.size != 1) {
        fmt::println("MAP size == 1 check failed for responses");
        return;
    }
    const auto& val = obj.via.map.ptr[0].val;
    if (val.type != msgpack::type::ARRAY) {
        fmt::println("ARRAY type check failed for responses");
        return;
    }
    if (val.via.array.size == 0) {
        return;
    }
    std::vector<Message::Ptr> unpackedResponses;
    size_t responseIdx{};
    std::map<size_t, std::string> responseIdxToError;
    for (const auto& response : val.via.array) {
        try {
            unpackedResponses.push_back(unpackResponse(response));
        } catch (const std::exception& e) {
            responseIdxToError[responseIdx] = e.what();
        }
        ++responseIdx;
    }
    if (responseIdxToError.size() > 0) {
        rapidjson::Document json{rapidjson::kObjectType};
        auto& allocator = json.GetAllocator();
        const auto errorRatio =
            static_cast<float>(responseIdxToError.size()) / val.via.array.size;
        json.AddMember(
            "messages",
            [&] {
                rapidjson::Value messagesJson{rapidjson::kArrayType};
                rapidjson::Value messageJson{rapidjson::kObjectType};
                messageJson.AddMember(
                    "type", rapidjson::Value{"RESPONSES_ERROR_REPORT", allocator}, allocator);
                messageJson.AddMember("timestamp", rapidjson::Value{now}, allocator);
                messageJson.AddMember("errorRatio", rapidjson::Value{errorRatio}, allocator);
                for (const auto& [key, val] : responseIdxToError) {
                    messageJson.AddMember(
                        rapidjson::Value{std::to_string(key).c_str(), allocator},
                        rapidjson::Value{val.c_str(), allocator},
                        allocator);
                }
                messagesJson.PushBack(messageJson, allocator);
                return messagesJson;
            }().Move(),
            allocator);
        rapidjson::Document res;
        net::asio::io_context io;
        net::asio::co_spawn(
            io,
            net::asyncSendOverNetwork(net::AsyncSendContext{
                .logger = *m_simulations.front(),
                .netInfo = m_netInfo,
                .reqBody = json,
                .endpoint = m_generalMsgEndpoint,
                .resJson = res
            }),
            net::asio::detached);
        io.run();
        if (res.HasMember("continue") && res["continue"].GetBool() == false) {
            throw std::runtime_error{fmt::format(
                "{}: Teardown requested by validator; latest error rate: {}; details: {{{}}}",
                ctx,
                errorRatio,
                fmt::join(
                    responseIdxToError
                    | views::transform([](const auto& pair) {
                        return fmt::format("{} -> {}", pair.first, pair.second);
                    }),
                    ", "
                ))};
        }
    }

    for (const auto& response : unpackedResponses) {
        const auto [msg, blockIdx] = decanonize(response, m_blockInfo.dimension);
        if (!blockIdx) {
            for (const auto& simulation : m_simulations) {
                simulation->queueMessage(msg);
            }
            continue;
        }
        m_simulations.at(*blockIdx)->queueMessage(msg);
    }
}

//-------------------------------------------------------------------------

bool SimulationManager::online() const noexcept
{
    return !m_replayMode && !m_netInfo.host.empty() && !m_netInfo.port.empty();
}

//-------------------------------------------------------------------------

bool SimulationManager::warmingUp() const noexcept
{
    return m_simulations.front()->currentTimestamp() < m_gracePeriod;
}

//-------------------------------------------------------------------------

std::unique_ptr<SimulationManager> SimulationManager::fromConfig(
    const fs::path& configPath, const fs::path& baseDir)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    pugi::xml_document doc;
    doc.load_file(configPath.c_str());
    fmt::println(" - '{}' loaded successfully", configPath.c_str());
    pugi::xml_node node = xml::rootNode(doc);

    auto mngr = std::make_unique<SimulationManager>();

    mngr->m_blockInfo = [&] -> SimulationBlockInfo {
        static constexpr const char* attrName = "blockCount";
        pugi::xml_attribute attr = node.attribute(attrName);
        const auto threadCount = [&] {
            const auto threadCount = attr.as_uint(1);
            if (threadCount > std::thread::hardware_concurrency()) {
                throw std::runtime_error{fmt::format(
                    "{}: requested thread count ({}) exceeds count available ({})",
                    ctx, threadCount, std::thread::hardware_concurrency()
                )};
            }
            return threadCount;
        }();
        const auto booksNode = node.child("Agents").child("MultiBookExchangeAgent").child("Books");
        if (!booksNode) {
            throw std::runtime_error{fmt::format(
                "{}: missing node 'Agents/MultiBookExchangeAgent/Books'", ctx
            )};
        }
        return {
            .count = threadCount,
            .dimension = booksNode.attribute("instanceCount").as_uint(1)
        };
    }();

    mngr->m_threadPool = std::make_unique<boost::asio::thread_pool>([&] {
        const auto ckptWorkerCount = node.attribute("ckptIntervalInSteps").as_ullong() > 0ull
            ? std::min(node.attribute("ckptNumWorkers").as_uint(1), mngr->m_blockInfo.count)
            : 0u;
        const auto requestedThreadCount = mngr->m_blockInfo.count + ckptWorkerCount;
        const auto maxAllowedThreadCount = std::thread::hardware_concurrency();
        if (requestedThreadCount > maxAllowedThreadCount) {
            throw SimulationError{fmt::format(
                "Requested thread count ({}) exceeds count available ({})",
                requestedThreadCount,
                maxAllowedThreadCount
            )};
        }
        return requestedThreadCount;
    }());

    boost::asio::signal_set{mngr->m_io, SIGINT, SIGTERM}.async_wait(
        [&](boost::system::error_code, int) {
            mngr->m_threadPool->stop();
            mngr->m_io.stop();
        });

    fmt::println(" - Setting up log directory...");
    mngr->setupLogDir(node, baseDir);

    fmt::println(" - Precomputing fundamental-price covariance factor "
                 "(blocks={}, books/block={})...",
                 mngr->m_blockInfo.count, mngr->m_blockInfo.dimension);
    taosim::process::helpers::initSharedResources(mngr->m_sharedResources, node);

    fmt::println(" - Configuring {} blocks (books/block={})...",
                 mngr->m_blockInfo.count, mngr->m_blockInfo.dimension);
    mngr->m_simulations.reserve(mngr->m_blockInfo.count);
    for (size_t blockIdx = 0; blockIdx < mngr->m_blockInfo.count; ++blockIdx) {
        auto simulation = std::make_unique<Simulation>(
            blockIdx, mngr->m_blockInfo.dimension, mngr->m_logDir, &mngr->m_sharedResources);
        simulation->configure(node);
        mngr->m_simulations.push_back(std::move(simulation));
        printProgress(blockIdx + 1, mngr->m_blockInfo.count, "Configuring blocks");
    }
    fmt::println(" - Block configuration complete.");

    mngr->m_gracePeriod = node.child("Agents")
        .child("MultiBookExchangeAgent")
        .attribute("gracePeriod")
        .as_ullong();

    mngr->m_netInfo = {
        .host = node.attribute("host").as_string(),
        .port = node.attribute("port").as_string(),
        .resolveTimeout = node.attribute("resolveTimeout").as_llong(1),
        .connectTimeout = node.attribute("connectTimeout").as_llong(3),
        .writeTimeout = node.attribute("writeTimeout").as_llong(15),
        .readTimeout = node.attribute("readTimeout").as_llong(60)
    };
    mngr->m_bookStateEndpoint = node.attribute("bookStateEndpoint").as_string("/");
    mngr->m_generalMsgEndpoint = node.attribute("generalMsgEndpoint").as_string("/");

    taosim::util::SLTPDebugger::enabled = node.attribute("sltpDebug").as_bool();


    mngr->m_stepSignal.connect([&] {
        for (auto& simulation : mngr->m_simulations) {
            simulation->exchange()->L3Record().clear();
        }
    });

    if (node.attribute("traceTime").as_bool()) {
        mngr->m_stepSignal.connect([&] {
            const auto& reprSimu = mngr->m_simulations.front();
            uint64_t total, seconds, hours, minutes, nanos;
            total = reprSimu->time().current / 1'000'000'000;
            minutes = total / 60;
            seconds = total % 60;
            hours = minutes / 60;
            minutes = minutes % 60;
            nanos = reprSimu->time().current % 1'000'000'000;
            fmt::println("TIME : {:02d}:{:02d}:{:02d}.{:09d}", hours, minutes, seconds, nanos); 
        });
    }

    mngr->m_validatorReqMessageQueue = std::make_unique<ipc::PosixMessageQueue>(
        ipc::PosixMessageQueueDesc{.name = s_validatorReqMessageQueueName.data()});
    mngr->m_validatorResMessageQueue = std::make_unique<ipc::PosixMessageQueue>(
        ipc::PosixMessageQueueDesc{.name = s_validatorResMessageQueueName.data()});

    mngr->m_useMessagePack = node.attribute("useMessagePack").as_bool();

    if (size_t ckptIntervalInSteps = node.attribute("ckptIntervalInSteps").as_ullong()) {
        mngr->m_checkpointManager =
            std::make_unique<checkpoint::CheckpointManager>(checkpoint::CheckpointingDesc{
                .simuMngr = mngr.get(),
                .runDir = mngr->m_logDir,
                .intervalInSteps = ckptIntervalInSteps,
                .numLastFilesToKeep = (ssize_t)node.attribute("ckptNumLastFilesToKeep").as_ullong(),
                .measureWallClockTime = node.attribute("ckptMeasureWallClockTime").as_bool()
            });
    }

    mngr->m_measureStepWallClockTime = node.attribute("measureStepWallClockTime").as_bool();

    return mngr;
}

//-------------------------------------------------------------------------

std::unique_ptr<SimulationManager> SimulationManager::fromCheckpoint(
    const checkpoint::CheckpointToken& ckptToken)
{
    const fs::path runDir = checkpoint::runDirFromToken(ckptToken);
    const fs::path ckptDir = checkpoint::ckptDirFromToken(ckptToken);
    const fs::path configPath = runDir / "config.xml";

    fmt::println("Loading checkpoint {}...", ckptDir.c_str());

    // Common.
    const auto commonCkptFile =
        ckptDir / fmt::format("common{}", checkpoint::CheckpointManager::s_fileExtension);
    std::ifstream ifs{commonCkptFile, std::ios::binary};
    const size_t commonCkptByteSize = fs::file_size(commonCkptFile);
    std::vector<char> commonCkptByteBuffer(commonCkptByteSize);
    ifs.read(commonCkptByteBuffer.data(), commonCkptByteSize);
    msgpack::object_handle commonOh =
        msgpack::unpack(commonCkptByteBuffer.data(), commonCkptByteSize);
    msgpack::object commonObj = commonOh.get();

    // Blocks.
    std::vector<msgpack::object_handle> blockObjHandles;
    static const std::regex blockCkptPattern{
        fmt::format("^\\d+\\{}$", checkpoint::CheckpointManager::s_fileExtension)};
    const auto blockCkptFilesSorted = filesystem::collectMatchingPaths(
        ckptDir,
        [&](auto&& p) {
            return std::regex_match(p.filename().string(), blockCkptPattern);
        })
        | ranges::actions::sort([](auto&& lhs, auto&& rhs) {
            return std::stoul(lhs.stem().string()) < std::stoul(rhs.stem().string());
        });
    const size_t nBlockFiles = blockCkptFilesSorted.size();
    for (auto&& [blockFileIdx, ckptFile] : views::enumerate(blockCkptFilesSorted)) {
        const size_t ckptByteSize = fs::file_size(ckptFile);
        std::ifstream ifs{ckptFile, std::ios::binary};
        std::vector<char> ckptByteBuffer(ckptByteSize);
        ifs.read(ckptByteBuffer.data(), ckptByteSize);
        blockObjHandles.push_back(msgpack::unpack(ckptByteBuffer.data(), ckptByteSize));
        printProgress(blockFileIdx + 1, nBlockFiles, "Loading blocks");
    }

    fmt::println("Checkpoint loaded successfully; initializing simulation...");

    auto ckptTimestamp = taosim::serialization::msgpackFindMap<Timestamp>(commonObj, "timestamp");
    if (!ckptTimestamp) {
        throw checkpoint::CheckpointError{};
    }

    pugi::xml_document doc;
    if (pugi::xml_parse_result result = doc.load_file(configPath.c_str()); !result) {
        throw checkpoint::CheckpointError{};
    }
    pugi::xml_node simuNode = xml::rootNode(doc);
    xml::setAttribute(simuNode, "current", *ckptTimestamp);
    const filesystem::TempPath tempConfigPath{"config.xml"};
    doc.save_file(tempConfigPath);

    auto mngr = SimulationManager::fromConfig(tempConfigPath, runDir.parent_path());

    fmt::println("Setting up simulation state according to the checkpoint...");

    checkpoint::setupUsingCkptData(mngr.get(), commonObj, blockObjHandles);

    fmt::println("Load from checkpoint successful.");

    return mngr;
}

//-------------------------------------------------------------------------

std::unique_ptr<SimulationManager> SimulationManager::fromReplay(const replay::ReplayDesc& desc)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    pugi::xml_document doc;
    const fs::path configPath = desc.dir / "config.xml";
    doc.load_file(configPath.c_str());
    fmt::println(" - '{}' loaded successfully", configPath.c_str());
    pugi::xml_node node = xml::rootNode(doc);
    node.attribute("id").set_value(
        fmt::format("{}-replay", desc.dir.filename().c_str()).c_str());
    
    static constexpr const char* replayNodeName = "Replay";
    auto replayLogNode = node.child("Agents")
        .child("MultiBookExchangeAgent")
        .child("Logging")
        .child(replayNodeName);
    if (replayLogNode) {
        replayLogNode.parent().remove_child(replayNodeName);
    }

    auto mngr = std::make_unique<SimulationManager>();

    mngr->m_blockInfo = [&] -> SimulationBlockInfo {
        static constexpr const char* attrName = "blockCount";
        pugi::xml_attribute attr = node.attribute(attrName);
        const auto threadCount = [&] {
            const auto threadCount = attr.as_uint(1);
            if (threadCount > std::thread::hardware_concurrency()) {
                throw std::runtime_error{fmt::format(
                    "{}: requested thread count ({}) exceeds count available ({})",
                    ctx, threadCount, std::thread::hardware_concurrency()
                )};
            }
            return threadCount;
        }();
        const auto booksNode = node.child("Agents").child("MultiBookExchangeAgent").child("Books");
        if (!booksNode) {
            throw std::runtime_error{fmt::format(
                "{}: missing node 'Agents/MultiBookExchangeAgent/Books'",
                ctx
            )};
        }
        return {
            .count = threadCount,
            .dimension = booksNode.attribute("instanceCount").as_uint(1)
        };
    }();
    mngr->m_threadPool = std::make_unique<boost::asio::thread_pool>(mngr->m_blockInfo.count);
    boost::asio::signal_set{mngr->m_io, SIGINT, SIGTERM}.async_wait(
        [&](boost::system::error_code, int) {
            mngr->m_threadPool->stop();
            mngr->m_io.stop();
        });

    mngr->m_replayManager = std::make_unique<replay::ReplayManager>(
        desc,
        [&](const replay::ReplayDesc& desc) {
            if (desc.bookId && !(*desc.bookId < mngr->m_blockInfo.count * mngr->m_blockInfo.dimension)) {
                throw replay::helpers::ReplayError{"bookId out of range"};
            }
        });

    mngr->setupLogDir(node, desc.dir.parent_path());
    taosim::process::helpers::initSharedResources(mngr->m_sharedResources, node);
    mngr->m_simulations = [&] {
        std::vector<std::unique_ptr<Simulation>> res;
        res.reserve(mngr->m_blockInfo.count);
        for (uint32_t blockIdx{}; blockIdx < mngr->m_blockInfo.count; ++blockIdx) {
            auto simulation = std::make_unique<Simulation>(
                blockIdx, mngr->m_blockInfo.dimension, mngr->m_logDir, &mngr->m_sharedResources, true, desc);
            simulation->configure(node);
            res.push_back(std::move(simulation));
            printProgress(blockIdx + 1, mngr->m_blockInfo.count, "Configuring blocks");
        }
        return res;
    }();

    mngr->m_gracePeriod = node.child("Agents")
        .child("MultiBookExchangeAgent")
        .attribute("gracePeriod")
        .as_ullong();

    if (node.attribute("traceTime").as_bool()) {
        mngr->m_stepSignal.connect([&] {
            const auto& reprSimu = mngr->m_simulations.front();
            uint64_t total, seconds, hours, minutes, nanos;
            total = reprSimu->time().current / 1'000'000'000;
            minutes = total / 60;
            seconds = total % 60;
            hours = minutes / 60;
            minutes = minutes % 60;
            nanos = reprSimu->time().current % 1'000'000'000;
            fmt::println("TIME : {:02d}:{:02d}:{:02d}.{:09d}", hours, minutes, seconds, nanos); 
        });
    }

    mngr->m_replayMode = true;

    mngr->m_useMessagePack = node.attribute("useMessagePack").as_bool();

    return mngr;
}

//-------------------------------------------------------------------------

void SimulationManager::setupLogDir(pugi::xml_node simuNode, const fs::path& baseDir)
{
    const std::string simuId = [&] {
        const std::string specifiedId = simuNode.attribute("id").as_string();
        if (!specifiedId.empty()) return specifiedId;
        using namespace std::chrono;
        const auto dateTimeId = date::format(
            "%Y%m%d_%H%M%S",
            date::make_zoned(
                date::current_zone(), time_point_cast<seconds>(system_clock::now())));
        xml::setAttribute(simuNode, "id", dateTimeId.c_str());
        return dateTimeId;
    }();

    m_logDir = baseDir / simuId;

    fs::create_directories(m_logDir);

    if (const auto configSavePath = m_logDir / "config.xml"; !fs::exists(configSavePath)) {
        pugi::xml_document doc;
        doc.append_copy(simuNode);
        doc.save_file(configSavePath.c_str());
    }
}

//-------------------------------------------------------------------------

rapidjson::Document SimulationManager::makeStateJson() const
{
    const auto& reprSimu = m_simulations.front();

    const auto bookStatePublishMsg = Message::create(
        reprSimu->currentTimestamp(),
        0,
        reprSimu->exchange()->name(),
        reprSimu->proxy()->name(),
        "MULTIBOOK_STATE_PUBLISH",
        MessagePayload::create<BookStateMessagePayload>(makeCollectiveBookStateJson()));

    rapidjson::Document json;
    auto& allocator = json.GetAllocator();
    bookStatePublishMsg->jsonSerialize(json);
    json["payload"].AddMember(
        "notices",
        [&] {
            rapidjson::Document noticesJson{rapidjson::kArrayType, &allocator};
            std::unordered_map<decltype(Message::type), uint32_t> msgTypeToCount{
                { "RESPONSE_DISTRIBUTED_RESET_AGENT", 0 },
                { "ERROR_RESPONSE_DISTRIBUTED_RESET_AGENT", 0 }
            };
            auto checkGlobalDuplicate = [&](Message::Ptr msg) -> bool {
                const auto payload = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                if (payload == nullptr) return false;
                auto relevantPayload = [&] {
                    const auto pld = payload->payload;
                    return std::dynamic_pointer_cast<ResetAgentsResponsePayload>(pld) != nullptr
                        || std::dynamic_pointer_cast<ResetAgentsErrorResponsePayload>(pld) != nullptr;
                };
                if (!relevantPayload()) return true;
                auto it = msgTypeToCount.find(msg->type);
                if (it == msgTypeToCount.end()) return true;
                if (it->second > 0) return false;
                it->second++;
                return true;
            };
            for (const auto& [blockIdx, simulation] : views::enumerate(m_simulations)) {
                for (const auto& msg : simulation->proxy()->messages()) {
                    if (!checkGlobalDuplicate(msg)) continue;
                    canonize(msg, blockIdx, m_blockInfo.dimension);
                    rapidjson::Document msgJson{&allocator};
                    msg->jsonSerialize(msgJson);
                    noticesJson.PushBack(msgJson, allocator);
                }
                simulation->proxy()->clearMessages();
            }
            return noticesJson;
        }().Move(),
        allocator);
    
    return json;
}

//-------------------------------------------------------------------------

rapidjson::Document SimulationManager::makeCollectiveBookStateJson() const
{
    auto serialize = [this](rapidjson::Document& json) {
        auto& allocator = json.GetAllocator();
        // Log directory.
        json.AddMember("logDir", rapidjson::Value{m_logDir.c_str(), allocator}, allocator);
        // Books.
        auto serializeBooks = [this](rapidjson::Document& json) {
            json.SetArray();
            auto& allocator = json.GetAllocator();
            for (const auto& [blockIdx, simulation] : views::enumerate(m_simulations)) {
                const auto exchange = simulation->exchange();
                for (const auto& book : exchange->books()) {
                    json.PushBack(
                        [&] {
                            rapidjson::Document bookJson{rapidjson::kObjectType, &allocator};
                            const BookId bookIdCanon = blockIdx * m_blockInfo.dimension + book->id();
                            bookJson.AddMember("bookId", rapidjson::Value{bookIdCanon}, allocator);
                            exchange->L3Record().at(book->id()).jsonSerialize(bookJson, "record");
                            rapidjson::Document bidAskJson{&allocator};
                            book->jsonSerialize(bidAskJson);
                            bookJson.AddMember("bid", bidAskJson["bid"], allocator);
                            bookJson.AddMember("ask", bidAskJson["ask"], allocator);
                            return bookJson;
                        }().Move(),
                        allocator);
                }
            }
        };
        json::serializeHelper(json, "books", serializeBooks);
        // Accounts.
        auto serializeAccounts = [this](rapidjson::Document& json) {
            json.SetObject();
            auto& allocator = json.GetAllocator();
            const auto& reprSimu = m_simulations.front();
            for (AgentId agentId : views::keys(reprSimu->exchange()->accounts())) {
                if (agentId < 0) continue;
                const auto agentIdStr = std::to_string(agentId);
                const char* agentIdCStr = agentIdStr.c_str();
                json.AddMember(
                    rapidjson::Value{agentIdCStr, allocator},
                    rapidjson::Document{rapidjson::kObjectType, &allocator}.Move(),
                    allocator);
                json[agentIdCStr].AddMember("agentId", rapidjson::Value{agentId}, allocator);
                json[agentIdCStr].AddMember("holdings", rapidjson::Document{rapidjson::kArrayType, &allocator}, allocator);
                json[agentIdCStr].AddMember("orders", rapidjson::Document{rapidjson::kArrayType, &allocator}, allocator);
                json[agentIdCStr].AddMember("loans", rapidjson::Document{rapidjson::kArrayType, &allocator}, allocator);
                rapidjson::Document feesJson{rapidjson::kObjectType, &allocator};
                for (const auto& [blockIdx, simulation] : views::enumerate(m_simulations)) {
                    const auto exchange = simulation->exchange();
                    const auto books = exchange->books();
                    const auto& account = exchange->accounts().at(agentId);
                    const auto feePolicy = exchange->clearingManager().feePolicy();
                    for (const auto& book : books) {
                        const BookId bookIdCanon = blockIdx * m_blockInfo.dimension + book->id();
                        json[agentIdCStr]["orders"].PushBack(
                            rapidjson::Document{rapidjson::kArrayType, &allocator}.Move(), allocator);
                        json[agentIdCStr]["holdings"].PushBack(
                            [&] {
                                rapidjson::Document holdingsJson{&allocator};
                                account.at(book->id()).jsonSerialize(holdingsJson);
                                return holdingsJson;
                            }().Move(),
                            allocator);
                        json[agentIdCStr]["loans"].PushBack(
                            [&] {
                                rapidjson::Document loansObjectJson{rapidjson::kObjectType, &allocator};
                                for (const auto& [id, loan] : account.at(book->id()).m_loans) {
                                    rapidjson::Document loanJson{rapidjson::kObjectType, &allocator};
                                    loanJson.AddMember("id", rapidjson::Value{id}, allocator);
                                    loanJson.AddMember("amount", rapidjson::Value{taosim::util::decimal2double(loan.amount())}, allocator);
                                    loanJson.AddMember("currency", rapidjson::Value{
                                        std::to_underlying(loan.direction() == OrderDirection::BUY ? Currency::QUOTE : Currency::BASE)
                                    }, allocator);
                                    loanJson.AddMember("baseCollateral", rapidjson::Value{taosim::util::decimal2double(loan.collateral().base())}, allocator);
                                    loanJson.AddMember("quoteCollateral", rapidjson::Value{taosim::util::decimal2double(loan.collateral().quote())}, allocator);
                                    const auto idStr = std::to_string(id);
                                    const char* idCStr = idStr.c_str();
                                    loansObjectJson.AddMember(rapidjson::Value{idCStr, allocator}, loanJson, allocator);
                                }
                                return loansObjectJson;
                            }().Move(),
                            allocator);
                        json::serializeHelper(
                            feesJson,
                            std::to_string(bookIdCanon).c_str(),
                            [&](rapidjson::Document& feeJson) {
                                feeJson.SetObject();
                                auto& allocator = feeJson.GetAllocator();
                                feeJson.AddMember(
                                    "volume",
                                    rapidjson::Value{util::decimal2double(
                                        feePolicy->agentVolume(book->id(), agentId))},
                                    allocator);
                                const auto rates = feePolicy->getRates(book->id(), agentId);
                                feeJson.AddMember(
                                    "makerFeeRate",
                                    rapidjson::Value{util::decimal2double(rates.maker)},
                                    allocator);
                                feeJson.AddMember(
                                    "takerFeeRate",
                                    rapidjson::Value{util::decimal2double(rates.taker)},
                                    allocator);
                            });
                    }
                }
                json[agentIdCStr].AddMember("fees", feesJson, allocator);
            }
            for (const auto& [blockIdx, simulation] : views::enumerate(m_simulations)) {
                const auto books = simulation->exchange()->books();
                for (const auto& book : books) {
                    const BookId bookIdCanon = blockIdx * m_blockInfo.dimension + book->id();
                    auto serializeSide = [&](OrderDirection side) {
                        const auto& levels =
                            side == OrderDirection::BUY ? book->buyQueue() : book->sellQueue();
                        for (const auto& level : levels) {
                            for (const auto& tick : level) {
                                const auto ctx = books[book->id()]->orderToClientInfo().at(tick->id());
                                if (ctx.agentId < 0) { continue; }
                                const auto agentIdStr = std::to_string(ctx.agentId);
                                const char* agentIdCStr = agentIdStr.c_str();
                                rapidjson::Document orderJson{&allocator};
                                tick->jsonSerialize(orderJson);
                                json::setOptionalMember(orderJson, "clientOrderId", ctx.clientOrderId);
                                json[agentIdCStr]["orders"][bookIdCanon].PushBack(orderJson, allocator);
                            }
                        }
                    };
                    serializeSide(OrderDirection::BUY);
                    serializeSide(OrderDirection::SELL);
                }
            }
        };
        json::serializeHelper(json, "accounts", serializeAccounts);
    };

    rapidjson::Document json{rapidjson::kObjectType};
    serialize(json);
    return json;
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
