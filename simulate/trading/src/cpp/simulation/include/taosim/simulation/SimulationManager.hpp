/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <Simulation.hpp>
#include <taosim/checkpoint/CheckpointToken.hpp>
#include <taosim/checkpoint/CheckpointManager.hpp>
#include <taosim/ipc/ipc.hpp>
#include <taosim/net/net.hpp>
#include <taosim/replay/ReplayManager.hpp>
#include <taosim/simulation/SharedResources.hpp>

#include <boost/asio.hpp>
#include <pugixml.hpp>

#include <atomic>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

//-------------------------------------------------------------------------


//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

struct SimulationBlockInfo
{
    uint32_t count;
    uint32_t dimension;
};

//-------------------------------------------------------------------------

// A STOP REQUEST IS PROCESS-WIDE, BECAUSE A SIGNAL IS.
//
// Installed with std::signal in SimulationManager.cpp. What used to sit there was
//
//     boost::asio::signal_set{mngr->m_io, SIGINT, SIGTERM}.async_wait(...)
//
// which could never fire, for two independent reasons: the signal_set is a TEMPORARY, destroyed at
// the end of the statement and cancelling its own pending wait; and m_io is never run -- its only
// two uses in the whole translation unit are those two constructions, and m_io.run() appears
// nowhere. So nothing here has ever handled SIGINT or SIGTERM, and the process died on the default
// disposition. That, rather than a handler that merely forgot to checkpoint, is why taosim has never
// written one on an intended stop.
//
// requestStop() only RECORDS the request. The barrier completion in runSimulations() is the one
// point at which every block has arrived and none has resumed, so it is the only place a checkpoint
// is consistent, and the only place allowed to act on it.
void requestStop() noexcept;
[[nodiscard]] bool stopRequested() noexcept;

//-------------------------------------------------------------------------

class SimulationManager
{
public:

    void runSimulations();
    void runReplay();
    void runReplayAdvanced();
    void publishStartInfo();
    void publishEndInfo();
    void publishState();
    
    [[nodiscard]] SimulationBlockInfo blockInfo() const noexcept { return m_blockInfo; }
    [[nodiscard]] auto&& threadPool(this auto&& self) noexcept { return self.m_threadPool; }
    [[nodiscard]] auto&& simulations(this auto&& self) noexcept { return self.m_simulations; }
    [[nodiscard]] const fs::path& logDir() const noexcept { return m_logDir; }
    [[nodiscard]] auto&& stepSignal(this auto&& self) noexcept { return self.m_stepSignal; }
    [[nodiscard]] auto&& checkpointManager(this auto&& self) noexcept { return self.m_checkpointManager; }
    [[nodiscard]] auto&& measurements(this auto&& self) noexcept { return self.m_measurements; }
    [[nodiscard]] auto&& sharedResources(this auto&& self) noexcept { return self.m_sharedResources; }

    [[nodiscard]] bool online() const noexcept;
    [[nodiscard]] bool warmingUp() const noexcept;



    static std::unique_ptr<SimulationManager> fromConfig(const fs::path& configPath, const fs::path& baseDir);
    static std::unique_ptr<SimulationManager> fromCheckpoint(const checkpoint::CheckpointToken& ckptToken);
    static std::unique_ptr<SimulationManager> fromReplay(const replay::ReplayDesc& desc);

    // IPC object names, optionally suffixed by the TAOSIM_IPC_SUFFIX env var so
    // multiple taosim runs can share one host with private, non-colliding POSIX
    // IPC (the namespace/unshare route needs privileges this host's AppArmor
    // policy denies). Empty/unset suffix => names unchanged (backward compatible).
    // The Python side (simbo/ipc.py) reads the SAME env var and applies the
    // SAME suffix, so the two processes agree on the object names.
    static std::string makeIpcName(std::string_view base)
    {
        const char* suffix = std::getenv("TAOSIM_IPC_SUFFIX");
        return suffix ? std::string{base} + suffix : std::string{base};
    }
    inline static const std::string s_validatorReqMessageQueueName = makeIpcName("taosim-req");
    inline static const std::string s_validatorResMessageQueueName = makeIpcName("taosim-res");
    inline static const std::string s_statePublishShmName = makeIpcName("state");
    inline static const std::string s_remoteResponsesShmName = makeIpcName("responses");

private:
    void setupLogDir(pugi::xml_node simuNode, const fs::path& logPath);
    void publishStateJson();
    void publishStateMessagePack();

    [[nodiscard]] rapidjson::Document makeStateJson() const;
    [[nodiscard]] rapidjson::Document makeCollectiveBookStateJson() const;

    SimulationBlockInfo m_blockInfo;
    boost::asio::io_context m_io;
    std::unique_ptr<boost::asio::thread_pool> m_threadPool;
    SharedResources m_sharedResources;
    std::vector<std::unique_ptr<Simulation>> m_simulations;
    fs::path m_logDir;
    Timestamp m_gracePeriod;
    taosim::net::NetworkingInfo m_netInfo;
    std::string m_bookStateEndpoint, m_generalMsgEndpoint;
    UnsyncSignal<void()> m_stepSignal;
    // Written ONLY inside the barrier completion and read ONLY after the barrier releases, so a
    // plain bool is correct: the completion happens-before every waiting block resumes. It exists
    // so the checkpoint and the decision to leave are the SAME observation -- sampling the request
    // separately in each block could let it flip after the completion had declined to checkpoint,
    // and the run would then exit without one.
    bool m_leaving{false};
    std::unique_ptr<ipc::PosixMessageQueue> m_validatorReqMessageQueue;
    std::unique_ptr<ipc::PosixMessageQueue> m_validatorResMessageQueue;
    bool m_useMessagePack{};
    bool m_replayMode{};
    std::unique_ptr<replay::ReplayManager> m_replayManager;
    std::unique_ptr<checkpoint::CheckpointManager> m_checkpointManager;
    bool m_measureStepWallClockTime{};
    struct {
        using Measurement = decltype(std::chrono::high_resolution_clock::now());
        std::optional<Measurement>
            t0parse, t1parse,
            t0proc, t1proc,
            t0state, t1state,
            t0ckptSave, t1ckptSave,
            t0ckptLoad, t1ckptLoad;
    } m_measurements;
};

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------