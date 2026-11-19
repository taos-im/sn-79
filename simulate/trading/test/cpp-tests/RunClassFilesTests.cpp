/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// A multi-asset run keeps the class files it started on. A fresh run copies each background's class file into its run
// directory and names the copies in the config.xml it persists there, so every resume reads the files the run started
// on, whatever the repository holds by then. A restore keeps the paths its config.xml names, so a run that an older
// engine started still resumes on the files it names.

#include "MultiAssetRunFixture.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <pugixml.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace
{

namespace fs = std::filesystem;
using taosim::test::multiasset_run::MultiAssetRun;
using taosim::test::multiasset_run::Orchestrator;

std::vector<std::string> backgroundPaths(const fs::path& configXml)
{
    pugi::xml_document doc;
    EXPECT_TRUE(doc.load_file(configXml.c_str())) << configXml;
    std::vector<std::string> out;
    for (auto bg : doc.child("MultiAssetSimulation").children("Background")) {
        out.emplace_back(bg.attribute("path").as_string());
    }
    return out;
}

std::string slurp(const fs::path& path)
{
    std::ifstream ifs{path};
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

}  // namespace

TEST(RunClassFiles, AFreshRunCopiesEachClassFileIntoItsRunDirectory)
{
    const MultiAssetRun run{"classfiles-fresh"};
    const auto source = run.root().parent_path() / "background.xml";
    (void)run.build();
    const auto copy = run.root() / "class_files" / "bg.xml";
    ASSERT_TRUE(fs::exists(copy)) << "no copy of the class file in the run directory";
    EXPECT_EQ(slurp(copy), slurp(source));
    EXPECT_EQ(backgroundPaths(run.root() / "config.xml"), std::vector<std::string>{fs::absolute(copy).string()});
}

TEST(RunClassFiles, AResumeReadsTheRunsCopyWhateverTheSourceHoldsByThen)
{
    const MultiAssetRun run{"classfiles-resume"};
    const auto source = run.root().parent_path() / "background.xml";
    run.build()->run();
    const auto ckpt = run.latestCheckpoint();
    ASSERT_FALSE(ckpt.empty()) << "the run wrote no checkpoint";
    const auto started = slurp(run.root() / "class_files" / "bg.xml");

    fs::remove(source);
    std::unique_ptr<Orchestrator> resumed;
    ASSERT_NO_THROW(resumed = Orchestrator::fromCheckpoint(ckpt.string())) << "the resume read the source file";
    EXPECT_EQ(slurp(run.root() / "class_files" / "bg.xml"), started) << "the restore rewrote the run's copy";
    EXPECT_EQ(backgroundPaths(run.root() / "config.xml"),
              std::vector<std::string>{fs::absolute(run.root() / "class_files" / "bg.xml").string()});
    resumed->run();
}

// A run's saved config.xml carries its id, so a fresh run from it opens the same run directory, where each class
// file it names already is the run's copy.
TEST(RunClassFiles, AFreshRunFromARunsSavedConfigKeepsTheRunsCopies)
{
    const MultiAssetRun run{"classfiles-saved"};
    (void)run.build();
    const auto copy = run.root() / "class_files" / "bg.xml";
    const auto started = slurp(copy);
    ASSERT_FALSE(started.empty());

    std::unique_ptr<Orchestrator> again;
    ASSERT_NO_THROW(again = Orchestrator::fromConfig(run.root() / "config.xml", run.root().parent_path()))
        << "a fresh run from the run's saved config could not take its class files";
    EXPECT_EQ(slurp(copy), started) << "the fresh run rewrote the copy";
    EXPECT_EQ(backgroundPaths(run.root() / "config.xml"), std::vector<std::string>{fs::absolute(copy).string()});
}

TEST(RunClassFiles, ARunAnOlderEngineStartedResumesOnTheFilesItsConfigNames)
{
    const MultiAssetRun run{"classfiles-older"};
    const auto source = fs::absolute(run.root().parent_path() / "background.xml");
    run.build()->run();
    const auto ckpt = run.latestCheckpoint();
    ASSERT_FALSE(ckpt.empty()) << "the run wrote no checkpoint";

    // as an older engine left it: config.xml names the source file, and there is no copy
    fs::remove_all(run.root() / "class_files");
    pugi::xml_document doc;
    ASSERT_TRUE(doc.load_file((run.root() / "config.xml").c_str()));
    for (auto bg : doc.child("MultiAssetSimulation").children("Background")) {
        bg.attribute("path").set_value(source.c_str());
    }
    ASSERT_TRUE(doc.save_file((run.root() / "config.xml").c_str()));

    (void)Orchestrator::fromCheckpoint(ckpt.string());
    EXPECT_EQ(backgroundPaths(run.root() / "config.xml"), std::vector<std::string>{source.string()});
    EXPECT_FALSE(fs::exists(run.root() / "class_files")) << "a restore copied the files an older run names";
}

// THE PUBLISHED CLASSES LOAD IN THE ENGINE: the four-class acceptance layout names every class's
// acceptance copy, which differs from the published class file only in its debug switches, so the engine building it
// reads all four markets, simulation_3 included. Skipped where run/config/ is reduced (the public tree).
//
// OFF THE LIVE QUEUES. An orchestrator built from a message-pack layout opens the validator queues under their default
// names (/taosim-req, /taosim-res) and unlinks them on teardown: built from the shipped files as they are, this test
// removed a running local simulator's queues and stalled its validator. It builds from copies with
// useMessagePack="0", so it opens none, and checks that the queues on this machine are as it found them.
TEST(RunClassFiles, TheFourClassAcceptanceLayoutLoadsEveryClass)
{
    const auto configDir = fs::path{__FILE__}.parent_path().parent_path().parent_path() / "run" / "config";
    const auto layout = configDir / "multiasset_acceptance_64_4.xml";
    if (!fs::exists(layout)) GTEST_SKIP() << "run/config/ is reduced here; the acceptance layouts are not distributed";
    const auto base = fs::temp_directory_path() / fmt::format("taosim-four-class-layout-{}", ::getpid());
    fs::remove_all(base);
    fs::create_directories(base);
    const auto offTheQueues = [](const fs::path& from, const fs::path& to) {
        pugi::xml_document doc;
        ASSERT_TRUE(doc.load_file(from.c_str())) << from;
        for (auto node = doc.first_child(); node; node = node.next_sibling()) {
            if (node.type() == pugi::node_element && node.attribute("useMessagePack")) {
                node.attribute("useMessagePack").set_value("0");
            }
        }
        ASSERT_TRUE(doc.save_file(to.c_str())) << to;
    };
    pugi::xml_document wrapper;
    ASSERT_TRUE(wrapper.load_file(layout.c_str()));
    offTheQueues(layout, base / layout.filename());
    for (const auto& bg : wrapper.document_element().children("Background")) {
        offTheQueues(configDir / bg.attribute("path").value(), base / bg.attribute("path").value());
    }
    const auto queues = [] {
        return std::vector<bool>{fs::exists("/dev/mqueue/taosim-req"), fs::exists("/dev/mqueue/taosim-res")};
    };
    const auto before = queues();
    std::unique_ptr<Orchestrator> orchestrator;
    ASSERT_NO_THROW(orchestrator = Orchestrator::fromConfig(base / layout.filename(), base))
        << "the engine could not build the four-class layout from its class files";
    ASSERT_NE(orchestrator, nullptr);
    orchestrator.reset();
    EXPECT_EQ(queues(), before) << "the test created or removed this machine's validator queues";
    fs::remove_all(base);
}

