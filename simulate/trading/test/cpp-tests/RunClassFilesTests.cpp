/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// A multi-asset run keeps the class files it started on. A fresh run copies each background's class file into its run
// directory and names the copies in the config.xml it persists there, so every resume reads the files the run started
// on, whatever the repository holds by then. A restore keeps the paths its config.xml names, so a run that an older
// engine started still resumes on the files it names.

#include "MultiAssetRunFixture.hpp"

#include <gtest/gtest.h>
#include <pugixml.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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
