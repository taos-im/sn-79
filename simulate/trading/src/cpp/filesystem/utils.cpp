/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/filesystem/utils.hpp>

#include <cerrno>
#include <fstream>
#include <ios>
#include <system_error>

//-------------------------------------------------------------------------

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace taosim::filesystem
{

//-------------------------------------------------------------------------

std::vector<fs::path> collectPaths(const fs::path& dir)
{
    return ranges::subrange(fs::directory_iterator{dir}, fs::directory_iterator{})
        | ranges::views::transform([](auto&& entry) { return entry.path(); })
        | ranges::to<std::vector>;
}

//-------------------------------------------------------------------------

void atomicWrite(const fs::path& path, std::span<const char> bytes)
{
    const fs::path tmp{path.native() + ".tmp"};

    auto discardTmp = [&] {
        std::error_code ignored;
        fs::remove(tmp, ignored);
    };

    try {
        std::ofstream out;
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out.open(tmp, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), std::ssize(bytes));
        // Explicit: a write error surfacing at close would be swallowed by the destructor.
        out.close();
    }
    catch (const std::ios_base::failure& e) {
        // The stream only reports a generic error; the OS reason is still in errno (best effort).
        const auto err = errno;
        discardTmp();
        throw fs::filesystem_error{
            "atomicWrite: writing the temp file failed",
            tmp,
            path,
            err != 0 ? std::error_code{err, std::generic_category()} : e.code()};
    }

    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        discardTmp();
        throw fs::filesystem_error{"atomicWrite: rename failed", tmp, path, ec};
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::filesystem

//-------------------------------------------------------------------------
