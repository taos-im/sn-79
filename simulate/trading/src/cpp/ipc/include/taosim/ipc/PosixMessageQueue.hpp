/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

extern "C" {
#include <fcntl.h>
#include <mqueue.h>
#include <sys/stat.h>
}

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>

//-------------------------------------------------------------------------

namespace taosim::ipc
{

//-------------------------------------------------------------------------

struct PosixMessageQueueDesc
{
    std::string name;
    int32_t oflag = O_CREAT | O_RDWR;
    mode_t mode = 0666;
    mq_attr attr = {
        .mq_flags = 0,
        .mq_maxmsg = 1,
        .mq_msgsize = sizeof(size_t),
        .mq_curmsgs = 0
    };
    std::optional<size_t> timeout{60'000'000'000};
};

//-------------------------------------------------------------------------

class PosixMessageQueue
{
public:
    explicit PosixMessageQueue(const PosixMessageQueueDesc& desc);
    ~PosixMessageQueue() noexcept;

    [[nodiscard]] mqd_t handle() const noexcept { return m_handle; }
    [[nodiscard]] std::optional<size_t> size() const noexcept;

    bool send(std::span<const char> msg, uint32_t priority = {}) noexcept;
    ptrdiff_t receive(std::span<char> msg, uint32_t* priority = {}) noexcept;
    ptrdiff_t blockingReceive(std::span<char> msg, uint32_t* priority = {}) noexcept;

    void flush() noexcept;

    static bool remove(std::string_view name);

private:
    mqd_t m_handle;
    PosixMessageQueueDesc m_desc;
};

//-------------------------------------------------------------------------

}  // namespace taosim::ipc

//-------------------------------------------------------------------------
