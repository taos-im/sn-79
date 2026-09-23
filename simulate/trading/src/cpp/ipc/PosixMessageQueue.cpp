/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/ipc/PosixMessageQueue.hpp>

#include <taosim/ipc/util.hpp>

#include <fmt/format.h>

#include <source_location>
#include <stdexcept>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::ipc
{

//-------------------------------------------------------------------------

PosixMessageQueue::PosixMessageQueue(const PosixMessageQueueDesc& desc)
    : m_desc{desc}
{
    m_desc.name = desc.name.starts_with("/") ? desc.name : "/" + desc.name;

    m_handle = mq_open(
        m_desc.name.c_str(),
        m_desc.oflag,
        m_desc.mode,
        &m_desc.attr);

    if (m_handle == static_cast<mqd_t>(-1)) {
        throw std::runtime_error{fmt::format(
            "{}: Failed to create POSIX mqueue with name '{}': {} ({})",
            std::source_location::current().function_name(),
            m_desc.name, errno, std::strerror(errno)
        )};
    }
}

//-------------------------------------------------------------------------

PosixMessageQueue::~PosixMessageQueue() noexcept
{
    mq_close(m_handle);
    mq_unlink(m_desc.name.c_str());
}

//-------------------------------------------------------------------------

std::optional<size_t> PosixMessageQueue::size() const noexcept
{
    struct mq_attr attr;
    if (mq_getattr(m_handle, &attr) == -1) {
        return std::nullopt;
    }
    return std::make_optional(attr.mq_curmsgs);
}

//-------------------------------------------------------------------------

bool PosixMessageQueue::send(std::span<const char> msg, uint32_t priority) noexcept
{
    if (!m_desc.timeout) {
        return mq_send(m_handle, msg.data(), msg.size(), priority) == 0;
    }
    const timespec ts = makeTimespec(*m_desc.timeout);
    return mq_timedsend(m_handle, msg.data(), msg.size(), priority, &ts) == 0;
}

//-------------------------------------------------------------------------

ptrdiff_t PosixMessageQueue::receive(std::span<char> msg, uint32_t* priority) noexcept
{
    if (!m_desc.timeout) {
        return blockingReceive(msg, priority);
    }
    const timespec ts = makeTimespec(*m_desc.timeout);
    return mq_timedreceive(m_handle, msg.data(), msg.size(), priority, &ts);
}

//-------------------------------------------------------------------------

ptrdiff_t PosixMessageQueue::blockingReceive(std::span<char> msg, uint32_t* priority) noexcept
{
    return mq_receive(m_handle, msg.data(), msg.size(), priority);
}

//-------------------------------------------------------------------------

void PosixMessageQueue::flush() noexcept
{
    // Drain every pending message without blocking, so a freshly-started
    // process never inherits stale frames a previous run left in a persistent
    // queue. Size the sink from the queue's own attributes — a peer may have
    // created it with a larger msgsize than our desc, and a too-small buffer
    // would make mq_receive fail with EMSGSIZE forever. An already-elapsed
    // absolute timeout makes each receive non-blocking: it returns a pending
    // message immediately, or fails at once on an empty queue, ending the loop.
    struct mq_attr attr;
    if (mq_getattr(m_handle, &attr) == -1) return;
    std::vector<char> sink(static_cast<size_t>(attr.mq_msgsize));
    const timespec ts{};
    while (mq_timedreceive(m_handle, sink.data(), sink.size(), nullptr, &ts) != -1) {}
}

//-------------------------------------------------------------------------

bool PosixMessageQueue::remove(std::string_view name)
{
    return mq_unlink(name.starts_with('/') ? name.data() : fmt::format("/{}", name).c_str()) == 0;
}

//-------------------------------------------------------------------------

}  // namespace taosim::ipc

//-------------------------------------------------------------------------