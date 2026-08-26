/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/ipc/PosixMessageQueue.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <msgpack.hpp>

#include <fmt/format.h>

#include <chrono>
#include <concepts>
#include <cstring>
#include <exception>
#include <functional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

//-------------------------------------------------------------------------
// Synchronous, request/response RPC over POSIX shared memory with
// msgpack-encoded payloads. Two roles (`MsgPackServer<Req, Res>` and
// `MsgPackClient<Req, Res>`) talk through three named primitives in
// each direction: a `PosixMessageQueue` carrying the next message's
// byte length, a `shared_memory_object` carrying the payload, and an
// optional `named_semaphore` for belt-and-braces synchronization
// (POSIX mq operations are already memory-sync points, so the semaphore
// is conservative: leave its name empty in `IpcChannelDesc` to omit).
//
// Wire protocol (server view of one iteration):
//   1. blocking-receive `size_t` on `reqMq`
//   2. (optional) wait on `reqSem`
//   3. open `reqShm`, read `size_t` bytes, msgpack-decode to `Request`
//   4. invoke handler, msgpack-encode result into a `HumanReadableStream`
//   5. open/truncate `resShm`, copy stream bytes in
//   6. (optional) post `resSem`
//   7. send `size_t` (response size) on `resMq`
//
// Client mirrors steps in reverse.
//
// Ownership of the named primitives: dtors only close per-process
// handles (mq_close, munmap, file descriptor release). They do NOT
// `*_unlink` the names — those are filesystem-style persistent objects
// shared across processes, and unlinking them out from under a peer
// (e.g. our simulator yanking shm that an external validator still
// has open) is the cross-process foot-gun we want to avoid. Call the
// static `cleanup(desc)` explicitly when you want the names gone;
// signal handlers are the usual driver.

namespace taosim::ipc
{

namespace bipc = boost::interprocess;

//-------------------------------------------------------------------------

struct IpcChannelDesc
{
    std::string reqMqName;
    std::string resMqName;
    std::string reqShmName;
    std::string resShmName;
    // Optional belt-and-braces synchronization atop the mq's POSIX
    // memory-sync guarantees. Set to empty to disable; both ends must
    // agree on whether to use them.
    std::string reqSemName;
    std::string resSemName;
};

//-------------------------------------------------------------------------

// Per-cycle timing + size breakdown handed to a server's `onServed` hook.
// Durations cover only the server-side work; the blocking wait for the next
// request is excluded. `publish` spans the response-shm truncate+copy, the
// optional semaphore post, and the size-frame send.
struct ServeTiming
{
    std::chrono::nanoseconds unpack{};   // request bytes available -> decoded
    std::chrono::nanoseconds handle{};   // handler invocation
    std::chrono::nanoseconds pack{};     // msgpack-encode of the response
    std::chrono::nanoseconds publish{};  // shm copy + sem post + size send
    size_t reqBytes{};
    size_t resBytes{};
    bool dropped{};                      // request undecodable or handler threw
};

// Optional hooks for `MsgPackServer::serveOne`. A default-constructed value
// preserves the legacy behavior: a decode/handler failure propagates out, and
// there is no post-cycle observer.
template<typename Request, typename Response>
struct ServeHooks
{
    // Produce a fallback response when the request cannot be decoded or the
    // handler throws. Answering anyway keeps the request/response handshake in
    // lock-step instead of orphaning a peer that is blocked awaiting a reply.
    // When unset, `serveOne` rethrows (the caller's loop owns the failure).
    std::function<Response(std::exception_ptr)> onError;
    // Invoked once the response has been published. `request` points at the
    // decoded request, or is nullptr when decoding failed. Must not throw.
    std::function<
        void(const Request* request, const Response& response, const ServeTiming&)> onServed;
    // Produce the response for a request whose BYTES are identical to the immediately preceding one,
    // WITHOUT running the handler. Opt-in: when unset there is no duplicate detection at all and the
    // handler sees every request, which is what every other user of this channel wants.
    //
    // WHY BYTES AND NOT A FIELD. A response timeout on the peer side once caused the same request to
    // be delivered twice; the handler had no idea and executed it twice, so a single order minted two
    // trades with the same engine timestamp and the second never settled. The peer-side fix stops
    // provoking it but does not make it impossible. The request's own step counter looked like the
    // natural key and is NOT unique per request: the peer sends a reconciliation and an instruction
    // batch under the same step, so guarding on it would silently drop every second request. The
    // packed bytes are unique per request by construction, because a re-send transmits the identical
    // buffer while any genuinely new request differs in at least its instruction sequence numbers.
    std::function<Response()> onDuplicate;
};

//-------------------------------------------------------------------------

namespace detail
{

// FNV-1a over the request bytes. Dependency-free and fast on a few hundred kilobytes; this is a
// duplicate-delivery discriminator, not a security primitive, and the two cases it must separate are
// "the identical buffer again" and "a different buffer". Returns non-zero for any non-empty input, so
// zero can mean "nothing served yet".
inline std::uint64_t hashRequestBytes(const void* data, std::size_t size) noexcept
{
    if (data == nullptr || size == 0) {
        return 0u;
    }
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < size; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= 1099511628211ull;
    }
    return h == 0u ? 1u : h;
}

// Pulls the next request: blocks on the req-mq for the byte count,
// optionally drains the req-sem, then opens and maps the req-shm.
// Returns the mapped region (keep it alive while reading) and the
// payload size as written by the client.
struct ReceivedShmPayload
{
    bipc::shared_memory_object shm;
    bipc::mapped_region region;
    size_t size{};
};

[[nodiscard]] inline ReceivedShmPayload waitForRequest(
    const IpcChannelDesc& desc, PosixMessageQueue& reqMq)
{
    size_t payloadSize{};
    reqMq.blockingReceive(std::span<char>{
        reinterpret_cast<char*>(&payloadSize), sizeof(payloadSize)});

    if (!desc.reqSemName.empty()) {
        bipc::named_semaphore sem{bipc::open_or_create, desc.reqSemName.c_str(), 0};
        sem.wait();
    }

    bipc::shared_memory_object shm{
        bipc::open_only, desc.reqShmName.c_str(), bipc::read_write};
    bipc::mapped_region region{shm, bipc::read_write};
    return {std::move(shm), std::move(region), payloadSize};
}

// Writes the response: opens (or creates) the res-shm at the right
// size, copies bytes in, optionally posts the res-sem, then signals
// the size on the res-mq.
inline void publishResponse(
    const IpcChannelDesc& desc, PosixMessageQueue& resMq, std::span<const char> bytes)
{
    bipc::shared_memory_object shm{
        bipc::open_or_create, desc.resShmName.c_str(), bipc::read_write};
    shm.truncate(bytes.size());
    bipc::mapped_region region{shm, bipc::read_write};
    std::memcpy(region.get_address(), bytes.data(), bytes.size());

    if (!desc.resSemName.empty()) {
        bipc::named_semaphore sem{bipc::open_or_create, desc.resSemName.c_str(), 0};
        sem.post();
    }

    const size_t payloadSize = bytes.size();
    resMq.send(std::span<const char>{
        reinterpret_cast<const char*>(&payloadSize), sizeof(payloadSize)});
}

}  // namespace detail

//-------------------------------------------------------------------------

template<
    serialization::MsgPackDeserializable Request,
    serialization::MsgPackSerializable Response>
class MsgPackServer
{
public:
    explicit MsgPackServer(const IpcChannelDesc& desc);
    ~MsgPackServer() noexcept = default;

    MsgPackServer(const MsgPackServer&) = delete;
    MsgPackServer& operator=(const MsgPackServer&) = delete;
    MsgPackServer(MsgPackServer&&) = delete;
    MsgPackServer& operator=(MsgPackServer&&) = delete;

    // Process exactly one request → response cycle. With default hooks the
    // handler's exception propagates, leaving the caller's loop / signal
    // logic in charge of program lifetime. Supply `hooks.onError` to convert
    // a decode/handler failure into a fallback response (so the channel stays
    // in lock-step), and `hooks.onServed` to observe the per-cycle timing.
    // The handler constraint is checked in-body via `static_assert` for a
    // clearer diagnostic than a failed-substitution error from a class-level
    // `requires`.
    template<typename H>
    void serveOne(H&& handler, const ServeHooks<Request, Response>& hooks = {});

    template<typename H>
    void serve(H&& handler, const ServeHooks<Request, Response>& hooks = {});

    [[nodiscard]] const IpcChannelDesc& desc() const noexcept { return m_desc; }

    // Static so signal handlers (which can't easily reach into member
    // state) can pass a captured-on-the-side desc and call this from
    // async-signal-safe context. All four underlying removes call
    // `*_unlink(3)` variants which are async-signal-safe.
    static void cleanup(const IpcChannelDesc& desc) noexcept;

private:
    IpcChannelDesc m_desc;
    PosixMessageQueue m_reqMq;
    PosixMessageQueue m_resMq;
    // Fingerprint of the previous request's bytes, for `ServeHooks::onDuplicate`. Zero means "nothing
    // served yet", which no real request collides with because an empty request is never sent.
    std::uint64_t m_lastReqHash{};
};

//-------------------------------------------------------------------------

template<
    serialization::MsgPackDeserializable Request,
    serialization::MsgPackSerializable Response>
MsgPackServer<Request, Response>::MsgPackServer(const IpcChannelDesc& desc)
    : m_desc{desc},
      m_reqMq{{.name = m_desc.reqMqName}},
      m_resMq{{.name = m_desc.resMqName}}
{}

template<
    serialization::MsgPackDeserializable Request,
    serialization::MsgPackSerializable Response>
template<typename H>
void MsgPackServer<Request, Response>::serveOne(
    H&& handler, const ServeHooks<Request, Response>& hooks)
{
    static_assert(std::invocable<H, const Request&>);
    static_assert(std::convertible_to<std::invoke_result_t<H, const Request&>, Response>);

    using Clock = std::chrono::steady_clock;

    // Blocks until the next request's bytes are mapped; the wait itself is
    // peer-side idle time and is deliberately excluded from the timing below.
    auto received = detail::waitForRequest(m_desc, m_reqMq);
    const auto t0 = Clock::now();

    ServeTiming timing;
    timing.reqBytes = received.size;

    Request req;
    const Request* reqPtr = nullptr;
    auto tUnpack = t0;
    auto tHandle = t0;

    // Computed before decoding, so a duplicate costs a hash rather than a decode plus an execution.
    const std::uint64_t reqHash = hooks.onDuplicate
        ? detail::hashRequestBytes(received.region.get_address(), received.size)
        : 0u;
    const bool isDuplicate = hooks.onDuplicate && reqHash != 0u && reqHash == m_lastReqHash;

    Response res = [&]() -> Response {
        try {
            msgpack::object_handle oh = msgpack::unpack(
                static_cast<const char*>(received.region.get_address()), received.size);
            oh.get().convert(req);
            reqPtr = &req;
            tUnpack = Clock::now();
            if (isDuplicate) {
                // Answer, but do not execute. Answering keeps the request/response handshake in lock-step
                // (the peer is blocked awaiting a reply); not executing is the whole point.
                Response r = hooks.onDuplicate();
                tHandle = Clock::now();
                return r;
            }
            Response r = handler(static_cast<const Request&>(req));
            tHandle = Clock::now();
            return r;
        }
        catch (...) {
            if (!hooks.onError) {
                throw;
            }
            timing.dropped = true;
            const auto tErr = Clock::now();
            // Attribute the failed work to the stage that owned it and keep the
            // remaining stage durations non-negative.
            if (reqPtr == nullptr) {
                tUnpack = tErr;  // decode failed before a request existed
            }
            tHandle = tErr;
            return hooks.onError(std::current_exception());
        }
    }();

    const auto tResReady = Clock::now();
    serialization::HumanReadableStream stream;
    msgpack::pack(stream, res);
    const auto tPack = Clock::now();
    timing.resBytes = stream.size();

    detail::publishResponse(m_desc, m_resMq,
        std::span<const char>{stream.data(), stream.size()});
    const auto tPublish = Clock::now();

    timing.unpack = tUnpack - t0;
    timing.handle = tHandle - tUnpack;
    timing.pack = tPack - tResReady;
    timing.publish = tPublish - tPack;

    // Recorded after the cycle, so a request that threw before answering does not mark itself served and
    // suppress its own legitimate retry.
    if (reqHash != 0u) {
        m_lastReqHash = reqHash;
    }

    if (hooks.onServed) {
        hooks.onServed(reqPtr, res, timing);
    }
}

template<
    serialization::MsgPackDeserializable Request,
    serialization::MsgPackSerializable Response>
template<typename H>
void MsgPackServer<Request, Response>::serve(
    H&& handler, const ServeHooks<Request, Response>& hooks)
{
    for (;;) {
        serveOne(handler, hooks);
    }
}

template<
    serialization::MsgPackDeserializable Request,
    serialization::MsgPackSerializable Response>
void MsgPackServer<Request, Response>::cleanup(const IpcChannelDesc& desc) noexcept
{
    if (!desc.reqSemName.empty()) {
        bipc::named_semaphore::remove(desc.reqSemName.c_str());
    }
    if (!desc.resSemName.empty()) {
        bipc::named_semaphore::remove(desc.resSemName.c_str());
    }
    PosixMessageQueue::remove(desc.reqMqName);
    PosixMessageQueue::remove(desc.resMqName);
    bipc::shared_memory_object::remove(desc.reqShmName.c_str());
    bipc::shared_memory_object::remove(desc.resShmName.c_str());
}

//-------------------------------------------------------------------------

// Both template parameters are unconstrained at the class level so
// `MsgPackClient<Fwd>` can be instantiated against a forward-declared
// Request type (otherwise the `MsgPackSerializable` concept would
// require Request's pack adaptor — and thus its full definition —
// at every site holding a `MsgPackClient` member, which forces
// awkward header cycles). The actual concept checks live inside
// `exchange` and `call` as `static_assert`s; Response also defaults
// to `void` for the raw-bytes-only use case.

template<typename Request, typename Response = void>
class MsgPackClient
{
public:
    explicit MsgPackClient(IpcChannelDesc desc);
    ~MsgPackClient() noexcept = default;

    MsgPackClient(const MsgPackClient&) = delete;
    MsgPackClient& operator=(const MsgPackClient&) = delete;
    MsgPackClient(MsgPackClient&&) = delete;
    MsgPackClient& operator=(MsgPackClient&&) = delete;

    // Send one request, block for the matching response. Throws on
    // msgpack decode failure (`msgpack::type_error`); mq timeouts are
    // surfaced by the underlying `PosixMessageQueue`.
    [[nodiscard]] Response call(const Request& req);

    // Raw-bytes escape hatch: pack the request, do the shm/mq dance,
    // hand the response bytes to `consumer` (a callable taking
    // `std::span<const char>`). Use this when the response shape is
    // too dynamic to express as a single `Response` type — for
    // example a map whose value is iterated manually with
    // `msgpack::object`. The bytes view is valid only inside the
    // callback; the mapping is torn down on return. The consumer
    // constraint is checked in-body via `static_assert` for the same
    // diagnostic-clarity reasons as `MsgPackServer::serveOne`.
    //
    // `Stream` defaults to `HumanReadableStream`; override with e.g.
    // `BinaryStream` when the Request's pack adaptor branches on
    // stream tag for a different on-wire encoding.
    template<
        serialization::MsgPackOutputStream Stream = serialization::HumanReadableStream,
        typename Consumer>
    void exchange(const Request& req, Consumer&& consumer);

    // Same shape but caller hands in pre-packed bytes. Useful when
    // the caller already produced a packed stream for other purposes
    // (e.g. measuring its size for a step-timing line) and doesn't
    // want this method to re-pack. The byte buffer must outlive the
    // call.
    template<typename Consumer>
    void exchange(std::span<const char> reqBytes, Consumer&& consumer);

    [[nodiscard]] const IpcChannelDesc& desc() const noexcept { return m_desc; }

private:
    IpcChannelDesc m_desc;
    PosixMessageQueue m_reqMq;
    PosixMessageQueue m_resMq;
};

//-------------------------------------------------------------------------

template<typename Request, typename Response>
MsgPackClient<Request, Response>::MsgPackClient(IpcChannelDesc desc)
    : m_desc{std::move(desc)},
      m_reqMq{{.name = m_desc.reqMqName}},
      m_resMq{{.name = m_desc.resMqName}}
{
    // Start from a clean slate: a previous run (or one interrupted mid-call)
    // can leave a stale length frame in either persistent queue, which would
    // knock the very first request/response round trip out of lock-step. We
    // drain rather than unlink — unlinking would orphan a server already
    // holding these names. The server drains on its side symmetrically.
    m_reqMq.flush();
    m_resMq.flush();
}

template<typename Request, typename Response>
template<serialization::MsgPackOutputStream Stream, typename Consumer>
void MsgPackClient<Request, Response>::exchange(const Request& req, Consumer&& consumer)
{
    static_assert(serialization::MsgPackSerializable<Request, Stream>);

    Stream stream;
    msgpack::pack(stream, req);
    exchange(
        std::span<const char>{stream.data(), stream.size()},
        std::forward<Consumer>(consumer));
}

template<typename Request, typename Response>
template<typename Consumer>
void MsgPackClient<Request, Response>::exchange(
    std::span<const char> reqBytes, Consumer&& consumer)
{
    static_assert(std::invocable<Consumer, std::span<const char>>);

    bipc::shared_memory_object reqShm{
        bipc::open_or_create, m_desc.reqShmName.c_str(), bipc::read_write};
    reqShm.truncate(reqBytes.size());
    bipc::mapped_region reqRegion{reqShm, bipc::read_write};
    std::memcpy(reqRegion.get_address(), reqBytes.data(), reqBytes.size());

    if (!m_desc.reqSemName.empty()) {
        bipc::named_semaphore sem{bipc::open_or_create, m_desc.reqSemName.c_str(), 0};
        sem.post();
    }

    const size_t reqSize = reqBytes.size();
    // mq operations are timed (see PosixMessageQueueDesc::timeout).
    // Flush before each send so a stale message from a previous run
    // doesn't block the queue; loop on timeout so a slow peer doesn't
    // drop the round-trip. Matches the existing retry shape in
    // SimulationManager::publishStateMessagePack.
    for (;;) {
        m_reqMq.flush();
        if (m_reqMq.send(std::span<const char>{
                reinterpret_cast<const char*>(&reqSize), sizeof(reqSize)})) {
            break;
        }
        fmt::println("MsgPackClient: send to /{} timed out, retrying...",
            m_desc.reqMqName);
    }

    size_t resSize{};
    for (;;) {
        if (m_resMq.receive(std::span<char>{
                reinterpret_cast<char*>(&resSize), sizeof(resSize)}) != -1) {
            break;
        }
        fmt::println("MsgPackClient: receive from /{} timed out, retrying...",
            m_desc.resMqName);
    }

    if (!m_desc.resSemName.empty()) {
        bipc::named_semaphore sem{bipc::open_or_create, m_desc.resSemName.c_str(), 0};
        sem.wait();
    }

    bipc::shared_memory_object resShm{
        bipc::open_only, m_desc.resShmName.c_str(), bipc::read_write};
    bipc::mapped_region resRegion{resShm, bipc::read_write};

    consumer(std::span<const char>{
        static_cast<const char*>(resRegion.get_address()), resSize});
}

template<typename Request, typename Response>
Response MsgPackClient<Request, Response>::call(const Request& req)
{
    static_assert(!std::is_void_v<Response>,
        "call() requires MsgPackClient to be instantiated with a non-void Response; "
        "use exchange() for the raw-bytes API instead");
    static_assert(serialization::MsgPackDeserializable<Response>,
        "Response must satisfy MsgPackDeserializable for call() to be available");

    Response res;
    exchange(req, [&](std::span<const char> bytes) {
        msgpack::object_handle oh = msgpack::unpack(bytes.data(), bytes.size());
        oh.get().convert(res);
    });
    return res;
}

//-------------------------------------------------------------------------

}  // namespace taosim::ipc

//-------------------------------------------------------------------------
