/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/BookProcessManager.hpp>
#include <taosim/book/serialization/UpdateCounter.hpp>
#include <taosim/process/FundamentalPrice.hpp>
#include <taosim/process/FuturesSignal.hpp>
#include <taosim/checkpoint/serialization/process/FundamentalPrice.hpp>
#include <taosim/checkpoint/serialization/process/FuturesSignal.hpp>
#include <taosim/checkpoint/serialization/process/MagneticField.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::book::BookProcessManager>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::book::BookProcessManager& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        auto convertProcesses = [&](const msgpack::object& o) {
            if (o.type != msgpack::type::MAP) {
                throw taosim::serialization::MsgPackError{};
            }
            for (const auto& [k, val] : o.via.map) {
                const auto key = k.as<std::string>();
                const auto& arr = val.via.array;

                auto& bookId2Process = v.container().at(key);

                for (size_t bookId{}; bookId < arr.size; ++bookId) {
                    auto* const p = bookId2Process.at(bookId).get();
                    const auto& obj = arr.ptr[bookId];
                    // Nil = packed without an adaptor; keep the constructed state.
                    // Type-dispatched to mirror the pack side (see there).
                    if (obj.is_nil()) continue;
                    if (auto* fp = dynamic_cast<taosim::process::FundamentalPrice*>(p)) {
                        obj.convert(*fp);
                    }
                    else if (auto* fs = dynamic_cast<taosim::process::FuturesSignal*>(p)) {
                        obj.convert(*fs);
                    }
                    else if (auto* mf = dynamic_cast<taosim::process::MagneticField*>(p)) {
                        obj.convert(*mf);
                    }
                }
            }
        };

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "processes") {
                convertProcesses(val);
            }
            else if (key == "updateCounters") {
                using T = std::remove_cvref_t<decltype(v.updateCounters())>;
                v.updateCounters() = val.as<T>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::book::BookProcessManager>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::book::BookProcessManager& v) const
    {
        o.pack_map(2);

        o.pack("processes");
        o.pack_map(v.container().size());

        for (auto&& [name, bookId2Process] : v.container()) {
            o.pack(name);
            o.pack_array(bookId2Process.size());

            for (const auto& process : bookId2Process) {
                auto* const proc = process.get();

                // Dispatch on the DYNAMIC type, never the configured name: any process
                // type can be registered under any name (e.g. a JumpDiffusion named
                // "fundamental"), and a name-assumed cast dereferences null.
                if (auto* fp = dynamic_cast<taosim::process::FundamentalPrice*>(proc)) {
                    o.pack(*fp);
                }
                else if (auto* fs = dynamic_cast<taosim::process::FuturesSignal*>(proc)) {
                    o.pack(*fs);
                }
                else if (auto* mf = dynamic_cast<taosim::process::MagneticField*>(proc)) {
                    o.pack(*mf);
                }
                else {
                    // No checkpoint adaptor for this process type (e.g. JumpDiffusion):
                    // pack nil so the checkpoint stays loadable; restore then keeps the
                    // freshly-constructed process state for it.
                    o.pack_nil();
                }
            }
        }

        o.pack("updateCounters");
        o.pack(v.updateCounters());

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
