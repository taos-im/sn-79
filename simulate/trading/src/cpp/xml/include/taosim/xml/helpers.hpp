/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <pugixml.hpp>

#include <string_view>

//-------------------------------------------------------------------------

namespace taosim::xml
{

//-------------------------------------------------------------------------

template<typename T>
requires requires(pugi::xml_attribute attr, T val) {
    { attr.set_value(val) } -> std::same_as<bool>;
}
void setAttribute(pugi::xml_node node, std::string_view name, const T& value)
{
    if (auto attr = node.attribute(name.data())) {
        attr.set_value(value);
    } else {
        node.append_attribute(name.data()) = value;
    }
}

//-------------------------------------------------------------------------

[[nodiscard]] inline pugi::xml_node findChildByName(pugi::xml_node node, std::string_view needle)
{
    return node.find_child([needle](pugi::xml_node child) { return needle == child.name(); });
}

//-------------------------------------------------------------------------

// The document's root, under either accepted name.
//
// Exchange-mode configs are rooted at <Exchange> because calling an exchange deployment a "Simulation"
// misdescribes it, while simulation configs stay at <Simulation>. BOTH must keep working, and not only
// for the configs on disk: every checkpoint ever written carries the root name it was written under, so
// a reader that accepted only the new name would fail to load existing checkpoints.
[[nodiscard]] inline pugi::xml_node rootNode(pugi::xml_node doc)
{
    if (auto node = doc.child("Simulation")) return node;
    return doc.child("Exchange");
}

//-------------------------------------------------------------------------

inline size_t removeChildren(pugi::xml_node node, std::function<bool(pugi::xml_node)> criterion)
{
    size_t removeCounter{};
    auto child = node.first_child();
    while (child) {
        const auto nextChild = child.next_sibling();
        if (criterion(child)) {
            node.remove_child(child);
            ++removeCounter;
        }
        child = nextChild;
    }
    return removeCounter;
}

//-------------------------------------------------------------------------

}  // namespace taosim::xml

//-------------------------------------------------------------------------