/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "util.hpp"

#include <taosim/xml/helpers.hpp>

#include <boost/algorithm/string.hpp>
#include <fmt/format.h>

#include <fstream>

//-------------------------------------------------------------------------

namespace taosim::util
{

//-------------------------------------------------------------------------

std::vector<std::string> split(std::string_view str, char delim) noexcept
{
    std::vector<std::string> res;
    boost::split(res, str, [delim](auto c) { return c == delim; });
    return res;
}

//-------------------------------------------------------------------------

std::vector<std::string> getLastLines( std::string const& filename, int lineCount ) noexcept
{
    size_t const granularity = 100 * lineCount;
    std::ifstream source( filename.c_str(), std::ios_base::binary );
    source.seekg( 0, std::ios_base::end );
    size_t size = static_cast<size_t>( source.tellg() );
    std::vector<char> buffer;
    int newlineCount = 0;
    while ( source 
            && buffer.size() != size
            && newlineCount < lineCount ) {
        buffer.resize( std::min( buffer.size() + granularity, size ) );
        source.seekg( -static_cast<std::streamoff>( buffer.size() ),
                      std::ios_base::end );
        source.read( buffer.data(), buffer.size() );
        newlineCount = std::count( buffer.begin(), buffer.end(), '\n');
    }
    std::vector<char>::iterator start = buffer.begin();
    while ( newlineCount > lineCount ) {
        start = std::find( start, buffer.end(), '\n' ) + 1;
        -- newlineCount;
    }
    std::vector<char>::iterator end = buffer.end();
    auto lines = std::string( start, end );
    return split(lines,'\n');
}

double getClosestPreviousEntry(const std::string& filename, long long currentTimestamp) {
    std::ifstream file(filename);
    std::string line;
    double closestRet =  0.0;
    long long closestDiff = std::numeric_limits<long long>::max();

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        try {
            
        // Get timestamp
        std::getline(ss, token, ',');
        long long timestamp = std::stoll(token);

        // Get value
        std::getline(ss, token, ',');
        double value = std::stod(token);

        // Only consider timestamps <= currentTimestamp
        if (timestamp <= currentTimestamp) {
            long long diff = currentTimestamp - timestamp;
            if (diff < closestDiff) {
                closestRet = value;
                closestDiff = diff;
            }
        }
        } catch (...) {
        }
    }

    return closestRet;
}

//-------------------------------------------------------------------------

Nodes parseSimulationFile(const fs::path& path)
{
    pugi::xml_document doc;
    pugi::xml_parse_result parseResult = doc.load_file(path.c_str());
    assert(parseResult);

    pugi::xml_node simulationNode = xml::rootNode(doc);
    pugi::xml_attribute start = simulationNode.attribute("start");
    pugi::xml_attribute duration = simulationNode.attribute("duration");
    assert(simulationNode && start && duration);

    pugi::xml_node exchangeNode = simulationNode.child("Agents").child("MultiBookExchangeAgent");
    assert(exchangeNode && exchangeNode.attribute("remoteAgentCount").as_uint() > 0);

    pugi::xml_node booksNode = exchangeNode.child("Books");
    assert(booksNode);
    assert(booksNode.attribute("instanceCount").as_uint() > 0);
    assert(std::string_view{booksNode.attribute("algorithm").as_string()} == "PriceTime");

    pugi::xml_node balancesNode = exchangeNode.child("Balances");
    assert(balancesNode);

    pugi::xml_node baseNode = balancesNode.child("Base");
    pugi::xml_node quoteNode = balancesNode.child("Quote");
    assert(baseNode && baseNode.attribute("total") && quoteNode && quoteNode.attribute("total"));

    return {.doc = std::move(doc), .simulation = simulationNode, .exchange = exchangeNode};
}

//-------------------------------------------------------------------------

}  // namespace taosim::util

//-------------------------------------------------------------------------