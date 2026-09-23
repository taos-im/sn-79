file(GLOB agent_sources CONFIGURE_DEPENDS "*.cpp")

add_library(agent STATIC ${agent_sources})
add_library(taosim::agent ALIAS agent)

target_include_directories(agent
    PUBLIC
        include
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SRC_DIR}"
        "${SRC_DIR}/base"
        "${SRC_DIR}/book"
        "${SRC_DIR}/decimal"
        "${SRC_DIR}/exchange"
        "${SRC_DIR}/mp"
        "${SRC_DIR}/net"
        "${SRC_DIR}/serialization"
)

find_package(TBB REQUIRED)

target_link_libraries(agent
    PUBLIC
        pugixml::static pugixml::pugixml
        taosim::accounting
        taosim::dsa
        taosim::event
        taosim::exchange
        taosim::message
        taosim::net
        taosim::process
        taosim::replay
        taosim::simulation
        taosim::stats
        taosim::statshub
        taosim::util
        Eigen3::Eigen
        date::date
        pybind11::embed
        TBB::tbb
)
