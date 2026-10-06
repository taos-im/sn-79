file(GLOB sources CONFIGURE_DEPENDS "*.cpp" "multiasset/*.cpp")
add_library(simulation STATIC ${sources})
add_library(taosim::simulation ALIAS simulation)

target_include_directories(simulation
    PUBLIC
        include
        "${SRC_DIR}"
        "${SRC_DIR}/accounting"
        "${SRC_DIR}/accounting/include"
        "${SRC_DIR}/agent"
        "${SRC_DIR}/base"
        "${SRC_DIR}/book"
        "${SRC_DIR}/book/include"
        "${SRC_DIR}/decimal"
        "${SRC_DIR}/exchange"
        "${SRC_DIR}/exchange/include"
        "${SRC_DIR}/mp"
        "${SRC_DIR}/net"
        "${SRC_DIR}/util"
)

target_link_libraries(simulation
    PUBLIC
        taosim::agent
        taosim::checkpoint
        taosim::decimal
        taosim::dsa
        taosim::event
        taosim::filesystem
        taosim::ipc
        taosim::message
        taosim::net
        taosim::process
        taosim::replay
        taosim::serialization
        taosim::util
        taosim::xml
        Boost::boost
        pybind11::embed
)