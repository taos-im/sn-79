add_library(util STATIC
    AgentReset.cpp
    Cancellation.cpp
    ClosePosition.cpp
    GBMValuationModel.hpp
    Order.cpp
    Timestamp.hpp
    Trade.cpp
    common.hpp
    util.cpp
    Flags.hpp
    InstructionLogger.cpp
)
add_library(taosim::util ALIAS util)

target_include_directories(util
    PUBLIC
        include
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SRC_DIR}/accounting"
        "${SRC_DIR}/agent"
        "${SRC_DIR}/base"
        "${SRC_DIR}/exchange"
        "${SRC_DIR}/mp"
        "${SRC_DIR}/serialization"
        "${SRC_DIR}/simulation/include"
)

target_link_libraries(util
    PUBLIC
        taosim::book
        taosim::decimal
        taosim::event
        taosim::message
        Boost::boost
        fmt::fmt
        taosim::serialization
)
