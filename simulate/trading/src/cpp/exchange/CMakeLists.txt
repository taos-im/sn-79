add_library(exchange STATIC
    ClearingManager.cpp
    Exchange.cpp
    ExchangeConfig.cpp
    ExchangeSignals.cpp
    FeePolicyWrapper.cpp
    OrderPlacementValidator.cpp
    FeePolicy.cpp
)
add_library(taosim::exchange ALIAS exchange)

target_compile_options(exchange PRIVATE ${internal_compile_options})
target_compile_definitions(exchange PRIVATE ${internal_compile_definitions})
target_link_options(exchange PRIVATE ${internal_link_options})

target_include_directories(exchange
    PUBLIC
        include
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${src_dir}"
        "${src_dir}/accounting"
        "${src_dir}/agent"
        "${src_dir}/book"
        "${src_dir}/base"
        "${src_dir}/decimal"
        "${src_dir}/exchange"
        "${src_dir}/message"
        "${src_dir}/mp"
        "${src_dir}/net"
        "${src_dir}/process"
        "${src_dir}/serialization"
        "${src_dir}/util"
)

target_link_libraries(exchange
    PUBLIC
        taosim::accounting
        taosim::book
        bdl
        pybind11::embed
)
