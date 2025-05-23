add_library(agent STATIC
    Agent.cpp
    DelayDistribution.cpp
    ALGOTraderAgent.cpp
    DistributedProxyAgent.cpp
    ExchangeAgentConfig.cpp
    InitializationAgent.cpp
    LocalAgentManager.cpp
    LocalAgentRoster.cpp
    MultiBookExchangeAgent.cpp
    OrderLogAgent.cpp
    PythonAgent.cpp
    StylizedTraderAgent.cpp
    TradeLogAgent.cpp
    HighFrequencyTraderAgent.cpp
    RandomTraderAgent.cpp
)
add_library(taosim::agent ALIAS agent)

target_compile_options(agent PRIVATE ${internal_compile_options})
target_compile_definitions(agent PRIVATE ${internal_compile_definitions})
target_link_options(agent PRIVATE ${internal_link_options})

target_include_directories(agent
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${src_dir}"
        "${src_dir}/accounting"
        "${src_dir}/base"
        "${src_dir}/book"
        "${src_dir}/decimal"
        "${src_dir}/exchange"
        "${src_dir}/message"
        "${src_dir}/mp"
        "${src_dir}/net"
        "${src_dir}/process"
        "${src_dir}/serialization"
        "${src_dir}/stats"
        "${src_dir}/util"
)

target_link_libraries(agent
    PUBLIC
        pugixml::static pugixml::pugixml
    PRIVATE
        taosim::accounting
        taosim::exchange
        taosim::stats
        Eigen3::Eigen
        date::date
        pybind11::embed
)
