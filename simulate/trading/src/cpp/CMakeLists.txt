add_subdirectory(accounting)
add_subdirectory(agent)
add_subdirectory(book)
add_subdirectory(base)
add_subdirectory(checkpoint)
add_subdirectory(decimal)
add_subdirectory(dsa)
add_subdirectory(event)
add_subdirectory(exchange)
add_subdirectory(matching)
add_subdirectory(filesystem)
add_subdirectory(fp)
add_subdirectory(ipc)
add_subdirectory(logging)
add_subdirectory(message)
add_subdirectory(mp)
add_subdirectory(net)
add_subdirectory(process)
add_subdirectory(replay)
add_subdirectory(serialization)
add_subdirectory(simulation)
add_subdirectory(stats)
add_subdirectory(statshub)
add_subdirectory(util)
add_subdirectory(xml)

#-------------------------------------------------------------------------

add_library(simulator STATIC Simulation.cpp)
add_library(taosim::simulator ALIAS simulator)

target_include_directories(simulator
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        accounting
        agent
        book
        base
        decimal
        exchange
        matching
        mp
        serialization
        stats
        util
)

target_link_libraries(simulator
    PUBLIC
        taosim::accounting
        taosim::agent
        taosim::book
        taosim::base
        taosim::checkpoint
        taosim::decimal
        taosim::event
        taosim::exchange
        taosim::matching
        taosim::filesystem
        taosim::fp
        taosim::ipc
        taosim::logging
        taosim::message
        taosim::mp
        taosim::net
        taosim::process
        taosim::replay
        taosim::serialization
        taosim::simulation
        taosim::stats
        taosim::util
        taosim::xml
        Boost::boost
        CLI11::CLI11
        bdl
        date::date date::date-tz
        fmt::fmt
        mimalloc
        pybind11::embed
)

#-------------------------------------------------------------------------

add_executable(${PROJECT_NAME}
    TheSimulatorModule.cpp
    main.cpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        $<TARGET_PROPERTY:taosim::simulator,INCLUDE_DIRECTORIES>
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        taosim::simulator
        ${Python3_LIBRARIES}
)

target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        $<$<BOOL:${OVERRIDE_NEW_DELETE}>:OVERRIDE_NEW_DELETE>
)

#-------------------------------------------------------------------------
