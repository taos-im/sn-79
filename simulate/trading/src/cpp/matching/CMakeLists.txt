file(GLOB sources CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE headers CONFIGURE_DEPENDS "include/*.hpp" "include/*.h")

add_library(matching STATIC)
add_library(taosim::matching ALIAS matching)

target_sources(matching PRIVATE ${sources})

target_sources(matching
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES ${headers}
)

target_include_directories(matching PUBLIC include)

target_link_libraries(matching
    PUBLIC
        taosim::accounting
        taosim::book
        taosim::decimal
        taosim::event
        taosim::logging
        taosim::message
        taosim::process
        taosim::replay
        taosim::serialization
        taosim::simulation
)
