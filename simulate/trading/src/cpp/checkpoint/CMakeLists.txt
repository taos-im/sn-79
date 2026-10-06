file(GLOB sources CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE headers CONFIGURE_DEPENDS "include/*.hpp" "include/*.h")

add_library(checkpoint STATIC)
add_library(taosim::checkpoint ALIAS checkpoint)

target_sources(checkpoint PRIVATE ${sources})

target_sources(checkpoint
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES ${headers}
)

target_include_directories(checkpoint PUBLIC include)

target_link_libraries(checkpoint
    PRIVATE
        taosim::accounting
        taosim::agent
        taosim::book
        taosim::simulation
        fmt::fmt
)
