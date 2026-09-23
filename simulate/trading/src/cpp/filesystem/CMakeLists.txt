find_package(range-v3 CONFIG REQUIRED)

file(GLOB sources CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE headers CONFIGURE_DEPENDS "include/*.hpp")

add_library(filesystem STATIC)
add_library(taosim::filesystem ALIAS filesystem)

target_sources(filesystem PRIVATE ${sources})

target_sources(filesystem
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES ${headers}
)

target_link_libraries(filesystem
    PUBLIC
        range-v3::range-v3
)