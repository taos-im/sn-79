file(GLOB sources CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE headers CONFIGURE_DEPENDS "include/*.hpp")

add_library(statshub STATIC)
add_library(taosim::statshub ALIAS statshub)

target_sources(statshub PRIVATE ${sources})

target_sources(statshub
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES ${headers}
)

target_link_libraries(statshub
    PUBLIC
        taosim::book
        taosim::decimal
        taosim::util
        Boost::boost
)
