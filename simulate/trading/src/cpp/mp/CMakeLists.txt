add_library(mp INTERFACE)
add_library(taosim::mp ALIAS mp)

target_include_directories(mp INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
