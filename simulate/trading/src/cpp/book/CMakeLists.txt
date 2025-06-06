add_library(book STATIC
    Book.cpp
    BookFactory.hpp
    BookProcessLogger.cpp
    BookProcessManager.cpp
    BookSignals.hpp
    L2Logger.cpp
    L3EventLogger.cpp
    PriceTimeBook.cpp
    UpdateCounter.cpp
    FeeLogger.cpp
)
add_library(taosim::book ALIAS book)

target_compile_options(book PRIVATE ${internal_compile_options})
target_compile_definitions(book PRIVATE ${internal_compile_definitions})
target_link_options(book PRIVATE ${internal_link_options})

target_include_directories(book
    PUBLIC
        include
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${src_dir}"
        "${src_dir}/agent"
        "${src_dir}/base"
        "${src_dir}/message"
        "${src_dir}/mp"
        "${src_dir}/net"
        "${src_dir}/process"
        "${src_dir}/serialization"
        "${src_dir}/util"
)

target_link_libraries(book
    PRIVATE
        taosim::exchange
        taosim::process
        fmt::fmt
        pybind11::embed
)
