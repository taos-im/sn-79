find_package(GTest CONFIG REQUIRED)

set(TEST_SOURCES "" CACHE INTERNAL "List of all collected test compilation units")

include(CollectTestSources)

add_subdirectory(test-common)
add_subdirectory(accounting)
add_subdirectory(agent)
add_subdirectory(decimal)
add_subdirectory(fp)
add_subdirectory(message)
add_subdirectory(serialization)
add_subdirectory(util)

add_executable(taosim-tests
    ${TEST_SOURCES}
    UtilTests.cpp
    ChainAmountPrecisionTests.cpp
    RequestFingerprintTests.cpp
    SelfTradePreventionTests.cpp
    PriceBandTests.cpp
    SweepInitiatorTests.cpp
    QuoteSharingTests.cpp
    SpecialOrderTypeTests.cpp
    SLTPContainerTests.cpp
    QuoteBasedOrderTests.cpp
    LoanSettlementTests.cpp
    L3FmtTests.cpp
    L3SignalWiringTests.cpp
    StatsHubTests.cpp
    WakeupChainTests.cpp
    BarSeriesTests.cpp
    StylizedTraderBarsTests.cpp
    HighFrequencyQuoteCycleTests.cpp
    MinQuoteOrderSizeTests.cpp
    BookAggregateInvariantTests.cpp
    ReferenceConfigTests.cpp
    BlockBookIdTests.cpp
    WakeupPayloadContractTests.cpp
    WakeupCheckpointRoundTripTests.cpp
    FuturesTraderActivityTests.cpp
)
# Some test sources are not present in every distribution of this tree; guard on existence so the
# target always configures.
foreach(_xs_test QuoteLimitPoolSizingTests.cpp ReconciliationDecodeTests.cpp)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_xs_test})
        target_sources(taosim-tests PRIVATE ${_xs_test})
    endif()
endforeach()

target_include_directories(taosim-tests
    PRIVATE
        "${src_dir}/net"
        "${src_dir}/util"
        test-common
)

target_link_libraries(taosim-tests
    PRIVATE
        GTest::gmock_main
        taosim_tests::common
        taosim::simulator
)

add_test(Tests taosim-tests)
