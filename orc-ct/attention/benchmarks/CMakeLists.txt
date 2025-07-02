# CMakeLists.txt for attention benchmarks
# Copyright (C) 2024 OpenCog Foundation

# Add benchmark library
SET(ATTENTION_BENCHMARK_SOURCES
    attention_benchmark.cc
)

SET(ATTENTION_BENCHMARK_HEADERS
    attention_benchmark.h
)

# Create the benchmark library
ADD_LIBRARY(attention-benchmark SHARED
    ${ATTENTION_BENCHMARK_SOURCES}
)

# Link with required libraries
TARGET_LINK_LIBRARIES(attention-benchmark
    ${ATOMSPACE_LIBRARIES}
    ${COGUTIL_LIBRARY}
    attention
    attentionbank
    ${Boost_LIBRARIES}
)

# Set include directories
TARGET_INCLUDE_DIRECTORIES(attention-benchmark PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ATOMSPACE_INCLUDE_DIR}
    ${COGUTIL_INCLUDE_DIR}
)

# Install the library
INSTALL(TARGETS attention-benchmark
    LIBRARY DESTINATION "lib${LIB_DIR_SUFFIX}/opencog"
)

# Install headers
INSTALL(FILES ${ATTENTION_BENCHMARK_HEADERS}
    DESTINATION "include/opencog/attention/benchmarks"
)

# Add benchmark test executable (if building tests)
IF(CXXTEST_FOUND)
    ADD_EXECUTABLE(attention-benchmark-test
        ../tests/attention_benchmark_test.cc
    )
    
    TARGET_LINK_LIBRARIES(attention-benchmark-test
        attention-benchmark
        ${ATOMSPACE_LIBRARIES}
        ${COGUTIL_LIBRARY}
        ${Boost_LIBRARIES}
    )
    
    TARGET_INCLUDE_DIRECTORIES(attention-benchmark-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ATOMSPACE_INCLUDE_DIR}
        ${COGUTIL_INCLUDE_DIR}
    )
    
    # Add test target
    ADD_TEST(NAME AttentionBenchmarkTest
        COMMAND attention-benchmark-test 100 5
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    INSTALL(TARGETS attention-benchmark-test
        RUNTIME DESTINATION "bin"
    )
ENDIF(CXXTEST_FOUND)