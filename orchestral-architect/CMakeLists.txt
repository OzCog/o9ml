cmake_minimum_required(VERSION 3.16)
project(orchestral-architect VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/src/attention
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)

# Source files
set(ORCHESTRAL_CORE_SOURCES
    src/core/AgenticKernel.cpp
    src/core/KernelRegistry.cpp
)

set(ORCHESTRAL_KERNEL_SOURCES
    src/kernels/TokenizationKernel.cpp
)

set(ORCHESTRAL_SYSTEM_SOURCES
    src/OrchestralSystem.cpp
)

# Create orchestral library
add_library(orchestral-core STATIC
    ${ORCHESTRAL_CORE_SOURCES}
    ${ORCHESTRAL_KERNEL_SOURCES}
    ${ORCHESTRAL_SYSTEM_SOURCES}
)

target_link_libraries(orchestral-core 
    Threads::Threads
)

# Demo executable
add_executable(orchestral-demo
    demo/orchestral_demo.cpp
)

target_link_libraries(orchestral-demo
    orchestral-core
    Threads::Threads
)

# Test executable (if tests exist)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    # Find test sources
    file(GLOB TEST_SOURCES "tests/*.cpp")
    
    if(TEST_SOURCES)
        add_executable(orchestral-tests ${TEST_SOURCES})
        target_link_libraries(orchestral-tests
            orchestral-core
            Threads::Threads
        )
        
        # Enable testing
        enable_testing()
        add_test(NAME orchestral_unit_tests COMMAND orchestral-tests)
    endif()
endif()

# Installation
install(TARGETS orchestral-core orchestral-demo
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(DIRECTORY src/
    DESTINATION include/orchestral-architect
    FILES_MATCHING PATTERN "*.h"
)

# Documentation target
find_program(DOXYGEN_EXECUTABLE doxygen)
if(DOXYGEN_EXECUTABLE)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY
    )
    
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Orchestral Architect Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  Binary directory: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  orchestral-core: Core library")
message(STATUS "  orchestral-demo: Live demonstration")
if(TEST_SOURCES)
    message(STATUS "  orchestral-tests: Unit tests")
endif()
if(DOXYGEN_EXECUTABLE)
    message(STATUS "  docs: API documentation")
endif()
message(STATUS "")