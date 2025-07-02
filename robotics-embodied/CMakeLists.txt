CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
SET(CMAKE_INSTALL_MESSAGE LAZY)

PROJECT(robotics-embodied)

# Cogutil
FIND_PACKAGE(CogUtil CONFIG REQUIRED)
IF(COGUTIL_FOUND)
    SET(HAVE_COGUTIL 1)
    list(APPEND CMAKE_MODULE_PATH ${COGUTIL_DATA_DIR}/cmake)
    if (NOT DEFINED ATOMSPACE_DATA_DIR)
        set (ATOMSPACE_DATA_DIR "${COGUTIL_DATA_DIR}")
    endif (NOT DEFINED ATOMSPACE_DATA_DIR)
    include(OpenCogGccOptions)
    include(OpenCogLibOptions) 
    include(OpenCogInstallOptions)
    include(Summary)
ENDIF()

# AtomSpace
FIND_PACKAGE(AtomSpace CONFIG REQUIRED)
IF(ATOMSPACE_FOUND)
    SET(HAVE_ATOMSPACE 1)
ENDIF()

# OpenCV for vision integration
FIND_PACKAGE(OpenCV QUIET)

# Include directories
INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../orc-gm/TinyCog/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../orc-ro/sensory/opencog
)

# Only add OpenCV if found
IF(OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    SET(HAVE_OPENCV ON)
ELSE()
    MESSAGE(STATUS "OpenCV not found - vision integration will be limited")
    SET(HAVE_OPENCV OFF)
ENDIF()

# Add subdirectories
ADD_SUBDIRECTORY(include/robotics)
ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(tests)

# Summary
SUMMARY_ADD("Robotics Embodied Cognition" "Embodied cognition layer for robotics" HAVE_COGUTIL)
SUMMARY_SHOW()