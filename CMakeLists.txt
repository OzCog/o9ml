#
# Master OpenCog Central CMake file.
#
# This file orchestrates the build of all OpenCog components in proper
# dependency order to ensure stable, correct operation.
#
# General organization:
# -- check for different compilers, OS'es
# -- search for various required & optional libraries/tools
# -- decide what to build based on above results.
# -- configure various config files.
# -- build components in dependency order
# -- print pretty summary
#

# CMake version requirement aligned with cogutil and atomspace
CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

IF (COMMAND CMAKE_POLICY)
	CMAKE_POLICY(SET CMP0003 NEW)
ENDIF (COMMAND CMAKE_POLICY)

IF(CMAKE_VERSION VERSION_GREATER 3.0.2)
	CMAKE_POLICY(SET CMP0037 OLD)
ENDIF(CMAKE_VERSION VERSION_GREATER 3.0.2)

PROJECT(opencog-central)

# ----------------------------------------------------------
# User-modifiable options. Feel free to change these!
#
# uncomment to be in Release mode [default]
# SET(CMAKE_BUILD_TYPE Release)

# uncomment to build in debug mode
# SET(CMAKE_BUILD_TYPE Debug)

# uncomment to be in coverage testing mode
# SET(CMAKE_BUILD_TYPE Coverage)

# uncomment to build in profile mode
# SET(CMAKE_BUILD_TYPE Profile)

# uncomment to build in release mode with debug information
# SET(CMAKE_BUILD_TYPE RelWithDebInfo)

# default build type
IF (CMAKE_BUILD_TYPE STREQUAL "")
	SET(CMAKE_BUILD_TYPE Release)
ENDIF (CMAKE_BUILD_TYPE STREQUAL "")

MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

ADD_DEFINITIONS(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
                -DPROJECT_BINARY_DIR="${CMAKE_BINARY_DIR}")

# ===============================================================
# OpenCog Central Components Build Order
#
# Components are built in dependency order to ensure proper installation
# 1. cogutil - base utilities for all OpenCog components
# 2. orc-as - AtomSpace orchestration (all atomspace components)
# 3. ure - Unified Rule Engine
# 4. attention - Attention allocation system
# 5. spacetime - Spatiotemporal reasoning
# 6. cogserver - Core server infrastructure
# 7. pln - Probabilistic Logic Networks
# 8. miner - Pattern mining
# 9. moses - Meta-Optimizing Semantic Evolutionary Search
# 10. asmoses - AtomSpace MOSES integration
# 11. learn - Learning systems
# 12. opencog - Main OpenCog framework
# 13. unify - Unification system
# 14. generate - Generation system
# 15. Optional components (vision, cheminformatics, etc.)

MESSAGE(STATUS "Building OpenCog Central - all components")

# ===============================================================
# Foundation Layer: Hardware Matrix & Tensor Configuration
# ===============================================================

# Include foundation layer hardware matrix configuration
INCLUDE(cmake/HardwareMatrix.cmake)

# Core foundation - Required
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/orc-dv/cogutil/CMakeLists.txt")
    MESSAGE(STATUS "Adding cogutil (foundation layer core utilities)")
    ADD_SUBDIRECTORY(orc-dv/cogutil)
    SET(HAVE_COGUTIL ON)
ELSE()
    MESSAGE(WARNING "cogutil not found in orc-dv - this is required!")
ENDIF()

# Foundation layer external tools
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/orc-dv/external-tools/CMakeLists.txt")
    MESSAGE(STATUS "Adding external-tools (foundation layer)")
    ADD_SUBDIRECTORY(orc-dv/external-tools)
ENDIF()

# Foundation layer MOSES (from orc-ai)
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/orc-ai/moses/CMakeLists.txt")
    MESSAGE(STATUS "Adding moses (foundation layer evolutionary algorithms)")
    ADD_SUBDIRECTORY(orc-ai/moses)
ENDIF()

# Advanced layer: Pattern Mining (from orc-ai)
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/orc-ai/miner/CMakeLists.txt")
    MESSAGE(STATUS "Adding miner (advanced layer pattern mining)")
    ADD_SUBDIRECTORY(orc-ai/miner)
ENDIF()

# Advanced layer: Probabilistic Logic Networks (from orc-ai)
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/orc-ai/pln/CMakeLists.txt")
    MESSAGE(STATUS "Adding pln (advanced layer probabilistic reasoning)")
    ADD_SUBDIRECTORY(orc-ai/pln)
ENDIF()

# Advanced layer: AtomSpace MOSES integration (from orc-ai)
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/orc-ai/asmoses/CMakeLists.txt")
    MESSAGE(STATUS "Adding asmoses (advanced layer AtomSpace MOSES)")
    ADD_SUBDIRECTORY(orc-ai/asmoses)
ENDIF()

# Advanced layer: Unified Rule Engine (from orc-ai)
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/orc-ai/ure/CMakeLists.txt")
    MESSAGE(STATUS "Adding ure (advanced layer unified reasoning)")
    ADD_SUBDIRECTORY(orc-ai/ure)
ENDIF()

# AtomSpace orchestration - Core infrastructure
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/orc-as/CMakeLists.txt")
    MESSAGE(STATUS "Adding orc-as (AtomSpace orchestration)")
    ADD_SUBDIRECTORY(orc-as)
ELSE()
    MESSAGE(WARNING "orc-as not found - AtomSpace components required!")
ENDIF()

# Unified Rule Engine - Core reasoning
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ure/CMakeLists.txt")
    MESSAGE(STATUS "Adding ure (Unified Rule Engine)")
    ADD_SUBDIRECTORY(ure)
ENDIF()

# Attention system
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/attention/CMakeLists.txt")
    MESSAGE(STATUS "Adding attention (attention allocation)")
    ADD_SUBDIRECTORY(attention)
ENDIF()

# SpaceTime reasoning
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/spacetime/CMakeLists.txt")
    MESSAGE(STATUS "Adding spacetime (spatiotemporal reasoning)")
    ADD_SUBDIRECTORY(spacetime)
ENDIF()

# CogServer - Core server infrastructure
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cogserver/CMakeLists.txt")
    MESSAGE(STATUS "Adding cogserver (server infrastructure)")
    ADD_SUBDIRECTORY(cogserver)
ENDIF()

# PLN - Probabilistic Logic Networks
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pln/CMakeLists.txt")
    MESSAGE(STATUS "Adding pln (Probabilistic Logic Networks)")
    ADD_SUBDIRECTORY(pln)
ENDIF()

# Pattern Miner
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/miner/CMakeLists.txt")
    MESSAGE(STATUS "Adding miner (pattern mining)")
    ADD_SUBDIRECTORY(miner)
ENDIF()

# MOSES - Meta-Optimizing Semantic Evolutionary Search
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/moses/CMakeLists.txt")
    MESSAGE(STATUS "Adding moses (evolutionary search)")
    ADD_SUBDIRECTORY(moses)
ENDIF()

# AS-MOSES - AtomSpace MOSES integration
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/asmoses/CMakeLists.txt")
    MESSAGE(STATUS "Adding asmoses (AtomSpace MOSES)")
    ADD_SUBDIRECTORY(asmoses)
ENDIF()

# Learning system
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/learn/CMakeLists.txt")
    MESSAGE(STATUS "Adding learn (learning systems)")
    ADD_SUBDIRECTORY(learn)
ENDIF()

# Main OpenCog framework
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/opencog/CMakeLists.txt")
    MESSAGE(STATUS "Adding opencog (main framework)")
    ADD_SUBDIRECTORY(opencog)
ENDIF()

# Unification system
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unify/CMakeLists.txt")
    MESSAGE(STATUS "Adding unify (unification system)")
    ADD_SUBDIRECTORY(unify)
ENDIF()

# Generation system
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generate/CMakeLists.txt")
    MESSAGE(STATUS "Adding generate (generation system)")
    ADD_SUBDIRECTORY(generate)
ENDIF()

# Optional specialized components
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vision/CMakeLists.txt")
    MESSAGE(STATUS "Adding vision (computer vision)")
    ADD_SUBDIRECTORY(vision)
ENDIF()

IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cheminformatics/CMakeLists.txt")
    MESSAGE(STATUS "Adding cheminformatics (chemical informatics)")
    ADD_SUBDIRECTORY(cheminformatics)
ENDIF()

IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lg-atomese/CMakeLists.txt")
    MESSAGE(STATUS "Adding lg-atomese (Link Grammar AtomESE)")
    ADD_SUBDIRECTORY(lg-atomese)
ENDIF()

IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sensory/CMakeLists.txt")
    MESSAGE(STATUS "Adding sensory (sensory processing)")
    ADD_SUBDIRECTORY(sensory)
ENDIF()

# ===================================================================
# Show a summary of what was built

MESSAGE(STATUS "")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "OpenCog Central Build Summary")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "All available OpenCog components have been configured.")
MESSAGE(STATUS "Components will be built in dependency order.")
MESSAGE(STATUS "Use 'make' to build all components.")
MESSAGE(STATUS "Use 'make install' to install all components.")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "")