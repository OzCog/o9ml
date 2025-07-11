#
# Foundation Layer: Hardware Matrix Configuration
# Multi-architecture support for cognitive kernel tensor operations
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

# ========================================================================
# Hardware Architecture Detection
# ========================================================================

MESSAGE(STATUS "Detecting hardware architecture for tensor optimization...")

# Get basic architecture info
EXECUTE_PROCESS(
    COMMAND uname -m
    OUTPUT_VARIABLE SYSTEM_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

MESSAGE(STATUS "System Architecture: ${SYSTEM_ARCH}")

# ========================================================================
# Multi-Architecture Matrix Configuration
# ========================================================================

# Architecture-specific tensor optimizations
IF(SYSTEM_ARCH MATCHES "x86_64|amd64")
    SET(TARGET_ARCH "x86_64")
    SET(TENSOR_SIMD_SUPPORT ON)
    SET(VECTOR_INSTRUCTION_SET "AVX2")
    
    # Check for advanced x86 features
    INCLUDE(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
    CHECK_CXX_COMPILER_FLAG("-mavx512f" COMPILER_SUPPORTS_AVX512)
    
    IF(COMPILER_SUPPORTS_AVX512)
        SET(VECTOR_INSTRUCTION_SET "AVX512")
        ADD_COMPILE_OPTIONS(-mavx512f)
        MESSAGE(STATUS "  ✓ AVX512 tensor acceleration enabled")
    ELSEIF(COMPILER_SUPPORTS_AVX2)
        ADD_COMPILE_OPTIONS(-mavx2)
        MESSAGE(STATUS "  ✓ AVX2 tensor acceleration enabled")
    ENDIF()
    
ELSEIF(SYSTEM_ARCH MATCHES "aarch64|arm64")
    SET(TARGET_ARCH "arm64")
    SET(TENSOR_SIMD_SUPPORT ON)
    SET(VECTOR_INSTRUCTION_SET "NEON")
    
    # ARM NEON optimizations
    CHECK_CXX_COMPILER_FLAG("-mfpu=neon" COMPILER_SUPPORTS_NEON)
    IF(COMPILER_SUPPORTS_NEON)
        ADD_COMPILE_OPTIONS(-mfpu=neon)
        MESSAGE(STATUS "  ✓ ARM NEON tensor acceleration enabled")
    ENDIF()
    
ELSEIF(SYSTEM_ARCH MATCHES "riscv64")
    SET(TARGET_ARCH "riscv64")
    SET(TENSOR_SIMD_SUPPORT OFF)  # Limited SIMD on RISC-V currently
    SET(VECTOR_INSTRUCTION_SET "SCALAR")
    MESSAGE(STATUS "  ✓ RISC-V scalar tensor operations enabled")
    
ELSE()
    SET(TARGET_ARCH "generic")
    SET(TENSOR_SIMD_SUPPORT OFF)
    SET(VECTOR_INSTRUCTION_SET "SCALAR")
    MESSAGE(STATUS "  Generic tensor operations (no SIMD)")
ENDIF()

# ========================================================================
# Tensor Operation Optimization Matrix
# ========================================================================

# Configure tensor operation parameters based on architecture
IF(TENSOR_SIMD_SUPPORT)
    ADD_DEFINITIONS(-DTENSOR_SIMD_ENABLED=1)
    
    # Set optimal tensor block sizes for SIMD operations
    IF(VECTOR_INSTRUCTION_SET STREQUAL "AVX512")
        SET(TENSOR_BLOCK_SIZE 512)
        SET(TENSOR_VECTOR_WIDTH 16)  # 512-bit / 32-bit float
    ELSEIF(VECTOR_INSTRUCTION_SET STREQUAL "AVX2")
        SET(TENSOR_BLOCK_SIZE 256)
        SET(TENSOR_VECTOR_WIDTH 8)   # 256-bit / 32-bit float
    ELSEIF(VECTOR_INSTRUCTION_SET STREQUAL "NEON")
        SET(TENSOR_BLOCK_SIZE 128)
        SET(TENSOR_VECTOR_WIDTH 4)   # 128-bit / 32-bit float
    ENDIF()
ELSE()
    ADD_DEFINITIONS(-DTENSOR_SIMD_ENABLED=0)
    SET(TENSOR_BLOCK_SIZE 64)
    SET(TENSOR_VECTOR_WIDTH 1)
ENDIF()

MESSAGE(STATUS "Tensor Configuration:")
MESSAGE(STATUS "  Block Size: ${TENSOR_BLOCK_SIZE}")
MESSAGE(STATUS "  Vector Width: ${TENSOR_VECTOR_WIDTH}")
MESSAGE(STATUS "  SIMD Support: ${TENSOR_SIMD_SUPPORT}")

# ========================================================================
# Memory Architecture Optimization
# ========================================================================

# Detect cache sizes for optimal tensor memory layout
IF(EXISTS "/proc/cpuinfo")
    # Try to get cache sizes from system
    EXECUTE_PROCESS(
        COMMAND grep -m1 "cache size" /proc/cpuinfo
        OUTPUT_VARIABLE CPU_CACHE_INFO
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    IF(CPU_CACHE_INFO)
        MESSAGE(STATUS "CPU Cache Info: ${CPU_CACHE_INFO}")
    ENDIF()
ENDIF()

# Set default cache-aware tensor parameters
IF(TARGET_ARCH STREQUAL "x86_64")
    SET(L1_CACHE_SIZE 32768)   # 32KB typical L1
    SET(L2_CACHE_SIZE 262144)  # 256KB typical L2
    SET(L3_CACHE_SIZE 8388608) # 8MB typical L3
ELSEIF(TARGET_ARCH STREQUAL "arm64")
    SET(L1_CACHE_SIZE 65536)   # 64KB typical L1 on ARM
    SET(L2_CACHE_SIZE 524288)  # 512KB typical L2
    SET(L3_CACHE_SIZE 4194304) # 4MB typical L3
ELSE()
    SET(L1_CACHE_SIZE 16384)   # Conservative defaults
    SET(L2_CACHE_SIZE 131072)
    SET(L3_CACHE_SIZE 1048576)
ENDIF()

# Calculate optimal tensor tile sizes based on cache
MATH(EXPR TENSOR_L1_TILE_SIZE "${L1_CACHE_SIZE} / 4")      # 1/4 of L1 for working set
MATH(EXPR TENSOR_L2_TILE_SIZE "${L2_CACHE_SIZE} / 2")      # 1/2 of L2 for data
MATH(EXPR TENSOR_L3_TILE_SIZE "${L3_CACHE_SIZE} / 2")      # 1/2 of L3 for large tensors

MESSAGE(STATUS "Cache-aware Tensor Tiling:")
MESSAGE(STATUS "  L1 Tile Size: ${TENSOR_L1_TILE_SIZE} bytes")
MESSAGE(STATUS "  L2 Tile Size: ${TENSOR_L2_TILE_SIZE} bytes") 
MESSAGE(STATUS "  L3 Tile Size: ${TENSOR_L3_TILE_SIZE} bytes")

# ========================================================================
# GPU Acceleration Detection
# ========================================================================

# Check for CUDA support
FIND_PACKAGE(CUDA QUIET)
IF(CUDA_FOUND)
    SET(TENSOR_CUDA_SUPPORT ON)
    MESSAGE(STATUS "  ✓ CUDA tensor acceleration available")
    
    # Set CUDA-specific tensor parameters
    SET(CUDA_BLOCK_SIZE 256)
    SET(CUDA_GRID_SIZE 65535)
    ADD_DEFINITIONS(-DTENSOR_CUDA_ENABLED=1)
ELSE()
    SET(TENSOR_CUDA_SUPPORT OFF)
    ADD_DEFINITIONS(-DTENSOR_CUDA_ENABLED=0)
ENDIF()

# Check for OpenCL support
FIND_PACKAGE(OpenCL QUIET)
IF(OpenCL_FOUND)
    SET(TENSOR_OPENCL_SUPPORT ON)
    MESSAGE(STATUS "  ✓ OpenCL tensor acceleration available")
    ADD_DEFINITIONS(-DTENSOR_OPENCL_ENABLED=1)
ELSE()
    SET(TENSOR_OPENCL_SUPPORT OFF)
    ADD_DEFINITIONS(-DTENSOR_OPENCL_ENABLED=0)
ENDIF()

# ========================================================================
# GGML Integration Configuration
# ========================================================================

OPTION(ENABLE_GGML "Enable GGML kernel integration" ON)

IF(ENABLE_GGML)
    MESSAGE(STATUS "Configuring GGML integration...")
    
    # GGML tensor format configuration
    SET(GGML_TENSOR_FORMATS "fp32,fp16,int8")
    SET(GGML_BLOCK_FORMATS "q4_0,q4_1,q5_0,q5_1,q8_0")
    
    # Architecture-specific GGML optimizations
    IF(TARGET_ARCH STREQUAL "x86_64")
        SET(GGML_BACKEND "cpu,avx2")
        IF(VECTOR_INSTRUCTION_SET STREQUAL "AVX512")
            SET(GGML_BACKEND "${GGML_BACKEND},avx512")
        ENDIF()
    ELSEIF(TARGET_ARCH STREQUAL "arm64")
        SET(GGML_BACKEND "cpu,neon")
    ELSE()
        SET(GGML_BACKEND "cpu")
    ENDIF()
    
    # Add GPU backends if available
    IF(TENSOR_CUDA_SUPPORT)
        SET(GGML_BACKEND "${GGML_BACKEND},cuda")
    ENDIF()
    
    IF(TENSOR_OPENCL_SUPPORT)
        SET(GGML_BACKEND "${GGML_BACKEND},opencl")
    ENDIF()
    
    MESSAGE(STATUS "GGML Configuration:")
    MESSAGE(STATUS "  Backends: ${GGML_BACKEND}")
    MESSAGE(STATUS "  Tensor Formats: ${GGML_TENSOR_FORMATS}")
    MESSAGE(STATUS "  Block Formats: ${GGML_BLOCK_FORMATS}")
    
    ADD_DEFINITIONS(
        -DGGML_ENABLED=1
        -DGGML_BACKEND="${GGML_BACKEND}"
        -DGGML_TENSOR_FORMATS="${GGML_TENSOR_FORMATS}"
    )
ELSE()
    ADD_DEFINITIONS(-DGGML_ENABLED=0)
ENDIF()

# ========================================================================
# Recursive Implementation Optimization
# ========================================================================

# Configure stack and recursion limits based on architecture
IF(TARGET_ARCH STREQUAL "x86_64")
    SET(MAX_RECURSION_DEPTH 10000)
    SET(STACK_SIZE_MB 8)
ELSEIF(TARGET_ARCH STREQUAL "arm64")
    SET(MAX_RECURSION_DEPTH 8000)
    SET(STACK_SIZE_MB 4)
ELSE()
    SET(MAX_RECURSION_DEPTH 5000)
    SET(STACK_SIZE_MB 2)
ENDIF()

ADD_DEFINITIONS(
    -DMAX_RECURSION_DEPTH=${MAX_RECURSION_DEPTH}
    -DSTACK_SIZE_MB=${STACK_SIZE_MB}
    -DRECURSIVE_COGNITIVE_KERNEL=1
)

MESSAGE(STATUS "Recursive Implementation:")
MESSAGE(STATUS "  Max Recursion Depth: ${MAX_RECURSION_DEPTH}")
MESSAGE(STATUS "  Stack Size: ${STACK_SIZE_MB}MB")

# ========================================================================
# Tensor Degrees of Freedom Configuration
# ========================================================================

# Define tensor dimensionality for different cognitive operations
SET(SPATIAL_TENSOR_DIM 3)      # 3D spatial reasoning
SET(TEMPORAL_TENSOR_DIM 1)     # Time series (1D sequence)
SET(SEMANTIC_TENSOR_DIM 256)   # Semantic embedding space
SET(LOGICAL_TENSOR_DIM 64)     # Logical inference chains

ADD_DEFINITIONS(
    -DSPATIAL_TENSOR_DIM=${SPATIAL_TENSOR_DIM}
    -DTEMPORAL_TENSOR_DIM=${TEMPORAL_TENSOR_DIM}
    -DSEMANTIC_TENSOR_DIM=${SEMANTIC_TENSOR_DIM}
    -DLOGICAL_TENSOR_DIM=${LOGICAL_TENSOR_DIM}
)

MESSAGE(STATUS "Tensor Degrees of Freedom:")
MESSAGE(STATUS "  Spatial: ${SPATIAL_TENSOR_DIM}D")
MESSAGE(STATUS "  Temporal: ${TEMPORAL_TENSOR_DIM}D") 
MESSAGE(STATUS "  Semantic: ${SEMANTIC_TENSOR_DIM}D")
MESSAGE(STATUS "  Logical: ${LOGICAL_TENSOR_DIM}D")

# ========================================================================
# Hardware Matrix Summary Export
# ========================================================================

# Create hardware configuration header for downstream components
CONFIGURE_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hardware_matrix_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hardware_matrix_config.h"
    @ONLY
)

# Export hardware matrix configuration for other components
SET(HARDWARE_MATRIX_CONFIG 
    "ARCH=${TARGET_ARCH};"
    "SIMD=${TENSOR_SIMD_SUPPORT};"
    "VECTOR_SET=${VECTOR_INSTRUCTION_SET};"
    "CUDA=${TENSOR_CUDA_SUPPORT};"
    "OPENCL=${TENSOR_OPENCL_SUPPORT};"
    "GGML=${ENABLE_GGML}"
    CACHE STRING "Hardware matrix configuration string"
)

MESSAGE(STATUS "")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "Hardware Matrix Configuration Complete")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "Target Architecture: ${TARGET_ARCH}")
MESSAGE(STATUS "SIMD Support: ${TENSOR_SIMD_SUPPORT}")
MESSAGE(STATUS "Vector Instructions: ${VECTOR_INSTRUCTION_SET}")
MESSAGE(STATUS "GPU Acceleration: CUDA=${TENSOR_CUDA_SUPPORT}, OpenCL=${TENSOR_OPENCL_SUPPORT}")
MESSAGE(STATUS "GGML Integration: ${ENABLE_GGML}")
MESSAGE(STATUS "Configuration exported to: ${CMAKE_CURRENT_BINARY_DIR}/hardware_matrix_config.h")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "")