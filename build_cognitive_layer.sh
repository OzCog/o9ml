#!/bin/bash

# Cognitive Layer Build Script
# Builds cogserver, attention, and spacetime modules for distributed cognition dynamics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_PREFIX="/usr/local"

echo "=============================================="
echo "Cognitive Layer: Distributed Cognition Build"
echo "=============================================="

# Parse command line arguments
VERBOSE=false
CLEAN=false
JOBS=$(nproc)

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--clean] [--verbose] [--jobs N] [--prefix PATH]"
            exit 1
            ;;
    esac
done

echo "Build configuration:"
echo "  Build directory: $BUILD_DIR"
echo "  Install prefix: $INSTALL_PREFIX"
echo "  Parallel jobs: $JOBS"
echo "  Clean build: $CLEAN"
echo "  Verbose: $VERBOSE"

# Clean if requested
if [[ "$CLEAN" == "true" ]]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Function to build a component
build_component() {
    local component_name="$1"
    local component_path="$2"
    
    echo ""
    echo "Building $component_name..."
    echo "----------------------------------------"
    
    # Create component build directory
    local comp_build_dir="$BUILD_DIR/$component_name"
    mkdir -p "$comp_build_dir"
    cd "$comp_build_dir"
    
    # Configure with CMake
    echo "Configuring $component_name..."
    cmake_args=(
        "$SCRIPT_DIR/$component_path"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
    )
    
    if [[ "$VERBOSE" == "true" ]]; then
        cmake_args+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
    fi
    
    cmake "${cmake_args[@]}"
    
    # Build
    echo "Building $component_name..."
    make -j"$JOBS"
    
    # Install
    echo "Installing $component_name..."
    make install
    
    echo "$component_name build completed successfully!"
}

# Function to check if component exists
check_component() {
    local component_name="$1"
    local component_path="$2"
    
    if [[ ! -d "$SCRIPT_DIR/$component_path" ]]; then
        echo "Warning: $component_name not found at $component_path"
        return 1
    fi
    
    if [[ ! -f "$SCRIPT_DIR/$component_path/CMakeLists.txt" ]]; then
        echo "Warning: $component_name CMakeLists.txt not found"
        return 1
    fi
    
    return 0
}

# Build cognitive layer components in dependency order
echo ""
echo "Checking component availability..."

# Check what components are available
HAVE_COGSERVER=false
HAVE_ATTENTION=false
HAVE_SPACETIME=false

if check_component "cogserver" "orc-sv/cogserver"; then
    HAVE_COGSERVER=true
    echo "  ✓ cogserver available"
else
    echo "  ✗ cogserver not available"
fi

if check_component "attention" "orc-ct/attention"; then
    HAVE_ATTENTION=true
    echo "  ✓ attention available"
else
    echo "  ✗ attention not available"
fi

if check_component "spacetime" "orc-ct/spacetime"; then
    HAVE_SPACETIME=true
    echo "  ✓ spacetime available"
else
    echo "  ✗ spacetime not available"
fi

# Build components that are available
echo ""
echo "Building available components..."

if [[ "$HAVE_COGSERVER" == "true" ]]; then
    build_component "cogserver" "orc-sv/cogserver"
fi

if [[ "$HAVE_ATTENTION" == "true" ]]; then
    build_component "attention" "orc-ct/attention"
fi

if [[ "$HAVE_SPACETIME" == "true" ]]; then
    build_component "spacetime" "orc-ct/spacetime"
fi

# Run tests if available
echo ""
echo "Running cognitive layer tests..."
echo "================================"

test_results=()

if [[ "$HAVE_ATTENTION" == "true" ]]; then
    echo "Testing attention allocation..."
    cd "$BUILD_DIR/attention"
    if make test; then
        echo "  ✓ Attention tests passed"
        test_results+=("attention:PASS")
    else
        echo "  ✗ Attention tests failed"
        test_results+=("attention:FAIL")
    fi
fi

if [[ "$HAVE_SPACETIME" == "true" ]]; then
    echo "Testing spacetime reasoning..."
    cd "$BUILD_DIR/spacetime"
    if make test; then
        echo "  ✓ Spacetime tests passed"
        test_results+=("spacetime:PASS")
    else
        echo "  ✗ Spacetime tests failed"
        test_results+=("spacetime:FAIL")
    fi
fi

if [[ "$HAVE_COGSERVER" == "true" ]]; then
    echo "Testing cogserver..."
    cd "$BUILD_DIR/cogserver"
    if make test; then
        echo "  ✓ CogServer tests passed"
        test_results+=("cogserver:PASS")
    else
        echo "  ✗ CogServer tests failed"
        test_results+=("cogserver:FAIL")
    fi
fi

# Performance benchmarking
echo ""
echo "Running performance benchmarks..."
echo "================================="

if [[ "$HAVE_ATTENTION" == "true" ]]; then
    echo "Running attention allocation benchmarks..."
    
    # Check if benchmark executable exists
    BENCHMARK_EXE="$BUILD_DIR/attention/benchmarks/attention-benchmark-test"
    if [[ -f "$BENCHMARK_EXE" ]]; then
        echo "Running attention performance measurement..."
        if "$BENCHMARK_EXE" 1000 5; then
            echo "  ✓ Attention benchmarks completed successfully"
            test_results+=("attention_benchmark:PASS")
        else
            echo "  ✗ Attention benchmarks failed"
            test_results+=("attention_benchmark:FAIL")
        fi
    else
        echo "  - Attention benchmark executable not found (this is expected if dependencies are missing)"
    fi
fi

# Final summary
echo ""
echo "=============================================="
echo "COGNITIVE LAYER BUILD SUMMARY"
echo "=============================================="

echo "Components built:"
[[ "$HAVE_COGSERVER" == "true" ]] && echo "  ✓ cogserver" || echo "  ✗ cogserver (not available)"
[[ "$HAVE_ATTENTION" == "true" ]] && echo "  ✓ attention" || echo "  ✗ attention (not available)"
[[ "$HAVE_SPACETIME" == "true" ]] && echo "  ✓ spacetime" || echo "  ✗ spacetime (not available)"

echo ""
echo "Test results:"
for result in "${test_results[@]}"; do
    component=$(echo "$result" | cut -d: -f1)
    status=$(echo "$result" | cut -d: -f2)
    if [[ "$status" == "PASS" ]]; then
        echo "  ✓ $component"
    else
        echo "  ✗ $component"
    fi
done

# Check if any builds failed
failed_tests=0
for result in "${test_results[@]}"; do
    if [[ "$result" == *":FAIL" ]]; then
        failed_tests=$((failed_tests + 1))
    fi
done

echo ""
if [[ $failed_tests -eq 0 ]]; then
    echo "✓ All cognitive layer components built and tested successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run './measure_attention_performance.py $BUILD_DIR' to measure ECAN performance"
    echo "2. Check ATTENTION_TENSOR_DOF.md for tensor degrees of freedom documentation"
    echo "3. Use the attention benchmarking framework for performance analysis"
    exit 0
else
    echo "✗ $failed_tests test(s) failed"
    echo ""
    echo "Some components may not have built correctly due to missing dependencies."
    echo "This is expected in a minimal build environment."
    exit 1
fi