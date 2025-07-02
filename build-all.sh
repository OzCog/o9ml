
#!/bin/bash
set -e

# OpenCog Unified Build Script
echo "=========================================="
echo "OpenCog Unified Framework Build Script"
echo "=========================================="

# Configuration
BUILD_TYPE=${BUILD_TYPE:-Release}
INSTALL_PREFIX=${INSTALL_PREFIX:-/usr/local}
JOBS=${JOBS:-$(nproc)}
BUILD_DIR=${BUILD_DIR:-$(pwd)/build-unified}

echo "Build Type: $BUILD_TYPE"
echo "Install Prefix: $INSTALL_PREFIX"
echo "Jobs: $JOBS"
echo "Build Directory: $BUILD_DIR"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Function to build component
build_component() {
    local component=$1
    local source_dir=$2
    
    echo "Building $component..."
    if [ ! -d "$source_dir" ]; then
        echo "Warning: $source_dir not found, skipping $component"
        return 0
    fi
    
    mkdir -p "$component"
    cd "$component"
    
    cmake "$source_dir" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX"
    
    make -j"$JOBS"
    make install
    
    cd ..
    echo "$component built successfully!"
    echo ""
}

# Build in dependency order
echo "Starting build process..."
echo ""

# Core foundation
build_component "cogutil" "../cogutil"

# AtomSpace orchestration - all components in proper order
build_component "atomspace-orchestration" "../orc-as"

# URE (depends on atomspace)
build_component "ure" "../ure"

# Attention system
build_component "attention" "../attention"

# SpaceTime
build_component "spacetime" "../spacetime"

# CogServer
build_component "cogserver" "../cogserver"

# PLN
build_component "pln" "../pln"

# Pattern Miner
build_component "miner" "../miner"

# MOSES
build_component "moses" "../moses"

# AS-MOSES
build_component "asmoses" "../asmoses"

# Learning system (if exists)
if [ -d "../learn" ]; then
    build_component "learn" "../learn"
fi

# Main OpenCog
build_component "opencog" "../opencog"

# Additional components
build_component "unify" "../unify"
build_component "generate" "../generate"

# Optional components
if [ -d "../vision" ]; then
    build_component "vision" "../vision"
fi

if [ -d "../cheminformatics" ]; then
    build_component "cheminformatics" "../cheminformatics"
fi

if [ -d "../lg-atomese" ]; then
    build_component "lg-atomese" "../lg-atomese"
fi

if [ -d "../sensory" ]; then
    build_component "sensory" "../sensory"
fi

if [ -d "../agents" ]; then
    build_component "agents" "../agents"
fi

# Build unified main application
echo "Building unified OpenCog application..."
cd "$BUILD_DIR"
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX"

make -j"$JOBS"

echo ""
echo "=========================================="
echo "OpenCog Unified Framework Build Complete!"
echo "=========================================="
echo ""
echo "Installation directory: $INSTALL_PREFIX"
echo "Build directory: $BUILD_DIR"
echo ""
echo "To run the unified system:"
echo "export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "export PKG_CONFIG_PATH=$INSTALL_PREFIX/lib/pkgconfig:\$PKG_CONFIG_PATH"
echo ""
echo "Then run: cogserver or other OpenCog binaries"
