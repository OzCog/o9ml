#!/bin/bash
# Local validation script for GitHub Actions workflow
# This script simulates the key steps from the GitHub Actions workflow locally

set -e

echo "🧠 OpenCog Central Local Build Validation"
echo "=========================================="

# Configuration
BUILD_TYPE=${BUILD_TYPE:-Release}
JOBS=${JOBS:-$(nproc)}
WORK_DIR="/tmp/opencog-local-build"

echo "Build Type: $BUILD_TYPE"
echo "Jobs: $JOBS"
echo "Work Directory: $WORK_DIR"
echo ""

# Clean up previous builds
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

# Function to build component
build_component() {
    local component=$1
    local source_dir=$2
    
    echo "🔧 Building $component..."
    if [ ! -d "$source_dir" ]; then
        echo "❌ Warning: $source_dir not found, skipping $component"
        return 1
    fi
    
    cd "$WORK_DIR"
    mkdir -p "$component"
    cd "$component"
    
    # Special handling for atomspace lib directory issue
    if [ "$component" = "atomspace" ]; then
        cd "$source_dir"
        if [ ! -d "lib" ]; then
            mkdir -p lib
            echo "# Empty lib directory for build compatibility" > lib/CMakeLists.txt
        fi
        cd "$WORK_DIR/$component"
    fi
    
    cmake "$source_dir" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$WORK_DIR/install" \
        -DCMAKE_PREFIX_PATH="$WORK_DIR/install"
    
    make -j"$JOBS"
    make install
    
    echo "✅ $component built successfully!"
    echo ""
    return 0
}

# Install system dependencies (Ubuntu/Debian)
echo "📦 Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake libboost-all-dev guile-3.0-dev
elif command -v yum &> /dev/null; then
    sudo yum install -y gcc-c++ cmake boost-devel
else
    echo "⚠️  Please install: build-essential cmake libboost-all-dev guile-3.0-dev"
fi

# Update library paths
export LD_LIBRARY_PATH="$WORK_DIR/install/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$WORK_DIR/install/lib/pkgconfig:$PKG_CONFIG_PATH"

echo ""
echo "🚀 Starting component builds..."
echo ""

# Build in dependency order (following GitHub Actions workflow)
REPO_ROOT=$(pwd)

# Foundation Layer
if build_component "cogutil" "$REPO_ROOT/orc-dv/cogutil"; then
    echo "✅ Foundation layer: cogutil"
else
    echo "❌ Foundation layer: cogutil FAILED"
    exit 1
fi

# Core Layer
if build_component "atomspace" "$REPO_ROOT/orc-as/atomspace"; then
    echo "✅ Core layer: atomspace"
else
    echo "❌ Core layer: atomspace FAILED"
    exit 1
fi

# Logic Layer
if build_component "ure" "$REPO_ROOT/orc-ai/ure"; then
    echo "✅ Logic layer: ure"
else
    echo "❌ Logic layer: ure FAILED"
    echo "⚠️  Continuing with available components..."
fi

echo ""
echo "🎉 Build validation complete!"
echo "=========================================="
echo ""
echo "📁 Build artifacts location: $WORK_DIR"
echo "📚 Installation prefix: $WORK_DIR/install"
echo ""
echo "💡 To use the built libraries:"
echo "export LD_LIBRARY_PATH=$WORK_DIR/install/lib:\$LD_LIBRARY_PATH"
echo "export PKG_CONFIG_PATH=$WORK_DIR/install/lib/pkgconfig:\$PKG_CONFIG_PATH"
echo ""
echo "🔗 This validates the same build process as the GitHub Actions workflow."