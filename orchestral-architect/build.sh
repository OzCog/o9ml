#!/bin/bash
# Build and run the Orchestral Architect demo

set -e

echo "🎼 Building Orchestral Architect..."
echo "================================="

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring build..."
cmake ..

# Build the project
echo "Building project..."
make -j$(nproc)

echo ""
echo "✅ Build complete!"
echo ""
echo "Available executables:"
echo "  ./orchestral-demo  - Live demonstration"
echo "  ./orchestral-tests - Unit tests"
echo ""

# Ask user what to run
read -p "Run demo? [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$|^$ ]]; then
    echo "🚀 Running Orchestral Architect Demo..."
    echo "======================================"
    ./orchestral-demo
fi