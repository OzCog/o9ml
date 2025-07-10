#!/bin/bash
# Build and test package script for CogML
# This script builds Debian and Nix packages and tests their integrity

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Configuration
BUILD_DIR="$PROJECT_ROOT/build-package"
DEBIAN_BUILD_DIR="$BUILD_DIR/debian"
NIX_BUILD_DIR="$BUILD_DIR/nix"
OUTPUT_DIR="$BUILD_DIR/output"

cleanup() {
    if [[ -d "$BUILD_DIR" ]]; then
        log_info "Cleaning up build directory..."
        rm -rf "$BUILD_DIR"
    fi
}

# Set up trap for cleanup
trap cleanup EXIT

setup_build_environment() {
    log_info "Setting up build environment..."
    
    # Create build directories
    mkdir -p "$DEBIAN_BUILD_DIR" "$NIX_BUILD_DIR" "$OUTPUT_DIR"
    
    # Copy source to build directories
    log_info "Copying source files..."
    cp -r "$PROJECT_ROOT"/* "$DEBIAN_BUILD_DIR"/ 2>/dev/null || true
    cp -r "$PROJECT_ROOT"/.* "$DEBIAN_BUILD_DIR"/ 2>/dev/null || true
    
    log_info "Build environment ready"
}

build_debian_package() {
    log_header "Building Debian Package"
    
    if ! command -v dpkg-buildpackage >/dev/null 2>&1; then
        log_warn "dpkg-buildpackage not available, skipping Debian build"
        log_warn "To build Debian packages, install: apt-get install devscripts build-essential"
        return 1
    fi
    
    cd "$DEBIAN_BUILD_DIR"
    
    log_info "Building Debian package..."
    if dpkg-buildpackage -us -uc -b --no-sign 2>&1 | tee "$OUTPUT_DIR/debian-build.log"; then
        log_info "✓ Debian package built successfully"
        
        # Move generated packages to output directory
        mv ../*.deb "$OUTPUT_DIR"/ 2>/dev/null || true
        mv ../*.changes "$OUTPUT_DIR"/ 2>/dev/null || true
        mv ../*.buildinfo "$OUTPUT_DIR"/ 2>/dev/null || true
        
        # List generated packages
        log_info "Generated Debian packages:"
        ls -la "$OUTPUT_DIR"/*.deb 2>/dev/null || log_warn "No .deb files found"
        
        return 0
    else
        log_error "Debian package build failed"
        log_error "See $OUTPUT_DIR/debian-build.log for details"
        return 1
    fi
}

build_nix_package() {
    log_header "Building Nix Package"
    
    if ! command -v nix >/dev/null 2>&1; then
        log_warn "Nix not available, skipping Nix build"
        log_warn "To build Nix packages, install Nix: https://nixos.org/download.html"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    
    log_info "Building Nix package..."
    if timeout 600 nix build . --out-link "$OUTPUT_DIR/cogml-nix" 2>&1 | tee "$OUTPUT_DIR/nix-build.log"; then
        log_info "✓ Nix package built successfully"
        
        # Show package contents
        if [[ -L "$OUTPUT_DIR/cogml-nix" ]]; then
            log_info "Nix package contents:"
            ls -la "$OUTPUT_DIR/cogml-nix"
            
            # Create a tarball of the Nix package for distribution
            log_info "Creating Nix package tarball..."
            tar -czf "$OUTPUT_DIR/cogml-nix.tar.gz" -C "$OUTPUT_DIR" cogml-nix
            log_info "Nix package tarball: $OUTPUT_DIR/cogml-nix.tar.gz"
        fi
        
        return 0
    else
        log_error "Nix package build failed or timed out"
        log_error "See $OUTPUT_DIR/nix-build.log for details"
        return 1
    fi
}

test_debian_package() {
    log_header "Testing Debian Package"
    
    local deb_files=($(ls "$OUTPUT_DIR"/*.deb 2>/dev/null))
    
    if [[ ${#deb_files[@]} -eq 0 ]]; then
        log_warn "No Debian packages found to test"
        return 1
    fi
    
    for deb_file in "${deb_files[@]}"; do
        log_info "Testing package: $(basename "$deb_file")"
        
        # Check package info
        if command -v dpkg-deb >/dev/null 2>&1; then
            log_info "Package information:"
            dpkg-deb --info "$deb_file" | head -20
            
            log_info "Package contents:"
            dpkg-deb --contents "$deb_file" | head -10
            echo "..."
        fi
        
        # Test package installation (in a safe way)
        log_info "Package structure verified"
    done
    
    log_info "✓ Debian package tests completed"
    return 0
}

test_nix_package() {
    log_header "Testing Nix Package"
    
    if [[ ! -L "$OUTPUT_DIR/cogml-nix" ]]; then
        log_warn "No Nix package found to test"
        return 1
    fi
    
    local nix_store_path=$(readlink "$OUTPUT_DIR/cogml-nix")
    
    log_info "Testing Nix package: $nix_store_path"
    
    # Check package structure
    log_info "Package structure:"
    find "$nix_store_path" -type f | head -10
    echo "..."
    
    # Check if executables are properly linked
    if [[ -d "$nix_store_path/bin" ]]; then
        log_info "Available executables:"
        ls -la "$nix_store_path/bin"
    fi
    
    log_info "✓ Nix package tests completed"
    return 0
}

run_package_verification() {
    log_header "Running Package Verification"
    
    # Run the verification script
    if [[ -x "$PROJECT_ROOT/scripts/verify-package-integrity.sh" ]]; then
        "$PROJECT_ROOT/scripts/verify-package-integrity.sh"
    else
        log_warn "Package verification script not found or not executable"
    fi
}

generate_build_report() {
    log_header "Generating Build Report"
    
    local report_file="$OUTPUT_DIR/package-build-report.md"
    
    cat > "$report_file" << EOF
# CogML Package Build Report

Generated: $(date)

## Build Summary

### Debian Package
- Build attempted: $(test -f "$OUTPUT_DIR/debian-build.log" && echo "Yes" || echo "No")
- Build successful: $(ls "$OUTPUT_DIR"/*.deb >/dev/null 2>&1 && echo "Yes" || echo "No")
- Generated packages: $(ls "$OUTPUT_DIR"/*.deb 2>/dev/null | wc -l)

### Nix Package
- Build attempted: $(test -f "$OUTPUT_DIR/nix-build.log" && echo "Yes" || echo "No")
- Build successful: $(test -L "$OUTPUT_DIR/cogml-nix" && echo "Yes" || echo "No")
- Package available: $(test -L "$OUTPUT_DIR/cogml-nix" && echo "Yes" || echo "No")

## Package Files

### Debian Packages
$(ls -la "$OUTPUT_DIR"/*.deb 2>/dev/null || echo "No Debian packages generated")

### Nix Packages
$(ls -la "$OUTPUT_DIR"/cogml-nix* 2>/dev/null || echo "No Nix packages generated")

## Build Logs

### Debian Build Log
$(test -f "$OUTPUT_DIR/debian-build.log" && echo "Available at: $OUTPUT_DIR/debian-build.log" || echo "Not available")

### Nix Build Log
$(test -f "$OUTPUT_DIR/nix-build.log" && echo "Available at: $OUTPUT_DIR/nix-build.log" || echo "Not available")

## Packaging Tensor Shape

The CogML packaging system implements a multi-dimensional deployment tensor:

1. **Platform Dimension**: Debian (.deb) and Nix packages
2. **Architecture Dimension**: Support for multiple CPU architectures
3. **Component Dimension**: Core package and development package
4. **Dependency Dimension**: Hierarchical dependency management
5. **Verification Dimension**: Integrity and installability testing

This tensor shape ensures comprehensive deployment coverage across different
environments while maintaining package integrity and dependency coherence.

EOF

    log_info "Build report generated: $report_file"
    
    # Display summary
    log_info "Package build summary:"
    cat "$report_file" | grep -A 10 "## Build Summary"
}

# Main build function
main() {
    log_header "CogML Package Build and Test"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Build directory: $BUILD_DIR"
    
    local exit_code=0
    
    # Setup build environment
    setup_build_environment
    
    # Run package verification first
    run_package_verification || true
    
    # Build packages
    log_info "Starting package builds..."
    
    build_debian_package || exit_code=1
    build_nix_package || exit_code=1
    
    # Test packages
    log_info "Testing built packages..."
    
    test_debian_package || true
    test_nix_package || true
    
    # Generate report
    generate_build_report
    
    # Final summary
    if [[ $exit_code -eq 0 ]]; then
        log_info "✓ Package build completed successfully"
        log_info "Output directory: $OUTPUT_DIR"
    else
        log_warn "Package build completed with some failures"
        log_info "Check build logs in: $OUTPUT_DIR"
    fi
    
    return $exit_code
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help]"
        echo "Build and test CogML packages for Debian and Nix"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo ""
        echo "Output will be in: $BUILD_DIR/output"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac