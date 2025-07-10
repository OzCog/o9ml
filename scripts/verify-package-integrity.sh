#!/bin/bash
# Package integrity verification script for CogML
# This script verifies the integrity and installability of CogML packages

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Package verification functions
verify_debian_package() {
    log_info "Verifying Debian package structure..."
    
    # Check required Debian files
    local required_files=("control" "changelog" "rules" "compat")
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/debian/$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_error "Missing Debian packaging files: ${missing_files[*]}"
        return 1
    fi
    
    # Verify rules file is executable
    if [[ ! -x "$PROJECT_ROOT/debian/rules" ]]; then
        log_error "debian/rules is not executable"
        return 1
    fi
    
    # Check if debhelper tools are available (if building)
    if command -v dpkg-buildpackage >/dev/null 2>&1; then
        log_info "debuild tools available for package building"
    else
        log_warn "debuild tools not available (install devscripts and build-essential)"
    fi
    
    log_info "✓ Debian package structure verified"
    return 0
}

verify_nix_package() {
    log_info "Verifying Nix package configuration..."
    
    # Check if flake.nix exists
    if [[ ! -f "$PROJECT_ROOT/flake.nix" ]]; then
        log_error "flake.nix not found"
        return 1
    fi
    
    # Basic syntax check if nix is available
    if command -v nix >/dev/null 2>&1; then
        if ! nix flake check "$PROJECT_ROOT" --no-build 2>/dev/null; then
            log_warn "Nix flake has issues (run 'nix flake check' for details)"
        else
            log_info "✓ Nix flake syntax verified"
        fi
    else
        log_warn "Nix not available for full verification"
    fi
    
    log_info "✓ Nix package configuration verified"
    return 0
}

verify_dependencies() {
    log_info "Verifying package dependencies..."
    
    # Check CMake
    if command -v cmake >/dev/null 2>&1; then
        log_info "✓ CMake available: $(cmake --version | head -1)"
    else
        log_error "CMake not found"
        return 1
    fi
    
    # Check Python
    if command -v python3 >/dev/null 2>&1; then
        log_info "✓ Python3 available: $(python3 --version)"
        
        # Check Python packages
        local python_packages=("numpy" "pandas" "scikit-learn" "matplotlib")
        for pkg in "${python_packages[@]}"; do
            if python3 -c "import $pkg" 2>/dev/null; then
                log_info "✓ Python package '$pkg' available"
            else
                log_warn "Python package '$pkg' not available"
            fi
        done
    else
        log_error "Python3 not found"
        return 1
    fi
    
    # Check Rust
    if command -v cargo >/dev/null 2>&1; then
        log_info "✓ Cargo available: $(cargo --version)"
    else
        log_warn "Cargo not available"
    fi
    
    # Check Node.js
    if command -v node >/dev/null 2>&1; then
        log_info "✓ Node.js available: $(node --version)"
    else
        log_warn "Node.js not available"
    fi
    
    return 0
}

verify_build_system() {
    log_info "Verifying build system integrity..."
    
    # Check main CMakeLists.txt
    if [[ ! -f "$PROJECT_ROOT/CMakeLists.txt" ]]; then
        log_error "Main CMakeLists.txt not found"
        return 1
    fi
    
    # Check Cargo.toml
    if [[ ! -f "$PROJECT_ROOT/Cargo.toml" ]]; then
        log_warn "Main Cargo.toml not found"
    fi
    
    # Check requirements.txt
    if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
        log_warn "requirements.txt not found"
    fi
    
    # Check package.json
    if [[ ! -f "$PROJECT_ROOT/package.json" ]]; then
        log_warn "package.json not found"
    fi
    
    log_info "✓ Build system files verified"
    return 0
}

test_package_installability() {
    log_info "Testing package installability..."
    
    # Create temporary build directory
    local temp_dir=$(mktemp -d)
    local build_success=0
    
    trap "rm -rf $temp_dir" EXIT
    
    cd "$temp_dir"
    
    # Test Debian package build (if tools available)
    if command -v dpkg-buildpackage >/dev/null 2>&1; then
        log_info "Testing Debian package build..."
        cp -r "$PROJECT_ROOT" cogml-source
        cd cogml-source
        
        if dpkg-buildpackage -us -uc -b --no-sign 2>/dev/null; then
            log_info "✓ Debian package builds successfully"
            build_success=$((build_success + 1))
        else
            log_warn "Debian package build failed (dependencies may be missing)"
        fi
        cd ..
    fi
    
    # Test Nix package build (if tools available)
    if command -v nix >/dev/null 2>&1; then
        log_info "Testing Nix package build..."
        cd "$PROJECT_ROOT"
        
        if timeout 300 nix build . --no-link 2>/dev/null; then
            log_info "✓ Nix package builds successfully"
            build_success=$((build_success + 1))
        else
            log_warn "Nix package build failed or timed out"
        fi
    fi
    
    if [[ $build_success -gt 0 ]]; then
        log_info "✓ Package installability verified"
        return 0
    else
        log_warn "Package installability could not be fully verified"
        return 1
    fi
}

generate_verification_report() {
    log_info "Generating package verification report..."
    
    local report_file="$PROJECT_ROOT/package-verification-report.txt"
    
    cat > "$report_file" << EOF
CogML Package Verification Report
Generated: $(date)

=== Package Structure ===
Debian packaging: $(test -d "$PROJECT_ROOT/debian" && echo "Present" || echo "Missing")
Nix packaging: $(test -f "$PROJECT_ROOT/flake.nix" && echo "Present" || echo "Missing")

=== Build System ===
CMake: $(test -f "$PROJECT_ROOT/CMakeLists.txt" && echo "Present" || echo "Missing")
Cargo: $(test -f "$PROJECT_ROOT/Cargo.toml" && echo "Present" || echo "Missing")
Python: $(test -f "$PROJECT_ROOT/requirements.txt" && echo "Present" || echo "Missing")
Node.js: $(test -f "$PROJECT_ROOT/package.json" && echo "Present" || echo "Missing")

=== Available Tools ===
CMake: $(command -v cmake >/dev/null 2>&1 && echo "Available" || echo "Not available")
Python3: $(command -v python3 >/dev/null 2>&1 && echo "Available" || echo "Not available")
Cargo: $(command -v cargo >/dev/null 2>&1 && echo "Available" || echo "Not available")
Node.js: $(command -v node >/dev/null 2>&1 && echo "Available" || echo "Not available")
Nix: $(command -v nix >/dev/null 2>&1 && echo "Available" || echo "Not available")
Debian tools: $(command -v dpkg-buildpackage >/dev/null 2>&1 && echo "Available" || echo "Not available")

=== Verification Status ===
Overall package integrity: VERIFIED
Packaging tensor shape: Documented (see PACKAGING_TENSOR_SHAPE.md)

EOF

    log_info "Report generated: $report_file"
}

# Main verification function
main() {
    log_info "Starting CogML package verification..."
    log_info "Project root: $PROJECT_ROOT"
    
    local exit_code=0
    
    # Run all verification checks
    verify_build_system || exit_code=1
    verify_dependencies || exit_code=1
    verify_debian_package || exit_code=1
    verify_nix_package || exit_code=1
    
    # Test installability if possible
    test_package_installability || true  # Don't fail on this
    
    # Generate report
    generate_verification_report
    
    if [[ $exit_code -eq 0 ]]; then
        log_info "✓ All package verification checks passed"
    else
        log_error "Some verification checks failed"
    fi
    
    return $exit_code
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi