#!/usr/bin/env python3
"""
CogML Command Line Interface

Provides command-line access to CogML cognitive architecture components.
"""

import sys
import argparse
from typing import List, Optional

def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    if args is None:
        args = sys.argv[1:]
    
    parser = argparse.ArgumentParser(
        prog="cogml",
        description="CogML - Comprehensive cognitive architecture for AGI"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="CogML 0.1.0"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system information"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify installation"
    )
    
    parsed_args = parser.parse_args(args)
    
    if parsed_args.info:
        show_system_info()
        return 0
    
    if parsed_args.verify:
        return verify_installation()
    
    # Default action
    print("CogML Cognitive Architecture")
    print("Use --help for available commands")
    return 0

def show_system_info():
    """Display system information."""
    import platform
    import sys
    
    print("CogML System Information")
    print("=" * 30)
    print(f"Version: 0.1.0")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check for available components
    print("\nAvailable Components:")
    try:
        import numpy
        print(f"  ✓ NumPy: {numpy.__version__}")
    except ImportError:
        print("  ✗ NumPy: Not available")
    
    try:
        import pandas
        print(f"  ✓ Pandas: {pandas.__version__}")
    except ImportError:
        print("  ✗ Pandas: Not available")

def verify_installation() -> int:
    """Verify CogML installation."""
    print("Verifying CogML installation...")
    
    errors = 0
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        errors += 1
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check core dependencies
    required_packages = ["numpy", "pandas", "scikit-learn", "matplotlib"]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            errors += 1
    
    if errors == 0:
        print("\n✓ Installation verified successfully")
        return 0
    else:
        print(f"\n✗ Installation verification failed ({errors} errors)")
        return 1

if __name__ == "__main__":
    sys.exit(main())