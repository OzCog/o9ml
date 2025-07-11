# ORC-AS Comprehensive Dependencies Documentation

## Overview

This document details the comprehensive dependency management implemented in the `orc-as-build.yml` workflow to ensure all prerequisites and optional prerequisites for all AtomSpace components are properly installed.

## Dependency Categories

### Core System Dependencies

These are the fundamental build tools and libraries required by all components:

- **build-essential**: Essential compilation tools (gcc, g++, make, libc-dev)
- **cmake**: Build system generator (minimum version varies by component)
- **pkg-config**: Library metadata management tool

### CogUtil Dependencies

CogUtil is the foundation library that all other components depend on:

- **libboost-all-dev**: Boost C++ libraries (filesystem, program_options, system, thread)
- **valgrind**: Memory debugging tool (optional but recommended)
- **doxygen**: Documentation generation (optional)

### AtomSpace Core Dependencies

Required for the core AtomSpace library:

- **guile-3.0-dev**: Scheme scripting language support
- **cython3**: Python C extension compilation
- **python3-nose**: Python testing framework
- **python3-dev**: Python development headers
- **libssl-dev**: SSL/TLS support for secure communications

### Testing Dependencies

For comprehensive unit testing across all components:

- **cxxtest**: C++ unit testing framework
- **libgtest-dev**: Google Test framework

### Storage Backend Dependencies

For persistence and database connectivity:

- **libpq-dev**: PostgreSQL client library development files
- **libmysqlclient-dev**: MySQL client library development files
- **librocksdb-dev**: RocksDB embedded database
- **unixodbc-dev**: ODBC database connectivity

### Network Interface Dependencies

For REST API and web service components:

- **libcpprest-dev**: C++ REST SDK for HTTP services

## Component-Specific Requirements

### atomspace-rocks
- **Primary**: librocksdb-dev
- **Use case**: High-performance embedded key-value storage

### atomspace-bridge
- **Primary**: libpq-dev, libmysqlclient-dev
- **Use case**: SQL database bridge interfaces

### atomspace-restful
- **Primary**: libcpprest-dev
- **Use case**: REST API web services

### atomspace-cog, atomspace-rpc, atomspace-websockets
- **Primary**: Standard networking libraries (included in base dependencies)
- **Use case**: Network communication protocols

### atomspace-agents, atomspace-dht, atomspace-ipfs, atomspace-metta
- **Primary**: Core AtomSpace dependencies
- **Use case**: Advanced AI and distributed computing features

### Node.js Components (atomspace-explorer, atomspace-typescript)
- **Primary**: Node.js 18+ with npm
- **Use case**: JavaScript/TypeScript development and web interfaces

### Rust Components (atomspace-js)
- **Primary**: Rust toolchain with Cargo
- **Use case**: WebAssembly bindings and high-performance components

## Workflow Implementation

The dependencies are installed at the beginning of each job to ensure:

1. **Consistency**: All jobs have the same base environment
2. **Completeness**: Optional dependencies are available if needed
3. **Efficiency**: Package cache is leveraged where possible
4. **Reliability**: Known working versions are used

## Version Requirements

Based on CMakeLists.txt analysis:

- **Boost**: Minimum 1.60 (cogutil requirement)
- **CMake**: Minimum 3.12 (atomspace requirement)
- **Guile**: Version 3.0+ preferred
- **Python**: Version 3.6+ with development headers
- **CogUtil**: Version 2.0.1+ required by components
- **AtomSpace**: Version 5.0.3+ required by dependent components

## Benefits

1. **Comprehensive Coverage**: All prerequisites and optional prerequisites are available
2. **Build Reliability**: Reduces failures due to missing dependencies
3. **Feature Completeness**: Optional features are available for testing
4. **Development Consistency**: Same environment for all components
5. **CI/CD Robustness**: Predictable build environment across all jobs

## Maintenance Notes

- Dependencies are reviewed and updated based on component CMakeLists.txt files
- Package availability is verified for Ubuntu runners
- Version compatibility is maintained for the OpenCog ecosystem
- Optional dependencies are included to enable full feature testing