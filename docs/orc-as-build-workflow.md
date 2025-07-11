# ORC-AS Build Workflow Documentation

## Overview

The `orc-as-build.yml` workflow provides comprehensive build and testing for all AtomSpace components in the `orc-as/` directory. This workflow ensures proper dependency management and build ordering across the entire AtomSpace ecosystem.

## Workflow Structure

### Dependency Graph
```
cogutil → atomspace → [storage backends, network interfaces, advanced components, JS/TS/Rust components]
```

### Job Stages

1. **build-foundation** - Builds and caches cogutil (foundation dependency)
2. **build-atomspace** - Builds and caches core atomspace library
3. **build-storage-backends** - Builds storage components (atomspace-rocks, atomspace-bridge)
4. **build-network-interfaces** - Builds network components (atomspace-cog, atomspace-restful, atomspace-rpc, atomspace-websockets)
5. **build-advanced-components** - Builds advanced components (atomspace-agents, atomspace-dht, atomspace-ipfs, atomspace-metta)
6. **build-js-ts-components** - Builds Node.js components (atomspace-explorer, atomspace-typescript)
7. **build-rust-components** - Builds Rust components (atomspace-js)
8. **integration-test** - Runs final integration tests

## Components Coverage

### CMake-based Components (11/14)
- atomspace (core)
- atomspace-agents
- atomspace-bridge
- atomspace-cog
- atomspace-dht
- atomspace-ipfs
- atomspace-metta
- atomspace-restful
- atomspace-rocks
- atomspace-rpc
- atomspace-websockets

### Node.js-based Components (2/14)
- atomspace-explorer (Angular-based web interface)
- atomspace-typescript (TypeScript bindings and visualizer)

### Rust-based Components (1/14)
- atomspace-js (Rust-based JavaScript bindings)

## Key Features

### Dependency Management
- Proper build ordering ensures dependencies are available
- Shared AtomSpace CMake configuration for all dependent components
- Caching of build artifacts to optimize CI performance

### Multi-Technology Support
- CMake/C++ builds with proper system dependencies
- Node.js/npm builds with package caching
- Rust/Cargo builds with registry caching

### Error Handling
- Tests are marked as non-failing to allow builds to complete
- Missing build commands are handled gracefully
- Integration test provides verification of successful builds

### Triggers
- Activated on changes to `orc-as/**` directories
- Activated on changes to `orc-dv/cogutil/**` (foundation dependency)
- Activated on changes to the workflow file itself

## Usage

The workflow runs automatically on:
- Push to main branch with relevant changes
- Pull requests targeting main branch with relevant changes

Manual trigger: Not currently supported, but can be added with `workflow_dispatch` event.

## Troubleshooting

### Common Issues
1. **Missing system dependencies**: Check component-specific dependency installation steps
2. **CMake configuration failures**: Verify AtomSpace CMake config is properly created
3. **Cache conflicts**: Clear caches if builds become inconsistent

### Debugging
- Each job outputs detailed build information
- Integration test provides verification of installed components
- Individual components can be tested locally using the same build commands