# GnuCash Cognitive Engine Integration

## Overview

The GnuCash Cognitive Engine (gnucashcog-v3) has been integrated into the OpenCog Central (cogml) repository to provide cognitive accounting capabilities. This integration combines traditional financial management with advanced cognitive architectures.

## Integration Details

### Repository Structure
- **Location**: `gnucashcog-v3/` folder within the main cogml repository
- **Build System**: Integrated into the main CMakeLists.txt build process
- **Dependencies**: Shares OpenCog core modules with the main cogml system

### Shared OpenCog Components

Both systems utilize the following OpenCog core modules:
- **cogutil**: Foundation utilities for cognitive operations
- **atomspace**: Hypergraph knowledge representation
- **cogserver**: Distributed cognitive processing
- **attention (ECAN)**: Economic Attention Allocation
- **pln**: Probabilistic Logic Networks
- **moses**: Meta-Optimizing Semantic Evolutionary Search
- **ure**: Uncertain Reasoning Engine

### Key Features

1. **Neural-Symbolic Accounting**: Transforms double-entry bookkeeping into cognitive tensor networks
2. **AtomSpace Integration**: Financial accounts represented as hypergraph nodes
3. **PLN Validation**: Probabilistic logic for transaction validation
4. **Attention Economics**: Dynamic resource allocation for financial processing
5. **Cognitive Pattern Recognition**: Emergent financial intelligence discovery

### Building the Integration

The cognitive accounting engine is automatically included when building the cogml system:

```bash
# Standard cogml build includes gnucashcog-v3
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Usage Examples

Key demonstration files:
- `gnucashcog-v3/cognitive-accounting-demo.cpp`: Comprehensive cognitive features demo
- `gnucashcog-v3/tensor-network-demo.cpp`: Tensor network processing demo
- `gnucashcog-v3/test-cognitive-accounting.sh`: Integration testing script

### Documentation

Detailed cognitive accounting documentation:
- `gnucashcog-v3/COGNITIVE_ACCOUNTING.md`: Framework architecture
- `gnucashcog-v3/TENSOR_NETWORK_ARCHITECTURE.md`: Distributed processing
- `gnucashcog-v3/IMPLEMENTATION_REPORT.md`: Technical implementation

### Integration Benefits

1. **Unified Cognitive Architecture**: Single system combining general AI and financial intelligence
2. **Shared Resources**: Efficient use of OpenCog modules across domains
3. **Cross-Domain Learning**: Financial patterns can inform general cognitive processing
4. **Emergent Capabilities**: Synergistic effects from combining cognitive systems

## Future Development

The integration enables:
- Cross-pollination between financial and general cognitive reasoning
- Unified attention allocation across all cognitive domains
- Enhanced pattern recognition capabilities
- Comprehensive cognitive testing and validation

## Testing

Run integration tests:
```bash
cd gnucashcog-v3
./test-cognitive-accounting.sh
```

## Notes

- The gnucashcog-v3 .git repository has been removed to avoid nested git issues
- Build artifacts are properly excluded via .gitignore
- The integration maintains full compatibility with both systems' existing features