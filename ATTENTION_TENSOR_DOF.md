# Attention Tensor Degrees of Freedom Documentation

## Overview

This document specifies the tensor degrees of freedom (DOF) for the OpenCog Cognitive Layer attention allocation mechanisms, specifically focusing on the Economic Attention Allocation Network (ECAN) and distributed cognition dynamics.

## Attention Tensor Architecture

The attention allocation system operates on multi-dimensional tensors that capture the various aspects of cognitive resource allocation across the hypergraph kernel. These tensors form the mathematical foundation for the attention membrane that dynamically weights cognitive resources.

### Core Tensor Dimensions

#### 1. Spatial Dimensions (3D)
- **DOF**: 3 (x, y, z coordinates)
- **Purpose**: Spatial attention allocation in embodied cognition scenarios
- **Use Cases**: 
  - 3D spatial reasoning and navigation
  - Coordinate-based attention focusing
  - Spatial pattern recognition in attention flow
- **Implementation**: `float spatial[3]` vectors
- **Range**: Typically [-1.0, 1.0] normalized coordinates

#### 2. Temporal Dimensions (1D)
- **DOF**: 1 (time sequence points)
- **Purpose**: Temporal attention patterns and time-series attention dynamics
- **Use Cases**:
  - Attention persistence over time
  - Temporal attention decay modeling
  - Sequential attention pattern learning
- **Implementation**: `float temporal[1]` sequences
- **Range**: Continuous time values or discrete time steps

#### 3. Semantic Dimensions (256D)
- **DOF**: 256 (semantic embedding space)
- **Purpose**: Semantic attention allocation based on concept similarity
- **Use Cases**:
  - Concept-based attention focusing
  - Semantic similarity-driven attention spreading
  - Cross-modal semantic attention integration
- **Implementation**: `float semantic[256]` dense vectors
- **Range**: Typically normalized L2 vectors in embedding space

#### 4. Importance Dimensions (3D)
- **DOF**: 3 (STI, LTI, VLTI)
- **Purpose**: Core ECAN importance value representation
- **Components**:
  - **STI (Short-Term Importance)**: Current attentional focus strength
  - **LTI (Long-Term Importance)**: Historical importance accumulation  
  - **VLTI (Very Long-Term Importance)**: Permanent importance markers
- **Implementation**: `float importance[3]` vectors
- **Range**: STI [0.0, 100.0], LTI [0.0, 1.0], VLTI {0, 1}

#### 5. Hebbian Dimensions (64D)
- **DOF**: 64 (synaptic strength patterns)
- **Purpose**: Hebbian learning and synaptic plasticity in attention networks
- **Use Cases**:
  - Attention connection strength modeling
  - Synaptic weight evolution in attention networks
  - Hebbian-based attention pattern reinforcement
- **Implementation**: `float hebbian[64]` vectors
- **Range**: [0.0, 1.0] representing synaptic strength

## Total Tensor Dimensionality

**Total DOF**: 327 dimensions (3 + 1 + 256 + 3 + 64)

The complete attention tensor combines all dimensions into a unified representation:
```
attention_tensor[327] = [spatial[3], temporal[1], semantic[256], importance[3], hebbian[64]]
```

## Attention Allocation Operations

### Core Operations

#### 1. Activation Spreading
- **Input**: Source attention tensor, neighbor connections
- **Output**: Updated attention tensors for connected atoms
- **Complexity**: O(n × d) where n = neighbors, d = tensor dimensions
- **Performance Target**: >10,000 atoms/second

#### 2. Importance Diffusion
- **Input**: Attention focus set, diffusion parameters
- **Output**: Redistributed importance values across network
- **Complexity**: O(f × c × d) where f = focus size, c = connections, d = dimensions
- **Performance Target**: >1,000 diffusion operations/second

#### 3. Tensor Computation
- **Input**: Multi-dimensional attention tensors
- **Output**: Computed attention weights and allocations
- **Operations**: Vector operations, matrix multiplications, normalization
- **Performance Target**: >100,000 tensor operations/second

### Mathematical Formulations

#### Attention Spreading Function
```
A'(t+1) = α × A(t) + β × Σ(neighbors) W(i,j) × A(j,t)
```
Where:
- A(t) = attention tensor at time t
- W(i,j) = connection weight between atoms i and j
- α, β = decay and spreading parameters

#### Importance Diffusion Function
```
STI'(i) = STI(i) × (1 - decay) + Σ(j∈neighbors) diffusion_rate × STI(j) × W(i,j)
```

#### Tensor Normalization
```
A_norm = A / ||A||₂
```

## Performance Specifications

### Throughput Requirements

| Operation | Target Rate | Unit |
|-----------|-------------|------|
| Activation Spreading | 10,000+ | atoms/second |
| Importance Diffusion | 1,000+ | operations/second |
| Tensor Computations | 100,000+ | operations/second |
| Memory Bandwidth | 1 GB/s | data throughput |

### Latency Requirements

| Operation | Target Latency | Unit |
|-----------|----------------|------|
| Single Atom Update | <100 | microseconds |
| Focus Set Update | <1 | millisecond |
| Full Network Cycle | <10 | milliseconds |

### Accuracy Requirements

| Metric | Target | Notes |
|---------|--------|-------|
| Convergence Rate | >95% | Within 100 iterations |
| Stability Measure | >0.9 | Attention pattern stability |
| Precision | 32-bit | Floating point precision |

## Implementation Guidelines

### Memory Layout
- Use structure-of-arrays (SoA) for better vectorization
- Align tensors to 32-byte boundaries for AVX2 operations
- Pre-allocate tensor pools to avoid memory fragmentation

### Optimization Strategies
- Utilize SIMD instructions for parallel tensor operations
- Implement sparse tensor representations for large networks
- Use attention focus pruning to limit computation scope
- Cache frequently accessed attention patterns

### Integration with ECAN
- Maintain compatibility with existing AttentionValue interface
- Extend AttentionBank to support tensor operations
- Integrate with ImportanceDiffusion agents
- Support for HebbianLink tensor updates

## Validation and Testing

### Unit Tests
- Tensor dimension consistency checks
- Attention spreading correctness validation
- Performance regression testing
- Memory usage monitoring

### Integration Tests
- Full ECAN cycle validation
- Multi-agent attention interaction testing
- Attention focus boundary testing
- Cross-modal attention integration

### Performance Benchmarks
- Throughput measurement under various loads
- Latency profiling for real-time applications
- Memory usage optimization validation
- Scalability testing with large atom spaces

## Future Extensions

### Adaptive Dimensions
- Dynamic tensor dimension adjustment based on problem complexity
- Learned optimal tensor configurations for specific domains
- Hierarchical tensor representations for multi-scale attention

### Hardware Acceleration
- GPU acceleration for large-scale tensor operations
- FPGA implementations for real-time applications
- Neuromorphic computing integration for biologically-inspired attention

### Advanced Operations
- Non-linear attention activation functions
- Attention tensor convolution operations
- Recursive attention pattern detection
- Meta-attention mechanisms for attention control

## References

1. OpenCog Foundation. "Attention Allocation in Cognitive Architectures"
2. Goertzel, B. et al. "Economic Attention Networks for Artificial General Intelligence"
3. ECAN Documentation: Economic Attention Allocation Network
4. AttentionBank API Reference
5. Tensor Operations Performance Guide