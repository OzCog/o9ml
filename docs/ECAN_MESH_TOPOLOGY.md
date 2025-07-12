# ECAN Mesh Topology and Dynamic State Propagation

## Overview

The ECAN (Economic Attention Allocation Network) system implements a sophisticated mesh topology for distributed attention allocation across cognitive agents. This document describes the mesh architecture, dynamic state propagation mechanisms, and cross-agent synchronization protocols.

## Mesh Topology Architecture

### Core Components

The ECAN mesh consists of interconnected cognitive nodes, each running attention allocation kernels that coordinate through economic protocols:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ECAN Mesh Network                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │ Agent A │◄──►│ Agent B │◄──►│ Agent C │◄──►│ Agent D │     │
│  │  ECAN   │    │  ECAN   │    │  ECAN   │    │  ECAN   │     │
│  │ Kernel  │    │ Kernel  │    │ Kernel  │    │ Kernel  │     │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│       ▲             ▲             ▲             ▲             │
│       │             │             │             │             │
│       ▼             ▼             ▼             ▼             │
│  ┌─────────────────────────────────────────────────────┐     │
│  │          Distributed Attention Coordinator          │     │
│  │                                                     │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │     │
│  │  │ Attention   │  │ Economic    │  │ Resource    │ │     │
│  │  │ Spreading   │  │ Allocator   │  │ Scheduler   │ │     │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │     │
│  └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### Network Topology Types

#### 1. Hierarchical Mesh
- **Structure**: Tree-like with central coordination nodes
- **Use Case**: Task-oriented cognitive processing
- **Advantages**: Clear command structure, efficient resource allocation
- **Disadvantages**: Single points of failure

#### 2. Peer-to-Peer Mesh
- **Structure**: Fully connected network of equal peers
- **Use Case**: Distributed problem solving, creative tasks
- **Advantages**: High resilience, balanced load distribution
- **Disadvantages**: Complex coordination protocols

#### 3. Hybrid Mesh
- **Structure**: Combines hierarchical and P2P elements
- **Use Case**: General-purpose cognitive architectures
- **Advantages**: Balances efficiency and resilience
- **Disadvantages**: Increased complexity

## Dynamic State Propagation

### Attention State Synchronization

The ECAN mesh maintains coherent attention state across all nodes through sophisticated propagation algorithms:

#### State Vector Format
```python
# ECAN Attention State Vector
attention_state = {
    'node_id': str,                    # Unique node identifier
    'timestamp': float,                # State timestamp
    'attention_tensors': {             # Active attention tensors
        'atom_id': {
            'short_term_importance': float,
            'long_term_importance': float,
            'urgency': float,
            'confidence': float,
            'spreading_factor': float,
            'decay_rate': float,
            'last_updated': float
        }
    },
    'focus_set': List[str],           # Current attention focus atoms
    'resource_allocation': {          # Economic resource state
        'total_budget': float,
        'allocated_budget': float,
        'allocation_efficiency': float
    },
    'mesh_connectivity': {            # Network connectivity state
        'connected_nodes': List[str],
        'link_strengths': Dict[str, float],
        'propagation_delays': Dict[str, float]
    }
}
```

### Propagation Algorithms

#### 1. Epidemic Propagation
- **Method**: Gossip-based state dissemination
- **Latency**: O(log N) rounds for full propagation
- **Bandwidth**: O(N log N) total messages
- **Resilience**: High fault tolerance

```python
def epidemic_propagate(state, neighbors, infection_probability=0.8):
    """
    Epidemic-style state propagation with configurable infection rates.
    """
    for neighbor in neighbors:
        if random.random() < infection_probability:
            send_state_update(neighbor, state)
            merge_received_state(receive_state_update(neighbor))
```

#### 2. Hierarchical Cascade
- **Method**: Tree-based top-down propagation
- **Latency**: O(log N) deterministic
- **Bandwidth**: O(N) total messages
- **Resilience**: Moderate (depends on root node)

#### 3. Ring-based Propagation
- **Method**: Token-passing state updates
- **Latency**: O(N) worst case
- **Bandwidth**: O(N) minimal overhead
- **Resilience**: High (self-healing ring)

### Conflict Resolution Protocols

When multiple nodes update the same attention atom simultaneously, the ECAN mesh employs conflict resolution:

#### Vector Clock Synchronization
```python
class VectorClock:
    """
    Vector clock for distributed attention state ordering.
    """
    def __init__(self, node_id, initial_nodes):
        self.node_id = node_id
        self.clock = {node: 0 for node in initial_nodes}
    
    def tick(self):
        """Increment local clock"""
        self.clock[self.node_id] += 1
        return self.clock.copy()
    
    def update(self, other_clock):
        """Update clock with received timestamp"""
        for node, time in other_clock.items():
            if node in self.clock:
                self.clock[node] = max(self.clock[node], time)
        self.tick()
    
    def compare(self, other_clock):
        """Compare vector clocks for causality"""
        # Returns: 'before', 'after', 'concurrent'
        all_leq = all(self.clock[n] <= other_clock.get(n, 0) for n in self.clock)
        all_geq = all(self.clock[n] >= other_clock.get(n, 0) for n in self.clock)
        
        if all_leq and not all_geq:
            return 'before'
        elif all_geq and not all_leq:
            return 'after'
        else:
            return 'concurrent'
```

#### Conflict Resolution Strategies

1. **Last-Writer-Wins**: Simple timestamp-based resolution
2. **Attention-Weighted Merge**: Merge based on attention salience
3. **Economic Priority**: Resolve based on economic value
4. **Consensus-Based**: Multi-node voting on conflicts

### Cross-Agent Attention Synchronization

#### Synchronization Triggers

1. **Periodic Sync**: Regular intervals (1-10 seconds)
2. **Threshold-Based**: When attention changes exceed threshold
3. **Focus-Change Sync**: When attention focus atoms change
4. **Economic Event Sync**: Resource allocation changes
5. **External Trigger Sync**: User or system commands

#### Synchronization Protocol

```python
class AttentionSynchronizer:
    """
    Manages cross-agent attention synchronization in ECAN mesh.
    """
    
    def __init__(self, node_id, mesh_topology):
        self.node_id = node_id
        self.mesh = mesh_topology
        self.vector_clock = VectorClock(node_id, mesh_topology.nodes())
        self.pending_updates = PriorityQueue()
        
    def sync_attention_state(self, attention_kernel):
        """
        Synchronize local attention state with mesh network.
        """
        # 1. Collect local state
        local_state = self._collect_local_state(attention_kernel)
        
        # 2. Generate state update message
        timestamp = self.vector_clock.tick()
        update_message = {
            'source_node': self.node_id,
            'timestamp': timestamp,
            'state_delta': local_state,
            'message_type': 'attention_sync'
        }
        
        # 3. Propagate to connected nodes
        self._propagate_update(update_message)
        
        # 4. Process pending updates
        self._process_pending_updates(attention_kernel)
        
        # 5. Resolve conflicts
        conflicts = self._detect_conflicts()
        if conflicts:
            self._resolve_conflicts(conflicts, attention_kernel)
    
    def _collect_local_state(self, attention_kernel):
        """Collect current local attention state"""
        focus_atoms = attention_kernel.get_attention_focus()
        distribution = attention_kernel.compute_global_attention_distribution()
        
        return {
            'focus_atoms': [atom_id for atom_id, _ in focus_atoms],
            'attention_distribution': distribution,
            'performance_metrics': attention_kernel.get_performance_metrics(),
            'last_updated': time.time()
        }
    
    def _propagate_update(self, message):
        """Propagate state update to mesh neighbors"""
        neighbors = self.mesh.get_neighbors(self.node_id)
        
        for neighbor in neighbors:
            try:
                self.mesh.send_message(neighbor, message)
            except NetworkException as e:
                # Handle network failures gracefully
                self._handle_propagation_failure(neighbor, message, e)
    
    def _process_pending_updates(self, attention_kernel):
        """Process received updates from other nodes"""
        while not self.pending_updates.empty():
            priority, message = self.pending_updates.get()
            
            # Check if update is still relevant
            if self._is_update_relevant(message):
                self._apply_remote_update(message, attention_kernel)
    
    def _resolve_conflicts(self, conflicts, attention_kernel):
        """Resolve attention allocation conflicts"""
        for conflict in conflicts:
            atom_id = conflict['atom_id']
            competing_updates = conflict['updates']
            
            # Strategy: Attention-weighted merge
            resolved_tensor = self._merge_attention_tensors(competing_updates)
            
            # Apply resolved state
            attention_kernel.allocate_attention(atom_id, resolved_tensor)
    
    def _merge_attention_tensors(self, tensor_updates):
        """Merge competing attention tensor updates"""
        total_weight = sum(update['salience'] for update in tensor_updates)
        
        if total_weight == 0:
            return tensor_updates[0]['tensor']  # Fallback to first
        
        # Weighted average merge
        merged = ECANAttentionTensor()
        
        for update in tensor_updates:
            weight = update['salience'] / total_weight
            tensor = update['tensor']
            
            merged.short_term_importance += tensor.short_term_importance * weight
            merged.long_term_importance += tensor.long_term_importance * weight
            merged.urgency += tensor.urgency * weight
            merged.confidence += tensor.confidence * weight
            merged.spreading_factor += tensor.spreading_factor * weight
            merged.decay_rate += tensor.decay_rate * weight
        
        return merged
```

## Mesh Resilience and Fault Tolerance

### Failure Detection

#### Heartbeat Monitoring
- **Frequency**: Every 5-30 seconds
- **Timeout**: 3x heartbeat interval
- **False Positive Rate**: <1%

#### Network Partition Detection
- **Method**: Consensus-based partition detection
- **Recovery**: Automatic mesh reformation
- **State Reconciliation**: Vector clock-based merge

### Recovery Mechanisms

#### Node Failure Recovery
1. **Detection**: Missed heartbeats, connection timeouts
2. **Isolation**: Remove failed node from active topology
3. **Redistribution**: Reallocate attention resources
4. **Rejoin**: Graceful reintegration when node recovers

#### Network Partition Recovery
1. **Detection**: Inconsistent global state, split consensus
2. **Leader Election**: Establish partition leaders
3. **Independent Operation**: Continue with reduced capabilities
4. **Merge Protocol**: Reconcile state when partitions heal

#### State Corruption Recovery
1. **Checksums**: Validate state integrity
2. **Rollback**: Revert to last known good state
3. **Reconstruction**: Rebuild from distributed backups
4. **Verification**: Consensus-based state validation

### Performance Optimization

#### Adaptive Topology
- **Dynamic Rewiring**: Optimize connections based on communication patterns
- **Load Balancing**: Distribute attention processing load
- **Bandwidth Management**: Prioritize critical synchronization messages

#### Compression and Caching
- **State Compression**: Reduce synchronization message size
- **Delta Updates**: Send only changes, not full state
- **Predictive Caching**: Pre-load likely needed attention states

## Implementation Guidelines

### Setting Up ECAN Mesh

```python
from ecan import AttentionKernel, EconomicAllocator, AttentionSpreading
from ecan.mesh import ECANMeshNode, MeshTopology

# 1. Initialize mesh node
node = ECANMeshNode(
    node_id="cognitive_agent_1",
    mesh_config={
        'topology_type': 'hybrid',
        'max_connections': 8,
        'heartbeat_interval': 10.0,
        'sync_threshold': 0.1
    }
)

# 2. Setup ECAN components
attention_kernel = AttentionKernel(max_atoms=1000)
economic_allocator = EconomicAllocator(total_attention_budget=200.0)
attention_spreading = AttentionSpreading()

# 3. Configure mesh coordination
node.setup_attention_systems(
    kernel=attention_kernel,
    allocator=economic_allocator,
    spreading=attention_spreading
)

# 4. Join mesh network
node.connect_to_mesh([
    "cognitive_agent_2:8080",
    "cognitive_agent_3:8080"
])

# 5. Start synchronization
node.start_synchronization()
```

### Monitoring and Diagnostics

```python
# Monitor mesh health
mesh_status = node.get_mesh_status()
print(f"Connected nodes: {mesh_status['connected_nodes']}")
print(f"Sync latency: {mesh_status['avg_sync_latency']:.2f}ms")
print(f"Message throughput: {mesh_status['messages_per_second']}")

# Monitor attention distribution
attention_stats = node.get_attention_statistics()
print(f"Local focus atoms: {attention_stats['local_focus_count']}")
print(f"Global attention coherence: {attention_stats['global_coherence']:.2f}")
print(f"Conflict resolution rate: {attention_stats['conflict_rate']:.1%}")
```

## Best Practices

### Mesh Design Principles

1. **Minimize Synchronization Overhead**
   - Use threshold-based synchronization
   - Batch multiple updates
   - Prioritize high-salience attention changes

2. **Design for Partial Failures**
   - Assume network partitions will occur
   - Implement graceful degradation
   - Maintain local autonomy

3. **Optimize for Common Cases**
   - Most attention changes are local
   - Focus synchronization on shared attention atoms
   - Use predictive pre-synchronization

4. **Monitor and Adapt**
   - Track synchronization performance
   - Adjust topology based on usage patterns
   - Implement adaptive algorithms

### Performance Tuning

- **Sync Frequency**: Balance responsiveness vs. overhead
- **Mesh Density**: Optimize connection count for resilience/performance
- **Message Batching**: Group updates to reduce network traffic
- **Compression**: Use efficient serialization for state updates

## Conclusion

The ECAN mesh topology provides a robust foundation for distributed attention allocation across cognitive agents. Through sophisticated synchronization protocols, conflict resolution mechanisms, and fault tolerance features, the mesh enables scalable and resilient cognitive architectures that can adapt to changing conditions while maintaining coherent attention state across the network.

The dynamic state propagation mechanisms ensure that attention allocation decisions are coordinated across the mesh, enabling emergent cognitive behaviors that transcend individual agent capabilities while preserving the economic efficiency principles of the ECAN attention allocation model.