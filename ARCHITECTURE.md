# OpenCog Central: Comprehensive Architecture Documentation

This document provides a comprehensive overview of the OpenCog Central cognitive architecture, featuring detailed Mermaid diagrams that illustrate the emergent neural-symbolic integration and hypergraph-centric processing patterns that define the MORK (Meta-Organizational Recursive Kernel) system.

## Table of Contents

1. [High-Level System Overview](#high-level-system-overview)
2. [Core Architectural Components](#core-architectural-components)
3. [Neural-Symbolic Integration Points](#neural-symbolic-integration-points)
4. [Data Flow and Signal Propagation](#data-flow-and-signal-propagation)
5. [Cognitive Synergy and Attention Allocation](#cognitive-synergy-and-attention-allocation)
6. [Recursive Processing Patterns](#recursive-processing-patterns)
7. [Module Interaction Mappings](#module-interaction-mappings)
8. [Implementation Architecture](#implementation-architecture)

---

## High-Level System Overview

The OpenCog Central architecture implements a distributed cognitive processing system with emergent neural-symbolic integration. The system is built around hypergraph pattern encoding that enables recursive system mapping and adaptive attention allocation.

```mermaid
graph TD
    subgraph "External Environment"
        ENV[Environment Stimuli]
        SENSORS[Sensory Inputs]
        MOTORS[Motor Outputs]
    end
    
    subgraph "Cognitive Core"
        AS[AtomSpace<br/>Hypergraph Knowledge Store]
        PLN[PLN<br/>Probabilistic Logic Network]
        URE[URE<br/>Unified Rule Engine]
        ATTENTION[Attention Allocation<br/>Mechanism]
    end
    
    subgraph "Sensory-Motor System"
        SM[Sensory-Motor Interface]
        LG[Link Grammar<br/>Connector Framework]
        PIPES[Processing Pipelines]
    end
    
    subgraph "Learning & Generation"
        LEARN[Structure Learning<br/>Engine]
        GEN[Generation System]
        AGENTS[Interactive Agents]
    end
    
    subgraph "Integration Layer"
        NEUSYM[Neural-Symbolic<br/>Integration]
        PATTERNS[Pattern Matching<br/>Engine]
        RECURSIVE[Recursive Processing<br/>Controller]
    end

    ENV --> SENSORS
    SENSORS --> SM
    SM --> LG
    LG --> PIPES
    PIPES --> AS
    
    AS <--> PLN
    AS <--> URE
    AS <--> ATTENTION
    AS <--> LEARN
    AS <--> GEN
    AS <--> AGENTS
    
    PLN --> NEUSYM
    LEARN --> NEUSYM
    NEUSYM --> PATTERNS
    PATTERNS --> RECURSIVE
    
    RECURSIVE --> ATTENTION
    ATTENTION --> AS
    
    AS --> PIPES
    PIPES --> MOTORS
    MOTORS --> ENV
    
    AGENTS <--> SM
    GEN --> SM
```

### Architecture Principles

The system implements several key cognitive principles:

- **Hypergraph Pattern Encoding**: All knowledge is represented as patterns in a hypergraph structure within the AtomSpace
- **Emergent Cognitive Synergy**: Cross-modal integration emerges from the interaction of specialized subsystems
- **Adaptive Attention Allocation**: Dynamic resource allocation based on salience and goal-directed processing
- **Neural-Symbolic Integration**: Seamless combination of statistical learning and symbolic reasoning
- **Recursive System Mapping**: Self-referential processing that enables meta-cognitive capabilities

---

## Core Architectural Components

### AtomSpace: The Central Hypergraph Knowledge Representation

The AtomSpace serves as the central knowledge representation system, implementing a generalized hypergraph (metagraph) database that provides the foundation for all cognitive processing.

```mermaid
graph LR
    subgraph "AtomSpace Core"
        ATOMS[Atom Objects]
        LINKS[Link Structures]
        VALUES[Value Attachments]
        TRUTH[Truth Values<br/>STVs & PTVs]
    end
    
    subgraph "Hypergraph Operations"
        PATTERN[Pattern Engine]
        QUERY[Query System]
        REWRITE[Graph Rewriting]
        EXECUTE[Execution Engine]
    end
    
    subgraph "Storage & Persistence"
        MEMORY[In-Memory Store]
        ROCKS[RocksDB Backend]
        NETWORK[Distributed Storage]
        COG[CogServer Interface]
    end
    
    ATOMS --> PATTERN
    LINKS --> PATTERN
    VALUES --> QUERY
    TRUTH --> REWRITE
    
    PATTERN --> EXECUTE
    QUERY --> EXECUTE
    REWRITE --> EXECUTE
    
    EXECUTE --> MEMORY
    MEMORY --> ROCKS
    ROCKS --> NETWORK
    NETWORK --> COG
    
    COG -.-> ATOMS
```

**Key Features:**
- **Metagraph Representation**: More flexible than traditional graph databases, supporting arbitrary arity relationships
- **Pattern Matching**: Advanced pattern engine with variable binding and conditional execution
- **Truth Value Propagation**: Probabilistic and fuzzy truth values with confidence intervals
- **Distributed Architecture**: Network-transparent storage and computation

### Probabilistic Logic Network (PLN)

PLN provides the probabilistic reasoning foundation, handling uncertainty through second-order probability distributions.

```mermaid
stateDiagram-v2
    [*] --> ObservationIntake
    ObservationIntake --> UncertaintyModeling
    UncertaintyModeling --> ProbabilisticInference
    ProbabilisticInference --> TruthValuePropagation
    TruthValuePropagation --> ConfidenceEstimation
    ConfidenceEstimation --> RuleApplication
    RuleApplication --> ConclusionGeneration
    ConclusionGeneration --> [*]
    
    UncertaintyModeling --> SecondOrderDistributions
    SecondOrderDistributions --> UncertaintyModeling
    
    ProbabilisticInference --> CommonSenseReasoning
    CommonSenseReasoning --> ProbabilisticInference
    
    RuleApplication --> ForwardChaining
    RuleApplication --> BackwardChaining
    ForwardChaining --> RuleApplication
    BackwardChaining --> RuleApplication
```

---

## Neural-Symbolic Integration Points

The architecture achieves emergent cognitive synergy through carefully designed integration points where neural processing meets symbolic reasoning.

```mermaid
graph TB
    subgraph "Neural Processing Layer"
        DEEPNET[Deep Neural Networks]
        EMBED[Embedding Spaces]
        ATTENTION_NEURAL[Neural Attention]
        PATTERNS_NEURAL[Pattern Recognition]
    end
    
    subgraph "Integration Interface"
        BRIDGE[Neural-Symbolic Bridge]
        CONFIDENCE[Confidence Fusion]
        TRANSLATION[Representation Translation]
        SYNERGY[Cognitive Synergy Detector]
    end
    
    subgraph "Symbolic Processing Layer"
        ATOMESE[Atomese Expressions]
        LOGIC[Logical Inference]
        ATTENTION_SYM[Symbolic Attention]
        PATTERNS_SYM[Symbolic Patterns]
    end
    
    DEEPNET --> BRIDGE
    EMBED --> CONFIDENCE
    ATTENTION_NEURAL --> TRANSLATION
    PATTERNS_NEURAL --> SYNERGY
    
    BRIDGE --> ATOMESE
    CONFIDENCE --> LOGIC
    TRANSLATION --> ATTENTION_SYM
    SYNERGY --> PATTERNS_SYM
    
    ATOMESE -.-> BRIDGE
    LOGIC -.-> CONFIDENCE
    ATTENTION_SYM -.-> TRANSLATION
    PATTERNS_SYM -.-> SYNERGY
```

### Integration Mechanisms

**Confidence Fusion**: Combines neural network confidence scores with symbolic truth values through weighted integration:
- Neural confidence scores → Probabilistic truth values
- Symbolic inference strength → Neural attention weights
- Cross-modal validation through bidirectional feedback

**Representation Translation**: Bi-directional mapping between vector embeddings and symbolic expressions:
- Vector → Symbol: Pattern matching in embedding space
- Symbol → Vector: Compositional embedding generation
- Attention synchronization between modalities

---

## Data Flow and Signal Propagation

The system implements sophisticated data flow patterns that enable both feedforward processing and recursive feedback loops.

```mermaid
sequenceDiagram
    participant ENV as Environment
    participant SENSOR as Sensory Interface
    participant LG as Link Grammar
    participant AS as AtomSpace
    participant PLN as PLN Engine
    participant ATTENTION as Attention System
    participant AGENT as Agent Controller
    participant MOTOR as Motor Interface

    ENV->>SENSOR: Sensory Stimuli
    SENSOR->>LG: Structured Input
    LG->>AS: Hypergraph Patterns
    
    AS->>PLN: Query Patterns
    PLN->>AS: Inference Results
    
    AS->>ATTENTION: Salience Computation
    ATTENTION->>AS: Priority Updates
    
    AS->>AGENT: Decision Context
    AGENT->>AS: Action Plans
    
    AS->>LG: Motor Commands
    LG->>MOTOR: Structured Output
    MOTOR->>ENV: Environmental Actions
    
    Note over ENV,MOTOR: Recursive Feedback Loop
    ENV-->>SENSOR: Environmental Response
    
    Note over AS,ATTENTION: Adaptive Attention Allocation
    ATTENTION-->>PLN: Focus Modulation
    PLN-->>ATTENTION: Confidence Feedback
```

### Signal Processing Pathways

**Forward Propagation Path**:
1. Environmental stimuli → Sensory processing
2. Link Grammar parsing → AtomSpace integration
3. PLN inference → Decision generation
4. Motor command synthesis → Environmental action

**Recursive Feedback Path**:
1. Environmental response → Prediction error computation
2. Attention reallocation → Priority updates
3. Pattern refinement → Knowledge base updates
4. Action adaptation → Improved motor commands

---

## Cognitive Synergy and Attention Allocation

The attention allocation mechanism implements dynamic resource management based on salience computation and goal-directed processing.

```mermaid
graph TD
    subgraph "Attention Allocation System"
        SALIENCE[Salience Computer]
        URGENCY[Urgency Evaluator]
        RELEVANCE[Relevance Assessor]
        RESOURCE[Resource Allocator]
    end
    
    subgraph "Cognitive Resources"
        PROC_POWER[Processing Power]
        MEMORY_BAND[Memory Bandwidth]
        INFERENCE_CAP[Inference Capacity]
        FOCUS_SPAN[Attention Span]
    end
    
    subgraph "Synergy Detection"
        CROSS_MODAL[Cross-Modal Integration]
        EMERGENCE[Emergent Pattern Detection]
        COHERENCE[Cognitive Coherence Monitor]
        OPTIMIZATION[Synergy Optimization]
    end
    
    subgraph "Feedback Loops"
        PERFORMANCE[Performance Monitor]
        ADAPTATION[Adaptive Controller]
        LEARNING[Learning Signal Generator]
        MEMORY[Memory Consolidation]
    end
    
    SALIENCE --> RESOURCE
    URGENCY --> RESOURCE
    RELEVANCE --> RESOURCE
    
    RESOURCE --> PROC_POWER
    RESOURCE --> MEMORY_BAND
    RESOURCE --> INFERENCE_CAP
    RESOURCE --> FOCUS_SPAN
    
    PROC_POWER --> CROSS_MODAL
    MEMORY_BAND --> EMERGENCE
    INFERENCE_CAP --> COHERENCE
    FOCUS_SPAN --> OPTIMIZATION
    
    CROSS_MODAL --> PERFORMANCE
    EMERGENCE --> PERFORMANCE
    COHERENCE --> PERFORMANCE
    OPTIMIZATION --> PERFORMANCE
    
    PERFORMANCE --> ADAPTATION
    ADAPTATION --> LEARNING
    LEARNING --> MEMORY
    
    MEMORY -.-> SALIENCE
    LEARNING -.-> URGENCY
    ADAPTATION -.-> RELEVANCE
```

### Adaptive Attention Mechanisms

**Salience Computation**: Dynamic weighting based on:
- Novelty detection (information-theoretic surprise)
- Goal relevance (utility function optimization)
- Temporal urgency (deadline-driven priorities)
- Cross-modal consistency (coherence rewards)

**Resource Allocation Strategy**:
- Processing power: Allocated based on problem complexity
- Memory bandwidth: Prioritized for high-salience patterns
- Inference capacity: Distributed across active reasoning chains
- Attention span: Dynamically adjusted for sustained focus

---

## Recursive Processing Patterns

The architecture implements recursive processing through self-referential pattern matching and meta-cognitive monitoring.

```mermaid
stateDiagram-v2
    [*] --> InitialProcessing
    InitialProcessing --> PatternRecognition
    PatternRecognition --> MetaAnalysis
    MetaAnalysis --> RecursiveCall
    RecursiveCall --> PatternRefinement
    PatternRefinement --> ConvergenceCheck
    ConvergenceCheck --> FinalIntegration
    FinalIntegration --> [*]
    
    ConvergenceCheck --> RecursiveCall : Not Converged
    PatternRefinement --> MetaAnalysis : Self-Monitoring
    MetaAnalysis --> InitialProcessing : Meta-Learning
    
    state RecursiveCall {
        [*] --> SubpatternAnalysis
        SubpatternAnalysis --> RecursiveInference
        RecursiveInference --> SubpatternIntegration
        SubpatternIntegration --> [*]
    }
    
    state MetaAnalysis {
        [*] --> SelfAssessment
        SelfAssessment --> ConfidenceEvaluation
        ConfidenceEvaluation --> StrategySelection
        StrategySelection --> [*]
    }
```

### Recursive Implementation Pathways

**Pattern Hierarchy Processing**:
1. **Base Level**: Direct pattern matching on sensory input
2. **Meta Level**: Pattern matching on pattern matching processes
3. **Meta-Meta Level**: Monitoring and optimization of meta-processes
4. **Recursive Termination**: Convergence detection and integration

**Self-Referential Mechanisms**:
- **Code-Data Equivalence**: Atomese programs stored as AtomSpace data
- **Self-Modification**: Programs that modify their own processing patterns
- **Meta-Cognitive Monitoring**: Attention allocation for self-assessment
- **Emergent Complexity**: Higher-order patterns emerging from recursive interactions

---

## Module Interaction Mappings

The following diagram illustrates the bidirectional interaction patterns between major architectural modules.

```mermaid
graph LR
    subgraph "Sensory-Motor Cluster"
        SENSORY[Sensory Processing]
        MOTOR[Motor Control]
        LG_SM[Link Grammar Interface]
    end
    
    subgraph "Cognitive Core Cluster"
        AS_CORE[AtomSpace Core]
        PLN_CORE[PLN Reasoning]
        URE_CORE[Rule Engine]
        ATT_CORE[Attention Manager]
    end
    
    subgraph "Learning Cluster"
        STRUCT_LEARN[Structure Learning]
        PATTERN_LEARN[Pattern Learning]
        AGENT_LEARN[Agent Learning]
    end
    
    subgraph "Generation Cluster"
        TEXT_GEN[Text Generation]
        PLAN_GEN[Plan Generation]
        RESPONSE_GEN[Response Generation]
    end
    
    SENSORY <--> LG_SM
    LG_SM <--> AS_CORE
    MOTOR <--> LG_SM
    
    AS_CORE <--> PLN_CORE
    AS_CORE <--> URE_CORE
    AS_CORE <--> ATT_CORE
    PLN_CORE <--> URE_CORE
    
    AS_CORE <--> STRUCT_LEARN
    PLN_CORE <--> PATTERN_LEARN
    ATT_CORE <--> AGENT_LEARN
    
    PLN_CORE <--> TEXT_GEN
    URE_CORE <--> PLAN_GEN
    AS_CORE <--> RESPONSE_GEN
    
    STRUCT_LEARN --> TEXT_GEN
    PATTERN_LEARN --> PLAN_GEN
    AGENT_LEARN --> RESPONSE_GEN
    
    TEXT_GEN -.-> SENSORY
    PLAN_GEN -.-> MOTOR
    RESPONSE_GEN -.-> LG_SM
```

### Inter-Module Communication Protocols

**Message Passing Architecture**:
- **AtomSpace Queries**: Pattern-based message routing
- **Value Streaming**: Continuous data flow between modules
- **Event Broadcasting**: Asynchronous notification system
- **Synchronization Points**: Coordinated processing stages

**Data Format Standards**:
- **Atomese Expressions**: Universal representation format
- **Truth Value Propagation**: Uncertainty information transfer
- **Connector Protocols**: Link Grammar interface specifications
- **Temporal Coordination**: Time-stamped processing sequences

---

## Implementation Architecture

The implementation follows a modular design that supports distributed processing and incremental development.

```mermaid
graph TB
    subgraph "Application Layer"
        CHATBOTS[Chatbot Interfaces]
        ROBOTICS[Robotic Control]
        RESEARCH[Research Tools]
        DEMOS[Demo Applications]
    end
    
    subgraph "API Layer"
        SCHEME_API[Scheme API]
        PYTHON_API[Python API]
        REST_API[REST API]
        WEBSOCKET[WebSocket API]
    end
    
    subgraph "Core Services"
        COGSERVER[CogServer]
        ATOMSPACE_SERVICE[AtomSpace Service]
        PLN_SERVICE[PLN Service]
        ATTENTION_SERVICE[Attention Service]
    end
    
    subgraph "Processing Engines"
        PATTERN_ENGINE[Pattern Engine]
        INFERENCE_ENGINE[Inference Engine]
        EXECUTION_ENGINE[Execution Engine]
        LEARNING_ENGINE[Learning Engine]
    end
    
    subgraph "Storage Layer"
        MEMORY_STORE[Memory Store]
        PERSISTENT_STORE[Persistent Store]
        DISTRIBUTED_STORE[Distributed Store]
        BACKUP_STORE[Backup Store]
    end
    
    subgraph "System Infrastructure"
        THREADING[Threading System]
        MESSAGING[Message Passing]
        MONITORING[System Monitoring]
        CONFIGURATION[Configuration Management]
    end
    
    CHATBOTS --> SCHEME_API
    ROBOTICS --> PYTHON_API
    RESEARCH --> REST_API
    DEMOS --> WEBSOCKET
    
    SCHEME_API --> COGSERVER
    PYTHON_API --> ATOMSPACE_SERVICE
    REST_API --> PLN_SERVICE
    WEBSOCKET --> ATTENTION_SERVICE
    
    COGSERVER --> PATTERN_ENGINE
    ATOMSPACE_SERVICE --> INFERENCE_ENGINE
    PLN_SERVICE --> EXECUTION_ENGINE
    ATTENTION_SERVICE --> LEARNING_ENGINE
    
    PATTERN_ENGINE --> MEMORY_STORE
    INFERENCE_ENGINE --> PERSISTENT_STORE
    EXECUTION_ENGINE --> DISTRIBUTED_STORE
    LEARNING_ENGINE --> BACKUP_STORE
    
    MEMORY_STORE --> THREADING
    PERSISTENT_STORE --> MESSAGING
    DISTRIBUTED_STORE --> MONITORING
    BACKUP_STORE --> CONFIGURATION
```

### Deployment Architecture

**Containerized Services**:
- Each major component runs in dedicated containers
- Docker-compose orchestration for development
- Kubernetes deployment for production scaling
- Service mesh for inter-component communication

**Scalability Patterns**:
- **Horizontal Scaling**: AtomSpace clustering across nodes
- **Vertical Scaling**: CPU/memory allocation per service
- **Load Balancing**: Request distribution across instances
- **Fault Tolerance**: Automatic failover and recovery

---

## Conclusion

The OpenCog Central architecture represents a sophisticated implementation of cognitive computing principles, integrating neural and symbolic processing through hypergraph-based knowledge representation. The recursive processing patterns and adaptive attention allocation mechanisms enable emergent cognitive behaviors that transcend the capabilities of individual components.

The modular design ensures extensibility while maintaining cognitive coherence, providing a robust foundation for artificial general intelligence research and applications. The comprehensive Mermaid documentation facilitates understanding and contribution from distributed development teams, supporting the collaborative evolution of this advanced cognitive architecture.

### Future Directions

- **Enhanced Neural Integration**: Deeper fusion of neural and symbolic processing via the Orchestral Architect framework
- **Distributed Cognition**: Multi-agent cognitive coordination through agentic kernel networks
- **Quantum Integration**: Quantum-classical hybrid processing capabilities
- **Biological Modeling**: Brain-inspired architectural refinements

## Orchestral Architect: Agentic Cognitive Grammar System

The Orchestral Architect represents the next evolution of the OpenCog architecture, implementing a distributed network of specialized cognitive kernels that operate as autonomous agents within the cognitive ecosystem.

```mermaid
graph TB
    subgraph "Orchestral Layer"
        ORC[Orchestral System]
        KR[Kernel Registry]
        EAA[Economic Attention Allocation]
        EDC[Event-Driven Communication]
    end
    
    subgraph "Cognitive Kernels"
        TK[Tokenization Kernel]
        AK[Attention Kernel] 
        RK[Reasoning Kernel]
        LK[Learning Kernel]
    end
    
    subgraph "Neural-Symbolic Bridge"
        NSI[Neural-Symbolic Integration]
        CF[Confidence Fusion]
        RT[Representation Translation]
    end
    
    subgraph "OpenCog Core"
        AS[AtomSpace]
        PLN[PLN]
        URE[URE]
        ECAN[ECAN]
    end
    
    ORC --> KR
    ORC --> EAA
    ORC --> EDC
    
    KR --> TK
    KR --> AK
    KR --> RK
    KR --> LK
    
    TK --> NSI
    RK --> NSI
    
    NSI --> CF
    NSI --> RT
    
    EAA --> AS
    CF --> PLN
    RT --> URE
    EDC --> ECAN
```

### Key Features of the Orchestral Architect

**🎼 Distributed Agentic Processing**: Autonomous cognitive kernels that communicate via event-driven messaging and coordinate through economic attention allocation.

**💰 Economic Attention Market**: ECAN-inspired attention allocation system that optimizes cognitive resource distribution based on value/cost analysis.

**🧠 Neural-Symbolic Integration**: Seamless bridge between neural network processing and symbolic reasoning, enabling hybrid cognitive architectures.

**⚡ High-Performance Processing**: Demonstrated processing speeds of ~14,773 operations/second with efficient attention weighting and tokenization.

**🔗 Event-Driven Architecture**: Asynchronous communication system that enables scalable coordination between distributed cognitive components.

### Live Demonstration Results

The Orchestral Architect system successfully processes cognitive inputs with measurable attention allocation:

```
Input: "hello world this is a simple test"
→ Multi-strategy tokenization with attention weighting
→ Attention allocation: 'simple' (0.83), 'hello' (0.90)
→ Economic analysis: Cost (0.221), Value (3.924), Efficiency (17.757)
→ Event-driven kernel coordination and real-time metrics
```

### Integration with OpenCog Ecosystem

The Orchestral Architect enhances the existing OpenCog architecture by:

- **Extending AtomSpace**: Providing agentic interfaces for hypergraph operations
- **Enhancing ECAN**: Adding market-based attention dynamics with economic optimization
- **Augmenting PLN**: Enabling distributed probabilistic reasoning across kernel networks
- **Scaling CogServer**: Supporting distributed cognitive processing with load balancing

For detailed technical documentation and live demonstration, see: `orchestral-architect/docs/TECHNICAL_ARCHITECTURE.md`

---

*This documentation is maintained as a living document, evolving with the architecture itself to ensure accuracy and completeness of the cognitive pattern descriptions.*