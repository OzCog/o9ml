# OpenCog Central GitHub Actions Build Architecture

This document describes the cognitive orchestration build system implemented in GitHub Actions, mapping the original CircleCI configuration to the actual repository structure.

## Cognitive Hypergraph Architecture

```mermaid
graph TB
    %% Foundation Layer - Tensor DOF = 1 (Pure Utilities)
    subgraph "Foundation Layer"
        CU[cogutil<br/>Base Utilities]
        MOSES[moses<br/>Evolutionary Search]
    end
    
    %% Core Layer - Tensor DOF = 2 (Graph + Hypergraph)
    subgraph "Core Layer - AtomSpace Orchestration"
        AS[atomspace<br/>Core Hypergraph]
        ASR[atomspace-rocks<br/>Persistent Storage]
        ASREST[atomspace-restful<br/>API Interface]
        ASWS[atomspace-websockets<br/>Real-time Interface]
        ASMETTA[atomspace-metta<br/>MeTTa Integration]
    end
    
    %% Logic Layer - Tensor DOF = 3 (Logic + Inference + Meta-reasoning)
    subgraph "Logic Layer"
        URE[ure<br/>Unified Rule Engine]
        UNIFY[unify<br/>Unification System]
    end
    
    %% Cognitive Layer - Tensor DOF = 4 (Attention + Space + Time + Emergence)
    subgraph "Cognitive Layer"
        ATT[attention<br/>Attention Allocation]
        ST[spacetime<br/>Spatiotemporal Reasoning]
        CS[cogserver<br/>Cognitive Server]
    end
    
    %% Advanced Layer - Tensor DOF = 5 (Pattern + Logic + Learning + Adaptation + Meta-learning)
    subgraph "Advanced Layer"
        PLN[pln<br/>Probabilistic Logic Networks]
        MINER[miner<br/>Pattern Mining]
        ASMOSES[asmoses<br/>AtomSpace MOSES]
    end
    
    %% Learning Layer - Tensor DOF = 6 (Multi-modal Learning)
    subgraph "Learning Layer"
        LEARN[learn<br/>Learning Systems]
        GEN[generate<br/>Generation Systems]
    end
    
    %% Language Layer - Tensor DOF = 7 (Syntax + Semantics + Pragmatics + Context + Generation + Understanding + Meta-language)
    subgraph "Language Layer"
        LGATOMESE[lg-atomese<br/>Link Grammar Integration]
        RELEX[relex<br/>Relation Extraction]
        LINKGRAM[link-grammar<br/>Syntactic Parser]
    end
    
    %% Embodiment Layer - Tensor DOF = 8 (Multi-sensory + Motor + Spatial + Temporal + Social + Emotional + Cognitive + Meta-cognitive)
    subgraph "Embodiment Layer"
        VISION[vision<br/>Computer Vision]
        PERCEPT[perception<br/>Perception Systems]
        SENSORY[sensory<br/>Sensory Processing]
    end
    
    %% Integration Layer - Tensor DOF = 9 (Complete Cognitive Integration)
    subgraph "Integration Layer"
        OC[opencog<br/>Central Integration]
    end
    
    %% Packaging Layer - Deployment orchestration
    subgraph "Packaging Layer"
        DEB[debian<br/>Package]
        NIX[nix<br/>Package]
    end
    
    %% Dependencies - Hypergraph edges encoding cognitive emergence
    CU --> AS
    MOSES --> AS
    AS --> ASR
    AS --> ASREST
    AS --> ASWS
    AS --> ASMETTA
    AS --> URE
    AS --> UNIFY
    URE --> ATT
    URE --> ST
    URE --> CS
    UNIFY --> ATT
    UNIFY --> ST
    UNIFY --> CS
    ATT --> PLN
    ST --> PLN
    CS --> PLN
    ATT --> MINER
    ST --> MINER
    CS --> MINER
    URE --> ASMOSES
    PLN --> LEARN
    MINER --> LEARN
    ASMOSES --> LEARN
    PLN --> GEN
    MINER --> GEN
    ASMOSES --> GEN
    AS --> LGATOMESE
    CS --> LGATOMESE
    AS --> RELEX
    CS --> RELEX
    AS --> LINKGRAM
    CS --> LINKGRAM
    AS --> VISION
    CS --> VISION
    AS --> PERCEPT
    CS --> PERCEPT
    AS --> SENSORY
    CS --> SENSORY
    LEARN --> OC
    GEN --> OC
    LGATOMESE --> OC
    RELEX --> OC
    LINKGRAM --> OC
    VISION --> OC
    PERCEPT --> OC
    SENSORY --> OC
    OC --> DEB
    OC --> NIX
```

## GitHub Actions Workflow Architecture

```mermaid
graph LR
    subgraph "Parallel Execution Membranes"
        subgraph "Foundation Jobs"
            J1[cogutil]
            J2[moses]
        end
        
        subgraph "Core Jobs"
            J3[atomspace]
            J4[atomspace-rocks]
            J5[atomspace-restful]
        end
        
        subgraph "Logic Jobs"
            J6[ure]
            J7[unify]
        end
        
        subgraph "Cognitive Jobs"
            J8[cogserver]
            J9[attention]
            J10[spacetime]
        end
        
        subgraph "Advanced Jobs"
            J11[pln]
            J12[miner]
            J13[asmoses]
        end
        
        subgraph "Learning Jobs"
            J14[learn]
            J15[generate]
        end
        
        subgraph "Language Jobs"
            J16[lg-atomese]
            J17[relex]
            J18[link-grammar]
        end
        
        subgraph "Embodiment Jobs"
            J19[vision]
            J20[perception]
            J21[sensory]
        end
        
        subgraph "Integration Jobs"
            J22[opencog]
        end
        
        subgraph "Packaging Jobs"
            J23[debian]
            J24[nix]
            J25[docs]
        end
    end
    
    %% Dependencies
    J1 --> J3
    J2 --> J3
    J3 --> J4
    J3 --> J5
    J3 --> J6
    J3 --> J7
    J6 --> J8
    J6 --> J9
    J6 --> J10
    J7 --> J8
    J7 --> J9
    J7 --> J10
    J8 --> J11
    J9 --> J11
    J10 --> J11
    J8 --> J12
    J9 --> J12
    J10 --> J12
    J6 --> J13
    J11 --> J14
    J12 --> J14
    J13 --> J14
    J11 --> J15
    J12 --> J15
    J13 --> J15
    J8 --> J16
    J9 --> J16
    J10 --> J16
    J8 --> J17
    J9 --> J17
    J10 --> J17
    J8 --> J18
    J9 --> J18
    J10 --> J18
    J8 --> J19
    J9 --> J19
    J10 --> J19
    J8 --> J20
    J9 --> J20
    J10 --> J20
    J8 --> J21
    J9 --> J21
    J10 --> J21
    J14 --> J22
    J15 --> J22
    J16 --> J22
    J17 --> J22
    J18 --> J22
    J19 --> J22
    J20 --> J22
    J21 --> J22
    J22 --> J23
    J22 --> J24
    J22 --> J25
```

## Repository Structure Mapping

The GitHub Actions workflow adapts to the actual repository structure:

```
cogml/
├── orc-dv/          # Development orchestration
│   └── cogutil/     # Foundation utilities
├── orc-as/          # AtomSpace orchestration  
│   ├── atomspace/
│   ├── atomspace-rocks/
│   └── atomspace-restful/
├── orc-ai/          # AI orchestration
│   ├── moses/
│   ├── miner/
│   ├── pln/
│   ├── ure/
│   ├── asmoses/
│   └── learn/
└── orc-*/           # Other orchestral components
```

## Implementation Details

### Cognitive Tensor Shape Design

Each layer represents a different cognitive tensor dimension:

1. **Foundation (DOF=1)**: Pure utilities and basic functions
2. **Core (DOF=2)**: Hypergraph representation and storage
3. **Logic (DOF=3)**: Reasoning and unification
4. **Cognitive (DOF=4)**: Attention, space, time, emergence
5. **Advanced (DOF=5)**: Pattern recognition, probabilistic logic, learning
6. **Learning (DOF=6)**: Multi-modal learning systems
7. **Language (DOF=7)**: Natural language processing
8. **Embodiment (DOF=8)**: Sensory and motor integration
9. **Integration (DOF=9)**: Complete cognitive system

### Parallelism and Dependencies

- **Parallel Execution**: Jobs within the same membrane (layer) without mutual dependencies run in parallel
- **Dependency Encoding**: The `needs` directive encodes hypergraph links between cognitive components
- **Attention Allocation**: Resource allocation optimized for parallel builds with proper dependency ordering

### Extension Points

Each job can be extended with:
- Matrix parallelism for multiple platforms/architectures
- Resource scaling based on component complexity
- Adaptive attention mechanisms for build optimization
- Caching strategies for dependency management

## GGML Customization Notes

The workflow is designed to support GGML integration:

1. **Hardware Detection**: Automatic detection of AVX2/AVX512 capabilities
2. **Tensor Optimization**: Component-specific optimization flags
3. **Memory Management**: Efficient caching and resource allocation
4. **Parallel Processing**: Optimized job distribution for cognitive workloads

## Usage

To trigger the workflow:
```bash
git push origin main
```

Or create a pull request targeting the main branch.

The workflow will automatically build all components in the correct dependency order, leveraging GitHub Actions' parallel execution capabilities while respecting the cognitive architecture constraints.