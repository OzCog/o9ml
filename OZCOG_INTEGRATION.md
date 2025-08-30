# OzCog Repository Integration

This document describes the integration of OzCog repositories into the o9ml monorepo for cognitive architecture development.

## Integrated Repositories

The following repositories have been cloned from the OzCog organization and prepared for monorepo integration:

### 1. m0 - Memory Systems Bridge
- **Source**: `https://github.com/OzCog/m0`
- **Purpose**: To bridge Frontend & Backend Memory Systems
- **Location**: `ozcog-repos/m0/`
- **Description**: Mem0 is a memory layer for personalized AI that provides intelligent memory management capabilities.

### 2. mlpn - ECAN Model Extension  
- **Source**: `https://github.com/OzCog/mlpn`
- **Purpose**: To extend ECAN model with Cognitive MLRP toward ERPCAN
- **Location**: `ozcog-repos/mlpn/`
- **Description**: ERPNext-based system for economic resource planning with cognitive attention networks.

### 3. ko6ml - OpenCog Core Extension
- **Source**: `https://github.com/OzCog/ko6ml`
- **Purpose**: To extend OpenCog Core with Narrative-Driven Local+API Inference Engine
- **Location**: `ozcog-repos/ko6ml/`
- **Description**: Advanced cognitive architecture for AI-assisted writing with 6-phase cognitive processing.

## Integration Process

### Clone Script
The integration was performed using `clone_ozcog_repos.sh` which:
1. Clones each repository from the OzCog organization
2. Removes `.git` directories to prepare for monorepo integration
3. Organizes repositories in the `ozcog-repos/` directory
4. Provides clear status reporting throughout the process

### Directory Structure
```
ozcog-repos/
├── m0/           # Memory systems bridge (Mem0)
├── mlpn/         # ECAN/ERPCAN extension (ERPNext-based)
└── ko6ml/        # Narrative inference engine (KoboldAI-based)
```

### Git Integration
- All `.git` directories have been removed from cloned repositories
- The `ozcog-repos/` directory is commented in `.gitignore` but tracked for integration
- Each repository maintains its original file structure and content

## Usage Instructions

### Accessing the Integrated Repositories
The cloned repositories are available in the `ozcog-repos/` directory:
- Navigate to specific repositories: `cd ozcog-repos/[repo-name]`
- Each repository contains its original README.md and documentation
- All dependencies and build instructions remain in their original locations

### Building and Testing
Each repository should be built according to its original instructions:
- **m0**: Follow instructions in `ozcog-repos/m0/README.md`
- **mlpn**: Follow instructions in `ozcog-repos/mlpn/README.md` 
- **ko6ml**: Follow instructions in `ozcog-repos/ko6ml/README.md`

### Integration with o9ml
These repositories are now ready for deeper integration with the o9ml cognitive architecture:
- Memory systems from m0 can be integrated with existing attention mechanisms
- ECAN extensions from mlpn can enhance economic attention allocation
- Narrative inference from ko6ml can provide advanced reasoning capabilities

## Technical Details

### Cognitive Architecture Integration Points

#### m0 (Memory Systems)
- Provides persistent memory layer for personalized AI
- Can be integrated with AtomSpace for cognitive memory management
- Supports both short-term and long-term memory patterns

#### mlpn (ECAN Extension)
- Extends Economic Attention Networks with cognitive MLRP
- Integrates with ERPNext for resource planning
- Provides attention allocation optimization

#### ko6ml (Narrative Inference)
- 6-phase cognitive architecture for text processing
- Hypergraph-based attention allocation
- Meta-cognitive learning capabilities
- Direct integration potential with OpenCog AtomSpace

## Future Integration Steps

1. **Phase 1**: Individual repository testing and validation
2. **Phase 2**: Interface definition between repositories and o9ml core
3. **Phase 3**: Gradual integration of memory systems (m0)
4. **Phase 4**: ECAN extension integration (mlpn)
5. **Phase 5**: Narrative inference integration (ko6ml)
6. **Phase 6**: Unified cognitive architecture testing

## Maintenance

- Repositories are snapshots from their original OzCog sources
- Updates should be managed through the monorepo development process
- Original repository histories are preserved in the OzCog organization
- For major updates, consider re-running the clone script or manual synchronization

---

*Generated during o9ml repository integration - $(date)*