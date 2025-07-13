Github Actions Continuous Integration Testing Workflows.

ci-tests.yml runs Ubuntu 18.04 and Ubuntu 20.04 using Github-provided virtual environments. The virtual environments are about twice as fast as Docker ones and getting Docker to run Ubuntu images was problematic. Note that updating Ubuntu 18.04 updates ICU4C from v60 to v65 and that causes conflicts with Boost, so don't.

ci-docker.yml runs tests in a Docker container running Arch Linux.

mac-tests.yml runs tests on macOS using a Github-provided virtual machine. Note that this test relies on a prebuilt tarball containing all of the dependencies. Instructions for building that tarball along with some support scripts may be found in utils/ci/macos-ci-deps.

## Cognitive Network Development Workflows

create-phase-issues.yml creates structured development issues for the Distributed Agentic Cognitive Grammar Network. This workflow automates the creation of 6 detailed phase issues plus a master coordination issue, enabling systematic cognitive development tracking and tutorial autogeneration integration.

### Features:
- **Automated Issue Creation**: Creates comprehensive development issues for each cognitive phase
- **Flexible Triggering**: Manual workflow dispatch with options for all phases or specific phases
- **Rich Content**: Each issue includes sub-steps, verification protocols, success criteria, and technical specifications
- **Proper Labeling**: Issues are labeled for easy filtering and project management
- **Tutorial Integration**: Structured content supports automated tutorial generation systems

### Usage:
1. Go to Actions tab → "Create Distributed Cognitive Network Phase Issues"
2. Click "Run workflow"
3. Choose to create all phases or specify a single phase (1-6)
4. View created issues in the Issues tab

The workflow supports the development of:
- Phase 1: Cognitive Primitives & Foundational Hypergraph Encoding
- Phase 2: ECAN Attention Allocation & Resource Kernel Construction
- Phase 3: Neural-Symbolic Synthesis via Custom ggml Kernels
- Phase 4: Distributed Cognitive Mesh API & Embodiment Layer
- Phase 5: Recursive Meta-Cognition & Evolutionary Optimization
- Phase 6: Rigorous Testing, Documentation, and Cognitive Unification
