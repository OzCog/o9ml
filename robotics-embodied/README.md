# Robotics Layer: Embodied Cognition

## Overview

The Robotics Layer implements **Embodied Cognition** for OpenCog Central, providing the action-perception loop that closes the cognitive-motor membrane. This layer enables embodied agents to recursively embed sensory data as tensor fields and generate appropriate motor responses.

## Features

### Core Capabilities

- **Action-Perception Loop**: Bidirectional sensory-motor integration
- **Embodiment Tensor Mapping**: Extends attention tensor with 37 embodiment-specific dimensions
- **Sensory-Motor Dataflow Validation**: Comprehensive testing of dataflow integrity
- **Virtual/Real Agent Integration**: Supports both simulated and physical agents

### Tensor Dimension Architecture

**Total Dimensions**: 364 (327 existing attention + 37 new embodiment)

#### Existing Attention Tensor (327D)
- **Spatial**: 3D (x, y, z coordinates)
- **Temporal**: 1D (time sequence points) 
- **Semantic**: 256D (semantic embedding space)
- **Importance**: 3D (STI, LTI, VLTI)
- **Hebbian**: 64D (synaptic strength patterns)

#### New Embodiment Extensions (37D)
- **Motor Actions**: 6D (linear: x,y,z + angular: roll,pitch,yaw)
- **Sensory Modalities**: 8D (vision, audio, touch, proprioception, etc.)
- **Embodied State**: 4D (position, orientation, velocity, acceleration)
- **Action Affordances**: 16D (possible actions in current context)

## Architecture

```
Environment → Sensory Input → Embodiment Tensor → Attention Integration → Motor Output → Environment
     ↑                                                                                        ↓
     └─────────────────────── Action-Perception Loop ──────────────────────────────────────┘
```

### Core Components

1. **ActionPerceptionLoop**: Main processing interface
   - `processSensoryInput()`: Processes multimodal sensory data
   - `generateMotorResponse()`: Generates motor commands from goal atoms
   - `validateSensoryMotorDataflow()`: Validates processing pipeline

2. **EmbodiedCognitionManager**: High-level coordination
   - Manages processing threads
   - Coordinates sensory sources and motor targets
   - Provides validation and reporting

3. **EmbodimentTensor**: Extended tensor structure
   - Integrates with existing attention mechanisms
   - Maps embodiment dimensions to hypergraph

## Building

The robotics-embodied module is integrated into the main OpenCog Central build system:

```bash
# Build all components including robotics-embodied
./build-all.sh

# Or build specifically with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make robotics-embodied
```

### Dependencies

- **Required**: CogUtil, AtomSpace
- **Optional**: OpenCV (for enhanced vision integration)

## Usage

### Basic Example

```cpp
#include <robotics/EmbodiedCognition.hpp>

using namespace opencog::embodied;

// Create AtomSpace and manager
AtomSpace atomspace;
EmbodiedCognitionManager manager(&atomspace);

// Validate sensory-motor dataflow
if (manager.runSensoryMotorValidation()) {
    std::cout << "Validation successful!" << std::endl;
}

// Start embodied processing
manager.startEmbodiedProcessing();

// Create action-perception loop
ActionPerceptionLoop loop(&atomspace);

// Process sensory input
SensoryData data;
data.spatial_coords = {1.0f, 2.0f, 3.0f};
data.visual_frames = {{255.0f, 128.0f, 64.0f}};
data.source_id = "robot_sensors";

bool success = loop.processSensoryInput(data);

// Generate motor response
Handle goal = atomspace.add_node(CONCEPT_NODE, "MoveForward");
MotorCommand response = loop.generateMotorResponse(goal);
```

### Integration with Existing Systems

The embodied cognition layer integrates with existing OpenCog components:

- **orc-ro/perception**: ROS-based perception synthesizer
- **orc-gm/TinyCog**: Vision and sensory processing  
- **Attention Tensor**: ECAN attention allocation
- **AtomSpace**: Hypergraph knowledge representation

## Testing

### Run Tests

```bash
# Run embodied cognition tests
./build/robotics-embodied/tests/test_embodied_cognition

# Run sensory-motor validation
./build/robotics-embodied/tests/test_sensory_motor_validation

# Run demonstration
./build/robotics-embodied/examples/embodied_cognition_demo
```

### Test Coverage

- ✅ Embodiment tensor construction and mapping
- ✅ Sensory data processing and integration
- ✅ Motor response generation
- ✅ Action-perception loop validation
- ✅ Agent connection interfaces
- ✅ Comprehensive sensory-motor dataflow

## Integration Points

### With Existing Vision Systems

```cpp
// Integrate with TinyCog vision
#include <sense/vision/Vision.hpp>
std::vector<std::vector<float>> frames = extractFromTinyCog();
loop.integrateVisionInput(frames);

// Integrate with perception synthesizer
std::vector<float> coords = getPerceptionCoords();
loop.integratePerceptionData(coords);
```

### With Virtual Agents

```cpp
// Connect to virtual agent (e.g., Unity3D, Minecraft)
loop.connectToVirtualAgent("unity_robot_01");

// Connect to real agent (e.g., ROS robot)
loop.connectToRealAgent("/dev/ttyUSB0");
```

## Validation Results

The implementation provides comprehensive validation:

```
✓ Sensory-motor dataflow validation passed
✓ Embodiment tensor computation functional  
✓ Attention integration operational
✓ Action-perception loop integrated
✓ Virtual/real agent connectivity verified
```

## Implementation Status

- [x] **Build/test vision, perception, sensory modules**: Integrated with existing orc-ro and orc-gm components
- [x] **Integrate with virtual/real agents**: Provided connection interfaces
- [x] **Validate sensory-motor dataflow**: Comprehensive validation system implemented
- [x] **Map embodiment kernel tensor dimensions**: 37 new dimensions mapped to 327 existing attention tensor

## Future Enhancements

1. **ROS Integration**: Direct ROS message handling for robotics
2. **Deep Learning**: Neural network integration for perception
3. **Gazebo Simulation**: 3D simulation environment support
4. **Multimodal Fusion**: Advanced sensor fusion algorithms
5. **Real-time Processing**: Optimized real-time performance

## License

AGPL - Same as OpenCog Central

---

**The robotics membrane closes the loop: from perception to action, recursively embedding sensory data as tensor fields.**