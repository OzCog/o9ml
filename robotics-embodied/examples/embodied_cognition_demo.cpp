/*
 * embodied_cognition_demo.cpp
 * 
 * Demonstration of the Robotics Layer: Embodied Cognition
 * Shows action-perception loop integration and tensor mapping
 * 
 * Author: OpenCog Central
 * License: AGPL  
 * Date: December 2024
 */

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

#include <opencog/atomspace/AtomSpace.h>
#include "../include/robotics/EmbodiedCognition.hpp"

using namespace opencog;
using namespace opencog::embodied;

void demonstrateEmbodiedCognition() {
    std::cout << "=============================================" << std::endl;
    std::cout << "Robotics Layer: Embodied Cognition Demo" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Create AtomSpace
    AtomSpace atomspace;
    std::cout << "✓ AtomSpace initialized" << std::endl;
    
    // Create Embodied Cognition Manager
    EmbodiedCognitionManager manager(&atomspace);
    std::cout << "✓ Embodied Cognition Manager created" << std::endl;
    
    // Demonstrate sensory-motor validation
    std::cout << "\n--- Sensory-Motor Dataflow Validation ---" << std::endl;
    bool validation_success = manager.runSensoryMotorValidation();
    
    auto validation_report = manager.getValidationReport();
    for (const auto& report_line : validation_report) {
        std::cout << report_line << std::endl;
    }
    
    if (!validation_success) {
        std::cerr << "Validation failed - aborting demo" << std::endl;
        return;
    }
    
    // Demonstrate action-perception loop
    std::cout << "\n--- Action-Perception Loop Demo ---" << std::endl;
    ActionPerceptionLoop loop(&atomspace);
    
    // Create sample sensory data
    SensoryData sensory_data;
    sensory_data.spatial_coords = {1.0f, 2.0f, 3.0f};
    sensory_data.visual_frames = {{255.0f, 128.0f, 64.0f}};
    sensory_data.audio_samples = {0.5f, 0.7f, 0.3f};
    sensory_data.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    sensory_data.source_id = "demo_sensor";
    
    // Process sensory input
    bool processing_success = loop.processSensoryInput(sensory_data);
    std::cout << "Sensory input processing: " << (processing_success ? "SUCCESS" : "FAILED") << std::endl;
    
    // Generate motor response
    Handle goal_atom = atomspace.add_node(CONCEPT_NODE, "MoveForward");
    MotorCommand motor_response = loop.generateMotorResponse(goal_atom);
    
    std::cout << "Motor response generated:" << std::endl;
    std::cout << "  Linear velocity: [" << motor_response.linear_velocity[0] 
              << ", " << motor_response.linear_velocity[1] 
              << ", " << motor_response.linear_velocity[2] << "]" << std::endl;
    std::cout << "  Target: " << motor_response.target_id << std::endl;
    
    // Demonstrate embodiment tensor computation
    std::cout << "\n--- Embodiment Tensor Mapping ---" << std::endl;
    EmbodimentTensor tensor = loop.computeEmbodimentTensor(sensory_data);
    
    std::cout << "Embodiment Tensor Dimensions:" << std::endl;
    std::cout << "  Spatial (3D): [" << tensor.spatial[0] 
              << ", " << tensor.spatial[1] 
              << ", " << tensor.spatial[2] << "]" << std::endl;
    std::cout << "  Motor Actions (6D): [" << tensor.motor_actions[0]
              << ", " << tensor.motor_actions[1] << ", ...]" << std::endl;
    std::cout << "  Sensory (8D): [" << tensor.sensory[0]
              << ", " << tensor.sensory[1] << ", ...]" << std::endl;
    std::cout << "  Embodied State (4D): [" << tensor.embodied_state[0]
              << ", " << tensor.embodied_state[1] << ", ...]" << std::endl;
    std::cout << "  Total Dimensions: 37 embodiment + 327 attention = 364" << std::endl;
    
    // Demonstrate agent connections
    std::cout << "\n--- Agent Integration Demo ---" << std::endl;
    bool virtual_connection = loop.connectToVirtualAgent("demo_virtual_agent");
    bool real_connection = loop.connectToRealAgent("/dev/demo_device");
    
    std::cout << "Virtual agent connection: " << (virtual_connection ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Real agent connection: " << (real_connection ? "SUCCESS" : "FAILED") << std::endl;
    
    // Demonstrate processing loop
    std::cout << "\n--- Processing Loop Demo ---" << std::endl;
    manager.startEmbodiedProcessing();
    
    std::cout << "Processing for 2 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    manager.stopEmbodiedProcessing();
    
    std::cout << "\n=============================================" << std::endl;
    std::cout << "Embodied Cognition Demo Complete!" << std::endl;
    std::cout << "=============================================" << std::endl;
}

void printEmbodimentKernelMapping() {
    std::cout << "\n=============================================" << std::endl;
    std::cout << "Embodiment Kernel Tensor Dimension Mapping" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    std::cout << "Total Tensor Dimensions: 364" << std::endl;
    std::cout << "\nExisting Attention Tensor (327D):" << std::endl;
    std::cout << "  - Spatial: 3D (x, y, z coordinates)" << std::endl;
    std::cout << "  - Temporal: 1D (time sequence)" << std::endl;
    std::cout << "  - Semantic: 256D (embedding space)" << std::endl;
    std::cout << "  - Importance: 3D (STI, LTI, VLTI)" << std::endl;
    std::cout << "  - Hebbian: 64D (synaptic strength)" << std::endl;
    
    std::cout << "\nNew Embodiment Extensions (37D):" << std::endl;
    std::cout << "  - Motor Actions: 6D (linear + angular velocity)" << std::endl;
    std::cout << "  - Sensory Modalities: 8D (vision, audio, touch, etc.)" << std::endl;
    std::cout << "  - Embodied State: 4D (position, orientation, velocity, acceleration)" << std::endl;
    std::cout << "  - Action Affordances: 16D (possible actions in context)" << std::endl;
    
    std::cout << "\nAction-Perception Loop Integration:" << std::endl;
    std::cout << "  Perception → Embodiment Tensor → Attention Update → Motor Response" << std::endl;
    std::cout << "  Sensory Data → Spatial Mapping → Action Selection → Motor Commands" << std::endl;
    
    std::cout << "=============================================" << std::endl;
}

int main(int argc, char** argv) {
    try {
        // Print tensor mapping information
        printEmbodimentKernelMapping();
        
        // Run embodied cognition demonstration
        demonstrateEmbodiedCognition();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Demo failed with exception: " << e.what() << std::endl;
        return 1;
    }
}