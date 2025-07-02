/*
 * test_sensory_motor_validation.cpp
 * 
 * Specific test for sensory-motor dataflow validation
 * Validates the action-perception loop integration
 * 
 * Author: OpenCog Central
 * License: AGPL
 * Date: December 2024
 */

#include <iostream>
#include <cassert>
#include <chrono>

#include <opencog/atomspace/AtomSpace.h>
#include "../include/robotics/EmbodiedCognition.hpp"

using namespace opencog;
using namespace opencog::embodied;

void validateSensoryMotorDataflow() {
    std::cout << "=======================================" << std::endl;
    std::cout << "Sensory-Motor Dataflow Validation Test" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    AtomSpace atomspace;
    EmbodiedCognitionManager manager(&atomspace);
    
    // Run comprehensive validation
    std::cout << "Running sensory-motor validation..." << std::endl;
    bool validation_result = manager.runSensoryMotorValidation();
    
    // Get detailed validation report
    auto report = manager.getValidationReport();
    std::cout << "\nValidation Report:" << std::endl;
    for (const auto& line : report) {
        std::cout << "  " << line << std::endl;
    }
    
    assert(validation_result == true);
    assert(!report.empty());
    
    std::cout << "\n✓ Sensory-motor dataflow validation: PASSED" << std::endl;
}

void validateEmbodimentTensorIntegration() {
    std::cout << "\n=======================================" << std::endl;
    std::cout << "Embodiment Tensor Integration Test" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    AtomSpace atomspace;
    ActionPerceptionLoop loop(&atomspace);
    
    // Create comprehensive sensory data
    SensoryData sensory_data;
    sensory_data.spatial_coords = {10.0f, 20.0f, 30.0f};
    sensory_data.visual_frames = {{255.0f, 200.0f, 150.0f, 100.0f}};
    sensory_data.audio_samples = {0.8f, 0.6f, 0.4f, 0.2f};
    sensory_data.detected_objects = {{50.0f, 60.0f, 70.0f, 80.0f}};
    sensory_data.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    sensory_data.source_id = "comprehensive_test";
    
    // Test tensor computation
    EmbodimentTensor tensor = loop.computeEmbodimentTensor(sensory_data);
    
    std::cout << "Computed Embodiment Tensor:" << std::endl;
    std::cout << "  Spatial: [" << tensor.spatial[0] << ", " 
              << tensor.spatial[1] << ", " << tensor.spatial[2] << "]" << std::endl;
    std::cout << "  Motor Actions: [" << tensor.motor_actions[0] << ", " 
              << tensor.motor_actions[1] << ", ...]" << std::endl;
    std::cout << "  Sensory: [" << tensor.sensory[0] << ", " 
              << tensor.sensory[1] << ", ...]" << std::endl;
    std::cout << "  Embodied State: [" << tensor.embodied_state[0] << ", " 
              << tensor.embodied_state[1] << ", ...]" << std::endl;
    
    // Validate tensor values
    assert(tensor.spatial[0] == 10.0f);
    assert(tensor.spatial[1] == 20.0f);
    assert(tensor.spatial[2] == 30.0f);
    assert(tensor.sensory[0] == 1.0f); // Vision active
    assert(tensor.sensory[2] == 1.0f); // Audio active
    
    // Test attention integration
    bool attention_update = loop.updateAttentionWithEmbodiment(tensor);
    assert(attention_update == true);
    
    std::cout << "\n✓ Embodiment tensor integration: PASSED" << std::endl;
}

void validateActionPerceptionIntegration() {
    std::cout << "\n=======================================" << std::endl;
    std::cout << "Action-Perception Integration Test" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    AtomSpace atomspace;
    ActionPerceptionLoop loop(&atomspace);
    
    // Test full action-perception cycle
    SensoryData input_data;
    input_data.spatial_coords = {5.0f, 10.0f, 15.0f};
    input_data.visual_frames = {{128.0f, 64.0f, 32.0f}};
    input_data.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    input_data.source_id = "integration_test";
    
    // Step 1: Process sensory input
    bool sensory_result = loop.processSensoryInput(input_data);
    assert(sensory_result == true);
    std::cout << "✓ Sensory input processing successful" << std::endl;
    
    // Step 2: Generate motor response
    Handle goal = atomspace.add_node(CONCEPT_NODE, "NavigateToTarget");
    MotorCommand motor_output = loop.generateMotorResponse(goal);
    
    assert(!motor_output.target_id.empty());
    assert(motor_output.execution_time > 0);
    std::cout << "✓ Motor response generation successful" << std::endl;
    
    // Step 3: Validate dataflow
    bool dataflow_valid = loop.validateSensoryMotorDataflow();
    assert(dataflow_valid == true);
    std::cout << "✓ Action-perception dataflow validation successful" << std::endl;
    
    std::cout << "\nMotor Response Details:" << std::endl;
    std::cout << "  Linear Velocity: [" << motor_output.linear_velocity[0] 
              << ", " << motor_output.linear_velocity[1] 
              << ", " << motor_output.linear_velocity[2] << "]" << std::endl;
    std::cout << "  Target: " << motor_output.target_id << std::endl;
    std::cout << "  Execution Time: " << motor_output.execution_time << std::endl;
    
    std::cout << "\n✓ Action-perception integration: PASSED" << std::endl;
}

int main(int argc, char** argv) {
    try {
        validateSensoryMotorDataflow();
        validateEmbodimentTensorIntegration();
        validateActionPerceptionIntegration();
        
        std::cout << "\n=======================================" << std::endl;
        std::cout << "All Validation Tests PASSED!" << std::endl;
        std::cout << "Sensory-Motor Dataflow: FUNCTIONAL" << std::endl;
        std::cout << "Embodiment Tensor Mapping: OPERATIONAL" << std::endl;
        std::cout << "Action-Perception Loop: INTEGRATED" << std::endl;
        std::cout << "=======================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Validation test failed: " << e.what() << std::endl;
        return 1;
    }
}