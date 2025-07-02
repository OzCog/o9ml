/*
 * standalone_test.cpp
 * 
 * Standalone test for embodied cognition without OpenCog dependencies
 * 
 * Author: OpenCog Central
 * License: AGPL
 * Date: December 2024
 */

#include "include/robotics/EmbodiedCognitionStandalone.hpp"
#include <iostream>
#include <cassert>
#include <chrono>

using namespace opencog::embodied;

void testEmbodimentTensorConstruction() {
    std::cout << "Testing embodiment tensor construction..." << std::endl;
    
    StandaloneAtomSpace atomspace;
    ActionPerceptionLoop loop(&atomspace);
    
    SensoryData data;
    data.spatial_coords = {1.0f, 2.0f, 3.0f};
    data.visual_frames = {{255.0f, 128.0f}};
    data.audio_samples = {0.5f, 0.7f};
    data.timestamp = 12345.0;
    data.source_id = "test_sensor";
    
    EmbodimentTensor tensor = loop.computeEmbodimentTensor(data);
    
    // Test spatial dimensions mapping
    assert(tensor.spatial[0] == 1.0f);
    assert(tensor.spatial[1] == 2.0f);
    assert(tensor.spatial[2] == 3.0f);
    
    // Test sensory modalities
    assert(tensor.sensory[0] == 1.0f); // Vision active
    assert(tensor.sensory[2] == 1.0f); // Audio active
    
    std::cout << "✓ Embodiment tensor construction test passed" << std::endl;
}

void testSensoryMotorValidation() {
    std::cout << "Testing sensory-motor validation..." << std::endl;
    
    StandaloneAtomSpace atomspace;
    EmbodiedCognitionManager manager(&atomspace);
    
    bool validation_result = manager.runSensoryMotorValidation();
    assert(validation_result == true);
    
    auto report = manager.getValidationReport();
    assert(!report.empty());
    
    std::cout << "✓ Sensory-motor validation test passed" << std::endl;
}

void testActionPerceptionLoop() {
    std::cout << "Testing action-perception loop..." << std::endl;
    
    StandaloneAtomSpace atomspace;
    ActionPerceptionLoop loop(&atomspace);
    
    // Test dataflow validation
    bool validation_result = loop.validateSensoryMotorDataflow();
    assert(validation_result == true);
    
    // Test vision integration
    std::vector<std::vector<float>> frames = {{1.0f, 2.0f, 3.0f}};
    bool vision_result = loop.integrateVisionInput(frames);
    assert(vision_result == true);
    
    // Test perception integration
    std::vector<float> coords = {4.0f, 5.0f, 6.0f};
    bool perception_result = loop.integratePerceptionData(coords);
    assert(perception_result == true);
    
    std::cout << "✓ Action-perception loop test passed" << std::endl;
}

void testTensorDimensionMapping() {
    std::cout << "Testing tensor dimension mapping..." << std::endl;
    
    // Test tensor structure sizes
    EmbodimentTensor tensor;
    
    // Verify dimension counts
    static_assert(sizeof(tensor.spatial) == 3 * sizeof(float), "Spatial dimensions should be 3D");
    static_assert(sizeof(tensor.motor_actions) == 6 * sizeof(float), "Motor actions should be 6D");
    static_assert(sizeof(tensor.sensory) == 8 * sizeof(float), "Sensory modalities should be 8D");
    static_assert(sizeof(tensor.embodied_state) == 4 * sizeof(float), "Embodied state should be 4D");
    static_assert(sizeof(tensor.affordances) == 16 * sizeof(float), "Affordances should be 16D");
    
    std::cout << "✓ Tensor dimension mapping test passed" << std::endl;
}

int main() {
    try {
        std::cout << "=========================================" << std::endl;
        std::cout << "Robotics Embodied Cognition: Standalone Tests" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        testEmbodimentTensorConstruction();
        testSensoryMotorValidation();
        testActionPerceptionLoop();
        testTensorDimensionMapping();
        
        std::cout << "\n=========================================" << std::endl;
        std::cout << "All standalone tests PASSED!" << std::endl;
        std::cout << "✓ Embodiment tensor construction: FUNCTIONAL" << std::endl;
        std::cout << "✓ Sensory-motor validation: OPERATIONAL" << std::endl;
        std::cout << "✓ Action-perception loop: INTEGRATED" << std::endl;
        std::cout << "✓ Tensor dimension mapping: VERIFIED" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}