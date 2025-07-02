/*
 * test_embodied_cognition.cpp
 * 
 * Unit tests for Robotics Layer: Embodied Cognition
 * Tests action-perception loop, tensor mapping, and integration
 * 
 * Author: OpenCog Central
 * License: AGPL
 * Date: December 2024
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>

#include <opencog/atomspace/AtomSpace.h>
#include "../include/robotics/EmbodiedCognition.hpp"

using namespace opencog;
using namespace opencog::embodied;

class EmbodiedCognitionTest {
public:
    static void runAllTests() {
        std::cout << "Running Embodied Cognition Tests..." << std::endl;
        
        testEmbodimentTensorConstruction();
        testSensoryDataProcessing();
        testMotorResponseGeneration();
        testActionPerceptionLoop();
        testEmbodiedCognitionManager();
        testTensorDimensionMapping();
        testAgentIntegration();
        
        std::cout << "All tests passed!" << std::endl;
    }

private:
    static void testEmbodimentTensorConstruction() {
        std::cout << "Testing embodiment tensor construction..." << std::endl;
        
        AtomSpace atomspace;
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
        
        // Test embodied state
        assert(tensor.embodied_state[0] == 1.0f); // Position from spatial
        
        std::cout << "✓ Embodiment tensor construction test passed" << std::endl;
    }
    
    static void testSensoryDataProcessing() {
        std::cout << "Testing sensory data processing..." << std::endl;
        
        AtomSpace atomspace;
        ActionPerceptionLoop loop(&atomspace);
        
        SensoryData data;
        data.spatial_coords = {0.0f, 1.0f, 2.0f};
        data.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        data.source_id = "test_processing";
        
        bool result = loop.processSensoryInput(data);
        assert(result == true);
        
        // Check that atoms were created in atomspace
        size_t atom_count = atomspace.get_size();
        assert(atom_count > 0);
        
        std::cout << "✓ Sensory data processing test passed" << std::endl;
    }
    
    static void testMotorResponseGeneration() {
        std::cout << "Testing motor response generation..." << std::endl;
        
        AtomSpace atomspace;
        ActionPerceptionLoop loop(&atomspace);
        
        Handle goal_atom = atomspace.add_node(CONCEPT_NODE, "TestGoal");
        MotorCommand command = loop.generateMotorResponse(goal_atom);
        
        // Test that motor command was generated
        assert(!command.target_id.empty());
        assert(command.execution_time > 0);
        
        std::cout << "✓ Motor response generation test passed" << std::endl;
    }
    
    static void testActionPerceptionLoop() {
        std::cout << "Testing action-perception loop..." << std::endl;
        
        AtomSpace atomspace;
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
    
    static void testEmbodiedCognitionManager() {
        std::cout << "Testing embodied cognition manager..." << std::endl;
        
        AtomSpace atomspace;
        EmbodiedCognitionManager manager(&atomspace);
        
        // Test validation
        bool validation_result = manager.runSensoryMotorValidation();
        assert(validation_result == true);
        
        auto report = manager.getValidationReport();
        assert(!report.empty());
        
        // Test processing control
        bool start_result = manager.startEmbodiedProcessing();
        assert(start_result == true);
        assert(manager.isProcessing() == true);
        
        bool stop_result = manager.stopEmbodiedProcessing();
        assert(stop_result == true);
        assert(manager.isProcessing() == false);
        
        std::cout << "✓ Embodied cognition manager test passed" << std::endl;
    }
    
    static void testTensorDimensionMapping() {
        std::cout << "Testing tensor dimension mapping..." << std::endl;
        
        // Test tensor structure sizes
        EmbodimentTensor tensor;
        
        // Verify dimension counts
        static_assert(sizeof(tensor.spatial) == 3 * sizeof(float), "Spatial dimensions should be 3D");
        static_assert(sizeof(tensor.motor_actions) == 6 * sizeof(float), "Motor actions should be 6D");
        static_assert(sizeof(tensor.sensory) == 8 * sizeof(float), "Sensory modalities should be 8D");
        static_assert(sizeof(tensor.embodied_state) == 4 * sizeof(float), "Embodied state should be 4D");
        static_assert(sizeof(tensor.affordances) == 16 * sizeof(float), "Affordances should be 16D");
        
        // Total embodiment dimensions: 3 + 6 + 8 + 4 + 16 = 37
        // Plus existing attention tensor: 327
        // Total: 364 dimensions
        
        std::cout << "✓ Tensor dimension mapping test passed" << std::endl;
    }
    
    static void testAgentIntegration() {
        std::cout << "Testing agent integration..." << std::endl;
        
        AtomSpace atomspace;
        ActionPerceptionLoop loop(&atomspace);
        
        // Test virtual agent connection
        bool virtual_result = loop.connectToVirtualAgent("test_virtual_agent");
        assert(virtual_result == true);
        
        // Test real agent connection
        bool real_result = loop.connectToRealAgent("/dev/test_device");
        assert(real_result == true);
        
        std::cout << "✓ Agent integration test passed" << std::endl;
    }
};

int main(int argc, char** argv) {
    try {
        EmbodiedCognitionTest::runAllTests();
        std::cout << "All embodied cognition tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}