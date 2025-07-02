/*
 * EmbodiedCognition.hpp
 * 
 * Robotics Layer: Embodied Cognition Integration
 * Implements action-perception loop for embodied agents
 * 
 * Author: OpenCog Central
 * License: AGPL
 * Date: December 2024
 */

#ifndef EMBODIED_COGNITION_HPP
#define EMBODIED_COGNITION_HPP

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <opencog/atomspace/AtomSpace.h>
#include <opencog/atoms/base/Handle.h>

namespace opencog {
namespace embodied {

/**
 * Embodiment Tensor Dimensions
 * Extends the existing attention tensor with embodiment-specific dimensions
 */
struct EmbodimentTensor {
    // Spatial dimensions (3D) - inherited from attention tensor
    float spatial[3];           // x, y, z coordinates
    
    // Motor action dimensions (6D) - new for embodiment  
    float motor_actions[6];     // linear: x,y,z + angular: roll,pitch,yaw
    
    // Sensory modalities (8D) - new for embodiment
    float sensory[8];           // vision, audio, touch, proprioception, etc.
    
    // Embodied state (4D) - new for embodiment
    float embodied_state[4];    // position, orientation, velocity, acceleration
    
    // Action affordances (16D) - new for embodiment
    float affordances[16];      // possible actions in current context
    
    // Total: 37 new dimensions + 327 existing attention tensor = 364 total
};

/**
 * Sensory Data Structure
 * Represents multimodal sensory input from the environment
 */
struct SensoryData {
    std::vector<std::vector<float>> visual_frames;    // Visual input as float arrays
    std::vector<float> audio_samples;                 // Audio input samples  
    std::vector<std::vector<float>> detected_objects; // Object detection results as bounding boxes
    std::vector<float> spatial_coords;                // 3D spatial coordinates
    double timestamp;                                 // Sensor timestamp
    std::string source_id;                           // Sensor source identifier
};

/**
 * Motor Command Structure  
 * Represents motor actions to be executed by the embodied agent
 */
struct MotorCommand {
    float linear_velocity[3];    // Linear velocity commands (x, y, z)
    float angular_velocity[3];   // Angular velocity commands (roll, pitch, yaw)
    float joint_positions[12];   // Joint position commands (up to 12 joints)
    double execution_time;       // Command execution timestamp
    std::string target_id;       // Target identifier for action
};

/**
 * Action-Perception Loop Interface
 * Core interface for embodied cognition processing
 */
class ActionPerceptionLoop {
public:
    ActionPerceptionLoop(AtomSpace* atomspace);
    virtual ~ActionPerceptionLoop() = default;
    
    // Core action-perception loop methods
    virtual bool processSensoryInput(const SensoryData& sensory_data);
    virtual MotorCommand generateMotorResponse(const Handle& goal_atom);
    virtual bool validateSensoryMotorDataflow();
    
    // Embodiment tensor operations
    EmbodimentTensor computeEmbodimentTensor(const SensoryData& input);
    bool updateAttentionWithEmbodiment(const EmbodimentTensor& tensor);
    
    // Integration with existing vision systems
    bool integrateVisionInput(const std::vector<std::vector<float>>& frames);
    bool integratePerceptionData(const std::vector<float>& spatial_coords);
    
    // Agent interface methods
    virtual bool connectToVirtualAgent(const std::string& agent_id);
    virtual bool connectToRealAgent(const std::string& device_path);
    
protected:
    AtomSpace* atomspace_;
    std::unique_ptr<EmbodimentTensor> current_tensor_;
    
    // Integration helper methods
    Handle createSensoryAtom(const SensoryData& data);
    Handle createMotorAtom(const MotorCommand& command);
    bool propagateAttentionFlow(const Handle& source, const Handle& target);
};

/**
 * Embodied Cognition Manager
 * High-level manager for embodied cognition operations
 */
class EmbodiedCognitionManager {
public:
    EmbodiedCognitionManager(AtomSpace* atomspace);
    ~EmbodiedCognitionManager();
    
    // Main processing loop
    bool startEmbodiedProcessing();
    bool stopEmbodiedProcessing();
    bool isProcessing() const { return is_processing_; }
    
    // Component integration
    bool addSensorySource(const std::string& source_id, 
                         std::function<SensoryData()> sensor_callback);
    bool addMotorTarget(const std::string& target_id,
                       std::function<bool(const MotorCommand&)> motor_callback);
    
    // Validation and testing
    bool runSensoryMotorValidation();
    std::vector<std::string> getValidationReport() const;
    
private:
    AtomSpace* atomspace_;
    std::unique_ptr<ActionPerceptionLoop> action_perception_loop_;
    bool is_processing_;
    std::vector<std::string> validation_report_;
    
    // Processing thread management
    void processingLoop();
    std::thread processing_thread_;
    std::mutex processing_mutex_;
    std::condition_variable processing_cv_;
    bool stop_requested_;
};

} // namespace embodied
} // namespace opencog

#endif // EMBODIED_COGNITION_HPP