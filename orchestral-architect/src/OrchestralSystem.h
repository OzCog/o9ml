/**
 * @file OrchestralSystem.h
 * @brief Main orchestral architect system coordinator
 * 
 * The OrchestralSystem provides the top-level interface for the distributed
 * agentic cognitive grammar system, coordinating kernels, attention allocation,
 * and neural-symbolic integration.
 */

#pragma once

#include "core/AgenticKernel.h"
#include "core/KernelRegistry.h"
#include "kernels/TokenizationKernel.h"
#include <memory>
#include <vector>
#include <string>
#include <map>

namespace orchestral {

/**
 * @brief Configuration for the orchestral system
 */
struct OrchestralConfig {
    bool enableNetworking = false;
    bool enableHealthMonitoring = true;
    double heartbeatInterval = 30.0;
    bool enableAttentionAllocation = true;
    bool enableNeuralSymbolicBridge = true;
    size_t maxConcurrentProcessing = 10;
    std::string logLevel = "INFO";
    
    // Attention configuration
    double attentionThreshold = 0.1;
    double attentionDecayRate = 0.95;
    
    // Performance tuning
    size_t processingThreads = 4;
    size_t eventQueueSize = 1000;
};

/**
 * @brief System-wide metrics and status
 */
struct SystemStatus {
    size_t activeKernels = 0;
    size_t totalProcessedItems = 0;
    double averageProcessingTime = 0.0;
    double systemLoad = 0.0;
    double cognitiveEfficiency = 0.0;
    std::chrono::system_clock::time_point startTime;
    std::chrono::system_clock::time_point lastUpdate;
    
    std::map<std::string, size_t> kernelCounts;  // By type
    std::map<std::string, double> kernelLoads;   // By name
};

/**
 * @brief Main orchestral architect system
 * 
 * Coordinates the distributed network of cognitive kernels, providing
 * high-level cognitive processing capabilities through specialized
 * agentic components.
 */
class OrchestralSystem {
public:
    /**
     * @brief Construct a new Orchestral System
     * @param config System configuration
     */
    explicit OrchestralSystem(const OrchestralConfig& config = OrchestralConfig());
    
    /**
     * @brief Destroy the Orchestral System
     */
    ~OrchestralSystem();
    
    /**
     * @brief Initialize the orchestral system
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Shutdown the system and all kernels
     */
    void shutdown();
    
    /**
     * @brief Register a cognitive kernel with the system
     * @param kernel Shared pointer to the kernel
     * @return true if registration successful
     */
    bool registerKernel(std::shared_ptr<AgenticKernel> kernel);
    
    /**
     * @brief Create and register a kernel by type
     * @param type Kernel type name
     * @param name Unique name for the kernel instance
     * @return true if creation and registration successful
     */
    bool createKernel(const std::string& type, const std::string& name);
    
    /**
     * @brief Process input using the orchestral system
     * @param input Cognitive input to process
     * @param preferredKernel Optional kernel to prefer for processing
     * @return Cognitive processing result
     */
    CognitiveResult processInput(const CognitiveInput& input, 
                                const std::string& preferredKernel = "");
    
    /**
     * @brief Process text input with automatic kernel selection
     * @param text Input text to process
     * @param urgency Processing urgency (0.0 to 1.0)
     * @return Processing result
     */
    CognitiveResult processText(const std::string& text, double urgency = 0.5);
    
    /**
     * @brief Get system status and metrics
     * @return Current system status
     */
    SystemStatus getSystemStatus() const;
    
    /**
     * @brief Get list of registered kernels
     * @return Vector of kernel names
     */
    std::vector<std::string> getRegisteredKernels() const;
    
    /**
     * @brief Find kernels by capability
     * @param capability Required capability
     * @return Vector of kernel names
     */
    std::vector<std::string> findKernelsByCapability(const std::string& capability) const;
    
    /**
     * @brief Broadcast event to all or specific kernels
     * @param event Event to broadcast
     * @param targetType Target kernel type (empty for all)
     * @return Number of kernels that received the event
     */
    size_t broadcastEvent(const KernelEvent& event, const std::string& targetType = "");
    
    /**
     * @brief Set system configuration
     * @param config New configuration
     * @return true if configuration applied successfully
     */
    bool configure(const OrchestralConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current system configuration
     */
    const OrchestralConfig& getConfiguration() const { return config_; }
    
    /**
     * @brief Enable or disable specific system features
     * @param feature Feature name
     * @param enabled Whether to enable the feature
     * @return true if feature toggle successful
     */
    bool setFeatureEnabled(const std::string& feature, bool enabled);
    
    /**
     * @brief Check if system is running
     * @return true if system is active
     */
    bool isRunning() const { return running_; }
    
    /**
     * @brief Get system uptime
     * @return Duration since system start
     */
    std::chrono::duration<double> getUptime() const;
    
    /**
     * @brief Create default kernel configuration
     * @return true if default kernels created successfully
     */
    bool createDefaultKernels();
    
    /**
     * @brief Save system state to file
     * @param filename File to save to
     * @return true if save successful
     */
    bool saveState(const std::string& filename) const;
    
    /**
     * @brief Load system state from file
     * @param filename File to load from
     * @return true if load successful
     */
    bool loadState(const std::string& filename);
    
private:
    /**
     * @brief Initialize kernel registry
     * @return true if registry initialization successful
     */
    bool initializeRegistry();
    
    /**
     * @brief Initialize default kernels
     * @return true if kernel initialization successful
     */
    bool initializeKernels();
    
    /**
     * @brief Handle system events
     * @param event System event to handle
     */
    void handleSystemEvent(const KernelEvent& event);
    
    /**
     * @brief Update system metrics
     */
    void updateSystemMetrics();
    
    /**
     * @brief Select best kernel for processing
     * @param input Input to process
     * @param preferredKernel Optional preferred kernel
     * @return Name of selected kernel, empty if none suitable
     */
    std::string selectKernelForProcessing(const CognitiveInput& input,
                                        const std::string& preferredKernel = "");
    
    /**
     * @brief Apply attention allocation across kernels
     * @param input Processing input
     * @return Attention weights by kernel
     */
    std::map<std::string, double> allocateAttention(const CognitiveInput& input);
    
    // Configuration and state
    OrchestralConfig config_;
    std::atomic<bool> running_{false};
    std::chrono::system_clock::time_point startTime_;
    
    // Core components
    KernelRegistry* registry_;
    
    // System metrics
    mutable std::mutex metricsMutex_;
    SystemStatus currentStatus_;
    
    // Default kernels
    std::vector<std::shared_ptr<AgenticKernel>> defaultKernels_;
    
    // Event handling
    std::function<void(const KernelEvent&)> eventHandler_;
};

/**
 * @brief Factory function to create a configured orchestral system
 * @param config System configuration
 * @return Unique pointer to configured system
 */
std::unique_ptr<OrchestralSystem> createOrchestralSystem(
    const OrchestralConfig& config = OrchestralConfig());

/**
 * @brief Helper to create a basic demo system
 * @return Configured system ready for demonstration
 */
std::unique_ptr<OrchestralSystem> createDemoSystem();

} // namespace orchestral