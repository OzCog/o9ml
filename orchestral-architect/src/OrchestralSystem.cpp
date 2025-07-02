/**
 * @file OrchestralSystem.cpp
 * @brief Implementation of the OrchestralSystem class
 */

#include "OrchestralSystem.h"
#include <algorithm>
#include <iostream>
#include <fstream>

namespace orchestral {

OrchestralSystem::OrchestralSystem(const OrchestralConfig& config)
    : config_(config), registry_(&KernelRegistry::getInstance()) {
    startTime_ = std::chrono::system_clock::now();
    currentStatus_.startTime = startTime_;
    
    // Set up event handler
    eventHandler_ = [this](const KernelEvent& event) {
        std::cout << "System event handler called for: " << event.eventType << std::endl;
        handleSystemEvent(event);
        std::cout << "System event handler completed for: " << event.eventType << std::endl;
    };
}

OrchestralSystem::~OrchestralSystem() {
    shutdown();
}

bool OrchestralSystem::initialize() {
    if (running_) {
        return true;  // Already running
    }
    
    std::cout << "Initializing orchestral system..." << std::endl;
    
    // Initialize kernel registry
    if (!initializeRegistry()) {
        std::cerr << "Failed to initialize kernel registry" << std::endl;
        return false;
    }
    
    std::cout << "Kernel registry initialized" << std::endl;
    
    // Set up global event handling
    registry_->setGlobalEventHandler(eventHandler_);
    
    // Configure health monitoring
    registry_->setHealthMonitoring(config_.enableHealthMonitoring, 
                                  config_.heartbeatInterval);
    
    // Mark as running before initializing kernels
    running_ = true;
    
    // Initialize default kernels if requested
    if (!initializeKernels()) {
        std::cerr << "Failed to initialize default kernels" << std::endl;
        running_ = false;
        return false;
    }
    
    std::cout << "Default kernels initialized" << std::endl;
    
    std::cout << "Orchestral System initialized successfully" << std::endl;
    return true;
}

void OrchestralSystem::shutdown() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    // Unregister all default kernels
    for (auto& kernel : defaultKernels_) {
        if (kernel) {
            registry_->unregisterKernel(kernel->getName());
        }
    }
    
    defaultKernels_.clear();
    
    // Shutdown registry (this will shutdown all kernels)
    registry_->shutdown();
    
    std::cout << "Orchestral System shutdown complete" << std::endl;
}

bool OrchestralSystem::registerKernel(std::shared_ptr<AgenticKernel> kernel) {
    if (!kernel || !running_) {
        std::cerr << "Cannot register kernel - invalid kernel or system not running" << std::endl;
        return false;
    }
    
    std::cout << "Registering kernel: " << kernel->getName() << std::endl;
    bool result = registry_->registerKernel(kernel);
    std::cout << "Registration result: " << (result ? "success" : "failed") << std::endl;
    return result;
}

bool OrchestralSystem::createKernel(const std::string& type, const std::string& name) {
    if (!running_) {
        return false;
    }
    
    auto kernel = KernelFactory::createKernel(type, name);
    if (!kernel) {
        return false;
    }
    
    return registerKernel(kernel);
}

CognitiveResult OrchestralSystem::processInput(const CognitiveInput& input, 
                                              const std::string& preferredKernel) {
    if (!running_) {
        CognitiveResult result;
        result.success = false;
        result.errorMessage = "System not running";
        return result;
    }
    
    // Select appropriate kernel
    std::string selectedKernel = selectKernelForProcessing(input, preferredKernel);
    if (selectedKernel.empty()) {
        CognitiveResult result;
        result.success = false;
        result.errorMessage = "No suitable kernel found for processing";
        return result;
    }
    
    // Get kernel info
    const KernelInfo* info = registry_->getKernelInfo(selectedKernel);
    if (!info || !info->isLocal || !info->kernel) {
        CognitiveResult result;
        result.success = false;
        result.errorMessage = "Selected kernel not available for local processing";
        return result;
    }
    
    // Allocate attention if enabled
    std::map<std::string, double> attentionWeights;
    if (config_.enableAttentionAllocation) {
        attentionWeights = allocateAttention(input);
    }
    
    // Create enhanced input with attention context
    CognitiveInput enhancedInput = input;
    enhancedInput.contextWeights = attentionWeights;
    
    // Process with selected kernel
    CognitiveResult result = info->kernel->process(enhancedInput);
    
    // Update system metrics
    updateSystemMetrics();
    
    return result;
}

CognitiveResult OrchestralSystem::processText(const std::string& text, double urgency) {
    CognitiveInput input(text, "text");
    input.urgency = urgency;
    
    return processInput(input);
}

SystemStatus OrchestralSystem::getSystemStatus() const {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    
    SystemStatus status = currentStatus_;
    status.lastUpdate = std::chrono::system_clock::now();
    
    // Get registry metrics
    KernelMetrics systemMetrics = registry_->getSystemMetrics();
    status.totalProcessedItems = systemMetrics.processedItems;
    status.averageProcessingTime = systemMetrics.averageProcessingTime;
    status.cognitiveEfficiency = systemMetrics.cognitiveEfficiency;
    
    // Calculate system load
    std::vector<std::string> kernels = registry_->getAllKernelNames();
    status.activeKernels = kernels.size();
    
    double totalLoad = 0.0;
    for (const std::string& kernelName : kernels) {
        const KernelInfo* info = registry_->getKernelInfo(kernelName);
        if (info) {
            totalLoad += info->currentLoad;
            status.kernelLoads[kernelName] = info->currentLoad;
            status.kernelCounts[info->type]++;
        }
    }
    
    status.systemLoad = status.activeKernels > 0 ? totalLoad / status.activeKernels : 0.0;
    
    return status;
}

std::vector<std::string> OrchestralSystem::getRegisteredKernels() const {
    if (!running_) {
        return {};
    }
    
    return registry_->getAllKernelNames();
}

std::vector<std::string> OrchestralSystem::findKernelsByCapability(const std::string& capability) const {
    if (!running_) {
        return {};
    }
    
    return registry_->findKernelsByCapability(capability);
}

size_t OrchestralSystem::broadcastEvent(const KernelEvent& event, const std::string& targetType) {
    if (!running_) {
        return 0;
    }
    
    return registry_->broadcastEvent(event, targetType);
}

bool OrchestralSystem::configure(const OrchestralConfig& config) {
    config_ = config;
    
    if (running_) {
        // Apply configuration changes
        registry_->setHealthMonitoring(config_.enableHealthMonitoring, 
                                      config_.heartbeatInterval);
    }
    
    return true;
}

bool OrchestralSystem::setFeatureEnabled(const std::string& feature, bool enabled) {
    if (feature == "networking") {
        config_.enableNetworking = enabled;
    } else if (feature == "health_monitoring") {
        config_.enableHealthMonitoring = enabled;
        if (running_) {
            registry_->setHealthMonitoring(enabled, config_.heartbeatInterval);
        }
    } else if (feature == "attention_allocation") {
        config_.enableAttentionAllocation = enabled;
    } else if (feature == "neural_symbolic_bridge") {
        config_.enableNeuralSymbolicBridge = enabled;
    } else {
        return false;  // Unknown feature
    }
    
    return true;
}

std::chrono::duration<double> OrchestralSystem::getUptime() const {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration<double>(now - startTime_);
}

bool OrchestralSystem::createDefaultKernels() {
    if (!running_) {
        std::cerr << "Cannot create kernels - system not running" << std::endl;
        return false;
    }
    
    std::cout << "Registering kernel types..." << std::endl;
    
    // Register kernel types
    KernelFactory::registerKernelType("tokenization", []() {
        return std::make_shared<TokenizationKernel>("default_tokenizer");
    });
    
    std::cout << "Creating default tokenization kernel..." << std::endl;
    
    // Create default tokenization kernel
    auto tokenizer = std::make_shared<TokenizationKernel>("primary_tokenizer");
    if (!registerKernel(tokenizer)) {
        std::cerr << "Failed to register tokenization kernel" << std::endl;
        return false;
    }
    defaultKernels_.push_back(tokenizer);
    
    std::cout << "Default kernels created successfully" << std::endl;
    return true;
}

bool OrchestralSystem::saveState(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Save basic state information
    file << "# Orchestral System State" << std::endl;
    file << "running: " << (running_ ? "true" : "false") << std::endl;
    file << "start_time: " << std::chrono::duration_cast<std::chrono::seconds>(
        startTime_.time_since_epoch()).count() << std::endl;
    
    // Save configuration
    file << "networking: " << (config_.enableNetworking ? "true" : "false") << std::endl;
    file << "health_monitoring: " << (config_.enableHealthMonitoring ? "true" : "false") << std::endl;
    file << "heartbeat_interval: " << config_.heartbeatInterval << std::endl;
    
    // Save kernel information
    auto kernels = getRegisteredKernels();
    file << "registered_kernels: " << kernels.size() << std::endl;
    for (const std::string& name : kernels) {
        const KernelInfo* info = registry_->getKernelInfo(name);
        if (info) {
            file << "kernel: " << name << " " << info->type << " " << info->currentLoad << std::endl;
        }
    }
    
    return true;
}

bool OrchestralSystem::loadState(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // This is a simplified implementation
    // In practice, you'd parse the file format properly
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("networking:") == 0) {
            config_.enableNetworking = (line.find("true") != std::string::npos);
        } else if (line.find("health_monitoring:") == 0) {
            config_.enableHealthMonitoring = (line.find("true") != std::string::npos);
        }
        // Parse other configuration...
    }
    
    return true;
}

bool OrchestralSystem::initializeRegistry() {
    return registry_->initialize(config_.enableNetworking);
}

bool OrchestralSystem::initializeKernels() {
    return createDefaultKernels();
}

void OrchestralSystem::handleSystemEvent(const KernelEvent& event) {
    std::cout << "Handling system event: " << event.eventType << " from " << event.sourceKernel << std::endl;
    
    if (event.eventType == "kernel_registered") {
        std::cout << "Kernel registered: " << event.sourceKernel << std::endl;
        updateSystemMetrics();
    } else if (event.eventType == "kernel_unregistered") {
        std::cout << "Kernel unregistered: " << event.sourceKernel << std::endl;
        updateSystemMetrics();
    } else if (event.eventType == "processing_error") {
        std::cout << "Processing error in kernel: " << event.sourceKernel << std::endl;
    }
    
    std::cout << "System event handling completed for: " << event.eventType << std::endl;
}

void OrchestralSystem::updateSystemMetrics() {
    std::cout << "Updating system metrics..." << std::endl;
    
    std::lock_guard<std::mutex> lock(metricsMutex_);
    
    std::cout << "Metrics lock acquired" << std::endl;
    
    // Update current status
    currentStatus_.lastUpdate = std::chrono::system_clock::now();
    
    // This would be updated more comprehensively in a real implementation
    auto kernels = registry_->getAllKernelNames();
    currentStatus_.activeKernels = kernels.size();
    
    std::cout << "System metrics updated - active kernels: " << currentStatus_.activeKernels << std::endl;
}

std::string OrchestralSystem::selectKernelForProcessing(const CognitiveInput& input,
                                                       const std::string& preferredKernel) {
    // If preferred kernel specified and available, use it
    if (!preferredKernel.empty()) {
        const KernelInfo* info = registry_->getKernelInfo(preferredKernel);
        if (info && info->isLocal && info->kernel && 
            info->kernel->canProcess(input.type)) {
            return preferredKernel;
        }
    }
    
    // Find kernels that can process this input type
    std::vector<std::string> candidates;
    auto allKernels = registry_->getAllKernelNames();
    
    for (const std::string& name : allKernels) {
        const KernelInfo* info = registry_->getKernelInfo(name);
        if (info && info->isLocal && info->kernel && 
            info->kernel->canProcess(input.type)) {
            candidates.push_back(name);
        }
    }
    
    if (candidates.empty()) {
        return "";
    }
    
    // Select kernel with lowest load
    std::string bestKernel;
    double lowestLoad = 1.0;
    
    for (const std::string& name : candidates) {
        const KernelInfo* info = registry_->getKernelInfo(name);
        if (info && info->currentLoad < lowestLoad) {
            lowestLoad = info->currentLoad;
            bestKernel = name;
        }
    }
    
    return bestKernel;
}

std::map<std::string, double> OrchestralSystem::allocateAttention(const CognitiveInput& input) {
    std::map<std::string, double> weights;
    
    // Simple attention allocation based on urgency and context
    double baseAttention = 0.5 + 0.5 * input.urgency;
    
    // For text input, give higher attention to certain words
    if (input.type == "text") {
        std::vector<std::string> highAttentionWords = {
            "important", "critical", "urgent", "simple", "hello"
        };
        
        for (const std::string& word : highAttentionWords) {
            if (input.data.find(word) != std::string::npos) {
                weights[word] = std::min(1.0, baseAttention + 0.3);
            }
        }
    }
    
    return weights;
}

// Factory functions

std::unique_ptr<OrchestralSystem> createOrchestralSystem(const OrchestralConfig& config) {
    auto system = std::make_unique<OrchestralSystem>(config);
    if (!system->initialize()) {
        std::cerr << "Failed to initialize orchestral system" << std::endl;
        return nullptr;
    }
    return system;
}

std::unique_ptr<OrchestralSystem> createDemoSystem() {
    OrchestralConfig config;
    config.enableNetworking = false;
    config.enableHealthMonitoring = false;  // Disable to avoid threading issues in demo
    config.enableAttentionAllocation = true;
    config.heartbeatInterval = 10.0;  // Faster updates for demo
    
    auto system = createOrchestralSystem(config);
    if (!system) {
        std::cerr << "Failed to create demo system" << std::endl;
    }
    return system;
}

} // namespace orchestral