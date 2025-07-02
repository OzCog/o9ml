/**
 * @file AgenticKernel.h
 * @brief Base interface for autonomous cognitive processing kernels
 * 
 * The AgenticKernel provides the foundational interface for all cognitive
 * processing units in the Orchestral Architect system. Each kernel is
 * responsible for specialized cognitive tasks while participating in
 * the distributed coordination network.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <atomic>
#include <chrono>
#include <mutex>

namespace orchestral {

/**
 * @brief Represents the result of cognitive processing
 */
struct CognitiveResult {
    std::string processedData;
    std::map<std::string, double> attentionWeights;
    double processingCost;
    double estimatedValue;
    std::chrono::milliseconds processingTime;
    bool success;
    std::string errorMessage;
    
    CognitiveResult() : processingCost(0.0), estimatedValue(0.0), 
                       processingTime(0), success(false) {}
};

/**
 * @brief Input data structure for cognitive processing
 */
struct CognitiveInput {
    std::string data;
    std::string type;
    std::map<std::string, double> contextWeights;
    double urgency;
    std::string sourceKernel;
    
    CognitiveInput(const std::string& d = "", const std::string& t = "text") 
        : data(d), type(t), urgency(0.5) {}
};

/**
 * @brief Event structure for inter-kernel communication
 */
struct KernelEvent {
    std::string eventType;
    std::string sourceKernel;
    std::string targetKernel;
    std::map<std::string, std::string> payload;
    std::chrono::system_clock::time_point timestamp;
    double priority;
    
    KernelEvent() : priority(0.5) {
        timestamp = std::chrono::system_clock::now();
    }
};

/**
 * @brief Kernel performance metrics
 */
struct KernelMetrics {
    uint64_t processedItems = 0;
    double averageProcessingTime = 0.0;
    double successRate = 0.0;
    double resourceUtilization = 0.0;
    double cognitiveEfficiency = 0.0;
    
    std::chrono::system_clock::time_point lastUpdate;
    
    KernelMetrics() {
        lastUpdate = std::chrono::system_clock::now();
    }
};

/**
 * @brief Base interface for all cognitive processing kernels
 * 
 * AgenticKernel defines the contract that all cognitive processing
 * units must implement to participate in the orchestral system.
 */
class AgenticKernel {
public:
    using EventCallback = std::function<void(const KernelEvent&)>;
    
    /**
     * @brief Construct a new Agentic Kernel
     * @param name Unique name for this kernel
     * @param type Type/category of cognitive processing
     */
    AgenticKernel(const std::string& name, const std::string& type);
    
    virtual ~AgenticKernel() = default;
    
    /**
     * @brief Initialize the kernel with necessary resources
     * @return true if initialization successful
     */
    virtual bool initialize() = 0;
    
    /**
     * @brief Shutdown and cleanup kernel resources
     */
    virtual void shutdown() = 0;
    
    /**
     * @brief Process cognitive input and produce result
     * @param input The cognitive data to process
     * @return Processing result with attention weights and metrics
     */
    virtual CognitiveResult process(const CognitiveInput& input) = 0;
    
    /**
     * @brief Handle incoming event from another kernel
     * @param event The event to process
     */
    virtual void handleEvent(const KernelEvent& event) = 0;
    
    /**
     * @brief Get kernel capabilities and specializations
     * @return Vector of capability strings
     */
    virtual std::vector<std::string> getCapabilities() const = 0;
    
    /**
     * @brief Check if kernel can process given input type
     * @param inputType The type of input to check
     * @return true if kernel can handle this input
     */
    virtual bool canProcess(const std::string& inputType) const = 0;
    
    /**
     * @brief Get current processing load (0.0 to 1.0)
     * @return Current load factor
     */
    virtual double getCurrentLoad() const = 0;
    
    /**
     * @brief Get kernel performance metrics
     * @return Current metrics snapshot
     */
    virtual KernelMetrics getMetrics() const;
    
    /**
     * @brief Register callback for kernel events
     * @param callback Function to call on events
     */
    void setEventCallback(EventCallback callback);
    
    /**
     * @brief Emit event to other kernels
     * @param event Event to emit
     */
    void emitEvent(const KernelEvent& event);
    
    // Accessors
    const std::string& getName() const { return name_; }
    const std::string& getType() const { return type_; }
    bool isActive() const { return active_; }
    std::chrono::system_clock::time_point getStartTime() const { return startTime_; }
    
protected:
    /**
     * @brief Update kernel metrics with processing result
     * @param result The processing result
     * @param processingTime Time taken for processing
     */
    void updateMetrics(const CognitiveResult& result, 
                      std::chrono::milliseconds processingTime);
    
    /**
     * @brief Calculate cognitive efficiency based on value/cost ratio
     * @param result Processing result
     * @return Efficiency score (0.0 to 1.0)
     */
    double calculateEfficiency(const CognitiveResult& result) const;
    
    /**
     * @brief Set the active state of the kernel
     * @param active Whether the kernel is active
     */
    void setActive(bool active) { active_ = active; }
    
private:
    std::string name_;
    std::string type_;
    std::atomic<bool> active_{false};
    std::chrono::system_clock::time_point startTime_;
    
    KernelMetrics metrics_;
    EventCallback eventCallback_;
    
    // Performance tracking
    std::atomic<uint64_t> totalProcessingTime_{0};
    std::atomic<uint64_t> successfulOperations_{0};
    std::atomic<uint64_t> totalOperations_{0};
};

/**
 * @brief Factory for creating specialized kernels
 */
class KernelFactory {
public:
    using KernelCreator = std::function<std::shared_ptr<AgenticKernel>()>;
    
    /**
     * @brief Register a kernel type with its creator function
     * @param type Kernel type name
     * @param creator Function that creates instances
     */
    static void registerKernelType(const std::string& type, KernelCreator creator);
    
    /**
     * @brief Create a kernel instance of specified type
     * @param type Type of kernel to create
     * @param name Name for the new kernel instance
     * @return Shared pointer to created kernel, null if type unknown
     */
    static std::shared_ptr<AgenticKernel> createKernel(const std::string& type, 
                                                      const std::string& name);
    
    /**
     * @brief Get list of registered kernel types
     * @return Vector of available kernel types
     */
    static std::vector<std::string> getRegisteredTypes();
    
private:
    static std::map<std::string, KernelCreator> creators_;
};

} // namespace orchestral