/**
 * @file KernelRegistry.h
 * @brief Distributed kernel discovery and communication management
 * 
 * The KernelRegistry manages the distributed network of cognitive kernels,
 * providing service discovery, event routing, and load balancing across
 * the orchestral system.
 */

#pragma once

#include "AgenticKernel.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <functional>

namespace orchestral {

/**
 * @brief Information about a registered kernel
 */
struct KernelInfo {
    std::string name;
    std::string type;
    std::vector<std::string> capabilities;
    double currentLoad;
    KernelMetrics metrics;
    std::chrono::system_clock::time_point lastHeartbeat;
    bool isLocal;
    std::string networkAddress;  // For remote kernels
    
    std::shared_ptr<AgenticKernel> kernel;  // null for remote kernels
    
    KernelInfo() : currentLoad(0.0), isLocal(true) {
        lastHeartbeat = std::chrono::system_clock::now();
    }
};

/**
 * @brief Event routing information
 */
struct EventRoute {
    std::string targetKernel;
    std::string routingStrategy;  // "direct", "load_balance", "broadcast"
    double priority;
    
    EventRoute(const std::string& target = "", const std::string& strategy = "direct")
        : targetKernel(target), routingStrategy(strategy), priority(0.5) {}
};

/**
 * @brief Manages the distributed network of cognitive kernels
 * 
 * The KernelRegistry provides centralized coordination for the orchestral
 * system, handling kernel discovery, event routing, load balancing, and
 * health monitoring.
 */
class KernelRegistry {
public:
    using EventHandler = std::function<void(const KernelEvent&)>;
    using LoadBalancer = std::function<std::string(const std::vector<std::string>&, 
                                                  const std::string&)>;
    
    /**
     * @brief Get the singleton instance of the registry
     * @return Reference to the global registry
     */
    static KernelRegistry& getInstance();
    
    /**
     * @brief Initialize the registry system
     * @param enableNetworking Whether to enable network discovery
     * @return true if initialization successful
     */
    bool initialize(bool enableNetworking = false);
    
    /**
     * @brief Shutdown the registry and all managed kernels
     */
    void shutdown();
    
    /**
     * @brief Register a local kernel with the system
     * @param kernel Shared pointer to the kernel
     * @return true if registration successful
     */
    bool registerKernel(std::shared_ptr<AgenticKernel> kernel);
    
    /**
     * @brief Unregister a kernel from the system
     * @param kernelName Name of kernel to remove
     * @return true if unregistration successful
     */
    bool unregisterKernel(const std::string& kernelName);
    
    /**
     * @brief Register a remote kernel
     * @param info Kernel information including network address
     * @return true if registration successful
     */
    bool registerRemoteKernel(const KernelInfo& info);
    
    /**
     * @brief Find kernels by capability
     * @param capability Required capability
     * @return Vector of kernel names that provide the capability
     */
    std::vector<std::string> findKernelsByCapability(const std::string& capability) const;
    
    /**
     * @brief Find kernels by type
     * @param type Required kernel type
     * @return Vector of kernel names of the specified type
     */
    std::vector<std::string> findKernelsByType(const std::string& type) const;
    
    /**
     * @brief Get information about a specific kernel
     * @param kernelName Name of the kernel
     * @return Pointer to kernel info, null if not found
     */
    const KernelInfo* getKernelInfo(const std::string& kernelName) const;
    
    /**
     * @brief Route an event to appropriate kernel(s)
     * @param event Event to route
     * @param route Routing strategy (optional)
     * @return true if event was routed successfully
     */
    bool routeEvent(const KernelEvent& event, const EventRoute& route = EventRoute());
    
    /**
     * @brief Broadcast event to all kernels of specified type
     * @param event Event to broadcast
     * @param targetType Type of kernels to target (empty = all)
     * @return Number of kernels that received the event
     */
    size_t broadcastEvent(const KernelEvent& event, const std::string& targetType = "");
    
    /**
     * @brief Get kernel with lowest load for capability
     * @param capability Required capability
     * @return Name of best kernel, empty if none available
     */
    std::string selectBestKernel(const std::string& capability) const;
    
    /**
     * @brief Get all registered kernel names
     * @return Vector of kernel names
     */
    std::vector<std::string> getAllKernelNames() const;
    
    /**
     * @brief Get system-wide metrics
     * @return Aggregated metrics across all kernels
     */
    KernelMetrics getSystemMetrics() const;
    
    /**
     * @brief Set custom load balancing strategy
     * @param balancer Function to select kernel from candidates
     */
    void setLoadBalancer(LoadBalancer balancer);
    
    /**
     * @brief Enable/disable automatic health monitoring
     * @param enabled Whether to monitor kernel health
     * @param heartbeatInterval Seconds between heartbeat checks
     */
    void setHealthMonitoring(bool enabled, double heartbeatInterval = 30.0);
    
    /**
     * @brief Register global event handler
     * @param handler Function to handle all events
     */
    void setGlobalEventHandler(EventHandler handler);
    
    /**
     * @brief Check if registry is running
     * @return true if registry is active
     */
    bool isRunning() const { return running_; }
    
private:
    KernelRegistry() = default;
    ~KernelRegistry();
    
    // Prevent copying
    KernelRegistry(const KernelRegistry&) = delete;
    KernelRegistry& operator=(const KernelRegistry&) = delete;
    
    /**
     * @brief Background thread for health monitoring
     */
    void healthMonitoringLoop();
    
    /**
     * @brief Background thread for event processing
     */
    void eventProcessingLoop();
    
    /**
     * @brief Remove unhealthy kernels
     */
    void pruneUnhealthyKernels();
    
    /**
     * @brief Default load balancer implementation
     */
    std::string defaultLoadBalancer(const std::vector<std::string>& candidates,
                                   const std::string& capability) const;
    
    // Core data structures
    mutable std::mutex registryMutex_;
    std::unordered_map<std::string, KernelInfo> kernels_;
    
    // Event processing
    std::queue<std::pair<KernelEvent, EventRoute>> eventQueue_;
    std::mutex eventMutex_;
    std::condition_variable eventCondition_;
    
    // Threading and lifecycle
    std::atomic<bool> running_{false};
    std::atomic<bool> healthMonitoringEnabled_{false};
    std::atomic<double> heartbeatInterval_{30.0};
    
    std::thread healthMonitorThread_;
    std::thread eventProcessingThread_;
    
    // Configuration
    LoadBalancer loadBalancer_;
    EventHandler globalEventHandler_;
    bool networkingEnabled_{false};
};

/**
 * @brief RAII helper for kernel registration
 */
class KernelRegistration {
public:
    KernelRegistration(std::shared_ptr<AgenticKernel> kernel);
    ~KernelRegistration();
    
    bool isRegistered() const { return registered_; }
    
private:
    std::shared_ptr<AgenticKernel> kernel_;
    bool registered_;
};

} // namespace orchestral