/**
 * @file KernelRegistry.cpp
 * @brief Implementation of the KernelRegistry class
 */

#include "KernelRegistry.h"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace orchestral {

KernelRegistry& KernelRegistry::getInstance() {
    static KernelRegistry instance;
    return instance;
}

KernelRegistry::~KernelRegistry() {
    shutdown();
}

bool KernelRegistry::initialize(bool enableNetworking) {
    std::lock_guard<std::mutex> lock(registryMutex_);
    
    if (running_) {
        return true;  // Already initialized
    }
    
    networkingEnabled_ = enableNetworking;
    
    // Set default load balancer
    if (!loadBalancer_) {
        loadBalancer_ = [this](const std::vector<std::string>& candidates, 
                               const std::string& capability) {
            return defaultLoadBalancer(candidates, capability);
        };
    }
    
    // Start background threads
    running_ = true;
    
    eventProcessingThread_ = std::thread(&KernelRegistry::eventProcessingLoop, this);
    
    if (healthMonitoringEnabled_) {
        healthMonitorThread_ = std::thread(&KernelRegistry::healthMonitoringLoop, this);
    }
    
    return true;
}

void KernelRegistry::shutdown() {
    {
        std::lock_guard<std::mutex> lock(registryMutex_);
        running_ = false;
    }
    
    // Notify event processing thread
    eventCondition_.notify_all();
    
    // Join threads
    if (eventProcessingThread_.joinable()) {
        eventProcessingThread_.join();
    }
    
    if (healthMonitorThread_.joinable()) {
        healthMonitorThread_.join();
    }
    
    // Shutdown all local kernels
    std::lock_guard<std::mutex> lock(registryMutex_);
    for (auto& pair : kernels_) {
        if (pair.second.isLocal && pair.second.kernel) {
            pair.second.kernel->shutdown();
        }
    }
    
    kernels_.clear();
}

bool KernelRegistry::registerKernel(std::shared_ptr<AgenticKernel> kernel) {
    if (!kernel) {
        std::cerr << "Cannot register null kernel" << std::endl;
        return false;
    }
    
    std::cout << "Registry: Registering kernel " << kernel->getName() << std::endl;
    
    std::unique_lock<std::mutex> lock(registryMutex_);
    
    const std::string& name = kernel->getName();
    
    // Check if already registered
    if (kernels_.find(name) != kernels_.end()) {
        std::cerr << "Kernel already registered: " << name << std::endl;
        return false;
    }
    
    std::cout << "Registry: Initializing kernel " << name << std::endl;
    
    // Initialize kernel if needed
    if (!kernel->initialize()) {
        std::cerr << "Failed to initialize kernel: " << name << std::endl;
        return false;
    }
    
    std::cout << "Registry: Creating kernel info for " << name << std::endl;
    
    // Create kernel info
    KernelInfo info;
    info.name = name;
    info.type = kernel->getType();
    info.capabilities = kernel->getCapabilities();
    info.currentLoad = kernel->getCurrentLoad();
    info.metrics = kernel->getMetrics();
    info.isLocal = true;
    info.kernel = kernel;
    info.lastHeartbeat = std::chrono::system_clock::now();
    
    std::cout << "Registry: Setting up event callback for " << name << std::endl;
    
    // Set up event callback
    kernel->setEventCallback([this](const KernelEvent& event) {
        std::cout << "Event received from kernel: " << event.sourceKernel << std::endl;
        routeEvent(event);
    });
    
    std::cout << "Registry: Storing kernel info for " << name << std::endl;
    
    kernels_[name] = std::move(info);
    
    std::cout << "Registry: Kernel " << name << " successfully registered" << std::endl;
    
    // Release the lock before calling event handler to avoid deadlock
    lock.unlock();
    
    std::cout << "Registry: Creating registration event for " << name << std::endl;
    
    // Emit registration event
    KernelEvent regEvent;
    regEvent.eventType = "kernel_registered";
    regEvent.sourceKernel = name;
    regEvent.payload["kernel_type"] = kernel->getType();
    
    if (globalEventHandler_) {
        std::cout << "Registry: Calling global event handler for " << name << std::endl;
        globalEventHandler_(regEvent);
    }
    
    return true;
}

bool KernelRegistry::unregisterKernel(const std::string& kernelName) {
    std::lock_guard<std::mutex> lock(registryMutex_);
    
    auto it = kernels_.find(kernelName);
    if (it == kernels_.end()) {
        return false;
    }
    
    // Shutdown local kernel
    if (it->second.isLocal && it->second.kernel) {
        it->second.kernel->shutdown();
    }
    
    kernels_.erase(it);
    
    // Emit unregistration event
    KernelEvent unregEvent;
    unregEvent.eventType = "kernel_unregistered";
    unregEvent.sourceKernel = kernelName;
    
    if (globalEventHandler_) {
        globalEventHandler_(unregEvent);
    }
    
    return true;
}

bool KernelRegistry::registerRemoteKernel(const KernelInfo& info) {
    std::lock_guard<std::mutex> lock(registryMutex_);
    
    if (kernels_.find(info.name) != kernels_.end()) {
        return false;  // Already registered
    }
    
    KernelInfo remoteInfo = info;
    remoteInfo.isLocal = false;
    remoteInfo.kernel = nullptr;
    remoteInfo.lastHeartbeat = std::chrono::system_clock::now();
    
    kernels_[info.name] = remoteInfo;
    return true;
}

std::vector<std::string> KernelRegistry::findKernelsByCapability(const std::string& capability) const {
    std::lock_guard<std::mutex> lock(registryMutex_);
    
    std::vector<std::string> result;
    
    for (const auto& pair : kernels_) {
        const auto& caps = pair.second.capabilities;
        if (std::find(caps.begin(), caps.end(), capability) != caps.end()) {
            result.push_back(pair.first);
        }
    }
    
    return result;
}

std::vector<std::string> KernelRegistry::findKernelsByType(const std::string& type) const {
    std::lock_guard<std::mutex> lock(registryMutex_);
    
    std::vector<std::string> result;
    
    for (const auto& pair : kernels_) {
        if (pair.second.type == type) {
            result.push_back(pair.first);
        }
    }
    
    return result;
}

const KernelInfo* KernelRegistry::getKernelInfo(const std::string& kernelName) const {
    std::lock_guard<std::mutex> lock(registryMutex_);
    
    auto it = kernels_.find(kernelName);
    return (it != kernels_.end()) ? &it->second : nullptr;
}

bool KernelRegistry::routeEvent(const KernelEvent& event, const EventRoute& route) {
    std::string targetKernel = route.targetKernel;
    
    // If no specific target, use event's target
    if (targetKernel.empty()) {
        targetKernel = event.targetKernel;
    }
    
    // If still no target, try to find appropriate kernel
    if (targetKernel.empty()) {
        std::vector<std::string> candidates = findKernelsByType("any");
        if (!candidates.empty()) {
            targetKernel = loadBalancer_(candidates, "general");
        }
    }
    
    // Queue the event for processing
    {
        std::lock_guard<std::mutex> lock(eventMutex_);
        eventQueue_.emplace(event, EventRoute(targetKernel, route.routingStrategy));
    }
    
    eventCondition_.notify_one();
    return true;
}

size_t KernelRegistry::broadcastEvent(const KernelEvent& event, const std::string& targetType) {
    std::lock_guard<std::mutex> lock(registryMutex_);
    
    size_t count = 0;
    
    for (const auto& pair : kernels_) {
        if (!targetType.empty() && pair.second.type != targetType) {
            continue;
        }
        
        if (pair.second.isLocal && pair.second.kernel) {
            pair.second.kernel->handleEvent(event);
            count++;
        }
        // TODO: Handle remote kernels via network
    }
    
    return count;
}

std::string KernelRegistry::selectBestKernel(const std::string& capability) const {
    auto candidates = findKernelsByCapability(capability);
    if (candidates.empty()) {
        return "";
    }
    
    return loadBalancer_(candidates, capability);
}

std::vector<std::string> KernelRegistry::getAllKernelNames() const {
    std::lock_guard<std::mutex> lock(registryMutex_);
    
    std::vector<std::string> names;
    names.reserve(kernels_.size());
    
    for (const auto& pair : kernels_) {
        names.push_back(pair.first);
    }
    
    return names;
}

KernelMetrics KernelRegistry::getSystemMetrics() const {
    std::lock_guard<std::mutex> lock(registryMutex_);
    
    KernelMetrics aggregated;
    size_t activeKernels = 0;
    
    double totalEfficiency = 0.0;
    double totalUtilization = 0.0;
    
    for (const auto& pair : kernels_) {
        const auto& metrics = pair.second.metrics;
        
        aggregated.processedItems += metrics.processedItems;
        
        if (pair.second.isLocal) {
            activeKernels++;
            totalEfficiency += metrics.cognitiveEfficiency;
            totalUtilization += metrics.resourceUtilization;
            
            // Average processing time weighted by items processed
            uint64_t items = metrics.processedItems;
            if (items > 0) {
                double weighted = metrics.averageProcessingTime * items;
                aggregated.averageProcessingTime = 
                    (aggregated.averageProcessingTime + weighted) / aggregated.processedItems;
            }
        }
    }
    
    if (activeKernels > 0) {
        aggregated.cognitiveEfficiency = totalEfficiency / activeKernels;
        aggregated.resourceUtilization = totalUtilization / activeKernels;
    }
    
    aggregated.lastUpdate = std::chrono::system_clock::now();
    
    return aggregated;
}

void KernelRegistry::setLoadBalancer(LoadBalancer balancer) {
    loadBalancer_ = balancer;
}

void KernelRegistry::setHealthMonitoring(bool enabled, double heartbeatInterval) {
    bool wasEnabled = healthMonitoringEnabled_.load();
    healthMonitoringEnabled_ = enabled;
    heartbeatInterval_ = heartbeatInterval;
    
    if (enabled && !wasEnabled && running_) {
        // Start health monitoring thread
        healthMonitorThread_ = std::thread(&KernelRegistry::healthMonitoringLoop, this);
    }
}

void KernelRegistry::setGlobalEventHandler(EventHandler handler) {
    globalEventHandler_ = handler;
}

void KernelRegistry::healthMonitoringLoop() {
    while (running_ && healthMonitoringEnabled_) {
        std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(heartbeatInterval_)));
        
        if (!running_) break;
        
        // Update metrics for local kernels
        {
            std::lock_guard<std::mutex> lock(registryMutex_);
            auto now = std::chrono::system_clock::now();
            
            for (auto& pair : kernels_) {
                if (pair.second.isLocal && pair.second.kernel) {
                    auto newMetrics = pair.second.kernel->getMetrics();
                    pair.second.metrics = newMetrics;
                    pair.second.currentLoad = pair.second.kernel->getCurrentLoad();
                    pair.second.lastHeartbeat = now;
                }
            }
        }
        
        // Prune unhealthy kernels
        pruneUnhealthyKernels();
    }
}

void KernelRegistry::eventProcessingLoop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(eventMutex_);
        
        eventCondition_.wait(lock, [this] {
            return !eventQueue_.empty() || !running_;
        });
        
        if (!running_) break;
        
        while (!eventQueue_.empty()) {
            auto [event, route] = eventQueue_.front();
            eventQueue_.pop();
            lock.unlock();
            
            // Process event
            {
                std::lock_guard<std::mutex> regLock(registryMutex_);
                auto it = kernels_.find(route.targetKernel);
                if (it != kernels_.end() && it->second.isLocal && it->second.kernel) {
                    it->second.kernel->handleEvent(event);
                }
            }
            
            lock.lock();
        }
    }
}

void KernelRegistry::pruneUnhealthyKernels() {
    std::lock_guard<std::mutex> lock(registryMutex_);
    
    auto now = std::chrono::system_clock::now();
    auto maxAge = std::chrono::seconds(static_cast<int>(heartbeatInterval_ * 3));
    
    auto it = kernels_.begin();
    while (it != kernels_.end()) {
        if (!it->second.isLocal) {
            // Check if remote kernel is stale
            if (now - it->second.lastHeartbeat > maxAge) {
                it = kernels_.erase(it);
                continue;
            }
        }
        ++it;
    }
}

std::string KernelRegistry::defaultLoadBalancer(const std::vector<std::string>& candidates,
                                               const std::string& capability) const {
    if (candidates.empty()) {
        return "";
    }
    
    // Find kernel with lowest load
    std::string bestKernel;
    double lowestLoad = 1.0;
    
    for (const std::string& name : candidates) {
        auto it = kernels_.find(name);
        if (it != kernels_.end()) {
            double load = it->second.currentLoad;
            if (load < lowestLoad) {
                lowestLoad = load;
                bestKernel = name;
            }
        }
    }
    
    return bestKernel;
}

// KernelRegistration implementation

KernelRegistration::KernelRegistration(std::shared_ptr<AgenticKernel> kernel)
    : kernel_(kernel), registered_(false) {
    if (kernel_) {
        registered_ = KernelRegistry::getInstance().registerKernel(kernel_);
    }
}

KernelRegistration::~KernelRegistration() {
    if (registered_ && kernel_) {
        KernelRegistry::getInstance().unregisterKernel(kernel_->getName());
    }
}

} // namespace orchestral