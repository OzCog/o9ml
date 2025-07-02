/**
 * @file AgenticKernel.cpp
 * @brief Implementation of the base AgenticKernel class
 */

#include "AgenticKernel.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace orchestral {

// Static member initialization
std::map<std::string, KernelFactory::KernelCreator> KernelFactory::creators_;

AgenticKernel::AgenticKernel(const std::string& name, const std::string& type)
    : name_(name), type_(type) {
    startTime_ = std::chrono::system_clock::now();
}

KernelMetrics AgenticKernel::getMetrics() const {
    return metrics_;
}

void AgenticKernel::setEventCallback(EventCallback callback) {
    eventCallback_ = callback;
}

void AgenticKernel::emitEvent(const KernelEvent& event) {
    if (eventCallback_) {
        eventCallback_(event);
    }
}

void AgenticKernel::updateMetrics(const CognitiveResult& result, 
                                 std::chrono::milliseconds processingTime) {
    // Update counters - using simple thread-safe updates
    static std::mutex metricsMutex;
    std::lock_guard<std::mutex> lock(metricsMutex);
    
    metrics_.processedItems++;
    totalOperations_.fetch_add(1);
    
    if (result.success) {
        successfulOperations_.fetch_add(1);
    }
    
    // Update processing time
    uint64_t currentTotal = totalProcessingTime_.load();
    uint64_t newTotal = currentTotal + processingTime.count();
    totalProcessingTime_.store(newTotal);
    
    uint64_t totalOps = totalOperations_.load();
    if (totalOps > 0) {
        metrics_.averageProcessingTime = 
            static_cast<double>(newTotal) / totalOps;
    }
    
    // Update success rate
    uint64_t successOps = successfulOperations_.load();
    metrics_.successRate = 
        totalOps > 0 ? static_cast<double>(successOps) / totalOps : 0.0;
    
    // Update cognitive efficiency
    double efficiency = calculateEfficiency(result);
    // Use exponential moving average for efficiency
    double currentEfficiency = metrics_.cognitiveEfficiency;
    double newEfficiency = 0.9 * currentEfficiency + 0.1 * efficiency;
    metrics_.cognitiveEfficiency = newEfficiency;
    
    // Update timestamp
    metrics_.lastUpdate = std::chrono::system_clock::now();
}

double AgenticKernel::calculateEfficiency(const CognitiveResult& result) const {
    if (!result.success || result.processingCost <= 0.0) {
        return 0.0;
    }
    
    // Efficiency = Value / Cost, normalized to [0,1]
    double rawEfficiency = result.estimatedValue / result.processingCost;
    
    // Apply sigmoid normalization to keep in reasonable range
    return 2.0 / (1.0 + std::exp(-rawEfficiency)) - 1.0;
}

// KernelFactory implementation

void KernelFactory::registerKernelType(const std::string& type, KernelCreator creator) {
    if (creator == nullptr) {
        throw std::invalid_argument("Creator function cannot be null");
    }
    creators_[type] = creator;
}

std::shared_ptr<AgenticKernel> KernelFactory::createKernel(const std::string& type, 
                                                          const std::string& name) {
    auto it = creators_.find(type);
    if (it == creators_.end()) {
        return nullptr;
    }
    
    auto kernel = it->second();
    if (kernel) {
        // Note: This assumes the kernel can be configured with a name after creation
        // In practice, you might need a different factory pattern
        return kernel;
    }
    
    return nullptr;
}

std::vector<std::string> KernelFactory::getRegisteredTypes() {
    std::vector<std::string> types;
    types.reserve(creators_.size());
    
    for (const auto& pair : creators_) {
        types.push_back(pair.first);
    }
    
    return types;
}

} // namespace orchestral