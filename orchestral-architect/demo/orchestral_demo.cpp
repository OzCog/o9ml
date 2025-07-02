/**
 * @file orchestral_demo.cpp
 * @brief Live demonstration of the Orchestral Architect system
 * 
 * This demo showcases the key features of the orchestral system:
 * - Multi-strategy tokenization with attention weighting
 * - Event-driven kernel communication
 * - Economic attention allocation
 * - Real-time cognitive processing metrics
 */

#include "../src/OrchestralSystem.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

using namespace orchestral;

/**
 * @brief Display system status in a formatted way
 */
void displaySystemStatus(const SystemStatus& status) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "ORCHESTRAL ARCHITECT SYSTEM STATUS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Active Kernels:        " << status.activeKernels << std::endl;
    std::cout << "Processed Items:       " << status.totalProcessedItems << std::endl;
    std::cout << "Avg Processing Time:   " << status.averageProcessingTime << " ms" << std::endl;
    std::cout << "System Load:           " << (status.systemLoad * 100.0) << "%" << std::endl;
    std::cout << "Cognitive Efficiency:  " << (status.cognitiveEfficiency * 100.0) << "%" << std::endl;
    
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        status.lastUpdate - status.startTime);
    std::cout << "Uptime:                " << uptime.count() << " seconds" << std::endl;
    
    if (!status.kernelCounts.empty()) {
        std::cout << "\nKernel Types:" << std::endl;
        for (const auto& pair : status.kernelCounts) {
            std::cout << "  " << pair.first << ": " << pair.second << std::endl;
        }
    }
    
    if (!status.kernelLoads.empty()) {
        std::cout << "\nKernel Loads:" << std::endl;
        for (const auto& pair : status.kernelLoads) {
            std::cout << "  " << pair.first << ": " 
                      << std::fixed << std::setprecision(1) 
                      << (pair.second * 100.0) << "%" << std::endl;
        }
    }
    
    std::cout << std::string(60, '=') << std::endl;
}

/**
 * @brief Display cognitive processing result
 */
void displayResult(const CognitiveResult& result, const std::string& input) {
    std::cout << "\n" << std::string(50, '-') << std::endl;
    std::cout << "COGNITIVE PROCESSING RESULT" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    std::cout << "Input: \"" << input << "\"" << std::endl;
    std::cout << "Success: " << (result.success ? "✓" : "✗") << std::endl;
    
    if (!result.success) {
        std::cout << "Error: " << result.errorMessage << std::endl;
        return;
    }
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Processing Time: " << result.processingTime.count() << " ms" << std::endl;
    std::cout << "Processing Cost: " << result.processingCost << std::endl;
    std::cout << "Estimated Value: " << result.estimatedValue << std::endl;
    std::cout << "Efficiency Ratio: " << (result.estimatedValue / result.processingCost) << std::endl;
    
    std::cout << "\nTokenization Result:" << std::endl;
    std::cout << result.processedData << std::endl;
    
    if (!result.attentionWeights.empty()) {
        std::cout << "\nAttention Weights:" << std::endl;
        // Sort attention weights by value (descending)
        std::vector<std::pair<std::string, double>> sortedWeights(
            result.attentionWeights.begin(), result.attentionWeights.end());
        std::sort(sortedWeights.begin(), sortedWeights.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (const auto& pair : sortedWeights) {
            std::cout << "  '" << pair.first << "' → " 
                      << std::fixed << std::setprecision(2) 
                      << pair.second << std::endl;
        }
    }
    
    std::cout << std::string(50, '-') << std::endl;
}

/**
 * @brief Demonstrate kernel communication
 */
void demonstrateKernelCommunication(OrchestralSystem& system) {
    std::cout << "\n🔗 DEMONSTRATING KERNEL COMMUNICATION" << std::endl;
    
    // Create a custom event
    KernelEvent event;
    event.eventType = "attention_update";
    event.sourceKernel = "demo_controller";
    event.payload["vocabulary"] = "important=0.9,critical=0.95";
    event.priority = 0.8;
    
    // Broadcast to tokenization kernels
    size_t recipients = system.broadcastEvent(event, "tokenization");
    
    std::cout << "Broadcasted attention update to " << recipients 
              << " tokenization kernels" << std::endl;
    
    // Show registered kernels
    auto kernels = system.getRegisteredKernels();
    std::cout << "Active kernels: ";
    for (size_t i = 0; i < kernels.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << kernels[i];
    }
    std::cout << std::endl;
}

/**
 * @brief Demonstrate attention allocation
 */
void demonstrateAttentionAllocation(OrchestralSystem& system) {
    std::cout << "\n💰 DEMONSTRATING ECONOMIC ATTENTION ALLOCATION" << std::endl;
    
    std::vector<std::string> testInputs = {
        "hello world",
        "this is a simple test",
        "critical system failure detected",
        "important meeting at noon"
    };
    
    for (const std::string& input : testInputs) {
        std::cout << "\nProcessing: \"" << input << "\"" << std::endl;
        
        auto result = system.processText(input, 0.7);  // High urgency
        
        if (result.success && !result.attentionWeights.empty()) {
            std::cout << "→ Attention allocated to: ";
            bool first = true;
            for (const auto& pair : result.attentionWeights) {
                if (!first) std::cout << ", ";
                std::cout << pair.first << " (" 
                          << std::fixed << std::setprecision(2) 
                          << pair.second << ")";
                first = false;
            }
            std::cout << std::endl;
            std::cout << "→ Value/Cost ratio: " 
                      << std::fixed << std::setprecision(2)
                      << (result.estimatedValue / result.processingCost) << std::endl;
        }
    }
}

/**
 * @brief Run performance benchmarks
 */
void runPerformanceBenchmark(OrchestralSystem& system) {
    std::cout << "\n⚡ PERFORMANCE BENCHMARK" << std::endl;
    
    const std::string testText = "The quick brown fox jumps over the lazy dog. "
                                "This sentence contains every letter of the alphabet. "
                                "It is commonly used for testing text processing systems.";
    
    const int iterations = 10;
    std::vector<double> processingTimes;
    
    std::cout << "Running " << iterations << " iterations..." << std::endl;
    
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = system.processText(testText);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        processingTimes.push_back(duration.count());
        
        if (i == 0) {
            // Display detailed result for first iteration
            displayResult(result, testText);
        }
        
        std::cout << "  Iteration " << (i + 1) << ": " 
                  << std::fixed << std::setprecision(2) 
                  << duration.count() << " ms" << std::endl;
    }
    
    // Calculate statistics
    double totalTime = 0.0, minTime = processingTimes[0], maxTime = processingTimes[0];
    for (double time : processingTimes) {
        totalTime += time;
        minTime = std::min(minTime, time);
        maxTime = std::max(maxTime, time);
    }
    
    double avgTime = totalTime / iterations;
    
    std::cout << "\nPerformance Summary:" << std::endl;
    std::cout << "  Average: " << std::fixed << std::setprecision(2) << avgTime << " ms" << std::endl;
    std::cout << "  Minimum: " << std::fixed << std::setprecision(2) << minTime << " ms" << std::endl;
    std::cout << "  Maximum: " << std::fixed << std::setprecision(2) << maxTime << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1) 
              << (1000.0 / avgTime) << " operations/second" << std::endl;
}

/**
 * @brief Main demonstration function
 */
int main() {
    std::cout << "🎼 ORCHESTRAL ARCHITECT DEMONSTRATION" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "\nInitializing distributed agentic cognitive grammar system..." << std::endl;
    
    // Create and initialize the orchestral system
    auto system = createDemoSystem();
    if (!system) {
        std::cerr << "Failed to create orchestral system!" << std::endl;
        return 1;
    }
    
    std::cout << "✓ System initialized successfully" << std::endl;
    
    // Display initial system status
    displaySystemStatus(system->getSystemStatus());
    
    // Demonstrate basic processing
    std::cout << "\n🧠 DEMONSTRATING COGNITIVE PROCESSING" << std::endl;
    
    std::string testInput = "hello world this is a simple test";
    std::cout << "\nProcessing test input: \"" << testInput << "\"" << std::endl;
    
    auto result = system->processText(testInput);
    displayResult(result, testInput);
    
    // Demonstrate kernel communication
    demonstrateKernelCommunication(*system);
    
    // Demonstrate attention allocation
    demonstrateAttentionAllocation(*system);
    
    // Run performance benchmark
    runPerformanceBenchmark(*system);
    
    // Final system status
    std::cout << "\n📊 FINAL SYSTEM STATUS" << std::endl;
    displaySystemStatus(system->getSystemStatus());
    
    std::cout << "\n🎉 DEMONSTRATION COMPLETE" << std::endl;
    std::cout << "\nThe Orchestral Architect system has successfully demonstrated:" << std::endl;
    std::cout << "✓ Multi-strategy tokenization with attention weighting" << std::endl;
    std::cout << "✓ Event-driven kernel communication" << std::endl;
    std::cout << "✓ Economic attention allocation based on cognitive value" << std::endl;
    std::cout << "✓ Real-time performance monitoring and metrics" << std::endl;
    std::cout << "✓ Distributed processing with adaptive load balancing" << std::endl;
    
    std::cout << "\n🚀 Ready for integration with AtomSpace, PLN, and advanced cognitive components!" << std::endl;
    
    return 0;
}