/**
 * @file test_orchestral_system.cpp
 * @brief Unit tests for the Orchestral Architect system
 */

#include "../src/OrchestralSystem.h"
#include "../src/kernels/TokenizationKernel.h"
#include <cassert>
#include <iostream>
#include <chrono>

using namespace orchestral;

// Simple test framework
class TestRunner {
public:
    void run_test(const std::string& name, std::function<void()> test) {
        std::cout << "Running test: " << name << "... ";
        try {
            test();
            std::cout << "✓ PASSED" << std::endl;
            passed_++;
        } catch (const std::exception& e) {
            std::cout << "✗ FAILED: " << e.what() << std::endl;
            failed_++;
        } catch (...) {
            std::cout << "✗ FAILED: Unknown exception" << std::endl;
            failed_++;
        }
        total_++;
    }
    
    void summary() {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Test Summary: " << passed_ << "/" << total_ << " passed";
        if (failed_ > 0) {
            std::cout << ", " << failed_ << " failed";
        }
        std::cout << std::endl;
        std::cout << std::string(50, '=') << std::endl;
    }
    
    bool all_passed() const { return failed_ == 0; }
    
private:
    size_t total_ = 0;
    size_t passed_ = 0;
    size_t failed_ = 0;
};

// Test helper macros
#define ASSERT_TRUE(condition) \
    if (!(condition)) { \
        throw std::runtime_error("Assertion failed: " #condition); \
    }

#define ASSERT_FALSE(condition) \
    if (condition) { \
        throw std::runtime_error("Assertion failed: " #condition " should be false"); \
    }

#define ASSERT_EQ(expected, actual) \
    if ((expected) != (actual)) { \
        throw std::runtime_error("Assertion failed: expected " + std::to_string(expected) + \
                               ", got " + std::to_string(actual)); \
    }

#define ASSERT_NOT_EMPTY(container) \
    if ((container).empty()) { \
        throw std::runtime_error("Assertion failed: container should not be empty"); \
    }

// Test functions
void test_agentic_kernel_creation() {
    auto kernel = std::make_shared<TokenizationKernel>("test_tokenizer");
    
    ASSERT_TRUE(kernel != nullptr);
    ASSERT_TRUE(kernel->getName() == "test_tokenizer");
    ASSERT_TRUE(kernel->getType() == "tokenization");
    ASSERT_FALSE(kernel->isActive());
}

void test_kernel_initialization() {
    auto kernel = std::make_shared<TokenizationKernel>("test_tokenizer");
    
    ASSERT_TRUE(kernel->initialize());
    ASSERT_TRUE(kernel->isActive());
    
    auto capabilities = kernel->getCapabilities();
    ASSERT_NOT_EMPTY(capabilities);
    
    ASSERT_TRUE(kernel->canProcess("text"));
    ASSERT_FALSE(kernel->canProcess("image"));
    
    kernel->shutdown();
    ASSERT_FALSE(kernel->isActive());
}

void test_tokenization_processing() {
    auto kernel = std::make_shared<TokenizationKernel>("test_tokenizer");
    ASSERT_TRUE(kernel->initialize());
    
    CognitiveInput input("hello world test", "text");
    input.urgency = 0.7;
    
    auto result = kernel->process(input);
    
    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.processingTime.count() >= 0);
    ASSERT_TRUE(result.processingCost > 0.0);
    ASSERT_TRUE(result.estimatedValue > 0.0);
    ASSERT_NOT_EMPTY(result.attentionWeights);
    
    kernel->shutdown();
}

void test_attention_weighting() {
    auto kernel = std::make_shared<TokenizationKernel>("test_tokenizer");
    ASSERT_TRUE(kernel->initialize());
    
    // Set attention vocabulary
    std::unordered_map<std::string, double> vocab = {
        {"important", 0.9},
        {"test", 0.6}
    };
    kernel->setAttentionVocabulary(vocab);
    
    CognitiveInput input("this is an important test", "text");
    auto result = kernel->process(input);
    
    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.attentionWeights.find("important") != result.attentionWeights.end());
    ASSERT_TRUE(result.attentionWeights.find("test") != result.attentionWeights.end());
    
    // Important should have higher attention than test
    ASSERT_TRUE(result.attentionWeights["important"] > result.attentionWeights["test"]);
    
    kernel->shutdown();
}

void test_kernel_registry() {
    auto& registry = KernelRegistry::getInstance();
    ASSERT_TRUE(registry.initialize());
    
    auto kernel = std::make_shared<TokenizationKernel>("registry_test");
    ASSERT_TRUE(registry.registerKernel(kernel));
    
    auto kernels = registry.getAllKernelNames();
    ASSERT_TRUE(std::find(kernels.begin(), kernels.end(), "registry_test") != kernels.end());
    
    auto capabilities = registry.findKernelsByCapability("text_tokenization");
    ASSERT_NOT_EMPTY(capabilities);
    
    ASSERT_TRUE(registry.unregisterKernel("registry_test"));
    
    registry.shutdown();
}

void test_orchestral_system_creation() {
    OrchestralConfig config;
    config.enableNetworking = false;
    config.enableHealthMonitoring = false;  // Disable for faster testing
    
    auto system = createOrchestralSystem(config);
    ASSERT_TRUE(system != nullptr);
    ASSERT_TRUE(system->isRunning());
    
    auto kernels = system->getRegisteredKernels();
    ASSERT_NOT_EMPTY(kernels);
    
    system->shutdown();
    ASSERT_FALSE(system->isRunning());
}

void test_orchestral_text_processing() {
    auto system = createDemoSystem();
    ASSERT_TRUE(system != nullptr);
    
    std::string testText = "hello world this is a simple test";
    auto result = system->processText(testText, 0.5);
    
    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.processingTime.count() >= 0);
    ASSERT_TRUE(result.processingCost > 0.0);
    ASSERT_TRUE(result.estimatedValue > 0.0);
    
    system->shutdown();
}

void test_system_status_monitoring() {
    auto system = createDemoSystem();
    ASSERT_TRUE(system != nullptr);
    
    auto status = system->getSystemStatus();
    ASSERT_TRUE(status.activeKernels > 0);
    ASSERT_TRUE(status.startTime <= status.lastUpdate);
    
    // Process some data to update metrics
    system->processText("test input", 0.5);
    
    auto newStatus = system->getSystemStatus();
    ASSERT_TRUE(newStatus.totalProcessedItems >= status.totalProcessedItems);
    
    system->shutdown();
}

void test_kernel_communication() {
    auto system = createDemoSystem();
    ASSERT_TRUE(system != nullptr);
    
    // Create test event
    KernelEvent event;
    event.eventType = "test_event";
    event.sourceKernel = "test_source";
    event.payload["test_data"] = "test_value";
    
    // Broadcast to tokenization kernels
    size_t recipients = system->broadcastEvent(event, "tokenization");
    ASSERT_TRUE(recipients > 0);
    
    system->shutdown();
}

void test_multi_strategy_tokenization() {
    auto kernel = std::make_shared<TokenizationKernel>("multi_strategy_test");
    ASSERT_TRUE(kernel->initialize());
    
    auto strategies = kernel->getAvailableStrategies();
    ASSERT_TRUE(strategies.size() >= 3);  // word, subword, linguistic
    
    std::string testText = "tokenization test example";
    
    // Test each strategy
    for (const std::string& strategy : strategies) {
        ASSERT_TRUE(kernel->setActiveStrategy(strategy));
        auto result = kernel->processText(testText, strategy);
        ASSERT_TRUE(result.success);
    }
    
    kernel->shutdown();
}

void test_performance_metrics() {
    auto system = createDemoSystem();
    ASSERT_TRUE(system != nullptr);
    
    // Process multiple inputs to generate metrics
    std::vector<std::string> inputs = {
        "first test input",
        "second test input",
        "third test input with more content"
    };
    
    for (const std::string& input : inputs) {
        auto result = system->processText(input);
        ASSERT_TRUE(result.success);
    }
    
    auto status = system->getSystemStatus();
    ASSERT_TRUE(status.totalProcessedItems >= inputs.size());
    
    system->shutdown();
}

// Main test runner
int main() {
    std::cout << "🧪 ORCHESTRAL ARCHITECT UNIT TESTS" << std::endl;
    std::cout << "===================================" << std::endl;
    
    TestRunner runner;
    
    // Core component tests
    runner.run_test("AgenticKernel Creation", test_agentic_kernel_creation);
    runner.run_test("Kernel Initialization", test_kernel_initialization);
    runner.run_test("Tokenization Processing", test_tokenization_processing);
    runner.run_test("Attention Weighting", test_attention_weighting);
    
    // Registry tests
    runner.run_test("Kernel Registry", test_kernel_registry);
    
    // System tests
    runner.run_test("Orchestral System Creation", test_orchestral_system_creation);
    runner.run_test("Text Processing", test_orchestral_text_processing);
    runner.run_test("System Status Monitoring", test_system_status_monitoring);
    runner.run_test("Kernel Communication", test_kernel_communication);
    
    // Advanced feature tests
    runner.run_test("Multi-Strategy Tokenization", test_multi_strategy_tokenization);
    runner.run_test("Performance Metrics", test_performance_metrics);
    
    runner.summary();
    
    return runner.all_passed() ? 0 : 1;
}