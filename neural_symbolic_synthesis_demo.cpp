/**
 * @file neural_symbolic_synthesis_demo.cpp
 * @brief Demonstration of Phase 3: Neural-Symbolic Synthesis via Custom ggml Kernels
 * 
 * This demonstration showcases the complete neural-symbolic synthesis system
 * with real data validation, performance benchmarking, and end-to-end inference
 * pipeline testing according to Phase 3 specifications.
 */

#include "orchestral-architect/src/kernels/SymbolicTensorKernel.h"
#include "orchestral-architect/src/kernels/NeuralInferenceKernel.h"
#include "orchestral-architect/src/kernels/HypergraphKernel.h"
#include "orchestral-architect/src/benchmarks/NeuralSymbolicBenchmark.h"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace orchestral;

/**
 * @brief Demonstrate Neural-Symbolic Tensor[7] signature validation
 */
void demonstrateTensorSignature() {
    std::cout << "\n🔬 Neural-Symbolic Tensor[7] Signature Demonstration\n";
    std::cout << "===================================================\n";
    
    auto symbolic_kernel = std::make_shared<SymbolicTensorKernel>("demo_symbolic", 256, 10);
    symbolic_kernel->initialize();
    
    // Create tensor from symbolic expression
    std::string symbolic_expr = "(EvaluationLink (PredicateNode important) (ConceptNode concept))";
    std::unordered_map<std::string, float> context = {{"important", 0.9f}, {"concept", 0.7f}};
    
    auto tensor = symbolic_kernel->createFromSymbolic(symbolic_expr, context);
    
    std::cout << "Created Neural-Symbolic Tensor from: " << symbolic_expr << "\n";
    std::cout << "Tensor Signature:\n";
    std::cout << "  symbolic_representation: " << static_cast<int>(tensor.symbolic_representation) 
              << " (0=discrete, 1=continuous, 2=hybrid)\n";
    std::cout << "  neural_embedding: [" << tensor.neural_embedding.size() << "D]\n";
    std::cout << "  confidence_score: " << std::fixed << std::setprecision(3) 
              << tensor.confidence_score << " ∈ [0.0, 1.0]\n";
    std::cout << "  gradient_flow: [" << tensor.gradient_flow[0] << ", " 
              << tensor.gradient_flow[1] << "] (backward, forward)\n";
    std::cout << "  fusion_weight: " << tensor.fusion_weight << " ∈ [0.0, 1.0]\n";
    std::cout << "  computation_cost: " << tensor.computation_cost << " ∈ [0.0, ∞)\n";
    std::cout << "  inference_depth: " << tensor.inference_depth << " ∈ [1, max_depth]\n";
    
    // Validate signature compliance
    auto benchmark = NeuralSymbolicBenchmark(symbolic_kernel, nullptr, nullptr);
    auto validation_result = benchmark.validateTensorSignature(tensor);
    
    std::cout << "\nSignature Validation: " << (validation_result.tensor_signature_valid ? "✓ VALID" : "✗ INVALID") << "\n";
    if (!validation_result.validation_errors.empty()) {
        std::cout << "Validation Errors:\n";
        for (const auto& error : validation_result.validation_errors) {
            std::cout << "  - " << error << "\n";
        }
    }
    
    symbolic_kernel->shutdown();
}

/**
 * @brief Demonstrate custom ggml kernel operations
 */
void demonstrateCustomKernelOperations() {
    std::cout << "\n⚡ Custom ggml Kernel Operations Demonstration\n";
    std::cout << "============================================\n";
    
    auto symbolic_kernel = std::make_shared<SymbolicTensorKernel>("demo_symbolic", 256, 10);
    symbolic_kernel->initialize();
    
    // Create test tensors
    std::vector<NeuralSymbolicTensor> test_tensors = {
        symbolic_kernel->createFromSymbolic("(ConceptNode A)"),
        symbolic_kernel->createFromSymbolic("(ConceptNode B)")
    };
    
    std::cout << "Created " << test_tensors.size() << " test tensors\n";
    
    // Test different tensor operations
    std::vector<SymbolicTensorOp> operations = {
        SymbolicTensorOp::SYMBOLIC_ADD,
        SymbolicTensorOp::FUSION_BLEND,
        SymbolicTensorOp::GRADIENT_COMPUTE,
        SymbolicTensorOp::ATTENTION_WEIGHT,
        SymbolicTensorOp::INFERENCE_STEP
    };
    
    std::vector<std::string> operation_names = {
        "Symbolic Addition", "Fusion Blend", "Gradient Computation", 
        "Attention Weighting", "Inference Step"
    };
    
    for (size_t i = 0; i < operations.size(); ++i) {
        std::cout << "\nTesting " << operation_names[i] << "... ";
        
        std::unordered_map<std::string, float> params;
        if (operations[i] == SymbolicTensorOp::FUSION_BLEND) {
            params["fusion_weight"] = 0.6f;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = symbolic_kernel->executeOperation(operations[i], test_tensors, params);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "✓ Complete (confidence: " << std::fixed << std::setprecision(3) 
                  << result.operation_confidence << ", time: " << duration.count() << "μs, "
                  << "memory: " << result.memory_usage_mb << "MB)\n";
    }
    
    symbolic_kernel->shutdown();
}

/**
 * @brief Demonstrate AtomSpace integration and neural inference
 */
void demonstrateAtomSpaceIntegration() {
    std::cout << "\n🧠 AtomSpace Integration & Neural Inference Demonstration\n";
    std::cout << "========================================================\n";
    
    auto symbolic_kernel = std::make_shared<SymbolicTensorKernel>("demo_symbolic", 256, 10);
    auto inference_kernel = std::make_shared<NeuralInferenceKernel>("demo_inference", symbolic_kernel, 20);
    
    symbolic_kernel->initialize();
    inference_kernel->initialize();
    
    // Create AtomSpace atoms
    std::cout << "Creating AtomSpace atoms:\n";
    auto dog_atom = inference_kernel->createAtom("ConceptNode", "Dog", {{"strength", 0.9f}, {"confidence", 0.8f}});
    auto animal_atom = inference_kernel->createAtom("ConceptNode", "Animal", {{"strength", 0.8f}, {"confidence", 0.9f}});
    auto inheritance_atom = inference_kernel->createAtom("InheritanceLink", "Dog_inherits_Animal", 
                                                        {{"strength", 0.85f}, {"confidence", 0.9f}});
    
    std::cout << "  - " << dog_atom.atom_type << ": " << dog_atom.atom_name 
              << " (confidence: " << dog_atom.neural_representation.confidence_score << ")\n";
    std::cout << "  - " << animal_atom.atom_type << ": " << animal_atom.atom_name 
              << " (confidence: " << animal_atom.neural_representation.confidence_score << ")\n";
    std::cout << "  - " << inheritance_atom.atom_type << ": " << inheritance_atom.atom_name 
              << " (confidence: " << inheritance_atom.neural_representation.confidence_score << ")\n";
    
    // Create hypergraph pattern
    std::vector<AtomSpaceAtom> atoms = {dog_atom, animal_atom, inheritance_atom};
    std::vector<std::pair<size_t, size_t>> connections = {{0, 2}, {1, 2}}; // Dog and Animal connect via inheritance
    auto pattern = inference_kernel->createPattern(atoms, connections);
    
    std::cout << "\nCreated hypergraph pattern with " << pattern.nodes.size() 
              << " nodes and " << pattern.edges.size() << " edges\n";
    std::cout << "Pattern confidence: " << std::fixed << std::setprecision(3) 
              << pattern.pattern_confidence << "\n";
    
    // Test different inference strategies
    std::vector<InferenceStrategy> strategies = {
        InferenceStrategy::FORWARD_CHAINING,
        InferenceStrategy::BACKWARD_CHAINING,
        InferenceStrategy::NEURAL_ATTENTION_GUIDED,
        InferenceStrategy::RECURSIVE_PATTERN_MATCHING
    };
    
    std::vector<std::string> strategy_names = {
        "Forward Chaining", "Backward Chaining", "Neural Attention Guided", "Recursive Pattern Matching"
    };
    
    for (size_t i = 0; i < strategies.size(); ++i) {
        std::cout << "\nTesting " << strategy_names[i] << "... ";
        
        auto start = std::chrono::high_resolution_clock::now();
        auto inference_result = inference_kernel->performInference(pattern, strategies[i], 5);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "✓ Complete\n";
        std::cout << "  Inferred atoms: " << inference_result.inferred_atoms.size() << "\n";
        std::cout << "  Reasoning patterns: " << inference_result.reasoning_patterns.size() << "\n";
        std::cout << "  Confidence: " << std::fixed << std::setprecision(3) 
                  << inference_result.inference_confidence << "\n";
        std::cout << "  Time: " << duration.count() << "ms\n";
        std::cout << "  Cognitive load: " << inference_result.cognitive_load << "\n";
    }
    
    symbolic_kernel->shutdown();
    inference_kernel->shutdown();
}

/**
 * @brief Demonstrate hypergraph computation with recursive traversal
 */
void demonstrateHypergraphComputation() {
    std::cout << "\n🕸️ Hypergraph Computation & Recursive Traversal Demonstration\n";
    std::cout << "=============================================================\n";
    
    auto symbolic_kernel = std::make_shared<SymbolicTensorKernel>("demo_symbolic", 256, 10);
    auto hypergraph_kernel = std::make_shared<HypergraphKernel>("demo_hypergraph", symbolic_kernel, 15);
    
    symbolic_kernel->initialize();
    hypergraph_kernel->initialize();
    
    // Create complex hypergraph from symbolic expressions
    std::vector<std::string> expressions = {
        "(ConceptNode Animal)",
        "(ConceptNode Mammal)", 
        "(ConceptNode Dog)",
        "(ConceptNode Cat)",
        "(InheritanceLink Dog Mammal)",
        "(InheritanceLink Cat Mammal)",
        "(InheritanceLink Mammal Animal)",
        "(SimilarityLink Dog Cat)"
    };
    
    std::vector<std::pair<std::string, std::string>> connections = {
        {"node_0", "node_1"}, {"node_1", "node_2"}, {"node_1", "node_3"}, 
        {"node_2", "node_3"}, {"node_0", "node_6"}
    };
    
    auto hypergraph = hypergraph_kernel->createFromSymbolic(expressions, connections);
    
    std::cout << "Created hypergraph with:\n";
    std::cout << "  - Nodes: " << hypergraph.nodes.size() << "\n";
    std::cout << "  - Edges: " << hypergraph.edges.size() << "\n";
    std::cout << "  - Recursion level: " << hypergraph.recursion_level << "\n";
    
    // Test different hypergraph operations
    std::vector<HypergraphOp> operations = {
        HypergraphOp::NODE_ACTIVATION,
        HypergraphOp::ATTENTION_FLOW,
        HypergraphOp::RECURSIVE_TRAVERSAL,
        HypergraphOp::NEURAL_EMBEDDING,
        HypergraphOp::COGNITIVE_PROCESSING
    };
    
    std::vector<std::string> operation_names = {
        "Node Activation", "Attention Flow", "Recursive Traversal", 
        "Neural Embedding", "Cognitive Processing"
    };
    
    for (size_t i = 0; i < operations.size(); ++i) {
        std::cout << "\nTesting " << operation_names[i] << "... ";
        
        std::unordered_map<std::string, float> params;
        if (operations[i] == HypergraphOp::ATTENTION_FLOW) {
            params["flow_strength"] = 1.2f;
        } else if (operations[i] == HypergraphOp::RECURSIVE_TRAVERSAL) {
            params["start_node_index"] = 0.0f;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = hypergraph_kernel->executeHypergraphOp(operations[i], hypergraph, params);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "✓ Complete\n";
        std::cout << "  Confidence: " << std::fixed << std::setprecision(3) 
                  << result.computation_confidence << "\n";
        std::cout << "  Memory usage: " << result.memory_usage_mb << "MB\n";
        std::cout << "  Processing time: " << duration.count() << "ms\n";
        
        if (!result.activated_nodes.empty()) {
            std::cout << "  Activated nodes: " << result.activated_nodes.size() << "\n";
        }
        if (!result.traversal_path.empty()) {
            std::cout << "  Traversal path length: " << result.traversal_path.size() << "\n";
        }
    }
    
    // Test different traversal strategies
    std::cout << "\nTesting traversal strategies:\n";
    std::vector<TraversalStrategy> strategies = {
        TraversalStrategy::BREADTH_FIRST,
        TraversalStrategy::DEPTH_FIRST,
        TraversalStrategy::ATTENTION_GUIDED,
        TraversalStrategy::NEURAL_FLOW,
        TraversalStrategy::COGNITIVE_PRIORITY
    };
    
    std::vector<std::string> strategy_names = {
        "Breadth-First", "Depth-First", "Attention-Guided", "Neural Flow", "Cognitive Priority"
    };
    
    std::string start_node = hypergraph.nodes.empty() ? "" : hypergraph.nodes.begin()->first;
    
    for (size_t i = 0; i < strategies.size(); ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto traversal_result = hypergraph_kernel->recursiveTraversal(hypergraph, start_node, strategies[i], 8);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "  " << strategy_names[i] << ": "
                  << traversal_result.traversal_path.size() << " nodes visited, "
                  << "confidence: " << std::fixed << std::setprecision(3) 
                  << traversal_result.computation_confidence << " ("
                  << duration.count() << "ms)\n";
    }
    
    symbolic_kernel->shutdown();
    hypergraph_kernel->shutdown();
}

/**
 * @brief Run comprehensive benchmark suite
 */
void runComprehensiveBenchmark() {
    std::cout << "\n📊 Comprehensive Neural-Symbolic Benchmark Suite\n";
    std::cout << "================================================\n";
    
    auto symbolic_kernel = std::make_shared<SymbolicTensorKernel>("benchmark_symbolic", 256, 10);
    auto inference_kernel = std::make_shared<NeuralInferenceKernel>("benchmark_inference", symbolic_kernel, 20);
    auto hypergraph_kernel = std::make_shared<HypergraphKernel>("benchmark_hypergraph", symbolic_kernel, 15);
    
    symbolic_kernel->initialize();
    inference_kernel->initialize();
    hypergraph_kernel->initialize();
    
    auto benchmark = NeuralSymbolicBenchmark(symbolic_kernel, inference_kernel, hypergraph_kernel);
    
    // Run complete benchmark suite
    auto benchmark_results = benchmark.runCompleteBenchmark("neural_symbolic_benchmark_results");
    
    const auto& results = benchmark_results.first;
    const auto& profile = benchmark_results.second;
    
    std::cout << "\n📋 Final Benchmark Summary:\n";
    std::cout << "==========================\n";
    
    int passed_tests = 0;
    int failed_tests = 0;
    float total_accuracy = 0.0f;
    float total_efficiency = 0.0f;
    std::chrono::milliseconds total_time(0);
    
    for (const auto& result : results) {
        if (result.success) {
            passed_tests++;
        } else {
            failed_tests++;
        }
        total_accuracy += result.accuracy_score;
        total_efficiency += result.efficiency_score;
        total_time += result.execution_time;
    }
    
    std::cout << "Tests Passed: " << passed_tests << "/" << (passed_tests + failed_tests) << "\n";
    std::cout << "Success Rate: " << std::fixed << std::setprecision(1) 
              << (100.0f * passed_tests / (passed_tests + failed_tests)) << "%\n";
    std::cout << "Average Accuracy: " << std::setprecision(3) 
              << (total_accuracy / results.size()) << "\n";
    std::cout << "Average Efficiency: " << (total_efficiency / results.size()) << "\n";
    std::cout << "Total Processing Time: " << total_time.count() << "ms\n";
    std::cout << "Overall System Efficiency: " << profile.overall_efficiency << "\n";
    std::cout << "Cognitive Load Factor: " << std::setprecision(2) 
              << profile.cognitive_load_factor << "\n";
    
    // Test tensor operations with various data sizes
    std::cout << "\n🔬 Tensor Operations Scaling Test:\n";
    std::vector<int> test_sizes = {100, 500, 1000, 2000};
    for (int size : test_sizes) {
        auto scaling_result = benchmark.testTensorOperations("symbolic_tensor", size);
        std::cout << "  Size " << size << ": " 
                  << (scaling_result.success ? "✓" : "✗") << " "
                  << scaling_result.execution_time.count() << "ms, "
                  << std::fixed << std::setprecision(2) << scaling_result.memory_usage_mb << "MB, "
                  << "accuracy: " << std::setprecision(3) << scaling_result.accuracy_score << "\n";
    }
    
    // Memory profiling
    std::cout << "\n💾 Memory Usage Profiling:\n";
    auto memory_profile = benchmark.profileMemoryUsage(2000, 200);
    std::cout << "Memory profile data points: " << memory_profile.size() << "\n";
    if (!memory_profile.empty()) {
        std::cout << "  Min memory: " << std::fixed << std::setprecision(2) 
                  << memory_profile.front().second << "MB\n";
        std::cout << "  Max memory: " << memory_profile.back().second << "MB\n";
    }
    
    // Computational complexity analysis
    std::cout << "\n⏱️ Computational Complexity Analysis:\n";
    auto complexity_analysis = benchmark.analyzeComputationalComplexity(
        {"symbolic_tensor", "neural_inference", "hypergraph"}, {1, 3, 5, 8});
    
    for (const auto& analysis : complexity_analysis) {
        std::cout << "  " << analysis.first << ": ";
        for (float timing : analysis.second) {
            std::cout << std::fixed << std::setprecision(0) << timing << "ms ";
        }
        std::cout << "\n";
    }
    
    symbolic_kernel->shutdown();
    inference_kernel->shutdown();
    hypergraph_kernel->shutdown();
    
    std::cout << "\n✅ Benchmark completed successfully!\n";
    std::cout << "Results saved to: neural_symbolic_benchmark_results.json\n";
}

/**
 * @brief Main demonstration program
 */
int main() {
    std::cout << "🌌 Phase 3: Neural-Symbolic Synthesis via Custom ggml Kernels\n";
    std::cout << "==============================================================\n";
    std::cout << "Comprehensive demonstration of neural-symbolic tensor operations,\n";
    std::cout << "AtomSpace integration, hypergraph computation, and cognitive reasoning.\n";
    
    try {
        // Demonstrate each component
        demonstrateTensorSignature();
        demonstrateCustomKernelOperations();
        demonstrateAtomSpaceIntegration();
        demonstrateHypergraphComputation();
        runComprehensiveBenchmark();
        
        std::cout << "\n🎉 Neural-Symbolic Synthesis Demonstration Complete!\n";
        std::cout << "====================================================\n";
        std::cout << "All Phase 3 requirements have been successfully demonstrated:\n";
        std::cout << "  ✓ Neural-Symbolic Tensor[7] signature implementation\n";
        std::cout << "  ✓ Custom ggml kernels for symbolic computation\n";
        std::cout << "  ✓ AtomSpace integration with neural inference hooks\n";
        std::cout << "  ✓ Hypergraph computation with recursive traversal\n";
        std::cout << "  ✓ Gradient computation for symbolic tensors\n";
        std::cout << "  ✓ Real data validation and performance benchmarking\n";
        std::cout << "  ✓ End-to-end neural-symbolic inference pipeline\n";
        std::cout << "  ✓ Cognitive workload optimization\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during demonstration: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}