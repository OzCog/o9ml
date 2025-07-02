/*
 * attention_benchmark_test.cc
 *
 * Copyright (C) 2024 OpenCog Foundation
 * All Rights Reserved
 *
 * Test application for attention allocation benchmarking
 */

#include <iostream>
#include <memory>
#include <opencog/atomspace/AtomSpace.h>
#include "../benchmarks/attention_benchmark.h"

using namespace opencog;

int main(int argc, char* argv[])
{
    std::cout << "OpenCog Attention Allocation Benchmark Test" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Parse command line arguments
    size_t num_atoms = 1000;
    size_t num_iterations = 10;
    
    if (argc > 1) {
        num_atoms = std::stoull(argv[1]);
    }
    if (argc > 2) {
        num_iterations = std::stoull(argv[2]);
    }
    
    std::cout << "Testing with " << num_atoms << " atoms and " 
              << num_iterations << " iterations." << std::endl;
    
    try {
        // Create AtomSpace
        AtomSpacePtr atomspace = std::make_shared<AtomSpace>();
        
        // Create benchmark instance
        AttentionBenchmark benchmark(atomspace, num_atoms, num_iterations);
        
        // Display tensor degrees of freedom
        auto tensor_dof = benchmark.get_tensor_degrees_of_freedom();
        std::cout << "\nAttention Tensor Degrees of Freedom:" << std::endl;
        std::cout << "- Spatial DOF: " << tensor_dof.SPATIAL_DOF << std::endl;
        std::cout << "- Temporal DOF: " << tensor_dof.TEMPORAL_DOF << std::endl;
        std::cout << "- Semantic DOF: " << tensor_dof.SEMANTIC_DOF << std::endl;
        std::cout << "- Importance DOF: " << tensor_dof.IMPORTANCE_DOF << std::endl;
        std::cout << "- Hebbian DOF: " << tensor_dof.HEBBIAN_DOF << std::endl;
        std::cout << "- Total DOF: " << tensor_dof.TOTAL_DOF << std::endl;
        
        // Run individual performance measurements
        std::cout << "\nRunning individual performance measurements..." << std::endl;
        
        double spreading_rate = benchmark.measure_activation_spreading_performance();
        std::cout << "Activation Spreading Rate: " << spreading_rate << " atoms/sec" << std::endl;
        
        double diffusion_rate = benchmark.measure_importance_diffusion_rate();
        std::cout << "Importance Diffusion Rate: " << diffusion_rate << " ops/sec" << std::endl;
        
        double tensor_rate = benchmark.measure_tensor_computation_throughput();
        std::cout << "Tensor Computation Rate: " << tensor_rate << " ops/sec" << std::endl;
        
        // Run comprehensive benchmark
        std::cout << "\nRunning comprehensive benchmark..." << std::endl;
        AttentionMetrics results = benchmark.run_comprehensive_benchmark();
        
        // Print detailed results
        benchmark.print_benchmark_results(results);
        
        // Get performance summary
        auto summary = benchmark.get_performance_summary();
        std::cout << "\nPerformance Summary:" << std::endl;
        for (const auto& pair : summary) {
            std::cout << "- " << pair.first << ": " << pair.second << std::endl;
        }
        
        // Performance validation
        std::cout << "\nPerformance Validation:" << std::endl;
        bool pass_spreading = summary["activation_spreading_rate"] > 1000.0; // 1K atoms/sec minimum
        bool pass_diffusion = summary["importance_diffusion_rate"] > 100.0;  // 100 ops/sec minimum
        bool pass_tensor = summary["tensor_computation_rate"] > 10000.0;     // 10K ops/sec minimum
        
        std::cout << "- Activation Spreading: " << (pass_spreading ? "PASS" : "FAIL") << std::endl;
        std::cout << "- Importance Diffusion: " << (pass_diffusion ? "PASS" : "FAIL") << std::endl;
        std::cout << "- Tensor Computation: " << (pass_tensor ? "PASS" : "FAIL") << std::endl;
        
        bool all_passed = pass_spreading && pass_diffusion && pass_tensor;
        std::cout << "\nOverall Result: " << (all_passed ? "PASS" : "FAIL") << std::endl;
        
        return all_passed ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}