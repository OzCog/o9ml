/*
 * attention_benchmark.h
 *
 * Copyright (C) 2024 OpenCog Foundation
 * All Rights Reserved
 *
 * Benchmarking framework for ECAN attention allocation mechanisms
 * Measures performance of activation spreading, importance diffusion,
 * and attention tensor operations.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 */

#ifndef _OPENCOG_ATTENTION_BENCHMARK_H
#define _OPENCOG_ATTENTION_BENCHMARK_H

#include <chrono>
#include <vector>
#include <memory>
#include <map>
#include <string>

#include <opencog/atomspace/AtomSpace.h>
#include <opencog/attentionbank/bank/AttentionBank.h>

namespace opencog
{

/**
 * Performance metrics structure for attention allocation operations
 */
struct AttentionMetrics
{
    // Timing metrics (in microseconds)
    double activation_spreading_time;
    double importance_diffusion_time;
    double attention_allocation_time;
    double tensor_operation_time;
    
    // Throughput metrics
    size_t atoms_processed;
    size_t diffusion_operations;
    size_t tensor_computations;
    
    // Memory metrics
    size_t memory_usage_bytes;
    size_t attention_focus_size;
    
    // Quality metrics
    double convergence_rate;
    double stability_measure;
    
    AttentionMetrics() : 
        activation_spreading_time(0.0),
        importance_diffusion_time(0.0),
        attention_allocation_time(0.0),
        tensor_operation_time(0.0),
        atoms_processed(0),
        diffusion_operations(0),
        tensor_computations(0),
        memory_usage_bytes(0),
        attention_focus_size(0),
        convergence_rate(0.0),
        stability_measure(0.0) {}
};

/**
 * Attention tensor degrees of freedom specification
 */
struct AttentionTensorDOF
{
    // Spatial dimensions (3D coordinate transformations)
    static constexpr size_t SPATIAL_DOF = 3;
    
    // Temporal dimensions (time-series attention patterns)
    static constexpr size_t TEMPORAL_DOF = 1;
    
    // Semantic dimensions (attention embedding space)
    static constexpr size_t SEMANTIC_DOF = 256;
    
    // Importance dimensions (STI, LTI, VLTI)
    static constexpr size_t IMPORTANCE_DOF = 3;
    
    // Hebbian dimensions (synaptic strength patterns)
    static constexpr size_t HEBBIAN_DOF = 64;
    
    // Total attention tensor dimensions
    static constexpr size_t TOTAL_DOF = SPATIAL_DOF + TEMPORAL_DOF + 
                                       SEMANTIC_DOF + IMPORTANCE_DOF + 
                                       HEBBIAN_DOF;
    
    // Tensor shape for different operations
    std::vector<size_t> spatial_shape = {3};
    std::vector<size_t> temporal_shape = {1};
    std::vector<size_t> semantic_shape = {256};
    std::vector<size_t> importance_shape = {3};
    std::vector<size_t> hebbian_shape = {64};
    std::vector<size_t> full_tensor_shape = {TOTAL_DOF};
};

/**
 * High-performance benchmarking framework for ECAN attention mechanisms
 */
class AttentionBenchmark
{
private:
    AtomSpacePtr _atomspace;
    AttentionBank* _attention_bank;
    
    // Benchmark configuration
    size_t _num_atoms;
    size_t _num_iterations;
    bool _enable_profiling;
    
    // Performance tracking
    std::vector<AttentionMetrics> _metrics_history;
    AttentionTensorDOF _tensor_dof;
    
    // Timing utilities
    std::chrono::high_resolution_clock::time_point _start_time;
    
    void start_timer();
    double end_timer(); // Returns elapsed time in microseconds
    
    // Test data generation
    void generate_test_atoms(size_t count);
    void setup_attention_patterns();
    
    // Core benchmarking methods
    AttentionMetrics benchmark_activation_spreading();
    AttentionMetrics benchmark_importance_diffusion();
    AttentionMetrics benchmark_attention_allocation();
    AttentionMetrics benchmark_tensor_operations();
    
public:
    AttentionBenchmark(AtomSpacePtr atomspace, 
                      size_t num_atoms = 10000,
                      size_t num_iterations = 100);
    ~AttentionBenchmark();
    
    // Main benchmark interface
    AttentionMetrics run_comprehensive_benchmark();
    std::vector<AttentionMetrics> run_performance_suite();
    
    // Specific benchmark methods
    double measure_activation_spreading_performance();
    double measure_importance_diffusion_rate();
    double measure_tensor_computation_throughput();
    
    // Configuration
    void set_num_atoms(size_t count) { _num_atoms = count; }
    void set_num_iterations(size_t iterations) { _num_iterations = iterations; }
    void enable_profiling(bool enable) { _enable_profiling = enable; }
    
    // Results and analysis
    void print_benchmark_results(const AttentionMetrics& metrics);
    void export_metrics_csv(const std::string& filename);
    AttentionTensorDOF get_tensor_degrees_of_freedom() const { return _tensor_dof; }
    
    // Statistical analysis
    double calculate_mean_performance();
    double calculate_performance_variance();
    std::map<std::string, double> get_performance_summary();
};

} // namespace opencog

#endif // _OPENCOG_ATTENTION_BENCHMARK_H