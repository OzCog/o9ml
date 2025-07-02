/*
 * attention_benchmark.cc
 *
 * Copyright (C) 2024 OpenCog Foundation
 * All Rights Reserved
 *
 * Implementation of benchmarking framework for ECAN attention allocation mechanisms
 */

#include "attention_benchmark.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstdlib>

#include <opencog/atoms/base/Node.h>
#include <opencog/atoms/base/Link.h>
#include <opencog/attentionbank/avalue/AttentionValue.h>

namespace opencog
{

AttentionBenchmark::AttentionBenchmark(AtomSpacePtr atomspace, 
                                     size_t num_atoms,
                                     size_t num_iterations)
    : _atomspace(atomspace), 
      _attention_bank(nullptr),
      _num_atoms(num_atoms),
      _num_iterations(num_iterations),
      _enable_profiling(true)
{
    // Get the attention bank from the atomspace
    _attention_bank = &(atomspace->get_attentional_focus_boundary());
    
    // Initialize tensor DOF structure
    _tensor_dof = AttentionTensorDOF();
    
    std::cout << "AttentionBenchmark initialized with " << _num_atoms 
              << " atoms and " << _num_iterations << " iterations." << std::endl;
}

AttentionBenchmark::~AttentionBenchmark()
{
    // Cleanup if needed
}

void AttentionBenchmark::start_timer()
{
    _start_time = std::chrono::high_resolution_clock::now();
}

double AttentionBenchmark::end_timer()
{
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - _start_time);
    return static_cast<double>(duration.count());
}

void AttentionBenchmark::generate_test_atoms(size_t count)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> sti_dist(0.0, 100.0);
    std::uniform_real_distribution<> lti_dist(0.0, 1.0);
    
    for (size_t i = 0; i < count; ++i) {
        // Create concept nodes with random names
        std::string atom_name = "TestAtom_" + std::to_string(i);
        Handle h = _atomspace->add_node(CONCEPT_NODE, atom_name);
        
        // Set random attention values
        AttentionValuePtr av = createAV(
            static_cast<AttentionValue::sti_t>(sti_dist(gen)),
            static_cast<AttentionValue::lti_t>(lti_dist(gen)),
            AttentionValue::NONNULL_VLTI()
        );
        
        _attention_bank->set_aw(h, av);
        
        // Create some random links between atoms for diffusion testing
        if (i > 0 && i % 10 == 0) {
            Handle prev = _atomspace->get_node(CONCEPT_NODE, 
                "TestAtom_" + std::to_string(i - 1));
            if (prev != Handle::UNDEFINED) {
                _atomspace->add_link(EVALUATION_LINK, h, prev);
            }
        }
    }
}

void AttentionBenchmark::setup_attention_patterns()
{
    // Create patterns for testing attention focus dynamics
    std::vector<Handle> high_attention_atoms;
    
    HandleSeq all_atoms;
    _atomspace->get_handles_by_type(all_atoms, CONCEPT_NODE);
    
    // Select 10% of atoms for high attention
    size_t high_attention_count = all_atoms.size() / 10;
    for (size_t i = 0; i < high_attention_count && i < all_atoms.size(); ++i) {
        Handle h = all_atoms[i];
        AttentionValuePtr high_av = createAV(95.0, 0.8, AttentionValue::NONNULL_VLTI());
        _attention_bank->set_aw(h, high_av);
        high_attention_atoms.push_back(h);
    }
}

AttentionMetrics AttentionBenchmark::benchmark_activation_spreading()
{
    AttentionMetrics metrics;
    
    start_timer();
    
    // Simulate activation spreading through the attention network
    HandleSeq focus_atoms;
    _attention_bank->get_handle_set_in_attentional_focus(focus_atoms);
    
    size_t operations = 0;
    for (const Handle& h : focus_atoms) {
        // Get incoming and outgoing atoms
        HandleSeq incoming = h->getIncomingSet();
        HandleSeq outgoing = h->getOutgoingSet();
        
        operations += incoming.size() + outgoing.size();
        
        // Simulate spreading activation
        for (const Handle& target : incoming) {
            AttentionValuePtr current_av = _attention_bank->get_aw(target);
            if (current_av) {
                // Simple spreading rule: increase STI by 1% of source STI
                AttentionValuePtr source_av = _attention_bank->get_aw(h);
                if (source_av) {
                    double spread_amount = source_av->getSTI() * 0.01;
                    AttentionValuePtr new_av = createAV(
                        current_av->getSTI() + spread_amount,
                        current_av->getLTI(),
                        current_av->getVLTI()
                    );
                    _attention_bank->set_aw(target, new_av);
                }
            }
        }
    }
    
    metrics.activation_spreading_time = end_timer();
    metrics.atoms_processed = focus_atoms.size();
    metrics.diffusion_operations = operations;
    metrics.attention_focus_size = focus_atoms.size();
    
    return metrics;
}

AttentionMetrics AttentionBenchmark::benchmark_importance_diffusion()
{
    AttentionMetrics metrics;
    
    start_timer();
    
    // Test importance diffusion algorithm performance
    HandleSeq all_atoms;
    _atomspace->get_handles_by_type(all_atoms, CONCEPT_NODE);
    
    size_t diffusion_ops = 0;
    for (const Handle& h : all_atoms) {
        AttentionValuePtr av = _attention_bank->get_aw(h);
        if (av && av->getSTI() > 50.0) {  // Only diffuse from high-STI atoms
            
            HandleSeq neighbors = h->getIncomingSet();
            for (const Handle& neighbor : neighbors) {
                diffusion_ops++;
                
                // Implement simple diffusion
                AttentionValuePtr neighbor_av = _attention_bank->get_aw(neighbor);
                if (neighbor_av) {
                    double diffusion_amount = av->getSTI() * 0.1;  // 10% diffusion
                    AttentionValuePtr new_av = createAV(
                        neighbor_av->getSTI() + diffusion_amount,
                        neighbor_av->getLTI(),
                        neighbor_av->getVLTI()
                    );
                    _attention_bank->set_aw(neighbor, new_av);
                }
            }
        }
    }
    
    metrics.importance_diffusion_time = end_timer();
    metrics.atoms_processed = all_atoms.size();
    metrics.diffusion_operations = diffusion_ops;
    
    return metrics;
}

AttentionMetrics AttentionBenchmark::benchmark_tensor_operations()
{
    AttentionMetrics metrics;
    
    start_timer();
    
    // Simulate tensor operations for attention computation
    size_t tensor_ops = 0;
    
    // Spatial tensor operations (3D)
    std::vector<double> spatial_tensor(_tensor_dof.SPATIAL_DOF, 0.0);
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t j = 0; j < _tensor_dof.SPATIAL_DOF; ++j) {
            spatial_tensor[j] = spatial_tensor[j] * 0.9 + (i * j) * 0.1;
            tensor_ops++;
        }
    }
    
    // Semantic tensor operations (256D)
    std::vector<double> semantic_tensor(_tensor_dof.SEMANTIC_DOF, 0.0);
    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < _tensor_dof.SEMANTIC_DOF; ++j) {
            semantic_tensor[j] = semantic_tensor[j] * 0.95 + (i + j) * 0.05;
            tensor_ops++;
        }
    }
    
    // Hebbian tensor operations (64D)
    std::vector<double> hebbian_tensor(_tensor_dof.HEBBIAN_DOF, 0.0);
    for (size_t i = 0; i < 500; ++i) {
        for (size_t j = 0; j < _tensor_dof.HEBBIAN_DOF; ++j) {
            hebbian_tensor[j] = hebbian_tensor[j] * 0.98 + i * 0.02;
            tensor_ops++;
        }
    }
    
    metrics.tensor_operation_time = end_timer();
    metrics.tensor_computations = tensor_ops;
    
    return metrics;
}

AttentionMetrics AttentionBenchmark::run_comprehensive_benchmark()
{
    std::cout << "Running comprehensive attention benchmark..." << std::endl;
    
    // Setup test environment
    generate_test_atoms(_num_atoms);
    setup_attention_patterns();
    
    AttentionMetrics total_metrics;
    
    for (size_t i = 0; i < _num_iterations; ++i) {
        // Run all benchmark components
        AttentionMetrics spreading_metrics = benchmark_activation_spreading();
        AttentionMetrics diffusion_metrics = benchmark_importance_diffusion();
        AttentionMetrics tensor_metrics = benchmark_tensor_operations();
        
        // Aggregate metrics
        total_metrics.activation_spreading_time += spreading_metrics.activation_spreading_time;
        total_metrics.importance_diffusion_time += diffusion_metrics.importance_diffusion_time;
        total_metrics.tensor_operation_time += tensor_metrics.tensor_operation_time;
        total_metrics.atoms_processed += spreading_metrics.atoms_processed;
        total_metrics.diffusion_operations += diffusion_metrics.diffusion_operations;
        total_metrics.tensor_computations += tensor_metrics.tensor_computations;
        
        if (i % 10 == 0) {
            std::cout << "Completed iteration " << i << "/" << _num_iterations << std::endl;
        }
    }
    
    // Calculate averages
    total_metrics.activation_spreading_time /= _num_iterations;
    total_metrics.importance_diffusion_time /= _num_iterations;
    total_metrics.tensor_operation_time /= _num_iterations;
    
    return total_metrics;
}

double AttentionBenchmark::measure_activation_spreading_performance()
{
    AttentionMetrics metrics = benchmark_activation_spreading();
    return metrics.atoms_processed / metrics.activation_spreading_time * 1000000.0; // atoms per second
}

double AttentionBenchmark::measure_importance_diffusion_rate()
{
    AttentionMetrics metrics = benchmark_importance_diffusion();
    return metrics.diffusion_operations / metrics.importance_diffusion_time * 1000000.0; // operations per second
}

double AttentionBenchmark::measure_tensor_computation_throughput()
{
    AttentionMetrics metrics = benchmark_tensor_operations();
    return metrics.tensor_computations / metrics.tensor_operation_time * 1000000.0; // computations per second
}

void AttentionBenchmark::print_benchmark_results(const AttentionMetrics& metrics)
{
    std::cout << "\n=== Attention Benchmark Results ===" << std::endl;
    std::cout << "Activation Spreading Time: " << metrics.activation_spreading_time << " μs" << std::endl;
    std::cout << "Importance Diffusion Time: " << metrics.importance_diffusion_time << " μs" << std::endl;
    std::cout << "Tensor Operation Time: " << metrics.tensor_operation_time << " μs" << std::endl;
    std::cout << "Atoms Processed: " << metrics.atoms_processed << std::endl;
    std::cout << "Diffusion Operations: " << metrics.diffusion_operations << std::endl;
    std::cout << "Tensor Computations: " << metrics.tensor_computations << std::endl;
    std::cout << "Attention Focus Size: " << metrics.attention_focus_size << std::endl;
    
    // Calculate performance rates
    if (metrics.activation_spreading_time > 0) {
        double spreading_rate = metrics.atoms_processed / metrics.activation_spreading_time * 1000000.0;
        std::cout << "Activation Spreading Rate: " << spreading_rate << " atoms/sec" << std::endl;
    }
    
    if (metrics.importance_diffusion_time > 0) {
        double diffusion_rate = metrics.diffusion_operations / metrics.importance_diffusion_time * 1000000.0;
        std::cout << "Importance Diffusion Rate: " << diffusion_rate << " ops/sec" << std::endl;
    }
    
    if (metrics.tensor_operation_time > 0) {
        double tensor_rate = metrics.tensor_computations / metrics.tensor_operation_time * 1000000.0;
        std::cout << "Tensor Computation Rate: " << tensor_rate << " ops/sec" << std::endl;
    }
    
    std::cout << "===================================" << std::endl;
}

std::map<std::string, double> AttentionBenchmark::get_performance_summary()
{
    AttentionMetrics metrics = run_comprehensive_benchmark();
    
    std::map<std::string, double> summary;
    summary["activation_spreading_rate"] = measure_activation_spreading_performance();
    summary["importance_diffusion_rate"] = measure_importance_diffusion_rate();
    summary["tensor_computation_rate"] = measure_tensor_computation_throughput();
    summary["total_atoms_processed"] = metrics.atoms_processed;
    summary["total_operations"] = metrics.diffusion_operations + metrics.tensor_computations;
    summary["attention_focus_size"] = metrics.attention_focus_size;
    
    return summary;
}

} // namespace opencog