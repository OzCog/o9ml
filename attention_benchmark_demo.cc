/*
 * attention_benchmark_demo.cc
 *
 * Standalone demonstration of attention allocation benchmarking
 * without external dependencies for validation purposes.
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <map>
#include <string>
#include <memory>
#include <algorithm>
#include <numeric>

// Simplified attention value structure for demo
struct SimpleAttentionValue {
    double sti;  // Short-term importance
    double lti;  // Long-term importance
    int vlti;    // Very long-term importance
    
    SimpleAttentionValue(double s = 0.0, double l = 0.0, int v = 0) 
        : sti(s), lti(l), vlti(v) {}
};

// Simplified atom structure for demo
struct SimpleAtom {
    std::string name;
    std::vector<size_t> connections;  // Indices to other atoms
    SimpleAttentionValue attention_value;
    
    SimpleAtom(const std::string& n) : name(n) {}
};

// Demo attention metrics structure
struct DemoAttentionMetrics {
    double activation_spreading_time;
    double importance_diffusion_time;
    double tensor_operation_time;
    size_t atoms_processed;
    size_t diffusion_operations;
    size_t tensor_computations;
    size_t attention_focus_size;
    
    DemoAttentionMetrics() : 
        activation_spreading_time(0.0),
        importance_diffusion_time(0.0),
        tensor_operation_time(0.0),
        atoms_processed(0),
        diffusion_operations(0),
        tensor_computations(0),
        attention_focus_size(0) {}
};

// Demo attention tensor DOF
struct DemoAttentionTensorDOF {
    static constexpr size_t SPATIAL_DOF = 3;
    static constexpr size_t TEMPORAL_DOF = 1;
    static constexpr size_t SEMANTIC_DOF = 256;
    static constexpr size_t IMPORTANCE_DOF = 3;
    static constexpr size_t HEBBIAN_DOF = 64;
    static constexpr size_t TOTAL_DOF = SPATIAL_DOF + TEMPORAL_DOF + 
                                       SEMANTIC_DOF + IMPORTANCE_DOF + 
                                       HEBBIAN_DOF;
};

// Demo attention benchmark class
class DemoAttentionBenchmark {
private:
    std::vector<SimpleAtom> atoms;
    std::vector<size_t> attention_focus;
    size_t num_iterations;
    std::chrono::high_resolution_clock::time_point start_time;
    
    void start_timer() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double end_timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        return static_cast<double>(duration.count());
    }
    
    void generate_test_atoms(size_t count) {
        atoms.clear();
        atoms.reserve(count);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> sti_dist(0.0, 100.0);
        std::uniform_real_distribution<> lti_dist(0.0, 1.0);
        std::uniform_int_distribution<> conn_dist(0, count - 1);
        
        // Create atoms
        for (size_t i = 0; i < count; ++i) {
            atoms.emplace_back("TestAtom_" + std::to_string(i));
            atoms[i].attention_value = SimpleAttentionValue(
                sti_dist(gen), lti_dist(gen), 0);
        }
        
        // Create random connections
        for (size_t i = 0; i < count; ++i) {
            size_t num_connections = std::min(size_t(10), count / 10);
            for (size_t j = 0; j < num_connections; ++j) {
                size_t target = conn_dist(gen);
                if (target != i) {
                    atoms[i].connections.push_back(target);
                }
            }
        }
        
        // Create attention focus (top 10% by STI)
        std::vector<size_t> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(), 
                  [this](size_t a, size_t b) {
                      return atoms[a].attention_value.sti > atoms[b].attention_value.sti;
                  });
        
        attention_focus.clear();
        size_t focus_size = count / 10;
        for (size_t i = 0; i < focus_size; ++i) {
            attention_focus.push_back(indices[i]);
        }
    }
    
public:
    DemoAttentionBenchmark(size_t num_atoms, size_t iterations) 
        : num_iterations(iterations) {
        generate_test_atoms(num_atoms);
        std::cout << "Demo benchmark initialized with " << num_atoms 
                  << " atoms and " << iterations << " iterations." << std::endl;
    }
    
    DemoAttentionMetrics benchmark_activation_spreading() {
        DemoAttentionMetrics metrics;
        
        start_timer();
        
        size_t operations = 0;
        for (size_t focus_atom : attention_focus) {
            for (size_t connected_atom : atoms[focus_atom].connections) {
                // Simulate activation spreading
                double spread_amount = atoms[focus_atom].attention_value.sti * 0.01;
                atoms[connected_atom].attention_value.sti += spread_amount;
                operations++;
            }
        }
        
        metrics.activation_spreading_time = end_timer();
        metrics.atoms_processed = attention_focus.size();
        metrics.diffusion_operations = operations;
        metrics.attention_focus_size = attention_focus.size();
        
        return metrics;
    }
    
    DemoAttentionMetrics benchmark_importance_diffusion() {
        DemoAttentionMetrics metrics;
        
        start_timer();
        
        size_t diffusion_ops = 0;
        for (size_t i = 0; i < atoms.size(); ++i) {
            if (atoms[i].attention_value.sti > 50.0) {
                for (size_t neighbor : atoms[i].connections) {
                    double diffusion_amount = atoms[i].attention_value.sti * 0.1;
                    atoms[neighbor].attention_value.sti += diffusion_amount;
                    diffusion_ops++;
                }
            }
        }
        
        metrics.importance_diffusion_time = end_timer();
        metrics.atoms_processed = atoms.size();
        metrics.diffusion_operations = diffusion_ops;
        
        return metrics;
    }
    
    DemoAttentionMetrics benchmark_tensor_operations() {
        DemoAttentionMetrics metrics;
        
        start_timer();
        
        size_t tensor_ops = 0;
        
        // Spatial tensor operations (3D)
        std::vector<double> spatial_tensor(DemoAttentionTensorDOF::SPATIAL_DOF, 0.0);
        for (size_t i = 0; i < 1000; ++i) {
            for (size_t j = 0; j < DemoAttentionTensorDOF::SPATIAL_DOF; ++j) {
                spatial_tensor[j] = spatial_tensor[j] * 0.9 + (i * j) * 0.1;
                tensor_ops++;
            }
        }
        
        // Semantic tensor operations (256D)
        std::vector<double> semantic_tensor(DemoAttentionTensorDOF::SEMANTIC_DOF, 0.0);
        for (size_t i = 0; i < 100; ++i) {
            for (size_t j = 0; j < DemoAttentionTensorDOF::SEMANTIC_DOF; ++j) {
                semantic_tensor[j] = semantic_tensor[j] * 0.95 + (i + j) * 0.05;
                tensor_ops++;
            }
        }
        
        // Hebbian tensor operations (64D)
        std::vector<double> hebbian_tensor(DemoAttentionTensorDOF::HEBBIAN_DOF, 0.0);
        for (size_t i = 0; i < 500; ++i) {
            for (size_t j = 0; j < DemoAttentionTensorDOF::HEBBIAN_DOF; ++j) {
                hebbian_tensor[j] = hebbian_tensor[j] * 0.98 + i * 0.02;
                tensor_ops++;
            }
        }
        
        metrics.tensor_operation_time = end_timer();
        metrics.tensor_computations = tensor_ops;
        
        return metrics;
    }
    
    DemoAttentionMetrics run_comprehensive_benchmark() {
        std::cout << "Running comprehensive attention benchmark..." << std::endl;
        
        DemoAttentionMetrics total_metrics;
        
        for (size_t i = 0; i < num_iterations; ++i) {
            DemoAttentionMetrics spreading_metrics = benchmark_activation_spreading();
            DemoAttentionMetrics diffusion_metrics = benchmark_importance_diffusion();
            DemoAttentionMetrics tensor_metrics = benchmark_tensor_operations();
            
            total_metrics.activation_spreading_time += spreading_metrics.activation_spreading_time;
            total_metrics.importance_diffusion_time += diffusion_metrics.importance_diffusion_time;
            total_metrics.tensor_operation_time += tensor_metrics.tensor_operation_time;
            total_metrics.atoms_processed += spreading_metrics.atoms_processed;
            total_metrics.diffusion_operations += diffusion_metrics.diffusion_operations;
            total_metrics.tensor_computations += tensor_metrics.tensor_computations;
            
            if (i % 10 == 0) {
                std::cout << "Completed iteration " << i << "/" << num_iterations << std::endl;
            }
        }
        
        // Calculate averages
        total_metrics.activation_spreading_time /= num_iterations;
        total_metrics.importance_diffusion_time /= num_iterations;
        total_metrics.tensor_operation_time /= num_iterations;
        
        return total_metrics;
    }
    
    void print_benchmark_results(const DemoAttentionMetrics& metrics) {
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
    
    DemoAttentionTensorDOF get_tensor_dof() const {
        return DemoAttentionTensorDOF();
    }
};

int main(int argc, char* argv[]) {
    std::cout << "OpenCog Attention Allocation Benchmark Demo" << std::endl;
    std::cout << "===========================================" << std::endl;
    
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
    
    // Display tensor degrees of freedom
    DemoAttentionTensorDOF tensor_dof;
    std::cout << "\nAttention Tensor Degrees of Freedom:" << std::endl;
    std::cout << "- Spatial DOF: " << tensor_dof.SPATIAL_DOF << std::endl;
    std::cout << "- Temporal DOF: " << tensor_dof.TEMPORAL_DOF << std::endl;
    std::cout << "- Semantic DOF: " << tensor_dof.SEMANTIC_DOF << std::endl;
    std::cout << "- Importance DOF: " << tensor_dof.IMPORTANCE_DOF << std::endl;
    std::cout << "- Hebbian DOF: " << tensor_dof.HEBBIAN_DOF << std::endl;
    std::cout << "- Total DOF: " << tensor_dof.TOTAL_DOF << std::endl;
    
    // Create benchmark
    DemoAttentionBenchmark benchmark(num_atoms, num_iterations);
    
    // Run comprehensive benchmark
    DemoAttentionMetrics results = benchmark.run_comprehensive_benchmark();
    
    // Print results
    benchmark.print_benchmark_results(results);
    
    // Performance validation
    std::cout << "\nPerformance Validation:" << std::endl;
    
    double spreading_rate = results.atoms_processed / results.activation_spreading_time * 1000000.0;
    double diffusion_rate = results.diffusion_operations / results.importance_diffusion_time * 1000000.0;
    double tensor_rate = results.tensor_computations / results.tensor_operation_time * 1000000.0;
    
    bool pass_spreading = spreading_rate > 1000.0;  // 1K atoms/sec minimum
    bool pass_diffusion = diffusion_rate > 100.0;   // 100 ops/sec minimum
    bool pass_tensor = tensor_rate > 10000.0;       // 10K ops/sec minimum
    
    std::cout << "- Activation Spreading: " << (pass_spreading ? "PASS" : "FAIL") 
              << " (" << spreading_rate << " atoms/sec)" << std::endl;
    std::cout << "- Importance Diffusion: " << (pass_diffusion ? "PASS" : "FAIL") 
              << " (" << diffusion_rate << " ops/sec)" << std::endl;
    std::cout << "- Tensor Computation: " << (pass_tensor ? "PASS" : "FAIL") 
              << " (" << tensor_rate << " ops/sec)" << std::endl;
    
    bool all_passed = pass_spreading && pass_diffusion && pass_tensor;
    std::cout << "\nOverall Result: " << (all_passed ? "PASS" : "FAIL") << std::endl;
    
    return all_passed ? 0 : 1;
}