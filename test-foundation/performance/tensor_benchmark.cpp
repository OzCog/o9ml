// Tensor operation performance benchmarks
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

class TensorBenchmark {
public:
    void run_benchmarks() {
        benchmark_tensor_operations();
        benchmark_recursive_calls();
        benchmark_memory_usage();
    }
    
private:
    void benchmark_tensor_operations() {
        const int tensor_size = 1000000;
        std::vector<float> tensor_a(tensor_size);
        std::vector<float> tensor_b(tensor_size);
        std::vector<float> result(tensor_size);
        
        // Initialize with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for(int i = 0; i < tensor_size; ++i) {
            tensor_a[i] = dis(gen);
            tensor_b[i] = dis(gen);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Tensor addition benchmark
        for(int i = 0; i < tensor_size; ++i) {
            result[i] = tensor_a[i] + tensor_b[i];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Tensor addition (" << tensor_size << " elements): " 
                  << duration.count() << " microseconds" << std::endl;
    }
    
    void benchmark_recursive_calls() {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Fibonacci as recursive benchmark
        int result = fibonacci(30);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "  Recursive calls (fibonacci(30)): " 
                  << duration.count() << " milliseconds, result: " << result << std::endl;
    }
    
    void benchmark_memory_usage() {
        const int num_allocations = 1000;
        std::vector<std::vector<float>*> allocations;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for(int i = 0; i < num_allocations; ++i) {
            allocations.push_back(new std::vector<float>(1000, 1.0f));
        }
        
        for(auto* alloc : allocations) {
            delete alloc;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Memory allocation/deallocation (" << num_allocations 
                  << " vectors): " << duration.count() << " microseconds" << std::endl;
    }
    
    int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n-1) + fibonacci(n-2);
    }
};

int main() {
    std::cout << "Running tensor performance benchmarks..." << std::endl;
    
    TensorBenchmark benchmark;
    benchmark.run_benchmarks();
    
    std::cout << "Performance benchmarks completed!" << std::endl;
    return 0;
}
