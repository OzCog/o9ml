#!/usr/bin/env python3
"""
Activation Spreading Performance Measurement Script

This script measures and analyzes the performance of attention allocation
mechanisms in the OpenCog cognitive architecture, specifically focusing
on ECAN (Economic Attention Allocation Network) dynamics.
"""

import sys
import os
import subprocess
import json
import time
import statistics
from pathlib import Path

class AttentionPerformanceAnalyzer:
    """Analyzes attention allocation performance metrics"""
    
    def __init__(self, build_dir="/tmp/cogml_build"):
        self.build_dir = Path(build_dir)
        self.results = []
        
    def run_benchmark(self, num_atoms=1000, num_iterations=10):
        """Run the attention benchmark with specified parameters"""
        
        benchmark_exe = self.build_dir / "orc-ct" / "attention" / "benchmarks" / "attention-benchmark-test"
        
        if not benchmark_exe.exists():
            print(f"Benchmark executable not found: {benchmark_exe}")
            return None
            
        try:
            # Run the benchmark
            result = subprocess.run(
                [str(benchmark_exe), str(num_atoms), str(num_iterations)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return self._parse_benchmark_output(result.stdout)
            else:
                print(f"Benchmark failed with return code {result.returncode}")
                print(f"Error output: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("Benchmark timed out after 5 minutes")
            return None
        except Exception as e:
            print(f"Error running benchmark: {e}")
            return None
    
    def _parse_benchmark_output(self, output):
        """Parse benchmark output to extract performance metrics"""
        
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            if "Activation Spreading Rate:" in line:
                metrics['activation_spreading_rate'] = float(line.split(':')[1].strip().split()[0])
            elif "Importance Diffusion Rate:" in line:
                metrics['importance_diffusion_rate'] = float(line.split(':')[1].strip().split()[0])
            elif "Tensor Computation Rate:" in line:
                metrics['tensor_computation_rate'] = float(line.split(':')[1].strip().split()[0])
            elif "Atoms Processed:" in line:
                metrics['atoms_processed'] = int(line.split(':')[1].strip())
            elif "Diffusion Operations:" in line:
                metrics['diffusion_operations'] = int(line.split(':')[1].strip())
            elif "Tensor Computations:" in line:
                metrics['tensor_computations'] = int(line.split(':')[1].strip())
            elif "Attention Focus Size:" in line:
                metrics['attention_focus_size'] = int(line.split(':')[1].strip())
                
        return metrics
    
    def run_performance_suite(self, atom_counts=[100, 500, 1000, 5000], iterations=5):
        """Run a comprehensive performance test suite"""
        
        print("Running ECAN Performance Test Suite")
        print("=" * 50)
        
        suite_results = []
        
        for atom_count in atom_counts:
            print(f"\nTesting with {atom_count} atoms...")
            
            run_results = []
            for i in range(iterations):
                print(f"  Run {i+1}/{iterations}...", end='', flush=True)
                
                metrics = self.run_benchmark(atom_count, 10)
                if metrics:
                    run_results.append(metrics)
                    print(" DONE")
                else:
                    print(" FAILED")
            
            if run_results:
                # Calculate statistics for this atom count
                avg_metrics = self._calculate_average_metrics(run_results)
                avg_metrics['atom_count'] = atom_count
                avg_metrics['successful_runs'] = len(run_results)
                suite_results.append(avg_metrics)
        
        return suite_results
    
    def _calculate_average_metrics(self, metrics_list):
        """Calculate average metrics from multiple runs"""
        
        if not metrics_list:
            return {}
            
        avg_metrics = {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        # Calculate averages
        for key in all_keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                avg_metrics[f'avg_{key}'] = statistics.mean(values)
                avg_metrics[f'std_{key}'] = statistics.stdev(values) if len(values) > 1 else 0
                avg_metrics[f'min_{key}'] = min(values)
                avg_metrics[f'max_{key}'] = max(values)
        
        return avg_metrics
    
    def print_performance_report(self, suite_results):
        """Print a detailed performance report"""
        
        print("\n" + "=" * 80)
        print("ECAN ATTENTION ALLOCATION PERFORMANCE REPORT")
        print("=" * 80)
        
        print(f"\n{'Atoms':<8} {'Spreading':<12} {'Diffusion':<12} {'Tensor':<12} {'Focus':<8}")
        print(f"{'Count':<8} {'Rate (a/s)':<12} {'Rate (o/s)':<12} {'Rate (o/s)':<12} {'Size':<8}")
        print("-" * 60)
        
        for result in suite_results:
            atom_count = result.get('atom_count', 0)
            spreading_rate = result.get('avg_activation_spreading_rate', 0)
            diffusion_rate = result.get('avg_importance_diffusion_rate', 0)
            tensor_rate = result.get('avg_tensor_computation_rate', 0)
            focus_size = result.get('avg_attention_focus_size', 0)
            
            print(f"{atom_count:<8} {spreading_rate:<12.0f} {diffusion_rate:<12.0f} "
                  f"{tensor_rate:<12.0f} {focus_size:<8.0f}")
        
        # Performance analysis
        print("\n" + "=" * 80)
        print("PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        if suite_results:
            # Analyze scaling
            largest_test = max(suite_results, key=lambda x: x.get('atom_count', 0))
            
            print(f"\nLargest test configuration:")
            print(f"  Atoms: {largest_test.get('atom_count', 0)}")
            print(f"  Activation Spreading Rate: {largest_test.get('avg_activation_spreading_rate', 0):.0f} atoms/sec")
            print(f"  Importance Diffusion Rate: {largest_test.get('avg_importance_diffusion_rate', 0):.0f} ops/sec")
            print(f"  Tensor Computation Rate: {largest_test.get('avg_tensor_computation_rate', 0):.0f} ops/sec")
            
            # Performance targets
            print(f"\nPerformance Target Analysis:")
            spreading_target = 10000  # atoms/sec
            diffusion_target = 1000   # ops/sec
            tensor_target = 100000    # ops/sec
            
            spreading_pass = largest_test.get('avg_activation_spreading_rate', 0) >= spreading_target
            diffusion_pass = largest_test.get('avg_importance_diffusion_rate', 0) >= diffusion_target
            tensor_pass = largest_test.get('avg_tensor_computation_rate', 0) >= tensor_target
            
            print(f"  Activation Spreading: {'PASS' if spreading_pass else 'FAIL'} "
                  f"(target: {spreading_target} atoms/sec)")
            print(f"  Importance Diffusion: {'PASS' if diffusion_pass else 'FAIL'} "
                  f"(target: {diffusion_target} ops/sec)")
            print(f"  Tensor Computation: {'PASS' if tensor_pass else 'FAIL'} "
                  f"(target: {tensor_target} ops/sec)")
            
            overall_pass = spreading_pass and diffusion_pass and tensor_pass
            print(f"\nOverall Performance: {'PASS' if overall_pass else 'FAIL'}")
            
            return overall_pass
            
        return False
    
    def export_results_json(self, suite_results, filename="attention_performance.json"):
        """Export results to JSON file"""
        
        with open(filename, 'w') as f:
            json.dump(suite_results, f, indent=2)
        
        print(f"\nResults exported to: {filename}")

def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        build_dir = sys.argv[1]
    else:
        build_dir = "/tmp/cogml_build"
    
    analyzer = AttentionPerformanceAnalyzer(build_dir)
    
    # Run performance suite
    results = analyzer.run_performance_suite()
    
    if results:
        # Print report
        performance_pass = analyzer.print_performance_report(results)
        
        # Export results
        analyzer.export_results_json(results)
        
        # Exit with appropriate code
        sys.exit(0 if performance_pass else 1)
    else:
        print("No results obtained from performance suite")
        sys.exit(1)

if __name__ == "__main__":
    main()