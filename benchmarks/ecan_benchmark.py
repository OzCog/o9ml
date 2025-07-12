"""
ECAN Attention Allocation Performance Benchmark

This benchmark suite tests the performance of the ECAN attention allocation
system under various loads and configurations. It provides comprehensive
metrics for attention flow, resource allocation, and system scalability.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import json
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ecan.attention_kernel import AttentionKernel, ECANAttentionTensor
from ecan.economic_allocator import EconomicAllocator, AttentionAllocationRequest
from ecan.resource_scheduler import ResourceScheduler, ScheduledTask
from ecan.attention_spreading import AttentionSpreading, AttentionLink
from ecan.decay_refresh import DecayRefresh


class ECANPerformanceBenchmark:
    """
    Comprehensive performance benchmark for ECAN attention allocation system.
    
    Tests performance under various loads, configurations, and usage patterns
    to validate that the system meets the specified performance targets.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }
        
        print(f"ECAN Performance Benchmark Suite")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("Running comprehensive ECAN performance benchmarks...")
        
        # Core performance benchmarks
        self.results['benchmarks']['attention_kernel'] = self.benchmark_attention_kernel()
        self.results['benchmarks']['economic_allocator'] = self.benchmark_economic_allocator()
        self.results['benchmarks']['resource_scheduler'] = self.benchmark_resource_scheduler()
        self.results['benchmarks']['attention_spreading'] = self.benchmark_attention_spreading()
        self.results['benchmarks']['decay_refresh'] = self.benchmark_decay_refresh()
        
        # Integration benchmarks
        self.results['benchmarks']['full_system_integration'] = self.benchmark_full_system()
        self.results['benchmarks']['scalability'] = self.benchmark_scalability()
        self.results['benchmarks']['real_world_simulation'] = self.benchmark_real_world_simulation()
        
        # Generate reports
        self._generate_performance_report()
        self._create_performance_visualizations()
        self._save_results()
        
        return self.results
    
    def benchmark_attention_kernel(self) -> Dict[str, Any]:
        """Benchmark attention kernel performance"""
        print("\n1. Benchmarking Attention Kernel...")
        
        results = {}
        
        # Test different atom counts
        atom_counts = [100, 500, 1000, 5000, 10000]
        
        for atom_count in atom_counts:
            print(f"  Testing with {atom_count} atoms...")
            
            kernel = AttentionKernel(max_atoms=atom_count * 2)
            
            # Allocation performance
            start_time = time.time()
            for i in range(atom_count):
                tensor = ECANAttentionTensor(
                    short_term_importance=np.random.uniform(0.1, 0.9),
                    long_term_importance=np.random.uniform(0.0, 0.5),
                    urgency=np.random.uniform(0.0, 1.0),
                    confidence=np.random.uniform(0.3, 1.0),
                    spreading_factor=np.random.uniform(0.2, 0.8),
                    decay_rate=np.random.uniform(0.05, 0.2)
                )
                kernel.allocate_attention(f"atom_{i}", tensor)
            
            allocation_time = time.time() - start_time
            
            # Update performance
            start_time = time.time()
            for i in range(min(1000, atom_count)):
                kernel.update_attention(
                    f"atom_{i}",
                    short_term_delta=np.random.uniform(-0.1, 0.1),
                    urgency_delta=np.random.uniform(-0.1, 0.1)
                )
            
            update_time = time.time() - start_time
            
            # Focus computation performance
            start_time = time.time()
            for _ in range(100):  # Multiple focus computations
                focus = kernel.get_attention_focus()
            focus_time = time.time() - start_time
            
            # Distribution computation performance
            start_time = time.time()
            for _ in range(50):  # Multiple distribution computations
                distribution = kernel.compute_global_attention_distribution()
            distribution_time = time.time() - start_time
            
            # Get performance metrics
            metrics = kernel.get_performance_metrics()
            
            results[atom_count] = {
                'allocation_time': allocation_time,
                'allocation_rate': atom_count / allocation_time,
                'update_time': update_time,
                'update_rate': min(1000, atom_count) / update_time,
                'focus_time': focus_time,
                'focus_rate': 100 / focus_time,
                'distribution_time': distribution_time,
                'distribution_rate': 50 / distribution_time,
                'final_metrics': metrics
            }
        
        # Performance target analysis
        results['performance_analysis'] = self._analyze_kernel_performance(results)
        
        return results
    
    def benchmark_economic_allocator(self) -> Dict[str, Any]:
        """Benchmark economic allocator performance"""
        print("\n2. Benchmarking Economic Allocator...")
        
        results = {}
        
        # Test different batch sizes
        batch_sizes = [10, 50, 100, 500, 1000]
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            kernel = AttentionKernel(max_atoms=batch_size * 2)
            allocator = EconomicAllocator(total_attention_budget=batch_size * 10.0)
            
            # Create allocation requests
            requests = []
            for i in range(batch_size):
                tensor = ECANAttentionTensor(
                    short_term_importance=np.random.uniform(0.2, 0.8),
                    urgency=np.random.uniform(0.0, 1.0),
                    confidence=np.random.uniform(0.4, 1.0)
                )
                
                request = AttentionAllocationRequest(
                    atom_id=f"batch_atom_{i}",
                    requested_attention=tensor,
                    value=np.random.uniform(5.0, 20.0),
                    cost=np.random.uniform(2.0, 10.0),
                    priority=np.random.uniform(0.2, 0.9),
                    deadline=time.time() + np.random.uniform(60, 300) if np.random.random() > 0.5 else None
                )
                requests.append(request)
            
            # Benchmark batch allocation
            start_time = time.time()
            allocation_result = allocator.allocate_attention_batch(requests, kernel)
            allocation_time = time.time() - start_time
            
            # Benchmark request evaluation
            start_time = time.time()
            for request in requests[:min(100, batch_size)]:
                evaluation = allocator.evaluate_allocation_request(request)
            evaluation_time = time.time() - start_time
            
            # Benchmark portfolio optimization
            start_time = time.time()
            optimization_result = allocator.optimize_attention_portfolio(kernel)
            optimization_time = time.time() - start_time
            
            results[batch_size] = {
                'allocation_time': allocation_time,
                'allocation_rate': batch_size / allocation_time,
                'evaluation_time': evaluation_time,
                'evaluation_rate': min(100, batch_size) / evaluation_time,
                'optimization_time': optimization_time,
                'allocation_result': {
                    'allocated_count': len(allocation_result['allocations']),
                    'rejected_count': len(allocation_result['rejected']),
                    'budget_utilization': allocation_result['metrics']['budget_utilization']
                },
                'optimization_result': optimization_result,
                'economic_metrics': allocator.get_economic_metrics()
            }
        
        results['performance_analysis'] = self._analyze_allocator_performance(results)
        
        return results
    
    def benchmark_resource_scheduler(self) -> Dict[str, Any]:
        """Benchmark resource scheduler performance"""
        print("\n3. Benchmarking Resource Scheduler...")
        
        results = {}
        
        # Test different task loads
        task_loads = [10, 50, 100, 200, 500]
        
        for task_count in task_loads:
            print(f"  Testing {task_count} tasks...")
            
            kernel = AttentionKernel(max_atoms=task_count * 2)
            allocator = EconomicAllocator(total_attention_budget=task_count * 5.0)
            scheduler = ResourceScheduler(
                max_concurrent_tasks=min(20, task_count // 5),
                enable_background_processing=False
            )
            
            # Create scheduled tasks
            tasks = []
            for i in range(task_count):
                tensor = ECANAttentionTensor(
                    short_term_importance=np.random.uniform(0.3, 0.9),
                    urgency=np.random.uniform(0.2, 1.0)
                )
                
                request = AttentionAllocationRequest(
                    atom_id=f"task_atom_{i}",
                    requested_attention=tensor,
                    value=np.random.uniform(8.0, 15.0),
                    cost=np.random.uniform(3.0, 8.0),
                    priority=np.random.uniform(0.1, 0.9)
                )
                
                task = ScheduledTask(
                    task_id=f"task_{i}",
                    attention_request=request,
                    max_execution_time=np.random.uniform(30, 120)
                )
                tasks.append(task)
            
            # Benchmark task scheduling
            start_time = time.time()
            for task in tasks:
                scheduler.schedule_task(task)
            scheduling_time = time.time() - start_time
            
            # Benchmark scheduler cycles
            start_time = time.time()
            cycle_count = 0
            while len(scheduler.priority_queue) > 0 and cycle_count < 100:
                cycle_result = scheduler.process_scheduler_cycle(kernel, allocator)
                cycle_count += 1
                if cycle_result['tasks_started'] == 0:
                    break  # No more tasks can be started
            
            processing_time = time.time() - start_time
            
            # Get final metrics
            scheduler_metrics = scheduler.get_scheduler_metrics()
            
            results[task_count] = {
                'scheduling_time': scheduling_time,
                'scheduling_rate': task_count / scheduling_time,
                'processing_time': processing_time,
                'cycle_count': cycle_count,
                'cycles_per_second': cycle_count / processing_time if processing_time > 0 else 0,
                'scheduler_metrics': scheduler_metrics
            }
        
        results['performance_analysis'] = self._analyze_scheduler_performance(results)
        
        return results
    
    def benchmark_attention_spreading(self) -> Dict[str, Any]:
        """Benchmark attention spreading performance"""
        print("\n4. Benchmarking Attention Spreading...")
        
        results = {}
        
        # Test different network sizes
        network_sizes = [50, 100, 200, 500, 1000]
        
        for network_size in network_sizes:
            print(f"  Testing network size: {network_size} atoms...")
            
            kernel = AttentionKernel(max_atoms=network_size)
            spreading = AttentionSpreading()
            
            # Create atoms with attention
            focus_atoms = []
            for i in range(network_size):
                tensor = ECANAttentionTensor(
                    short_term_importance=np.random.uniform(0.1, 0.9),
                    spreading_factor=np.random.uniform(0.3, 0.8),
                    confidence=np.random.uniform(0.4, 1.0)
                )
                kernel.allocate_attention(f"spread_atom_{i}", tensor)
                
                if i < network_size // 10:  # 10% focus atoms
                    focus_atoms.append(f"spread_atom_{i}")
            
            # Create attention topology
            link_count = 0
            for i in range(network_size):
                # Create links to random neighbors
                neighbor_count = min(5, np.random.poisson(3))
                for _ in range(neighbor_count):
                    target_idx = np.random.randint(0, network_size)
                    if target_idx != i:
                        link = AttentionLink(
                            source_atom=f"spread_atom_{i}",
                            target_atom=f"spread_atom_{target_idx}",
                            link_strength=np.random.uniform(0.2, 0.8),
                            bidirectional=np.random.random() > 0.5
                        )
                        spreading.add_attention_link(link)
                        link_count += 1
            
            # Benchmark attention spreading
            start_time = time.time()
            spread_result = spreading.spread_attention(
                kernel, 
                source_atoms=focus_atoms,
                spread_focus_only=True
            )
            spreading_time = time.time() - start_time
            
            # Benchmark topology creation
            similarities = {}
            for i in range(min(100, network_size)):
                for j in range(i + 1, min(100, network_size)):
                    similarities[(f"spread_atom_{i}", f"spread_atom_{j}")] = np.random.uniform(0.0, 1.0)
            
            start_time = time.time()
            semantic_links = spreading.create_semantic_topology(similarities, min_similarity=0.5)
            topology_time = time.time() - start_time
            
            # Analyze attention flow
            start_time = time.time()
            flow_analysis = spreading.analyze_attention_flow(kernel)
            analysis_time = time.time() - start_time
            
            results[network_size] = {
                'spreading_time': spreading_time,
                'spreading_rate': spread_result.atoms_affected / spreading_time if spreading_time > 0 else 0,
                'topology_creation_time': topology_time,
                'topology_rate': len(similarities) / topology_time if topology_time > 0 else 0,
                'analysis_time': analysis_time,
                'spread_result': {
                    'atoms_affected': spread_result.atoms_affected,
                    'spread_iterations': spread_result.spread_iterations,
                    'convergence_achieved': spread_result.convergence_achieved,
                    'total_spread': spread_result.total_spread
                },
                'network_metrics': {
                    'link_count': link_count,
                    'semantic_links_created': semantic_links,
                    'network_density': flow_analysis.get('network_density', 0)
                },
                'spreading_metrics': spreading.get_spreading_metrics()
            }
        
        results['performance_analysis'] = self._analyze_spreading_performance(results)
        
        return results
    
    def benchmark_decay_refresh(self) -> Dict[str, Any]:
        """Benchmark decay and refresh mechanisms"""
        print("\n5. Benchmarking Decay and Refresh...")
        
        results = {}
        
        # Test different atom populations
        atom_populations = [100, 500, 1000, 2000, 5000]
        
        for atom_count in atom_populations:
            print(f"  Testing {atom_count} atoms...")
            
            kernel = AttentionKernel(max_atoms=atom_count)
            decay_refresh = DecayRefresh()
            
            # Populate with atoms
            for i in range(atom_count):
                tensor = ECANAttentionTensor(
                    short_term_importance=np.random.uniform(0.1, 0.9),
                    long_term_importance=np.random.uniform(0.0, 0.5),
                    urgency=np.random.uniform(0.0, 1.0),
                    confidence=np.random.uniform(0.3, 1.0)
                )
                kernel.allocate_attention(f"decay_atom_{i}", tensor)
                
                # Add some with access history
                if np.random.random() > 0.7:
                    for _ in range(np.random.randint(1, 5)):
                        decay_refresh.add_access_trigger(f"decay_atom_{i}")
            
            # Benchmark decay cycle
            start_time = time.time()
            decay_result = decay_refresh.process_decay_cycle(kernel)
            decay_time = time.time() - start_time
            
            # Benchmark refresh trigger processing
            start_time = time.time()
            for i in range(min(200, atom_count)):
                if np.random.random() > 0.5:
                    decay_refresh.add_access_trigger(f"decay_atom_{i}", np.random.uniform(0.3, 0.8))
            
            refresh_processing_time = time.time() - start_time
            
            # Benchmark parameter optimization
            start_time = time.time()
            optimization_result = decay_refresh.optimize_decay_schedule(kernel)
            optimization_time = time.time() - start_time
            
            results[atom_count] = {
                'decay_time': decay_time,
                'decay_rate': atom_count / decay_time if decay_time > 0 else 0,
                'refresh_processing_time': refresh_processing_time,
                'optimization_time': optimization_time,
                'decay_result': {
                    'atoms_decayed': decay_result.atoms_decayed,
                    'atoms_refreshed': decay_result.atoms_refreshed,
                    'atoms_removed': decay_result.atoms_removed,
                    'decay_efficiency': decay_result.decay_efficiency
                },
                'optimization_result': optimization_result,
                'decay_metrics': decay_refresh.get_decay_refresh_metrics()
            }
        
        results['performance_analysis'] = self._analyze_decay_performance(results)
        
        return results
    
    def benchmark_full_system(self) -> Dict[str, Any]:
        """Benchmark full integrated system performance"""
        print("\n6. Benchmarking Full System Integration...")
        
        results = {}
        
        # Test different system scales
        scales = [
            {'atoms': 100, 'requests': 50, 'tasks': 20},
            {'atoms': 500, 'requests': 200, 'tasks': 100},
            {'atoms': 1000, 'requests': 500, 'tasks': 200},
            {'atoms': 2000, 'requests': 1000, 'tasks': 400}
        ]
        
        for i, scale in enumerate(scales):
            print(f"  Testing scale {i+1}: {scale['atoms']} atoms, {scale['requests']} requests, {scale['tasks']} tasks")
            
            # Initialize all components
            kernel = AttentionKernel(max_atoms=scale['atoms'])
            allocator = EconomicAllocator(total_attention_budget=scale['atoms'] * 2.0)
            scheduler = ResourceScheduler(max_concurrent_tasks=20, enable_background_processing=False)
            spreading = AttentionSpreading()
            decay_refresh = DecayRefresh()
            
            start_time = time.time()
            
            # Phase 1: Initial allocation
            requests = []
            for j in range(scale['requests']):
                tensor = ECANAttentionTensor(
                    short_term_importance=np.random.uniform(0.2, 0.9),
                    urgency=np.random.uniform(0.1, 1.0),
                    confidence=np.random.uniform(0.4, 1.0)
                )
                
                request = AttentionAllocationRequest(
                    atom_id=f"sys_atom_{j}",
                    requested_attention=tensor,
                    value=np.random.uniform(5.0, 20.0),
                    cost=np.random.uniform(2.0, 10.0)
                )
                requests.append(request)
            
            allocation_result = allocator.allocate_attention_batch(requests, kernel)
            phase1_time = time.time() - start_time
            
            # Phase 2: Build attention topology
            start_time = time.time()
            
            allocated_atoms = [req.atom_id for req, _ in allocation_result['allocations']]
            for j in range(len(allocated_atoms)):
                for k in range(j + 1, min(j + 5, len(allocated_atoms))):
                    if np.random.random() > 0.5:
                        link = AttentionLink(
                            source_atom=allocated_atoms[j],
                            target_atom=allocated_atoms[k],
                            link_strength=np.random.uniform(0.3, 0.8)
                        )
                        spreading.add_attention_link(link)
            
            phase2_time = time.time() - start_time
            
            # Phase 3: Attention spreading
            start_time = time.time()
            
            spread_result = spreading.spread_attention(kernel, spread_focus_only=True)
            
            phase3_time = time.time() - start_time
            
            # Phase 4: Task scheduling and processing
            start_time = time.time()
            
            tasks = []
            for j in range(scale['tasks']):
                if j < len(allocated_atoms):
                    atom_id = allocated_atoms[j]
                    existing_tensor = kernel.get_attention(atom_id)
                else:
                    atom_id = f"task_atom_{j}"
                    existing_tensor = ECANAttentionTensor(short_term_importance=0.5)
                
                request = AttentionAllocationRequest(
                    atom_id=atom_id,
                    requested_attention=existing_tensor,
                    value=np.random.uniform(8.0, 15.0),
                    cost=np.random.uniform(3.0, 8.0)
                )
                
                task = ScheduledTask(task_id=f"sys_task_{j}", attention_request=request)
                scheduler.schedule_task(task)
                tasks.append(task)
            
            # Process scheduler cycles
            cycle_count = 0
            while len(scheduler.priority_queue) > 0 and cycle_count < 50:
                scheduler.process_scheduler_cycle(kernel, allocator)
                cycle_count += 1
            
            phase4_time = time.time() - start_time
            
            # Phase 5: Decay and refresh
            start_time = time.time()
            
            # Add some refresh triggers
            for j in range(min(100, scale['atoms'])):
                if np.random.random() > 0.6:
                    decay_refresh.add_access_trigger(f"sys_atom_{j}")
            
            decay_result = decay_refresh.process_decay_cycle(kernel)
            
            phase5_time = time.time() - start_time
            
            total_time = phase1_time + phase2_time + phase3_time + phase4_time + phase5_time
            
            results[f"scale_{i+1}"] = {
                'scale': scale,
                'total_time': total_time,
                'phase_times': {
                    'allocation': phase1_time,
                    'topology': phase2_time,
                    'spreading': phase3_time,
                    'scheduling': phase4_time,
                    'decay_refresh': phase5_time
                },
                'throughput': {
                    'atoms_per_second': scale['atoms'] / total_time,
                    'requests_per_second': scale['requests'] / total_time,
                    'tasks_per_second': scale['tasks'] / total_time
                },
                'final_metrics': {
                    'focus_size': len(kernel.get_attention_focus()),
                    'spread_atoms': spread_result.atoms_affected,
                    'scheduler_cycles': cycle_count,
                    'atoms_decayed': decay_result.atoms_decayed
                }
            }
        
        results['performance_analysis'] = self._analyze_system_performance(results)
        
        return results
    
    def benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark system scalability under increasing load"""
        print("\n7. Benchmarking System Scalability...")
        
        results = {}
        
        # Progressive load increases
        load_factors = [1, 2, 5, 10, 20]
        base_atoms = 200
        
        for factor in load_factors:
            print(f"  Testing load factor {factor}x...")
            
            atoms = base_atoms * factor
            kernel = AttentionKernel(max_atoms=atoms)
            allocator = EconomicAllocator(total_attention_budget=atoms * 1.5)
            
            # Measure allocation scalability
            start_time = time.time()
            
            for i in range(atoms):
                tensor = ECANAttentionTensor(
                    short_term_importance=np.random.uniform(0.1, 0.9)
                )
                kernel.allocate_attention(f"scale_atom_{i}", tensor)
            
            allocation_time = time.time() - start_time
            
            # Measure batch processing scalability
            requests = []
            for i in range(atoms // 2):
                tensor = ECANAttentionTensor(short_term_importance=np.random.uniform(0.3, 0.8))
                request = AttentionAllocationRequest(
                    atom_id=f"batch_atom_{i}",
                    requested_attention=tensor,
                    value=10.0,
                    cost=5.0
                )
                requests.append(request)
            
            start_time = time.time()
            batch_result = allocator.allocate_attention_batch(requests, kernel)
            batch_time = time.time() - start_time
            
            # Measure focus computation scalability
            start_time = time.time()
            for _ in range(10):
                focus = kernel.get_attention_focus()
            focus_time = time.time() - start_time
            
            results[f"factor_{factor}"] = {
                'atoms': atoms,
                'allocation_time': allocation_time,
                'allocation_throughput': atoms / allocation_time,
                'batch_time': batch_time,
                'batch_throughput': len(requests) / batch_time,
                'focus_time': focus_time,
                'focus_throughput': 10 / focus_time,
                'memory_usage': kernel._estimate_memory_usage()
            }
        
        # Analyze scalability trends
        results['scalability_analysis'] = self._analyze_scalability(results)
        
        return results
    
    def benchmark_real_world_simulation(self) -> Dict[str, Any]:
        """Benchmark real-world usage simulation"""
        print("\n8. Benchmarking Real-World Task Simulation...")
        
        # Simulate a cognitive task with multiple phases
        kernel = AttentionKernel(max_atoms=1000)
        allocator = EconomicAllocator(total_attention_budget=200.0)
        scheduler = ResourceScheduler(max_concurrent_tasks=10, enable_background_processing=False)
        spreading = AttentionSpreading()
        decay_refresh = DecayRefresh()
        
        simulation_results = []
        total_start_time = time.time()
        
        # Simulation: Language processing task
        print("  Simulating language processing task...")
        
        # Phase 1: Word attention allocation
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        word_importance = [0.3, 0.7, 0.8, 0.9, 0.8, 0.4, 0.6, 0.7]
        
        start_time = time.time()
        
        for word, importance in zip(words, word_importance):
            tensor = ECANAttentionTensor(
                short_term_importance=importance,
                urgency=0.8,
                confidence=0.7
            )
            kernel.allocate_attention(f"word_{word}", tensor)
        
        word_allocation_time = time.time() - start_time
        
        # Phase 2: Semantic connections
        start_time = time.time()
        
        semantic_pairs = [
            ("word_quick", "word_fox", 0.6),
            ("word_brown", "word_fox", 0.8),
            ("word_fox", "word_jumps", 0.7),
            ("word_jumps", "word_over", 0.6),
            ("word_lazy", "word_dog", 0.8)
        ]
        
        for source, target, strength in semantic_pairs:
            link = AttentionLink(source, target, strength)
            spreading.add_attention_link(link)
        
        semantic_time = time.time() - start_time
        
        # Phase 3: Attention spreading for comprehension
        start_time = time.time()
        
        spread_result = spreading.spread_attention(kernel, source_atoms=["word_fox", "word_jumps"])
        
        spreading_time = time.time() - start_time
        
        # Phase 4: Task scheduling for analysis
        start_time = time.time()
        
        analysis_tasks = ["parse", "semantics", "syntax", "pragmatics"]
        for task_type in analysis_tasks:
            tensor = ECANAttentionTensor(
                short_term_importance=0.7,
                urgency=0.9,
                confidence=0.8
            )
            
            request = AttentionAllocationRequest(
                atom_id=f"analysis_{task_type}",
                requested_attention=tensor,
                value=15.0,
                cost=8.0,
                priority=0.8
            )
            
            task = ScheduledTask(f"task_{task_type}", request)
            scheduler.schedule_task(task)
        
        # Process tasks
        cycles = 0
        while len(scheduler.priority_queue) > 0 and cycles < 20:
            scheduler.process_scheduler_cycle(kernel, allocator)
            cycles += 1
        
        task_processing_time = time.time() - start_time
        
        # Phase 5: Memory consolidation
        start_time = time.time()
        
        # Add access patterns for important words
        for word in ["fox", "jumps"]:
            for _ in range(3):
                decay_refresh.add_access_trigger(f"word_{word}", 0.8)
        
        decay_result = decay_refresh.process_decay_cycle(kernel)
        
        consolidation_time = time.time() - start_time
        
        total_simulation_time = time.time() - total_start_time
        
        simulation_results = {
            'total_time': total_simulation_time,
            'phase_times': {
                'word_allocation': word_allocation_time,
                'semantic_connections': semantic_time,
                'attention_spreading': spreading_time,
                'task_processing': task_processing_time,
                'memory_consolidation': consolidation_time
            },
            'final_state': {
                'focus_atoms': len(kernel.get_attention_focus()),
                'total_atoms': len(kernel.attention_tensors),
                'spread_coverage': spread_result.atoms_affected,
                'tasks_completed': cycles,
                'consolidated_memories': decay_result.atoms_refreshed
            },
            'performance_metrics': {
                'words_per_second': len(words) / word_allocation_time,
                'connections_per_second': len(semantic_pairs) / semantic_time,
                'spread_atoms_per_second': spread_result.atoms_affected / spreading_time,
                'tasks_per_second': len(analysis_tasks) / task_processing_time,
                'consolidation_rate': decay_result.atoms_refreshed / consolidation_time if consolidation_time > 0 else 0
            }
        }
        
        return simulation_results
    
    def _analyze_kernel_performance(self, results: Dict) -> Dict[str, Any]:
        """Analyze attention kernel performance against targets"""
        # Target: >10,000 atoms/second allocation
        max_allocation_rate = max(r['allocation_rate'] for r in results.values() if isinstance(r, dict))
        allocation_target_met = max_allocation_rate >= 10000
        
        # Target: >100,000 tensor operations/second
        max_update_rate = max(r['update_rate'] for r in results.values() if isinstance(r, dict))
        update_target_met = max_update_rate >= 100000
        
        return {
            'allocation_performance': {
                'max_rate': max_allocation_rate,
                'target': 10000,
                'target_met': allocation_target_met
            },
            'update_performance': {
                'max_rate': max_update_rate,
                'target': 100000,
                'target_met': update_target_met
            },
            'overall_grade': 'PASS' if allocation_target_met and update_target_met else 'FAIL'
        }
    
    def _analyze_allocator_performance(self, results: Dict) -> Dict[str, Any]:
        """Analyze economic allocator performance"""
        max_allocation_rate = max(r['allocation_rate'] for r in results.values() if isinstance(r, dict))
        max_evaluation_rate = max(r['evaluation_rate'] for r in results.values() if isinstance(r, dict))
        
        return {
            'allocation_rate': max_allocation_rate,
            'evaluation_rate': max_evaluation_rate,
            'efficiency_achieved': max_allocation_rate >= 1000,  # Target: 1000 allocations/sec
            'overall_grade': 'PASS' if max_allocation_rate >= 1000 else 'FAIL'
        }
    
    def _analyze_scheduler_performance(self, results: Dict) -> Dict[str, Any]:
        """Analyze resource scheduler performance"""
        max_scheduling_rate = max(r['scheduling_rate'] for r in results.values() if isinstance(r, dict))
        max_cycle_rate = max(r['cycles_per_second'] for r in results.values() if isinstance(r, dict))
        
        return {
            'scheduling_rate': max_scheduling_rate,
            'cycle_rate': max_cycle_rate,
            'target_met': max_scheduling_rate >= 1000,  # Target: 1000 tasks/sec scheduling
            'overall_grade': 'PASS' if max_scheduling_rate >= 1000 else 'FAIL'
        }
    
    def _analyze_spreading_performance(self, results: Dict) -> Dict[str, Any]:
        """Analyze attention spreading performance"""
        max_spreading_rate = max(r['spreading_rate'] for r in results.values() if isinstance(r, dict))
        convergence_rates = [r['spread_result']['convergence_achieved'] for r in results.values() if isinstance(r, dict)]
        convergence_success = sum(convergence_rates) / len(convergence_rates) if convergence_rates else 0
        
        return {
            'max_spreading_rate': max_spreading_rate,
            'convergence_success_rate': convergence_success,
            'target_met': max_spreading_rate >= 1000,  # Target: 1000 atoms/sec spreading
            'overall_grade': 'PASS' if max_spreading_rate >= 1000 and convergence_success >= 0.9 else 'FAIL'
        }
    
    def _analyze_decay_performance(self, results: Dict) -> Dict[str, Any]:
        """Analyze decay and refresh performance"""
        max_decay_rate = max(r['decay_rate'] for r in results.values() if isinstance(r, dict))
        avg_efficiency = np.mean([r['decay_result']['decay_efficiency'] for r in results.values() if isinstance(r, dict)])
        
        return {
            'max_decay_rate': max_decay_rate,
            'average_efficiency': avg_efficiency,
            'target_met': max_decay_rate >= 5000,  # Target: 5000 atoms/sec decay processing
            'overall_grade': 'PASS' if max_decay_rate >= 5000 and avg_efficiency >= 0.7 else 'FAIL'
        }
    
    def _analyze_system_performance(self, results: Dict) -> Dict[str, Any]:
        """Analyze full system integration performance"""
        throughputs = []
        for result in results.values():
            if isinstance(result, dict) and 'throughput' in result:
                throughputs.append(result['throughput']['atoms_per_second'])
        
        max_throughput = max(throughputs) if throughputs else 0
        
        return {
            'max_system_throughput': max_throughput,
            'target_met': max_throughput >= 500,  # Target: 500 atoms/sec full system
            'overall_grade': 'PASS' if max_throughput >= 500 else 'FAIL'
        }
    
    def _analyze_scalability(self, results: Dict) -> Dict[str, Any]:
        """Analyze scalability trends"""
        factors = []
        throughputs = []
        
        for key, result in results.items():
            if key.startswith('factor_') and isinstance(result, dict):
                factor = int(key.split('_')[1])
                factors.append(factor)
                throughputs.append(result['allocation_throughput'])
        
        # Calculate scalability efficiency (ideally should remain constant)
        if len(factors) >= 2:
            efficiency_trend = []
            for i in range(1, len(factors)):
                expected = throughputs[0]  # Baseline
                actual = throughputs[i]
                efficiency = actual / expected
                efficiency_trend.append(efficiency)
            
            avg_efficiency = np.mean(efficiency_trend)
        else:
            avg_efficiency = 1.0
        
        return {
            'scalability_factors': factors,
            'throughput_trend': throughputs,
            'scalability_efficiency': avg_efficiency,
            'scalability_grade': 'PASS' if avg_efficiency >= 0.5 else 'FAIL'  # Should maintain at least 50% efficiency
        }
    
    def _generate_performance_report(self):
        """Generate comprehensive performance report"""
        report_path = os.path.join(self.output_dir, "ECAN_Performance_Report.md")
        
        with open(report_path, 'w') as f:
            f.write("# ECAN Attention Allocation Performance Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System info
            f.write("## System Information\n")
            for key, value in self.results['system_info'].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Performance summary
            f.write("## Performance Summary\n\n")
            
            benchmarks = self.results['benchmarks']
            
            # Extract performance grades
            grades = {}
            for component, results in benchmarks.items():
                if 'performance_analysis' in results:
                    analysis = results['performance_analysis']
                    if 'overall_grade' in analysis:
                        grades[component] = analysis['overall_grade']
            
            f.write("### Component Performance Grades\n")
            for component, grade in grades.items():
                status = "✅" if grade == "PASS" else "❌"
                f.write(f"- **{component}**: {grade} {status}\n")
            f.write("\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            
            for component, results in benchmarks.items():
                f.write(f"### {component.replace('_', ' ').title()}\n\n")
                
                if 'performance_analysis' in results:
                    analysis = results['performance_analysis']
                    f.write("**Performance Analysis:**\n")
                    for key, value in analysis.items():
                        if isinstance(value, dict):
                            f.write(f"- **{key}**:\n")
                            for subkey, subvalue in value.items():
                                f.write(f"  - {subkey}: {subvalue}\n")
                        else:
                            f.write(f"- **{key}**: {value}\n")
                    f.write("\n")
        
        print(f"Performance report generated: {report_path}")
    
    def _create_performance_visualizations(self):
        """Create performance visualization charts"""
        try:
            # Attention Kernel Performance
            self._plot_kernel_performance()
            
            # System Scalability
            self._plot_scalability()
            
            # Integration Performance
            self._plot_integration_performance()
            
            print(f"Performance visualizations saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
    
    def _plot_kernel_performance(self):
        """Plot attention kernel performance"""
        kernel_results = self.results['benchmarks']['attention_kernel']
        
        atom_counts = []
        allocation_rates = []
        update_rates = []
        
        for count, result in kernel_results.items():
            if isinstance(result, dict):
                atom_counts.append(count)
                allocation_rates.append(result['allocation_rate'])
                update_rates.append(result['update_rate'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Allocation performance
        ax1.plot(atom_counts, allocation_rates, 'b-o', label='Allocation Rate')
        ax1.axhline(y=10000, color='r', linestyle='--', label='Target (10K/s)')
        ax1.set_xlabel('Number of Atoms')
        ax1.set_ylabel('Allocations per Second')
        ax1.set_title('Attention Allocation Performance')
        ax1.legend()
        ax1.grid(True)
        
        # Update performance
        ax2.plot(atom_counts, update_rates, 'g-o', label='Update Rate')
        ax2.axhline(y=100000, color='r', linestyle='--', label='Target (100K/s)')
        ax2.set_xlabel('Number of Atoms')
        ax2.set_ylabel('Updates per Second')
        ax2.set_title('Attention Update Performance')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'kernel_performance.png'), dpi=150)
        plt.close()
    
    def _plot_scalability(self):
        """Plot system scalability"""
        scalability_results = self.results['benchmarks']['scalability']
        
        factors = []
        throughputs = []
        
        for key, result in scalability_results.items():
            if key.startswith('factor_') and isinstance(result, dict):
                factor = int(key.split('_')[1])
                factors.append(factor)
                throughputs.append(result['allocation_throughput'])
        
        if factors and throughputs:
            plt.figure(figsize=(10, 6))
            plt.plot(factors, throughputs, 'b-o', label='Actual Throughput')
            
            # Ideal linear scaling
            if throughputs:
                ideal = [throughputs[0] for _ in factors]  # Constant performance
                plt.plot(factors, ideal, 'r--', label='Ideal Scaling')
            
            plt.xlabel('Load Factor')
            plt.ylabel('Allocation Throughput (atoms/s)')
            plt.title('ECAN System Scalability')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'scalability.png'), dpi=150)
            plt.close()
    
    def _plot_integration_performance(self):
        """Plot integration performance across phases"""
        integration_results = self.results['benchmarks']['full_system_integration']
        
        scales = []
        phase_times = {'allocation': [], 'topology': [], 'spreading': [], 'scheduling': [], 'decay_refresh': []}
        
        for key, result in integration_results.items():
            if key.startswith('scale_') and isinstance(result, dict):
                scales.append(result['scale']['atoms'])
                for phase, time_val in result['phase_times'].items():
                    if phase in phase_times:
                        phase_times[phase].append(time_val)
        
        if scales:
            plt.figure(figsize=(12, 6))
            
            bottom = np.zeros(len(scales))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, (phase, times) in enumerate(phase_times.items()):
                if times:
                    plt.bar(range(len(scales)), times, bottom=bottom, 
                           label=phase.replace('_', ' ').title(), color=colors[i % len(colors)])
                    bottom = np.add(bottom, times)
            
            plt.xlabel('System Scale (atoms)')
            plt.ylabel('Processing Time (seconds)')
            plt.title('ECAN Integration Performance by Phase')
            plt.xticks(range(len(scales)), scales)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'integration_performance.png'), dpi=150)
            plt.close()
    
    def _save_results(self):
        """Save complete results to JSON"""
        results_path = os.path.join(self.output_dir, "ecan_benchmark_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Complete results saved: {results_path}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for the report"""
        try:
            import psutil
            import platform
            
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except ImportError:
            return {
                'platform': 'Unknown',
                'python_version': 'Unknown',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }


def main():
    """Run the complete ECAN performance benchmark suite"""
    print("ECAN Attention Allocation Performance Benchmark")
    print("=" * 60)
    
    benchmark = ECANPerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)
    
    benchmarks = results['benchmarks']
    passed = 0
    total = 0
    
    for component, result in benchmarks.items():
        if 'performance_analysis' in result:
            analysis = result['performance_analysis']
            if 'overall_grade' in analysis:
                total += 1
                if analysis['overall_grade'] == 'PASS':
                    passed += 1
                status = "✅ PASS" if analysis['overall_grade'] == 'PASS' else "❌ FAIL"
                print(f"{component.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {passed}/{total} components passed")
    print(f"Success Rate: {passed/total*100:.1f}%" if total > 0 else "No results")
    
    return results


if __name__ == "__main__":
    main()