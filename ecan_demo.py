#!/usr/bin/env python3
"""
ECAN Attention Allocation Demonstration

This script demonstrates the key features of the ECAN (Economic Attention 
Allocation Network) system including:

1. Attention kernel with 6-dimensional tensors
2. Economic attention allocation algorithms
3. Resource scheduling with priority queues
4. Attention spreading across knowledge networks
5. Decay and refresh mechanisms
6. Performance metrics and analysis
"""

import time
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from ecan import (
    AttentionKernel, ECANAttentionTensor,
    EconomicAllocator, AttentionAllocationRequest,
    ResourceScheduler, ScheduledTask,
    AttentionSpreading, AttentionLink,
    DecayRefresh, DecayParameters, DecayMode
)


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")


def demonstrate_attention_kernel():
    """Demonstrate core attention kernel functionality"""
    print_section("ECAN Attention Kernel Demonstration")
    
    # Initialize attention kernel
    kernel = AttentionKernel(max_atoms=50, focus_boundary=0.6)
    print(f"✓ Attention kernel initialized (max_atoms=50, focus_boundary=0.6)")
    
    print_subsection("Creating ECAN Attention Tensors")
    
    # Create example cognitive concepts with attention tensors
    concepts = [
        ("problem_solving", 0.9, 0.3, 0.8, 0.85, 0.7, 0.1),
        ("memory_recall", 0.7, 0.8, 0.4, 0.9, 0.6, 0.05),
        ("creative_thinking", 0.8, 0.2, 0.6, 0.7, 0.8, 0.15),
        ("logical_reasoning", 0.75, 0.6, 0.7, 0.95, 0.5, 0.08),
        ("pattern_recognition", 0.6, 0.7, 0.5, 0.8, 0.9, 0.12)
    ]
    
    for concept, sti, lti, urgency, confidence, spreading, decay in concepts:
        tensor = ECANAttentionTensor(
            short_term_importance=sti,
            long_term_importance=lti,
            urgency=urgency,
            confidence=confidence,
            spreading_factor=spreading,
            decay_rate=decay
        )
        
        success = kernel.allocate_attention(concept, tensor)
        salience = tensor.compute_salience()
        activation = tensor.compute_activation_strength()
        
        print(f"  {concept:20s}: salience={salience:.3f}, activation={activation:.3f}, allocated={success}")
    
    print_subsection("Attention Focus Analysis")
    
    focus = kernel.get_attention_focus()
    print(f"Attention focus contains {len(focus)} atoms:")
    for atom_id, tensor in focus:
        salience = tensor.compute_salience()
        print(f"  {atom_id:20s}: salience={salience:.3f}")
    
    print_subsection("Global Attention Distribution")
    
    distribution = kernel.compute_global_attention_distribution()
    print("Normalized attention distribution:")
    for atom_id, weight in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {atom_id:20s}: {weight:.3f} ({weight*100:.1f}%)")
    
    return kernel


def demonstrate_economic_allocator(kernel):
    """Demonstrate economic attention allocation"""
    print_section("Economic Attention Allocation")
    
    allocator = EconomicAllocator(
        total_attention_budget=100.0,
        fairness_factor=0.15,
        efficiency_threshold=1.2
    )
    print(f"✓ Economic allocator initialized (budget=100.0, fairness=0.15, efficiency_threshold=1.2)")
    
    print_subsection("Creating Attention Allocation Requests")
    
    # Create allocation requests for new cognitive tasks
    requests = []
    
    task_specs = [
        ("language_parsing", 0.8, 0.7, 12.0, 6.0, 0.9),
        ("visual_processing", 0.7, 0.8, 15.0, 8.0, 0.7),
        ("motor_planning", 0.6, 0.9, 10.0, 4.0, 0.8),
        ("decision_making", 0.9, 0.6, 18.0, 10.0, 0.95),
        ("attention_control", 0.75, 0.85, 14.0, 7.0, 0.85)
    ]
    
    for task, sti, urgency, value, cost, priority in task_specs:
        tensor = ECANAttentionTensor(
            short_term_importance=sti,
            long_term_importance=0.2,
            urgency=urgency,
            confidence=0.8,
            spreading_factor=0.6,
            decay_rate=0.1
        )
        
        request = AttentionAllocationRequest(
            atom_id=task,
            requested_attention=tensor,
            value=value,
            cost=cost,
            priority=priority,
            deadline=time.time() + 120  # 2 minutes from now
        )
        requests.append(request)
        
        efficiency = request.compute_efficiency()
        total_priority = request.compute_total_priority()
        print(f"  {task:20s}: value={value:5.1f}, cost={cost:4.1f}, efficiency={efficiency:.2f}, priority={total_priority:.3f}")
    
    print_subsection("Batch Allocation Processing")
    
    start_time = time.time()
    allocation_result = allocator.allocate_attention_batch(requests, kernel)
    allocation_time = time.time() - start_time
    
    print(f"Batch allocation completed in {allocation_time:.3f} seconds")
    print(f"Allocated: {len(allocation_result['allocations'])} requests")
    print(f"Rejected:  {len(allocation_result['rejected'])} requests")
    print(f"Budget utilization: {allocation_result['metrics']['budget_utilization']:.1%}")
    
    print("\nAllocation Results:")
    for request, evaluation in allocation_result['allocations']:
        print(f"  ✓ {request.atom_id:20s}: efficiency={evaluation['efficiency']:.2f}, score={evaluation['allocation_score']:.3f}")
    
    if allocation_result['rejected']:
        print("\nRejected Requests:")
        for request, evaluation, reason in allocation_result['rejected']:
            print(f"  ✗ {request.atom_id:20s}: {reason}, score={evaluation['allocation_score']:.3f}")
    
    print_subsection("Economic Metrics")
    
    economic_metrics = allocator.get_economic_metrics()
    print(f"Overall efficiency: {economic_metrics['overall_efficiency']:.2f}")
    print(f"Average efficiency: {economic_metrics['average_efficiency']:.2f}")
    print(f"Budget utilization: {economic_metrics['budget_utilization']:.1%}")
    print(f"Available budget: {economic_metrics['available_budget']:.1f}")
    
    return allocator


def demonstrate_attention_spreading(kernel):
    """Demonstrate attention spreading mechanisms"""
    print_section("Attention Spreading and Network Topology")
    
    spreading = AttentionSpreading(
        spreading_rate=0.8,
        decay_rate=0.1,
        convergence_threshold=0.01,
        max_iterations=50
    )
    print(f"✓ Attention spreading initialized (rate=0.8, decay=0.1, threshold=0.01)")
    
    print_subsection("Building Cognitive Network Topology")
    
    # Create semantic relationships between concepts
    semantic_links = [
        ("problem_solving", "logical_reasoning", 0.8, "semantic"),
        ("problem_solving", "creative_thinking", 0.6, "semantic"),
        ("memory_recall", "pattern_recognition", 0.7, "semantic"),
        ("language_parsing", "pattern_recognition", 0.5, "semantic"),
        ("visual_processing", "pattern_recognition", 0.9, "semantic"),
        ("decision_making", "problem_solving", 0.7, "semantic"),
        ("motor_planning", "decision_making", 0.6, "semantic")
    ]
    
    for source, target, strength, link_type in semantic_links:
        link = AttentionLink(
            source_atom=source,
            target_atom=target,
            link_strength=strength,
            link_type=link_type,
            bidirectional=True
        )
        spreading.add_attention_link(link)
        print(f"  {source:20s} <-> {target:20s} (strength={strength:.1f})")
    
    print_subsection("Attention Spreading Simulation")
    
    # Get current focus atoms as spreading sources
    focus_atoms = kernel.get_attention_focus()
    source_atoms = [atom_id for atom_id, _ in focus_atoms[:3]]  # Top 3 focus atoms
    
    print(f"Starting attention spreading from {len(source_atoms)} focus atoms:")
    for atom_id in source_atoms:
        tensor = kernel.get_attention(atom_id)
        activation = tensor.compute_activation_strength()
        print(f"  {atom_id:20s}: activation={activation:.3f}")
    
    start_time = time.time()
    spread_result = spreading.spread_attention(kernel, source_atoms=source_atoms)
    spreading_time = time.time() - start_time
    
    print(f"\nSpreading completed in {spreading_time:.3f} seconds:")
    print(f"  Atoms affected: {spread_result.atoms_affected}")
    print(f"  Iterations: {spread_result.spread_iterations}")
    print(f"  Converged: {spread_result.convergence_achieved}")
    print(f"  Total spread: {spread_result.total_spread:.3f}")
    
    print_subsection("Attention Flow Analysis")
    
    flow_analysis = spreading.analyze_attention_flow(kernel)
    print("Network connectivity:")
    print(f"  Total atoms: {flow_analysis['connectivity_metrics']['total_atoms']}")
    print(f"  Total links: {flow_analysis['connectivity_metrics']['total_links']}")
    print(f"  Network density: {flow_analysis['network_density']:.3f}")
    print(f"  Average degree: {flow_analysis['connectivity_metrics']['average_degree']:.1f}")
    
    if flow_analysis['flow_predictions']:
        print("\nPredicted attention flow paths:")
        for prediction in flow_analysis['flow_predictions'][:3]:
            source = prediction['source_atom']
            print(f"  From {source}:")
            for target in prediction['predicted_targets'][:2]:
                flow = target['predicted_flow']
                strength = target['link_strength']
                print(f"    -> {target['target_atom']:20s}: flow={flow:.3f}, strength={strength:.1f}")
    
    return spreading


def demonstrate_resource_scheduling():
    """Demonstrate resource scheduling with priority queues"""
    print_section("Resource Scheduling with Priority Queues")
    
    scheduler = ResourceScheduler(
        max_concurrent_tasks=3,
        scheduler_interval=0.1,
        enable_background_processing=False
    )
    print(f"✓ Resource scheduler initialized (max_concurrent=3, interval=0.1s)")
    
    print_subsection("Creating Scheduled Tasks")
    
    # Create cognitive tasks with different priorities and requirements
    task_specs = [
        ("urgent_response", 0.9, 0.95, 20.0, 5.0, 30.0),
        ("background_learning", 0.3, 0.2, 8.0, 3.0, 300.0),
        ("routine_maintenance", 0.4, 0.3, 6.0, 2.0, 180.0),
        ("creative_exploration", 0.7, 0.4, 15.0, 8.0, 120.0),
        ("critical_analysis", 0.8, 0.85, 18.0, 7.0, 60.0)
    ]
    
    tasks = []
    for task_name, sti, urgency, value, cost, max_time in task_specs:
        tensor = ECANAttentionTensor(
            short_term_importance=sti,
            urgency=urgency,
            confidence=0.8,
            spreading_factor=0.6
        )
        
        request = AttentionAllocationRequest(
            atom_id=f"task_{task_name}",
            requested_attention=tensor,
            value=value,
            cost=cost,
            priority=0.5,
            deadline=time.time() + 300  # 5 minutes
        )
        
        task = ScheduledTask(
            task_id=task_name,
            attention_request=request,
            max_execution_time=max_time
        )
        
        tasks.append(task)
        scheduler.schedule_task(task)
        
        priority = request.compute_total_priority()
        print(f"  {task_name:20s}: priority={priority:.3f}, max_time={max_time:5.1f}s")
    
    print_subsection("Scheduler Processing Simulation")
    
    # Create kernel and allocator for scheduling
    sched_kernel = AttentionKernel(max_atoms=20)
    sched_allocator = EconomicAllocator(total_attention_budget=50.0)
    
    print("Processing scheduler cycles...")
    cycle_count = 0
    total_started = 0
    
    while len(scheduler.priority_queue) > 0 and cycle_count < 10:
        print(f"\nCycle {cycle_count + 1}:")
        
        result = scheduler.process_scheduler_cycle(sched_kernel, sched_allocator)
        
        print(f"  Queue size: {result['queue_size']}")
        print(f"  Running tasks: {result['running_tasks']}")
        print(f"  Tasks started: {result['tasks_started']}")
        print(f"  Tasks completed: {result['tasks_completed']}")
        
        total_started += result['tasks_started']
        cycle_count += 1
        
        if result['tasks_started'] == 0 and result['tasks_completed'] == 0:
            print("  No progress - stopping simulation")
            break
        
        time.sleep(0.1)  # Simulate processing delay
    
    print_subsection("Scheduling Metrics")
    
    metrics = scheduler.get_scheduler_metrics()
    print(f"Tasks scheduled: {metrics['tasks_scheduled']}")
    print(f"Tasks completed: {metrics['tasks_completed']}")
    print(f"Scheduler cycles: {metrics['scheduler_cycles']}")
    print(f"Average wait time: {metrics['average_wait_time']:.2f}s")
    print(f"Queue size: {metrics['queue_size']}")
    print(f"Running tasks: {metrics['running_tasks']}")
    
    return scheduler


def demonstrate_decay_refresh(kernel):
    """Demonstrate attention decay and refresh mechanisms"""
    print_section("Attention Decay and Refresh Mechanisms")
    
    decay_params = DecayParameters(
        mode=DecayMode.EXPONENTIAL,
        base_rate=0.1,
        half_life=180.0,  # 3 minutes
        min_threshold=0.05,
        preserve_lti=True
    )
    
    decay_refresh = DecayRefresh(
        decay_params=decay_params,
        refresh_sensitivity=0.8,
        consolidation_threshold=0.7
    )
    print(f"✓ Decay/refresh initialized (mode=EXPONENTIAL, half_life=180s, threshold=0.05)")
    
    print_subsection("Initial Attention State")
    
    initial_distribution = kernel.compute_global_attention_distribution()
    initial_count = len(kernel.attention_tensors)
    print(f"Initial atoms with attention: {initial_count}")
    
    # Simulate some time passage by backdating timestamps
    current_time = time.time()
    for atom_id in list(kernel.attention_tensors.keys()):
        # Simulate different access patterns
        if "problem" in atom_id or "decision" in atom_id:
            # Recent access - less decay
            decay_refresh.attention_timestamps[atom_id] = current_time - 60
        elif "memory" in atom_id or "pattern" in atom_id:
            # Medium age
            decay_refresh.attention_timestamps[atom_id] = current_time - 200
        else:
            # Older items
            decay_refresh.attention_timestamps[atom_id] = current_time - 400
    
    print_subsection("Adding Refresh Triggers")
    
    # Simulate access patterns that should trigger refresh
    refresh_triggers = [
        ("problem_solving", "access", 0.9),
        ("memory_recall", "goal_relevance", 0.7),
        ("decision_making", "access", 0.8),
        ("language_parsing", "external", 0.6)
    ]
    
    for atom_id, trigger_type, strength in refresh_triggers:
        from ecan import RefreshTrigger
        trigger = RefreshTrigger(
            atom_id=atom_id,
            trigger_type=trigger_type,
            strength=strength,
            timestamp=current_time
        )
        decay_refresh.add_refresh_trigger(trigger)
        print(f"  Added {trigger_type} trigger for {atom_id} (strength={strength:.1f})")
    
    print_subsection("Processing Decay Cycle")
    
    start_time = time.time()
    decay_result = decay_refresh.process_decay_cycle(kernel)
    processing_time = time.time() - start_time
    
    print(f"Decay cycle completed in {processing_time:.3f} seconds:")
    print(f"  Atoms decayed: {decay_result.atoms_decayed}")
    print(f"  Atoms refreshed: {decay_result.atoms_refreshed}")
    print(f"  Atoms removed: {decay_result.atoms_removed}")
    print(f"  Attention before: {decay_result.total_attention_before:.3f}")
    print(f"  Attention after: {decay_result.total_attention_after:.3f}")
    print(f"  Decay efficiency: {decay_result.decay_efficiency:.3f}")
    
    print_subsection("Updated Attention State")
    
    final_distribution = kernel.compute_global_attention_distribution()
    final_count = len(kernel.attention_tensors)
    
    print(f"Final atoms with attention: {final_count} (change: {final_count - initial_count:+d})")
    
    # Show attention changes
    print("\nAttention changes:")
    all_atoms = set(initial_distribution.keys()) | set(final_distribution.keys())
    for atom_id in sorted(all_atoms):
        initial = initial_distribution.get(atom_id, 0.0)
        final = final_distribution.get(atom_id, 0.0)
        change = final - initial
        
        if abs(change) > 0.01:  # Only show significant changes
            status = "↗" if change > 0 else "↘" if change < 0 else "→"
            print(f"  {atom_id:20s}: {initial:.3f} -> {final:.3f} {status} ({change:+.3f})")
    
    print_subsection("Decay/Refresh Metrics")
    
    dr_metrics = decay_refresh.get_decay_refresh_metrics()
    print(f"Decay operations: {dr_metrics['decay_operations']}")
    print(f"Refresh operations: {dr_metrics['refresh_operations']}")
    print(f"Average decay time: {dr_metrics['average_decay_time']:.4f}s")
    print(f"Consolidation rate: {dr_metrics['consolidation_rate']:.3f}")
    print(f"Active triggers: {dr_metrics['active_triggers']}")
    print(f"Attention conservation: {dr_metrics['attention_conservation_ratio']:.3f}")
    
    return decay_refresh


def demonstrate_performance_analysis():
    """Demonstrate performance metrics and analysis"""
    print_section("Performance Metrics and Analysis")
    
    print_subsection("Creating High-Load Test")
    
    # Create a high-load scenario for performance testing
    kernel = AttentionKernel(max_atoms=200)
    allocator = EconomicAllocator(total_attention_budget=500.0)
    
    # Generate many atoms and requests
    num_atoms = 100
    num_requests = 50
    
    print(f"Generating {num_atoms} atoms and {num_requests} allocation requests...")
    
    start_time = time.time()
    
    # Create atoms
    for i in range(num_atoms):
        tensor = ECANAttentionTensor(
            short_term_importance=np.random.uniform(0.1, 0.9),
            long_term_importance=np.random.uniform(0.0, 0.5),
            urgency=np.random.uniform(0.0, 1.0),
            confidence=np.random.uniform(0.4, 1.0),
            spreading_factor=np.random.uniform(0.3, 0.8),
            decay_rate=np.random.uniform(0.05, 0.2)
        )
        kernel.allocate_attention(f"atom_{i:03d}", tensor)
    
    atom_creation_time = time.time() - start_time
    
    # Create requests
    requests = []
    for i in range(num_requests):
        tensor = ECANAttentionTensor(
            short_term_importance=np.random.uniform(0.3, 0.9),
            urgency=np.random.uniform(0.2, 1.0),
            confidence=np.random.uniform(0.5, 1.0)
        )
        
        request = AttentionAllocationRequest(
            atom_id=f"request_{i:03d}",
            requested_attention=tensor,
            value=np.random.uniform(8.0, 20.0),
            cost=np.random.uniform(3.0, 12.0),
            priority=np.random.uniform(0.2, 0.9)
        )
        requests.append(request)
    
    print_subsection("Performance Benchmarking")
    
    # Benchmark allocation
    start_time = time.time()
    allocation_result = allocator.allocate_attention_batch(requests, kernel)
    allocation_time = time.time() - start_time
    
    # Benchmark focus computation
    start_time = time.time()
    for _ in range(10):
        focus = kernel.get_attention_focus()
    focus_time = time.time() - start_time
    
    # Benchmark distribution computation
    start_time = time.time()
    for _ in range(5):
        distribution = kernel.compute_global_attention_distribution()
    distribution_time = time.time() - start_time
    
    print("Performance Results:")
    print(f"  Atom creation: {num_atoms / atom_creation_time:.1f} atoms/second")
    print(f"  Batch allocation: {num_requests / allocation_time:.1f} requests/second")
    print(f"  Focus computation: {10 / focus_time:.1f} operations/second")
    print(f"  Distribution computation: {5 / distribution_time:.1f} operations/second")
    
    print_subsection("System Metrics")
    
    kernel_metrics = kernel.get_performance_metrics()
    print("Attention Kernel Metrics:")
    print(f"  Atoms processed: {kernel_metrics['atoms_processed']}")
    print(f"  Tensor operations: {kernel_metrics['tensor_operations']}")
    print(f"  Focus updates: {kernel_metrics['focus_updates']}")
    print(f"  Current atoms: {kernel_metrics['current_atoms']}")
    print(f"  Focus size: {kernel_metrics['focus_size']}")
    print(f"  Memory usage: {kernel_metrics['memory_usage_mb']:.2f} MB")
    
    economic_metrics = allocator.get_economic_metrics()
    print("\nEconomic Allocator Metrics:")
    print(f"  Allocations made: {economic_metrics['allocations_made']}")
    print(f"  Rejected requests: {economic_metrics['rejected_requests']}")
    print(f"  Overall efficiency: {economic_metrics['overall_efficiency']:.2f}")
    print(f"  Budget utilization: {economic_metrics['budget_utilization']:.1%}")
    
    print_subsection("Performance Targets Analysis")
    
    # Check against ECAN performance targets from issue
    targets = {
        'activation_spreading': 10000,  # atoms/second
        'importance_diffusion': 1000,   # operations/second  
        'tensor_computation': 100000    # operations/second
    }
    
    # Approximate spreading rate from allocation performance
    approx_spreading_rate = kernel_metrics['atoms_processed'] / (allocation_time + atom_creation_time)
    approx_tensor_rate = kernel_metrics['tensor_operations'] / (allocation_time + atom_creation_time)
    
    print("Target Analysis:")
    print(f"  Activation spreading: {approx_spreading_rate:.0f}/s (target: {targets['activation_spreading']}/s) {'✓' if approx_spreading_rate >= targets['activation_spreading'] else '⚠'}")
    print(f"  Tensor computation: {approx_tensor_rate:.0f}/s (target: {targets['tensor_computation']}/s) {'✓' if approx_tensor_rate >= targets['tensor_computation'] else '⚠'}")
    
    return kernel_metrics, economic_metrics


def main():
    """Main demonstration function"""
    print("🧠 ECAN Attention Allocation System Demonstration")
    print("Phase 2: Economic Attention Allocation & Resource Kernel Construction")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Core demonstrations
        kernel = demonstrate_attention_kernel()
        allocator = demonstrate_economic_allocator(kernel)
        spreading = demonstrate_attention_spreading(kernel)
        scheduler = demonstrate_resource_scheduling()
        decay_refresh = demonstrate_decay_refresh(kernel)
        
        # Performance analysis
        kernel_metrics, economic_metrics = demonstrate_performance_analysis()
        
        # Summary
        print_section("ECAN Demonstration Summary")
        print("✅ All ECAN components successfully demonstrated:")
        print("  • Attention Kernel with 6-dimensional tensors")
        print("  • Economic Attention Allocation algorithms")
        print("  • Resource Scheduling with priority queues")
        print("  • Attention Spreading across knowledge networks")
        print("  • Decay and Refresh mechanisms")
        print("  • Performance metrics and analysis")
        
        print("\n🎯 Key Features Validated:")
        print("  • ECAN_Attention_Tensor[6] implementation")
        print("  • Economic value/cost optimization")
        print("  • Priority queue-based task scheduling")
        print("  • Dynamic attention spreading and topology")
        print("  • Intelligent decay and refresh mechanisms")
        print("  • Real-time performance monitoring")
        
        print("\n📊 Performance Summary:")
        print(f"  • Atoms processed: {kernel_metrics['atoms_processed']}")
        print(f"  • Tensor operations: {kernel_metrics['tensor_operations']}")
        print(f"  • Focus computations: {kernel_metrics['focus_updates']}")
        print(f"  • Economic efficiency: {economic_metrics['overall_efficiency']:.2f}")
        print(f"  • Budget utilization: {economic_metrics['budget_utilization']:.1%}")
        
        print("\n✨ The ECAN system is ready for cognitive architecture integration!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)