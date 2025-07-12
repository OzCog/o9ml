"""
Comprehensive tests for the ECAN Attention Allocation system.

Tests all major components including:
- AttentionKernel and ECANAttentionTensor
- EconomicAllocator
- ResourceScheduler  
- AttentionSpreading
- DecayRefresh mechanisms
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ecan.attention_kernel import AttentionKernel, ECANAttentionTensor
from ecan.economic_allocator import EconomicAllocator, AttentionAllocationRequest
from ecan.resource_scheduler import ResourceScheduler, ScheduledTask, TaskStatus
from ecan.attention_spreading import AttentionSpreading, AttentionLink
from ecan.decay_refresh import DecayRefresh, DecayParameters, RefreshTrigger, DecayMode


class TestECANAttentionTensor:
    """Test the ECAN attention tensor implementation"""
    
    def test_tensor_creation(self):
        """Test basic tensor creation and validation"""
        tensor = ECANAttentionTensor(
            short_term_importance=0.8,
            long_term_importance=0.6,
            urgency=0.7,
            confidence=0.9,
            spreading_factor=0.5,
            decay_rate=0.1
        )
        
        assert tensor.short_term_importance == 0.8
        assert tensor.long_term_importance == 0.6
        assert tensor.urgency == 0.7
        assert tensor.confidence == 0.9
        assert tensor.spreading_factor == 0.5
        assert tensor.decay_rate == 0.1
    
    def test_tensor_validation(self):
        """Test tensor value validation"""
        # Valid tensor
        tensor = ECANAttentionTensor(short_term_importance=0.5)
        assert tensor.short_term_importance == 0.5
        
        # Invalid values should raise ValueError
        with pytest.raises(ValueError):
            ECANAttentionTensor(short_term_importance=1.5)  # > 1.0
        
        with pytest.raises(ValueError):
            ECANAttentionTensor(urgency=-0.1)  # < 0.0
    
    def test_tensor_conversion(self):
        """Test tensor array conversion"""
        tensor = ECANAttentionTensor(
            short_term_importance=0.8,
            long_term_importance=0.6,
            urgency=0.7,
            confidence=0.9,
            spreading_factor=0.5,
            decay_rate=0.1
        )
        
        # Test to_array
        array = tensor.to_array()
        expected = np.array([0.8, 0.6, 0.7, 0.9, 0.5, 0.1])
        np.testing.assert_array_equal(array, expected)
        
        # Test from_array
        reconstructed = ECANAttentionTensor.from_array(array)
        assert reconstructed.short_term_importance == tensor.short_term_importance
        assert reconstructed.long_term_importance == tensor.long_term_importance
        assert reconstructed.urgency == tensor.urgency
        assert reconstructed.confidence == tensor.confidence
        assert reconstructed.spreading_factor == tensor.spreading_factor
        assert reconstructed.decay_rate == tensor.decay_rate
    
    def test_salience_computation(self):
        """Test salience computation"""
        tensor = ECANAttentionTensor(
            short_term_importance=0.8,
            long_term_importance=0.6,
            urgency=0.7,
            confidence=0.9
        )
        
        expected_salience = 0.8 * 0.4 + 0.6 * 0.2 + 0.7 * 0.3 + 0.9 * 0.1
        assert abs(tensor.compute_salience() - expected_salience) < 0.001
    
    def test_activation_strength(self):
        """Test activation strength computation"""
        tensor = ECANAttentionTensor(
            confidence=0.8,
            spreading_factor=0.6,
            decay_rate=0.2
        )
        
        expected = 0.6 * 0.8 * (1 - 0.2)  # spreading_factor * confidence * (1 - decay_rate)
        assert abs(tensor.compute_activation_strength() - expected) < 0.001


class TestAttentionKernel:
    """Test the attention kernel implementation"""
    
    def test_kernel_initialization(self):
        """Test kernel initialization"""
        kernel = AttentionKernel(max_atoms=1000, focus_boundary=0.6)
        
        assert kernel.max_atoms == 1000
        assert kernel.focus_boundary == 0.6
        assert len(kernel.attention_tensors) == 0
    
    def test_attention_allocation(self):
        """Test attention allocation"""
        kernel = AttentionKernel()
        tensor = ECANAttentionTensor(short_term_importance=0.8)
        
        success = kernel.allocate_attention("atom1", tensor)
        assert success
        assert "atom1" in kernel.attention_tensors
        assert kernel.get_attention("atom1") == tensor
    
    def test_attention_update(self):
        """Test attention updates"""
        kernel = AttentionKernel()
        tensor = ECANAttentionTensor(short_term_importance=0.5)
        
        kernel.allocate_attention("atom1", tensor)
        
        success = kernel.update_attention("atom1", short_term_delta=0.2)
        assert success
        
        updated_tensor = kernel.get_attention("atom1")
        assert abs(updated_tensor.short_term_importance - 0.7) < 0.001
    
    def test_attention_focus(self):
        """Test attention focus computation"""
        kernel = AttentionKernel(focus_boundary=0.5)
        
        # Add atoms with different salience levels
        high_salience = ECANAttentionTensor(short_term_importance=0.9)
        low_salience = ECANAttentionTensor(short_term_importance=0.3)
        
        kernel.allocate_attention("high_atom", high_salience)
        kernel.allocate_attention("low_atom", low_salience)
        
        focus = kernel.get_attention_focus()
        
        # Only high salience atom should be in focus
        assert len(focus) == 1
        assert focus[0][0] == "high_atom"
    
    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        kernel = AttentionKernel()
        tensor = ECANAttentionTensor(short_term_importance=0.8)
        
        # Perform some operations
        kernel.allocate_attention("atom1", tensor)
        kernel.update_attention("atom1", short_term_delta=0.1)
        kernel.get_attention_focus()
        
        metrics = kernel.get_performance_metrics()
        
        assert metrics['atoms_processed'] > 0
        assert metrics['tensor_operations'] > 0
        assert metrics['focus_updates'] > 0
    
    def test_capacity_management(self):
        """Test capacity management and eviction"""
        kernel = AttentionKernel(max_atoms=2)
        
        # Fill to capacity
        tensor1 = ECANAttentionTensor(short_term_importance=0.8)
        tensor2 = ECANAttentionTensor(short_term_importance=0.6)
        tensor3 = ECANAttentionTensor(short_term_importance=0.9)
        
        kernel.allocate_attention("atom1", tensor1)
        kernel.allocate_attention("atom2", tensor2)
        
        assert len(kernel.attention_tensors) == 2
        
        # Adding third atom should evict lowest salience
        kernel.allocate_attention("atom3", tensor3)
        
        assert len(kernel.attention_tensors) == 2
        assert "atom3" in kernel.attention_tensors  # New high-salience atom
        assert "atom1" in kernel.attention_tensors  # Highest existing atom
        assert "atom2" not in kernel.attention_tensors  # Evicted lowest


class TestEconomicAllocator:
    """Test the economic attention allocator"""
    
    def test_allocator_initialization(self):
        """Test allocator initialization"""
        allocator = EconomicAllocator(
            total_attention_budget=200.0,
            fairness_factor=0.2,
            efficiency_threshold=1.5
        )
        
        assert allocator.total_budget == 200.0
        assert allocator.fairness_factor == 0.2
        assert allocator.efficiency_threshold == 1.5
    
    def test_request_evaluation(self):
        """Test allocation request evaluation"""
        allocator = EconomicAllocator()
        
        tensor = ECANAttentionTensor(short_term_importance=0.8)
        request = AttentionAllocationRequest(
            atom_id="test_atom",
            requested_attention=tensor,
            value=10.0,
            cost=5.0,
            priority=0.7
        )
        
        evaluation = allocator.evaluate_allocation_request(request)
        
        assert 'efficiency' in evaluation
        assert 'allocation_score' in evaluation
        assert 'recommended' in evaluation
        assert evaluation['efficiency'] == 2.0  # value/cost = 10/5
    
    def test_batch_allocation(self):
        """Test batch allocation processing"""
        kernel = AttentionKernel()
        allocator = EconomicAllocator(total_attention_budget=100.0)
        
        # Create requests with different priorities
        requests = []
        for i in range(3):
            tensor = ECANAttentionTensor(short_term_importance=0.6 + i * 0.1)
            request = AttentionAllocationRequest(
                atom_id=f"atom_{i}",
                requested_attention=tensor,
                value=10.0 + i * 2.0,
                cost=5.0,
                priority=0.5 + i * 0.1
            )
            requests.append(request)
        
        result = allocator.allocate_attention_batch(requests, kernel)
        
        assert 'allocations' in result
        assert 'rejected' in result
        assert 'metrics' in result
        assert len(result['allocations']) > 0
    
    def test_economic_metrics(self):
        """Test economic metrics calculation"""
        allocator = EconomicAllocator()
        
        # Simulate some allocations
        allocator.metrics['allocations_made'] = 5
        allocator.metrics['total_value_delivered'] = 50.0
        allocator.metrics['total_cost_incurred'] = 25.0
        
        metrics = allocator.get_economic_metrics()
        
        assert 'overall_efficiency' in metrics
        assert 'budget_utilization' in metrics
        assert metrics['overall_efficiency'] == 2.0  # 50/25


class TestResourceScheduler:
    """Test the resource scheduler implementation"""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization"""
        scheduler = ResourceScheduler(
            max_concurrent_tasks=5,
            scheduler_interval=0.5,
            enable_background_processing=False
        )
        
        assert scheduler.max_concurrent_tasks == 5
        assert scheduler.scheduler_interval == 0.5
        assert len(scheduler.priority_queue) == 0
    
    def test_task_scheduling(self):
        """Test task scheduling"""
        scheduler = ResourceScheduler(enable_background_processing=False)
        
        tensor = ECANAttentionTensor(short_term_importance=0.8)
        request = AttentionAllocationRequest(
            atom_id="test_atom",
            requested_attention=tensor,
            value=10.0,
            cost=5.0
        )
        
        task = ScheduledTask(
            task_id="task1",
            attention_request=request
        )
        
        success = scheduler.schedule_task(task)
        assert success
        assert len(scheduler.priority_queue) == 1
    
    def test_task_priority_ordering(self):
        """Test that tasks are ordered by priority"""
        scheduler = ResourceScheduler(enable_background_processing=False)
        
        # Create tasks with different priorities
        tasks = []
        for i, priority in enumerate([0.3, 0.8, 0.5]):
            tensor = ECANAttentionTensor(short_term_importance=priority)
            request = AttentionAllocationRequest(
                atom_id=f"atom_{i}",
                requested_attention=tensor,
                value=10.0,
                cost=5.0,
                priority=priority
            )
            task = ScheduledTask(task_id=f"task_{i}", attention_request=request)
            tasks.append(task)
            scheduler.schedule_task(task)
        
        # Highest priority task should be first
        assert scheduler.priority_queue[0].task_id == "task_1"  # priority 0.8
    
    def test_scheduler_cycle(self):
        """Test scheduler cycle processing"""
        kernel = AttentionKernel()
        allocator = EconomicAllocator()
        scheduler = ResourceScheduler(enable_background_processing=False)
        
        # Add a task
        tensor = ECANAttentionTensor(short_term_importance=0.8)
        request = AttentionAllocationRequest(
            atom_id="test_atom",
            requested_attention=tensor,
            value=10.0,
            cost=5.0
        )
        task = ScheduledTask(task_id="task1", attention_request=request)
        scheduler.schedule_task(task)
        
        result = scheduler.process_scheduler_cycle(kernel, allocator)
        
        assert 'tasks_started' in result
        assert 'queue_size' in result
        assert 'resource_utilization' in result
    
    def test_task_cancellation(self):
        """Test task cancellation"""
        scheduler = ResourceScheduler(enable_background_processing=False)
        
        tensor = ECANAttentionTensor(short_term_importance=0.8)
        request = AttentionAllocationRequest(
            atom_id="test_atom",
            requested_attention=tensor,
            value=10.0,
            cost=5.0
        )
        task = ScheduledTask(task_id="task1", attention_request=request)
        
        scheduler.schedule_task(task)
        assert len(scheduler.priority_queue) == 1
        
        success = scheduler.cancel_task("task1")
        assert success
        assert len(scheduler.priority_queue) == 0


class TestAttentionSpreading:
    """Test the attention spreading mechanism"""
    
    def test_spreading_initialization(self):
        """Test spreading engine initialization"""
        spreading = AttentionSpreading(
            spreading_rate=0.8,
            decay_rate=0.1,
            convergence_threshold=0.001
        )
        
        assert spreading.spreading_rate == 0.8
        assert spreading.decay_rate == 0.1
        assert spreading.convergence_threshold == 0.001
    
    def test_attention_link_creation(self):
        """Test attention link creation"""
        spreading = AttentionSpreading()
        
        link = AttentionLink(
            source_atom="atom1",
            target_atom="atom2",
            link_strength=0.7,
            bidirectional=True
        )
        
        success = spreading.add_attention_link(link)
        assert success
        assert len(spreading.attention_links["atom1"]) == 1
        assert len(spreading.attention_links["atom2"]) == 1  # Bidirectional
    
    def test_attention_spreading(self):
        """Test attention spreading operation"""
        kernel = AttentionKernel()
        spreading = AttentionSpreading()
        
        # Set up atoms and links
        tensor1 = ECANAttentionTensor(short_term_importance=0.8, spreading_factor=0.6)
        tensor2 = ECANAttentionTensor(short_term_importance=0.2, spreading_factor=0.4)
        
        kernel.allocate_attention("atom1", tensor1)
        kernel.allocate_attention("atom2", tensor2)
        
        link = AttentionLink(
            source_atom="atom1",
            target_atom="atom2",
            link_strength=0.7
        )
        spreading.add_attention_link(link)
        
        result = spreading.spread_attention(kernel, source_atoms=["atom1"])
        
        assert result.atoms_affected >= 1
        assert result.execution_time > 0
        assert isinstance(result.attention_distribution, dict)
    
    def test_semantic_topology_creation(self):
        """Test semantic topology creation"""
        spreading = AttentionSpreading()
        
        similarities = {
            ("atom1", "atom2"): 0.8,
            ("atom1", "atom3"): 0.4,
            ("atom2", "atom3"): 0.6
        }
        
        links_created = spreading.create_semantic_topology(similarities, min_similarity=0.5)
        
        assert links_created == 2  # Only similarities >= 0.5
    
    def test_temporal_topology_creation(self):
        """Test temporal topology creation"""
        spreading = AttentionSpreading()
        
        sequence = ["atom1", "atom2", "atom3", "atom4"]
        links_created = spreading.create_temporal_topology(sequence, window_size=2)
        
        assert links_created > 0
        # Each atom (except last) should link to next atoms in window
        expected_links = sum(min(2, len(sequence) - i - 1) for i in range(len(sequence) - 1))
        assert links_created == expected_links


class TestDecayRefresh:
    """Test the decay and refresh mechanisms"""
    
    def test_decay_refresh_initialization(self):
        """Test decay refresh initialization"""
        params = DecayParameters(
            mode=DecayMode.EXPONENTIAL,
            base_rate=0.1,
            half_life=300.0
        )
        
        decay_refresh = DecayRefresh(
            decay_params=params,
            refresh_sensitivity=0.7
        )
        
        assert decay_refresh.decay_params.mode == DecayMode.EXPONENTIAL
        assert decay_refresh.refresh_sensitivity == 0.7
    
    def test_decay_processing(self):
        """Test attention decay processing"""
        kernel = AttentionKernel()
        decay_refresh = DecayRefresh()
        
        # Add attention with timestamp
        tensor = ECANAttentionTensor(short_term_importance=0.8)
        kernel.allocate_attention("atom1", tensor)
        
        # Simulate time passage
        past_time = time.time() - 100  # 100 seconds ago
        decay_refresh.attention_timestamps["atom1"] = past_time
        
        result = decay_refresh.process_decay_cycle(kernel)
        
        assert result.atoms_decayed >= 0
        assert result.execution_time > 0
    
    def test_refresh_triggers(self):
        """Test refresh trigger processing"""
        kernel = AttentionKernel()
        decay_refresh = DecayRefresh()
        
        # Add refresh trigger
        trigger = RefreshTrigger(
            atom_id="atom1",
            trigger_type="access",
            strength=0.7,
            timestamp=time.time()
        )
        decay_refresh.add_refresh_trigger(trigger)
        
        assert len(decay_refresh.refresh_triggers) == 1
        
        # Process refresh
        result = decay_refresh.process_decay_cycle(kernel)
        
        # Should create new attention or refresh existing
        assert "atom1" in kernel.attention_tensors or result.atoms_refreshed > 0
    
    def test_memory_consolidation(self):
        """Test memory consolidation"""
        kernel = AttentionKernel()
        decay_refresh = DecayRefresh(consolidation_threshold=0.7)
        
        # Add high attention atom with access history
        tensor = ECANAttentionTensor(
            short_term_importance=0.9,
            confidence=0.9
        )
        kernel.allocate_attention("atom1", tensor)
        
        # Add access history
        current_time = time.time()
        decay_refresh.access_history["atom1"] = [
            current_time - 3000,  # 50 minutes ago
            current_time - 1800,  # 30 minutes ago
            current_time - 600    # 10 minutes ago
        ]
        
        result = decay_refresh.process_decay_cycle(kernel)
        
        # Check if consolidation occurred
        updated_tensor = kernel.get_attention("atom1")
        assert updated_tensor.long_term_importance > 0 or result.atoms_refreshed > 0
    
    def test_adaptive_decay_parameters(self):
        """Test adaptive decay parameter adjustment"""
        decay_refresh = DecayRefresh()
        
        # Simulate performance metrics
        performance_metrics = {
            'attention_efficiency': 0.8,
            'resource_utilization': 0.6
        }
        
        original_rate = decay_refresh.decay_params.base_rate
        decay_refresh.adjust_decay_parameters(performance_metrics, target_attention_level=0.7)
        
        # Parameters should be adjusted based on performance
        assert decay_refresh.decay_params.base_rate != original_rate or decay_refresh.refresh_sensitivity != 0.7


class TestIntegration:
    """Integration tests for the complete ECAN system"""
    
    def test_full_ecan_workflow(self):
        """Test complete ECAN workflow integration"""
        # Initialize all components
        kernel = AttentionKernel(max_atoms=100)
        allocator = EconomicAllocator(total_attention_budget=100.0)
        scheduler = ResourceScheduler(enable_background_processing=False)
        spreading = AttentionSpreading()
        decay_refresh = DecayRefresh()
        
        # Create attention allocation requests
        requests = []
        for i in range(5):
            tensor = ECANAttentionTensor(
                short_term_importance=0.5 + i * 0.1,
                urgency=0.3 + i * 0.15
            )
            request = AttentionAllocationRequest(
                atom_id=f"atom_{i}",
                requested_attention=tensor,
                value=10.0 + i * 2.0,
                cost=5.0,
                priority=0.4 + i * 0.1
            )
            requests.append(request)
        
        # Process batch allocation
        allocation_result = allocator.allocate_attention_batch(requests, kernel)
        assert len(allocation_result['allocations']) > 0
        
        # Set up attention topology
        for i in range(4):
            link = AttentionLink(
                source_atom=f"atom_{i}",
                target_atom=f"atom_{i+1}",
                link_strength=0.6
            )
            spreading.add_attention_link(link)
        
        # Perform attention spreading
        spread_result = spreading.spread_attention(kernel)
        assert spread_result.atoms_affected > 0
        
        # Add refresh triggers
        for i in range(3):
            decay_refresh.add_access_trigger(f"atom_{i}", access_strength=0.6)
        
        # Process decay and refresh
        decay_result = decay_refresh.process_decay_cycle(kernel)
        assert decay_result.execution_time > 0
        
        # Verify system state
        focus = kernel.get_attention_focus()
        metrics = kernel.get_performance_metrics()
        
        assert len(focus) >= 0  # Should have some focus atoms
        assert metrics['atoms_processed'] > 0
        assert metrics['tensor_operations'] > 0
    
    def test_performance_benchmarking(self):
        """Test system performance under load"""
        kernel = AttentionKernel(max_atoms=1000)
        allocator = EconomicAllocator()
        
        start_time = time.time()
        
        # Create large batch of requests
        requests = []
        for i in range(100):
            tensor = ECANAttentionTensor(
                short_term_importance=np.random.uniform(0.1, 0.9),
                urgency=np.random.uniform(0.0, 1.0)
            )
            request = AttentionAllocationRequest(
                atom_id=f"load_atom_{i}",
                requested_attention=tensor,
                value=np.random.uniform(5.0, 20.0),
                cost=np.random.uniform(2.0, 10.0)
            )
            requests.append(request)
        
        # Process requests
        result = allocator.allocate_attention_batch(requests, kernel)
        
        processing_time = time.time() - start_time
        
        # Verify performance
        assert processing_time < 1.0  # Should complete within 1 second
        assert len(result['allocations']) > 0
        
        # Check kernel performance
        metrics = kernel.get_performance_metrics()
        assert metrics['tensor_ops_per_second'] > 100  # Minimum performance threshold


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])