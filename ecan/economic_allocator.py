"""
Economic Attention Allocator

Implements economic attention allocation algorithms that optimize cognitive 
resource distribution based on value/cost analysis and market dynamics.

Key features:
- Value/cost optimization for attention allocation
- Economic efficiency metrics and analysis  
- Market-based attention dynamics
- Resource utilization optimization
- Fairness and equity considerations
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from queue import PriorityQueue
import logging
from .attention_kernel import ECANAttentionTensor, AttentionKernel

logger = logging.getLogger(__name__)


@dataclass
class AttentionAllocationRequest:
    """Request for attention allocation with economic parameters"""
    atom_id: str
    requested_attention: ECANAttentionTensor
    value: float  # Expected value/utility from allocation
    cost: float   # Resource cost of allocation
    deadline: Optional[float] = None  # Optional deadline (timestamp)
    priority: float = 0.5  # Base priority [0.0, 1.0]
    
    def compute_efficiency(self) -> float:
        """Compute value-to-cost efficiency ratio"""
        return self.value / max(self.cost, 0.001)  # Avoid division by zero
    
    def compute_urgency_factor(self) -> float:
        """Compute urgency factor based on deadline"""
        if self.deadline is None:
            return 1.0
        
        current_time = time.time()
        time_remaining = self.deadline - current_time
        
        if time_remaining <= 0:
            return 10.0  # Very urgent, past deadline
        elif time_remaining < 60:  # Less than 1 minute
            return 5.0
        elif time_remaining < 300:  # Less than 5 minutes
            return 2.0
        else:
            return 1.0
    
    def compute_total_priority(self) -> float:
        """Compute total priority considering all factors"""
        efficiency = self.compute_efficiency()
        urgency = self.compute_urgency_factor()
        salience = self.requested_attention.compute_salience()
        
        # Weighted combination of factors
        total_priority = (
            efficiency * 0.3 +
            urgency * 0.25 +
            salience * 0.25 +
            self.priority * 0.2
        )
        
        return total_priority


class EconomicAllocator:
    """
    Economic attention allocator implementing market-based resource allocation.
    
    Uses economic principles to optimize attention allocation across cognitive
    resources, considering value, cost, efficiency, and fairness.
    """
    
    def __init__(self, 
                 total_attention_budget: float = 100.0,
                 fairness_factor: float = 0.1,
                 efficiency_threshold: float = 1.0):
        """
        Initialize economic allocator.
        
        Args:
            total_attention_budget: Total attention resources available
            fairness_factor: Weight for fairness considerations [0.0, 1.0]
            efficiency_threshold: Minimum efficiency for allocation consideration
        """
        self.total_budget = total_attention_budget
        self.fairness_factor = fairness_factor
        self.efficiency_threshold = efficiency_threshold
        
        # Current allocations: atom_id -> allocated_attention_amount
        self.current_allocations: Dict[str, float] = {}
        
        # Allocation history for fairness analysis
        self.allocation_history: List[Tuple[str, float, float]] = []  # (atom_id, amount, timestamp)
        
        # Performance metrics
        self.metrics = {
            'allocations_made': 0,
            'total_value_delivered': 0.0,
            'total_cost_incurred': 0.0,
            'rejected_requests': 0,
            'average_efficiency': 0.0,
            'fairness_violations': 0
        }
        
        logger.info(f"EconomicAllocator initialized: budget={total_attention_budget}, "
                   f"fairness_factor={fairness_factor}, efficiency_threshold={efficiency_threshold}")
    
    def evaluate_allocation_request(self, request: AttentionAllocationRequest) -> Dict[str, float]:
        """
        Evaluate an allocation request and return economic metrics.
        
        Args:
            request: Attention allocation request to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        efficiency = request.compute_efficiency()
        total_priority = request.compute_total_priority()
        urgency = request.compute_urgency_factor()
        
        # Compute required budget
        required_budget = request.requested_attention.compute_salience() * 10.0  # Scale to budget units
        
        # Check fairness impact
        fairness_score = self._compute_fairness_score(request.atom_id, required_budget)
        
        # Overall allocation score
        allocation_score = (
            total_priority * 0.4 +
            efficiency * 0.3 +
            fairness_score * self.fairness_factor +
            min(1.0, urgency / 5.0) * 0.2
        )
        
        evaluation = {
            'efficiency': efficiency,
            'total_priority': total_priority,
            'urgency_factor': urgency,
            'required_budget': required_budget,
            'fairness_score': fairness_score,
            'allocation_score': allocation_score,
            'recommended': allocation_score > 0.5 and efficiency >= self.efficiency_threshold
        }
        
        logger.debug(f"Evaluated request for {request.atom_id}: score={allocation_score:.3f}, "
                    f"efficiency={efficiency:.3f}, fairness={fairness_score:.3f}")
        
        return evaluation
    
    def allocate_attention_batch(self, 
                                requests: List[AttentionAllocationRequest],
                                attention_kernel: AttentionKernel) -> Dict[str, Any]:
        """
        Process a batch of allocation requests using economic optimization.
        
        Args:
            requests: List of attention allocation requests
            attention_kernel: Attention kernel to apply allocations to
            
        Returns:
            Results dictionary with allocation decisions and metrics
        """
        if not requests:
            return {'allocations': [], 'rejected': [], 'metrics': {}}
        
        # Evaluate all requests
        evaluated_requests = []
        for request in requests:
            evaluation = self.evaluate_allocation_request(request)
            evaluated_requests.append((request, evaluation))
        
        # Sort by allocation score (highest first)
        evaluated_requests.sort(key=lambda x: x[1]['allocation_score'], reverse=True)
        
        # Allocate within budget constraints
        allocated = []
        rejected = []
        remaining_budget = self._get_available_budget()
        
        for request, evaluation in evaluated_requests:
            required_budget = evaluation['required_budget']
            
            if (evaluation['recommended'] and 
                required_budget <= remaining_budget and
                evaluation['efficiency'] >= self.efficiency_threshold):
                
                # Make allocation
                success = attention_kernel.allocate_attention(
                    request.atom_id, 
                    request.requested_attention
                )
                
                if success:
                    # Update budget and tracking
                    self.current_allocations[request.atom_id] = required_budget
                    remaining_budget -= required_budget
                    
                    # Record allocation
                    self.allocation_history.append((
                        request.atom_id, 
                        required_budget, 
                        time.time()
                    ))
                    
                    # Update metrics
                    self.metrics['allocations_made'] += 1
                    self.metrics['total_value_delivered'] += request.value
                    self.metrics['total_cost_incurred'] += request.cost
                    
                    allocated.append((request, evaluation))
                    
                    logger.debug(f"Allocated attention to {request.atom_id}: "
                               f"budget={required_budget:.1f}, efficiency={evaluation['efficiency']:.3f}")
                else:
                    rejected.append((request, evaluation, "allocation_failed"))
            else:
                # Determine rejection reason
                if not evaluation['recommended']:
                    reason = "low_score"
                elif required_budget > remaining_budget:
                    reason = "budget_exceeded"
                else:
                    reason = "efficiency_threshold"
                
                rejected.append((request, evaluation, reason))
                self.metrics['rejected_requests'] += 1
        
        # Update average efficiency
        if allocated:
            efficiencies = [eval['efficiency'] for _, eval in allocated]
            self.metrics['average_efficiency'] = np.mean(efficiencies)
        
        results = {
            'allocations': allocated,
            'rejected': rejected,
            'metrics': {
                'allocated_count': len(allocated),
                'rejected_count': len(rejected),
                'total_budget_used': self.total_budget - remaining_budget,
                'remaining_budget': remaining_budget,
                'budget_utilization': (self.total_budget - remaining_budget) / self.total_budget,
                'average_efficiency': self.metrics['average_efficiency']
            }
        }
        
        logger.info(f"Batch allocation complete: {len(allocated)} allocated, {len(rejected)} rejected, "
                   f"budget utilization: {results['metrics']['budget_utilization']:.1%}")
        
        return results
    
    def optimize_attention_portfolio(self, 
                                   attention_kernel: AttentionKernel,
                                   target_efficiency: float = 2.0) -> Dict[str, Any]:
        """
        Optimize current attention portfolio for better efficiency.
        
        Args:
            attention_kernel: Attention kernel with current allocations
            target_efficiency: Target efficiency threshold for reallocation
            
        Returns:
            Optimization results and metrics
        """
        focus_atoms = attention_kernel.get_attention_focus()
        if not focus_atoms:
            return {'optimizations': 0, 'efficiency_gain': 0.0}
        
        optimizations = 0
        total_efficiency_gain = 0.0
        
        for atom_id, tensor in focus_atoms:
            current_salience = tensor.compute_salience()
            current_allocation = self.current_allocations.get(atom_id, 0.0)
            
            if current_allocation > 0:
                # Estimate current efficiency (simplified)
                current_efficiency = current_salience / (current_allocation / 10.0)
                
                if current_efficiency < target_efficiency:
                    # Try to optimize by adjusting attention parameters
                    optimized_tensor = self._optimize_tensor_parameters(tensor, target_efficiency)
                    
                    if optimized_tensor:
                        success = attention_kernel.allocate_attention(atom_id, optimized_tensor)
                        if success:
                            new_salience = optimized_tensor.compute_salience()
                            new_efficiency = new_salience / (current_allocation / 10.0)
                            efficiency_gain = new_efficiency - current_efficiency
                            
                            optimizations += 1
                            total_efficiency_gain += efficiency_gain
                            
                            logger.debug(f"Optimized {atom_id}: efficiency {current_efficiency:.3f} -> {new_efficiency:.3f}")
        
        results = {
            'optimizations': optimizations,
            'efficiency_gain': total_efficiency_gain,
            'average_gain_per_optimization': total_efficiency_gain / max(optimizations, 1)
        }
        
        logger.info(f"Portfolio optimization complete: {optimizations} optimizations, "
                   f"total efficiency gain: {total_efficiency_gain:.3f}")
        
        return results
    
    def _compute_fairness_score(self, atom_id: str, requested_budget: float) -> float:
        """
        Compute fairness score for allocation considering historical distribution.
        
        Args:
            atom_id: Atom requesting allocation
            requested_budget: Budget amount requested
            
        Returns:
            Fairness score [0.0, 1.0] where 1.0 is most fair
        """
        if not self.allocation_history:
            return 1.0  # No history, fully fair
        
        # Calculate historical allocation for this atom
        atom_history = [amount for aid, amount, _ in self.allocation_history if aid == atom_id]
        atom_total = sum(atom_history)
        
        # Calculate average allocation across all atoms
        all_allocations = [amount for _, amount, _ in self.allocation_history]
        average_allocation = np.mean(all_allocations) if all_allocations else 0.0
        
        # Fairness is higher if atom has received less than average
        if atom_total < average_allocation:
            fairness_score = 1.0  # Give preference to under-allocated atoms
        else:
            # Reduce fairness score for over-allocated atoms
            excess_ratio = atom_total / max(average_allocation, 1.0)
            fairness_score = max(0.0, 1.0 - (excess_ratio - 1.0) * 0.5)
        
        return fairness_score
    
    def _get_available_budget(self) -> float:
        """Get currently available budget"""
        allocated_total = sum(self.current_allocations.values())
        return max(0.0, self.total_budget - allocated_total)
    
    def _optimize_tensor_parameters(self, 
                                   tensor: ECANAttentionTensor, 
                                   target_efficiency: float) -> Optional[ECANAttentionTensor]:
        """
        Optimize tensor parameters to improve efficiency.
        
        Args:
            tensor: Current attention tensor
            target_efficiency: Target efficiency to achieve
            
        Returns:
            Optimized tensor or None if no improvement possible
        """
        # Simple optimization: adjust spreading factor and confidence
        optimized = ECANAttentionTensor(
            short_term_importance=tensor.short_term_importance,
            long_term_importance=tensor.long_term_importance,
            urgency=tensor.urgency,
            confidence=min(1.0, tensor.confidence * 1.1),  # Boost confidence slightly
            spreading_factor=min(1.0, tensor.spreading_factor * 1.05),  # Boost spreading
            decay_rate=max(0.0, tensor.decay_rate * 0.95)  # Reduce decay slightly
        )
        
        # Check if optimization improves salience
        if optimized.compute_salience() > tensor.compute_salience():
            return optimized
        else:
            return None
    
    def release_attention(self, atom_id: str, attention_kernel: AttentionKernel) -> bool:
        """
        Release attention allocation for an atom.
        
        Args:
            atom_id: Atom to release attention from
            attention_kernel: Attention kernel to update
            
        Returns:
            True if successfully released, False otherwise
        """
        if atom_id in self.current_allocations:
            released_budget = self.current_allocations[atom_id]
            del self.current_allocations[atom_id]
            
            # Try to remove from attention kernel (if it has such functionality)
            # For now, we'll just log the release
            logger.info(f"Released attention allocation for {atom_id}: budget={released_budget:.1f}")
            return True
        
        return False
    
    def get_economic_metrics(self) -> Dict[str, Any]:
        """Get comprehensive economic metrics"""
        total_efficiency = self.metrics['total_value_delivered'] / max(self.metrics['total_cost_incurred'], 0.001)
        
        # Calculate budget utilization
        allocated_total = sum(self.current_allocations.values())
        budget_utilization = allocated_total / self.total_budget
        
        # Calculate fairness metrics
        fairness_metrics = self._calculate_fairness_metrics()
        
        return {
            'allocations_made': self.metrics['allocations_made'],
            'rejected_requests': self.metrics['rejected_requests'],
            'total_value_delivered': self.metrics['total_value_delivered'],
            'total_cost_incurred': self.metrics['total_cost_incurred'],
            'overall_efficiency': total_efficiency,
            'average_efficiency': self.metrics['average_efficiency'],
            'budget_utilization': budget_utilization,
            'available_budget': self._get_available_budget(),
            'current_allocations': len(self.current_allocations),
            'fairness_violations': self.metrics['fairness_violations'],
            **fairness_metrics
        }
    
    def _calculate_fairness_metrics(self) -> Dict[str, float]:
        """Calculate fairness distribution metrics"""
        if not self.allocation_history:
            return {'gini_coefficient': 0.0, 'allocation_variance': 0.0}
        
        # Group allocations by atom
        atom_totals = {}
        for atom_id, amount, _ in self.allocation_history:
            atom_totals[atom_id] = atom_totals.get(atom_id, 0.0) + amount
        
        allocations = list(atom_totals.values())
        
        if len(allocations) < 2:
            return {'gini_coefficient': 0.0, 'allocation_variance': 0.0}
        
        # Calculate Gini coefficient (measure of inequality)
        sorted_allocations = sorted(allocations)
        n = len(sorted_allocations)
        cumsum = np.cumsum(sorted_allocations)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        # Calculate variance
        variance = np.var(allocations)
        
        return {
            'gini_coefficient': gini,
            'allocation_variance': variance
        }
    
    def reset_allocations(self):
        """Reset all current allocations and history"""
        self.current_allocations.clear()
        self.allocation_history.clear()
        
        # Reset metrics
        self.metrics = {
            'allocations_made': 0,
            'total_value_delivered': 0.0,
            'total_cost_incurred': 0.0,
            'rejected_requests': 0,
            'average_efficiency': 0.0,
            'fairness_violations': 0
        }
        
        logger.info("Reset all allocations and metrics")