"""
Attention Decay and Refresh Mechanisms

Implements dynamic attention persistence management with decay and refresh
algorithms for maintaining optimal attention allocation over time.

Key features:
- Exponential and linear attention decay models
- Smart refresh mechanisms based on relevance
- Attention persistence optimization
- Memory consolidation for long-term importance
- Dynamic decay rate adjustment
"""

import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from .attention_kernel import ECANAttentionTensor, AttentionKernel

logger = logging.getLogger(__name__)


class DecayMode(Enum):
    """Attention decay modes"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    ADAPTIVE = "adaptive"


@dataclass
class DecayParameters:
    """Parameters for attention decay"""
    mode: DecayMode = DecayMode.EXPONENTIAL
    base_rate: float = 0.1  # Base decay rate per time unit
    half_life: float = 300.0  # Time in seconds for 50% decay
    min_threshold: float = 0.01  # Minimum attention before removal
    preserve_lti: bool = True  # Preserve long-term importance during decay
    urgency_protection: float = 0.8  # Urgency threshold for decay protection


@dataclass
class RefreshTrigger:
    """Trigger condition for attention refresh"""
    atom_id: str
    trigger_type: str  # "access", "reference", "goal_relevance", "external"
    strength: float  # Strength of refresh trigger [0.0, 1.0]
    timestamp: float
    context: Optional[Dict[str, Any]] = None


@dataclass
class DecayRefreshResult:
    """Result of decay/refresh operation"""
    atoms_decayed: int
    atoms_refreshed: int
    atoms_removed: int
    total_attention_before: float
    total_attention_after: float
    execution_time: float
    decay_efficiency: float


class DecayRefresh:
    """
    Attention decay and refresh manager for dynamic attention persistence.
    
    Manages the temporal dynamics of attention allocation, implementing
    sophisticated decay and refresh mechanisms that optimize cognitive
    resource utilization over time.
    """
    
    def __init__(self,
                 decay_params: Optional[DecayParameters] = None,
                 refresh_sensitivity: float = 0.7,
                 consolidation_threshold: float = 0.9,
                 background_processing: bool = True,
                 processing_interval: float = 1.0):
        """
        Initialize decay and refresh manager.
        
        Args:
            decay_params: Parameters for attention decay
            refresh_sensitivity: Sensitivity to refresh triggers [0.0, 1.0]
            consolidation_threshold: Threshold for memory consolidation
            background_processing: Enable background decay processing
            processing_interval: Interval for background processing (seconds)
        """
        self.decay_params = decay_params or DecayParameters()
        self.refresh_sensitivity = refresh_sensitivity
        self.consolidation_threshold = consolidation_threshold
        self.background_processing = background_processing
        self.processing_interval = processing_interval
        
        # Attention tracking
        self.attention_timestamps: Dict[str, float] = {}  # atom_id -> last_update_time
        self.refresh_triggers: List[RefreshTrigger] = []
        self.access_history: Dict[str, List[float]] = {}  # atom_id -> access_times
        
        # Performance metrics
        self.metrics = {
            'decay_operations': 0,
            'refresh_operations': 0,
            'atoms_consolidated': 0,
            'total_decay_time': 0.0,
            'total_refresh_time': 0.0,
            'attention_conservation_ratio': 1.0
        }
        
        # Background processing state
        self._last_processing_time = time.time()
        
        logger.info(f"DecayRefresh initialized: mode={self.decay_params.mode}, "
                   f"base_rate={self.decay_params.base_rate}, "
                   f"half_life={self.decay_params.half_life}s")
    
    def process_decay_cycle(self, attention_kernel: AttentionKernel) -> DecayRefreshResult:
        """
        Process one decay and refresh cycle.
        
        Args:
            attention_kernel: Attention kernel to process
            
        Returns:
            DecayRefreshResult with operation metrics
        """
        start_time = time.time()
        current_time = time.time()
        
        # Calculate total attention before processing
        attention_before = self._calculate_total_attention(attention_kernel)
        
        # Process decay for all atoms
        atoms_decayed, atoms_removed = self._process_attention_decay(attention_kernel, current_time)
        
        # Process refresh triggers
        atoms_refreshed = self._process_refresh_triggers(attention_kernel, current_time)
        
        # Perform memory consolidation
        atoms_consolidated = self._perform_memory_consolidation(attention_kernel)
        
        # Calculate total attention after processing
        attention_after = self._calculate_total_attention(attention_kernel)
        
        # Calculate efficiency metrics
        execution_time = time.time() - start_time
        decay_efficiency = self._calculate_decay_efficiency(attention_before, attention_after)
        
        # Update metrics
        self.metrics['decay_operations'] += 1
        self.metrics['total_decay_time'] += execution_time
        self.metrics['atoms_consolidated'] += atoms_consolidated
        
        if attention_before > 0:
            self.metrics['attention_conservation_ratio'] = attention_after / attention_before
        
        result = DecayRefreshResult(
            atoms_decayed=atoms_decayed,
            atoms_refreshed=atoms_refreshed,
            atoms_removed=atoms_removed,
            total_attention_before=attention_before,
            total_attention_after=attention_after,
            execution_time=execution_time,
            decay_efficiency=decay_efficiency
        )
        
        logger.debug(f"Decay cycle: {atoms_decayed} decayed, {atoms_refreshed} refreshed, "
                    f"{atoms_removed} removed, efficiency={decay_efficiency:.3f}")
        
        return result
    
    def _process_attention_decay(self, 
                               attention_kernel: AttentionKernel, 
                               current_time: float) -> Tuple[int, int]:
        """Process attention decay for all atoms"""
        atoms_decayed = 0
        atoms_removed = 0
        
        # Get all atoms with attention
        atoms_to_process = list(attention_kernel.attention_tensors.keys())
        
        for atom_id in atoms_to_process:
            tensor = attention_kernel.get_attention(atom_id)
            if not tensor:
                continue
            
            # Calculate time since last update
            last_update = self.attention_timestamps.get(atom_id, current_time)
            time_delta = current_time - last_update
            
            # Calculate decay factor
            decay_factor = self._calculate_decay_factor(tensor, time_delta)
            
            if decay_factor < 1.0:
                # Apply decay to appropriate components
                new_sti = tensor.short_term_importance * decay_factor
                new_urgency = tensor.urgency * decay_factor
                new_confidence = tensor.confidence * (0.5 + 0.5 * decay_factor)  # Confidence decays slower
                
                # Preserve long-term importance if configured
                new_lti = tensor.long_term_importance
                if not self.decay_params.preserve_lti:
                    new_lti *= (0.9 + 0.1 * decay_factor)  # Very slow LTI decay
                
                # Check if attention has fallen below threshold
                new_salience = (
                    new_sti * 0.4 + 
                    new_lti * 0.2 + 
                    new_urgency * 0.3 + 
                    new_confidence * 0.1
                )
                
                if new_salience < self.decay_params.min_threshold:
                    # Remove atom from attention if below threshold
                    attention_kernel.attention_tensors.pop(atom_id, None)
                    self.attention_timestamps.pop(atom_id, None)
                    atoms_removed += 1
                else:
                    # Update tensor with decayed values
                    success = attention_kernel.update_attention(
                        atom_id,
                        short_term_delta=new_sti - tensor.short_term_importance,
                        long_term_delta=new_lti - tensor.long_term_importance,
                        urgency_delta=new_urgency - tensor.urgency,
                        confidence_delta=new_confidence - tensor.confidence
                    )
                    
                    if success:
                        atoms_decayed += 1
                        self.attention_timestamps[atom_id] = current_time
        
        return atoms_decayed, atoms_removed
    
    def _calculate_decay_factor(self, tensor: ECANAttentionTensor, time_delta: float) -> float:
        """Calculate decay factor based on decay mode and parameters"""
        if time_delta <= 0:
            return 1.0
        
        # Check urgency protection
        if tensor.urgency >= self.decay_params.urgency_protection:
            return 0.95  # Minimal decay for urgent items
        
        # Apply decay based on mode
        if self.decay_params.mode == DecayMode.EXPONENTIAL:
            # Exponential decay: factor = e^(-lambda * t)
            lambda_rate = math.log(2) / self.decay_params.half_life
            decay_factor = math.exp(-lambda_rate * time_delta)
            
        elif self.decay_params.mode == DecayMode.LINEAR:
            # Linear decay: factor = 1 - rate * t
            decay_factor = max(0.0, 1.0 - self.decay_params.base_rate * time_delta)
            
        elif self.decay_params.mode == DecayMode.LOGARITHMIC:
            # Logarithmic decay: factor = 1 / (1 + rate * log(1 + t))
            decay_factor = 1.0 / (1.0 + self.decay_params.base_rate * math.log(1.0 + time_delta))
            
        elif self.decay_params.mode == DecayMode.ADAPTIVE:
            # Adaptive decay based on tensor properties
            base_decay = math.exp(-self.decay_params.base_rate * time_delta)
            
            # Adjust based on confidence and spreading factor
            confidence_factor = 0.5 + 0.5 * tensor.confidence
            spreading_factor = 0.5 + 0.5 * tensor.spreading_factor
            
            decay_factor = base_decay * confidence_factor * spreading_factor
            
        else:
            decay_factor = 1.0
        
        # Apply tensor's own decay rate
        tensor_decay = math.exp(-tensor.decay_rate * time_delta)
        final_decay_factor = decay_factor * tensor_decay
        
        return max(0.0, min(1.0, final_decay_factor))
    
    def _process_refresh_triggers(self, 
                                attention_kernel: AttentionKernel, 
                                current_time: float) -> int:
        """Process pending refresh triggers"""
        atoms_refreshed = 0
        
        # Process recent triggers (within last 10 seconds)
        recent_triggers = [
            trigger for trigger in self.refresh_triggers
            if current_time - trigger.timestamp <= 10.0
        ]
        
        # Group triggers by atom
        atom_triggers = {}
        for trigger in recent_triggers:
            if trigger.atom_id not in atom_triggers:
                atom_triggers[trigger.atom_id] = []
            atom_triggers[trigger.atom_id].append(trigger)
        
        # Apply refresh to each atom
        for atom_id, triggers in atom_triggers.items():
            if self._apply_refresh(atom_id, triggers, attention_kernel, current_time):
                atoms_refreshed += 1
        
        # Clean up old triggers
        self.refresh_triggers = [
            trigger for trigger in self.refresh_triggers
            if current_time - trigger.timestamp <= 60.0  # Keep triggers for 1 minute
        ]
        
        return atoms_refreshed
    
    def _apply_refresh(self, 
                      atom_id: str, 
                      triggers: List[RefreshTrigger],
                      attention_kernel: AttentionKernel,
                      current_time: float) -> bool:
        """Apply refresh to a specific atom based on triggers"""
        tensor = attention_kernel.get_attention(atom_id)
        if not tensor:
            # Create new tensor for refreshed atom
            refresh_strength = sum(trigger.strength for trigger in triggers) / len(triggers)
            refresh_strength *= self.refresh_sensitivity
            
            new_tensor = ECANAttentionTensor(
                short_term_importance=min(1.0, refresh_strength),
                long_term_importance=0.0,
                urgency=min(1.0, refresh_strength * 0.8),
                confidence=0.6,
                spreading_factor=0.7,
                decay_rate=0.1
            )
            
            success = attention_kernel.allocate_attention(atom_id, new_tensor)
            if success:
                self.attention_timestamps[atom_id] = current_time
                return True
        else:
            # Refresh existing tensor
            total_refresh_strength = sum(trigger.strength for trigger in triggers)
            refresh_boost = total_refresh_strength * self.refresh_sensitivity
            
            # Calculate refresh deltas
            sti_boost = refresh_boost * 0.6
            urgency_boost = refresh_boost * 0.4
            confidence_boost = refresh_boost * 0.2
            
            success = attention_kernel.update_attention(
                atom_id,
                short_term_delta=sti_boost,
                urgency_delta=urgency_boost,
                confidence_delta=confidence_boost
            )
            
            if success:
                self.attention_timestamps[atom_id] = current_time
                return True
        
        return False
    
    def _perform_memory_consolidation(self, attention_kernel: AttentionKernel) -> int:
        """Perform memory consolidation for stable attention patterns"""
        atoms_consolidated = 0
        
        # Get atoms with high stable attention
        stable_atoms = []
        for atom_id, tensor in attention_kernel.attention_tensors.items():
            # Check if atom has been stable (high attention for extended period)
            if (tensor.short_term_importance > self.consolidation_threshold and
                tensor.confidence > 0.8):
                
                # Check access history for stability
                access_times = self.access_history.get(atom_id, [])
                if len(access_times) >= 3:  # Multiple accesses
                    recent_accesses = [t for t in access_times if time.time() - t <= 3600]  # Last hour
                    if len(recent_accesses) >= 2:  # Recent activity
                        stable_atoms.append((atom_id, tensor))
        
        # Consolidate stable atoms
        for atom_id, tensor in stable_atoms:
            # Transfer some STI to LTI
            consolidation_amount = tensor.short_term_importance * 0.1
            
            success = attention_kernel.update_attention(
                atom_id,
                short_term_delta=-consolidation_amount,
                long_term_delta=consolidation_amount
            )
            
            if success:
                atoms_consolidated += 1
                logger.debug(f"Consolidated memory for atom {atom_id}: STI->LTI transfer={consolidation_amount:.3f}")
        
        return atoms_consolidated
    
    def add_refresh_trigger(self, trigger: RefreshTrigger):
        """Add a refresh trigger"""
        self.refresh_triggers.append(trigger)
        
        # Update access history
        if trigger.atom_id not in self.access_history:
            self.access_history[trigger.atom_id] = []
        self.access_history[trigger.atom_id].append(trigger.timestamp)
        
        # Keep only recent access history (last 24 hours)
        cutoff_time = trigger.timestamp - 86400  # 24 hours
        self.access_history[trigger.atom_id] = [
            t for t in self.access_history[trigger.atom_id] if t > cutoff_time
        ]
        
        logger.debug(f"Added refresh trigger for {trigger.atom_id}: "
                    f"type={trigger.trigger_type}, strength={trigger.strength:.3f}")
    
    def add_access_trigger(self, atom_id: str, access_strength: float = 0.5):
        """Convenience method to add an access-based refresh trigger"""
        trigger = RefreshTrigger(
            atom_id=atom_id,
            trigger_type="access",
            strength=access_strength,
            timestamp=time.time()
        )
        self.add_refresh_trigger(trigger)
    
    def add_goal_relevance_trigger(self, atom_id: str, relevance_score: float):
        """Add a goal relevance-based refresh trigger"""
        trigger = RefreshTrigger(
            atom_id=atom_id,
            trigger_type="goal_relevance",
            strength=relevance_score,
            timestamp=time.time(),
            context={"relevance_score": relevance_score}
        )
        self.add_refresh_trigger(trigger)
    
    def adjust_decay_parameters(self, 
                              performance_metrics: Dict[str, float],
                              target_attention_level: float = 0.7):
        """
        Dynamically adjust decay parameters based on performance metrics.
        
        Args:
            performance_metrics: Current system performance metrics
            target_attention_level: Target attention conservation ratio
        """
        current_conservation = self.metrics.get('attention_conservation_ratio', 1.0)
        
        # Adjust decay rate based on conservation ratio
        if current_conservation > target_attention_level + 0.1:
            # Too much attention being preserved, increase decay
            self.decay_params.base_rate = min(1.0, self.decay_params.base_rate * 1.1)
            logger.debug(f"Increased decay rate to {self.decay_params.base_rate:.3f}")
            
        elif current_conservation < target_attention_level - 0.1:
            # Too much attention being lost, decrease decay
            self.decay_params.base_rate = max(0.01, self.decay_params.base_rate * 0.9)
            logger.debug(f"Decreased decay rate to {self.decay_params.base_rate:.3f}")
        
        # Adjust refresh sensitivity based on refresh success rate
        refresh_ratio = self.metrics.get('refresh_operations', 0) / max(self.metrics.get('decay_operations', 1), 1)
        
        if refresh_ratio < 0.1:  # Very few refreshes
            self.refresh_sensitivity = min(1.0, self.refresh_sensitivity * 1.05)
        elif refresh_ratio > 0.5:  # Too many refreshes
            self.refresh_sensitivity = max(0.1, self.refresh_sensitivity * 0.95)
    
    def _calculate_total_attention(self, attention_kernel: AttentionKernel) -> float:
        """Calculate total attention across all atoms"""
        total = 0.0
        for tensor in attention_kernel.attention_tensors.values():
            total += tensor.compute_salience()
        return total
    
    def _calculate_decay_efficiency(self, attention_before: float, attention_after: float) -> float:
        """Calculate decay efficiency (how well decay preserved important attention)"""
        if attention_before <= 0:
            return 1.0
        
        conservation_ratio = attention_after / attention_before
        
        # Efficiency is higher when conservation ratio is close to target (0.8)
        target_ratio = 0.8
        efficiency = 1.0 - abs(conservation_ratio - target_ratio) / target_ratio
        
        return max(0.0, min(1.0, efficiency))
    
    def optimize_decay_schedule(self, 
                               attention_kernel: AttentionKernel,
                               optimization_window: float = 3600.0) -> Dict[str, Any]:
        """
        Optimize decay schedule based on attention patterns.
        
        Args:
            attention_kernel: Attention kernel to analyze
            optimization_window: Time window for optimization analysis (seconds)
            
        Returns:
            Optimization results and recommended parameters
        """
        current_time = time.time()
        cutoff_time = current_time - optimization_window
        
        # Analyze attention patterns
        stable_atoms = []
        volatile_atoms = []
        
        for atom_id, tensor in attention_kernel.attention_tensors.items():
            access_times = self.access_history.get(atom_id, [])
            recent_accesses = [t for t in access_times if t > cutoff_time]
            
            if len(recent_accesses) >= 3:
                # Calculate access frequency
                if len(recent_accesses) > 1:
                    intervals = [recent_accesses[i] - recent_accesses[i-1] 
                               for i in range(1, len(recent_accesses))]
                    avg_interval = sum(intervals) / len(intervals)
                    
                    if avg_interval < 600:  # Less than 10 minutes between accesses
                        stable_atoms.append((atom_id, tensor, avg_interval))
                    else:
                        volatile_atoms.append((atom_id, tensor, avg_interval))
        
        # Recommend optimized parameters
        optimized_params = DecayParameters(
            mode=self.decay_params.mode,
            base_rate=self.decay_params.base_rate,
            half_life=self.decay_params.half_life,
            min_threshold=self.decay_params.min_threshold,
            preserve_lti=self.decay_params.preserve_lti,
            urgency_protection=self.decay_params.urgency_protection
        )
        
        # Adjust based on patterns
        if len(stable_atoms) > len(volatile_atoms):
            # More stable patterns - can afford slightly faster decay
            optimized_params.base_rate *= 1.1
            optimized_params.half_life *= 0.9
        else:
            # More volatile patterns - use slower decay
            optimized_params.base_rate *= 0.9
            optimized_params.half_life *= 1.1
        
        results = {
            'stable_atoms': len(stable_atoms),
            'volatile_atoms': len(volatile_atoms),
            'current_parameters': self.decay_params,
            'optimized_parameters': optimized_params,
            'optimization_benefit': abs(len(stable_atoms) - len(volatile_atoms)) / max(len(stable_atoms) + len(volatile_atoms), 1)
        }
        
        logger.info(f"Decay optimization: {len(stable_atoms)} stable, {len(volatile_atoms)} volatile atoms")
        return results
    
    def get_decay_refresh_metrics(self) -> Dict[str, Any]:
        """Get comprehensive decay and refresh metrics"""
        total_operations = self.metrics['decay_operations'] + self.metrics['refresh_operations']
        
        return {
            **self.metrics,
            'average_decay_time': self.metrics['total_decay_time'] / max(self.metrics['decay_operations'], 1),
            'average_refresh_time': self.metrics['total_refresh_time'] / max(self.metrics['refresh_operations'], 1),
            'refresh_to_decay_ratio': self.metrics['refresh_operations'] / max(self.metrics['decay_operations'], 1),
            'consolidation_rate': self.metrics['atoms_consolidated'] / max(total_operations, 1),
            'active_triggers': len(self.refresh_triggers),
            'tracked_atoms': len(self.attention_timestamps),
            'atoms_with_history': len(self.access_history)
        }
    
    def reset_decay_refresh(self):
        """Reset decay and refresh state"""
        self.attention_timestamps.clear()
        self.refresh_triggers.clear()
        self.access_history.clear()
        
        self.metrics = {
            'decay_operations': 0,
            'refresh_operations': 0,
            'atoms_consolidated': 0,
            'total_decay_time': 0.0,
            'total_refresh_time': 0.0,
            'attention_conservation_ratio': 1.0
        }
        
        self._last_processing_time = time.time()
        
        logger.info("Reset decay and refresh state and metrics")