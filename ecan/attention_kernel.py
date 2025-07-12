"""
ECAN Attention Kernel

Implements the core 6-dimensional attention tensor and basic attention kernel
operations as specified in the ECAN attention allocation system.

Tensor Structure:
ECAN_Attention_Tensor[6] = {
  short_term_importance: [0.0, 1.0],
  long_term_importance: [0.0, 1.0], 
  urgency: [0.0, 1.0],
  confidence: [0.0, 1.0],
  spreading_factor: [0.0, 1.0],
  decay_rate: [0.0, 1.0]
}
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from threading import Lock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ECANAttentionTensor:
    """
    6-dimensional ECAN attention tensor representing cognitive resource allocation.
    
    Attributes:
        short_term_importance: Current attentional focus strength [0.0, 1.0]
        long_term_importance: Historical importance accumulation [0.0, 1.0]  
        urgency: Time-sensitive priority level [0.0, 1.0]
        confidence: Certainty in attention allocation [0.0, 1.0]
        spreading_factor: Attention propagation strength [0.0, 1.0]
        decay_rate: Rate of attention dissipation [0.0, 1.0]
    """
    short_term_importance: float = 0.0
    long_term_importance: float = 0.0
    urgency: float = 0.0
    confidence: float = 0.0
    spreading_factor: float = 0.0
    decay_rate: float = 0.1
    
    def __post_init__(self):
        """Validate tensor values are within [0.0, 1.0] range"""
        for field, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in range [0.0, 1.0], got {value}")
    
    def to_array(self) -> np.ndarray:
        """Convert tensor to numpy array"""
        return np.array([
            self.short_term_importance,
            self.long_term_importance,
            self.urgency,
            self.confidence,
            self.spreading_factor,
            self.decay_rate
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'ECANAttentionTensor':
        """Create tensor from numpy array"""
        if len(array) != 6:
            raise ValueError(f"Array must have 6 elements, got {len(array)}")
        
        return cls(
            short_term_importance=float(array[0]),
            long_term_importance=float(array[1]),
            urgency=float(array[2]),
            confidence=float(array[3]),
            spreading_factor=float(array[4]),
            decay_rate=float(array[5])
        )
    
    def compute_salience(self) -> float:
        """
        Compute overall salience score from tensor components.
        
        Formula: salience = (STI * 0.4 + LTI * 0.2 + urgency * 0.3 + confidence * 0.1)
        """
        return (
            self.short_term_importance * 0.4 +
            self.long_term_importance * 0.2 +
            self.urgency * 0.3 +
            self.confidence * 0.1
        )
    
    def compute_activation_strength(self) -> float:
        """
        Compute activation strength for spreading.
        
        Formula: activation = spreading_factor * confidence * (1 - decay_rate)
        """
        return self.spreading_factor * self.confidence * (1 - self.decay_rate)


class AttentionKernel:
    """
    Core ECAN attention kernel managing attention allocation and tensor operations.
    
    This kernel provides the foundation for economic attention allocation,
    resource scheduling, and activation spreading in the cognitive architecture.
    """
    
    def __init__(self, max_atoms: int = 10000, focus_boundary: float = 0.5):
        """
        Initialize attention kernel.
        
        Args:
            max_atoms: Maximum number of atoms to track
            focus_boundary: Threshold for attention focus inclusion
        """
        self.max_atoms = max_atoms
        self.focus_boundary = focus_boundary
        
        # Attention tensor storage: atom_id -> ECANAttentionTensor
        self.attention_tensors: Dict[str, ECANAttentionTensor] = {}
        
        # Performance metrics
        self.metrics = {
            'atoms_processed': 0,
            'tensor_operations': 0,
            'focus_updates': 0,
            'last_update': time.time()
        }
        
        # Thread safety
        self._lock = Lock()
        
        logger.info(f"AttentionKernel initialized: max_atoms={max_atoms}, focus_boundary={focus_boundary}")
    
    def allocate_attention(self, atom_id: str, tensor: ECANAttentionTensor) -> bool:
        """
        Allocate attention tensor to an atom.
        
        Args:
            atom_id: Unique identifier for the atom
            tensor: ECAN attention tensor to allocate
            
        Returns:
            True if allocation successful, False otherwise
        """
        with self._lock:
            if len(self.attention_tensors) >= self.max_atoms and atom_id not in self.attention_tensors:
                # Remove lowest salience atom if at capacity
                self._evict_lowest_salience()
            
            self.attention_tensors[atom_id] = tensor
            self.metrics['atoms_processed'] += 1
            self.metrics['tensor_operations'] += 1
            
            logger.debug(f"Allocated attention to atom {atom_id}: salience={tensor.compute_salience():.3f}")
            return True
    
    def get_attention(self, atom_id: str) -> Optional[ECANAttentionTensor]:
        """Get attention tensor for an atom"""
        with self._lock:
            return self.attention_tensors.get(atom_id)
    
    def update_attention(self, atom_id: str, 
                        short_term_delta: float = 0.0,
                        long_term_delta: float = 0.0,
                        urgency_delta: float = 0.0,
                        confidence_delta: float = 0.0) -> bool:
        """
        Update attention tensor components for an atom.
        
        Args:
            atom_id: Atom identifier
            short_term_delta: Change in short-term importance
            long_term_delta: Change in long-term importance  
            urgency_delta: Change in urgency
            confidence_delta: Change in confidence
            
        Returns:
            True if update successful, False if atom not found
        """
        with self._lock:
            if atom_id not in self.attention_tensors:
                return False
            
            tensor = self.attention_tensors[atom_id]
            
            # Apply deltas with bounds checking
            tensor.short_term_importance = np.clip(
                tensor.short_term_importance + short_term_delta, 0.0, 1.0)
            tensor.long_term_importance = np.clip(
                tensor.long_term_importance + long_term_delta, 0.0, 1.0)
            tensor.urgency = np.clip(
                tensor.urgency + urgency_delta, 0.0, 1.0)
            tensor.confidence = np.clip(
                tensor.confidence + confidence_delta, 0.0, 1.0)
            
            self.metrics['tensor_operations'] += 1
            
            logger.debug(f"Updated attention for atom {atom_id}: salience={tensor.compute_salience():.3f}")
            return True
    
    def get_attention_focus(self) -> List[Tuple[str, ECANAttentionTensor]]:
        """
        Get current attention focus (atoms above focus boundary).
        
        Returns:
            List of (atom_id, tensor) tuples for atoms in focus
        """
        with self._lock:
            focus_atoms = []
            for atom_id, tensor in self.attention_tensors.items():
                if tensor.compute_salience() >= self.focus_boundary:
                    focus_atoms.append((atom_id, tensor))
            
            # Sort by salience descending
            focus_atoms.sort(key=lambda x: x[1].compute_salience(), reverse=True)
            
            self.metrics['focus_updates'] += 1
            return focus_atoms
    
    def compute_global_attention_distribution(self) -> Dict[str, float]:
        """
        Compute normalized attention distribution across all atoms.
        
        Returns:
            Dictionary mapping atom_id to normalized attention weight
        """
        with self._lock:
            if not self.attention_tensors:
                return {}
            
            # Compute salience for all atoms
            saliences = {
                atom_id: tensor.compute_salience() 
                for atom_id, tensor in self.attention_tensors.items()
            }
            
            # Normalize to sum to 1.0
            total_salience = sum(saliences.values())
            if total_salience > 0:
                normalized = {
                    atom_id: salience / total_salience
                    for atom_id, salience in saliences.items()
                }
            else:
                normalized = {atom_id: 0.0 for atom_id in saliences.keys()}
            
            self.metrics['tensor_operations'] += len(saliences)
            return normalized
    
    def _evict_lowest_salience(self) -> Optional[str]:
        """Remove atom with lowest salience to make room for new allocation"""
        if not self.attention_tensors:
            return None
        
        # Find atom with lowest salience
        lowest_atom = min(
            self.attention_tensors.items(),
            key=lambda x: x[1].compute_salience()
        )
        
        atom_id = lowest_atom[0]
        del self.attention_tensors[atom_id]
        
        logger.debug(f"Evicted atom {atom_id} with salience {lowest_atom[1].compute_salience():.3f}")
        return atom_id
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._lock:
            current_time = time.time()
            time_delta = current_time - self.metrics['last_update']
            
            metrics = self.metrics.copy()
            metrics.update({
                'current_atoms': len(self.attention_tensors),
                'focus_size': len(self.get_attention_focus()),
                'tensor_ops_per_second': self.metrics['tensor_operations'] / max(time_delta, 0.001),
                'atoms_per_second': self.metrics['atoms_processed'] / max(time_delta, 0.001),
                'memory_usage_mb': self._estimate_memory_usage()
            })
            
            return metrics
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        # Rough estimate: each tensor ~100 bytes + overhead
        return len(self.attention_tensors) * 0.0001  # MB
    
    def reset_metrics(self):
        """Reset performance metrics"""
        with self._lock:
            self.metrics = {
                'atoms_processed': 0,
                'tensor_operations': 0,
                'focus_updates': 0,
                'last_update': time.time()
            }
    
    def export_attention_state(self) -> Dict[str, Dict[str, float]]:
        """
        Export complete attention state for serialization.
        
        Returns:
            Dictionary mapping atom_id to tensor component values
        """
        with self._lock:
            state = {}
            for atom_id, tensor in self.attention_tensors.items():
                state[atom_id] = {
                    'short_term_importance': tensor.short_term_importance,
                    'long_term_importance': tensor.long_term_importance,
                    'urgency': tensor.urgency,
                    'confidence': tensor.confidence,
                    'spreading_factor': tensor.spreading_factor,
                    'decay_rate': tensor.decay_rate,
                    'salience': tensor.compute_salience()
                }
            return state
    
    def import_attention_state(self, state: Dict[str, Dict[str, float]]) -> int:
        """
        Import attention state from serialized data.
        
        Args:
            state: Dictionary mapping atom_id to tensor components
            
        Returns:
            Number of atoms imported
        """
        with self._lock:
            imported_count = 0
            for atom_id, tensor_data in state.items():
                try:
                    tensor = ECANAttentionTensor(
                        short_term_importance=tensor_data['short_term_importance'],
                        long_term_importance=tensor_data['long_term_importance'],
                        urgency=tensor_data['urgency'],
                        confidence=tensor_data['confidence'],
                        spreading_factor=tensor_data['spreading_factor'],
                        decay_rate=tensor_data['decay_rate']
                    )
                    self.attention_tensors[atom_id] = tensor
                    imported_count += 1
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to import attention for atom {atom_id}: {e}")
            
            logger.info(f"Imported attention state for {imported_count} atoms")
            return imported_count