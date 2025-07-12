"""
Cognitive Primitives & Tensor Architecture

This module implements the foundational cognitive primitives with tensor encoding
for representing agent states and cognitive processes in hypergraph structures.

Core Features:
- 5-dimensional cognitive primitive tensors: [modality, depth, context, salience, autonomy_index]
- Type-safe enum definitions for tensor dimensions
- Prime factorization mapping for efficient encoding
- Validation and transformation utilities
"""

import numpy as np
from enum import Enum, IntEnum
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass, field
import json
import math


class ModalityType(IntEnum):
    """
    Sensory and cognitive modality types for tensor encoding.
    
    These represent the different ways information can be perceived,
    processed, and represented in the cognitive system.
    """
    VISUAL = 0      # Visual perception and imagery
    AUDITORY = 1    # Auditory processing and sound
    TEXTUAL = 2     # Language and text processing
    SYMBOLIC = 3    # Abstract symbolic reasoning


class DepthType(IntEnum):
    """
    Cognitive processing depth levels for hierarchical representation.
    
    Represents the level of abstraction and processing depth
    from surface-level perception to deep pragmatic understanding.
    """
    SURFACE = 0     # Surface-level perception and immediate sensory input
    SEMANTIC = 1    # Semantic meaning and conceptual understanding
    PRAGMATIC = 2   # Pragmatic context and intentional understanding


class ContextType(IntEnum):
    """
    Contextual scope and temporal dimensions for cognitive processing.
    
    Defines the spatial and temporal scope of contextual information
    that influences cognitive processing and decision making.
    """
    LOCAL = 0       # Local immediate context
    GLOBAL = 1      # Global system-wide context
    TEMPORAL = 2    # Temporal historical and predictive context


@dataclass
class TensorSignature:
    """
    Signature specification for cognitive primitive tensors.
    
    Provides metadata and validation rules for tensor shapes,
    including dimension constraints, prime factorization mappings,
    and cognitive semantics.
    """
    modality: ModalityType
    depth: DepthType
    context: ContextType
    salience: float = field(default=0.0)  # Range [0.0, 1.0]
    autonomy_index: float = field(default=0.0)  # Range [0.0, 1.0]
    
    # Metadata
    prime_factors: Optional[List[int]] = field(default=None)
    semantic_tags: List[str] = field(default_factory=list)
    creation_timestamp: Optional[float] = field(default=None)
    
    def __post_init__(self):
        """Validate tensor signature constraints."""
        self._validate_ranges()
        if self.prime_factors is None:
            self.prime_factors = self._compute_prime_factors()
        if self.creation_timestamp is None:
            import time
            self.creation_timestamp = time.time()
    
    def _validate_ranges(self):
        """Validate that continuous values are in correct ranges."""
        if not (0.0 <= self.salience <= 1.0):
            raise ValueError(f"Salience must be in [0.0, 1.0], got {self.salience}")
        if not (0.0 <= self.autonomy_index <= 1.0):
            raise ValueError(f"Autonomy index must be in [0.0, 1.0], got {self.autonomy_index}")
    
    def _compute_prime_factors(self) -> List[int]:
        """
        Compute prime factorization mapping for efficient tensor encoding.
        
        Uses the discrete enum values to create a unique prime factorization
        that can be used for fast indexing and retrieval in hypergraph structures.
        """
        # Create a composite number from enum values
        composite = (
            (int(self.modality) + 1) * 2 +
            (int(self.depth) + 1) * 3 +
            (int(self.context) + 1) * 5
        )
        
        factors = []
        d = 2
        while d * d <= composite:
            while composite % d == 0:
                factors.append(d)
                composite //= d
            d += 1
        if composite > 1:
            factors.append(composite)
        
        return factors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signature to dictionary for serialization."""
        return {
            "modality": int(self.modality),
            "depth": int(self.depth),
            "context": int(self.context),
            "salience": self.salience,
            "autonomy_index": self.autonomy_index,
            "prime_factors": self.prime_factors,
            "semantic_tags": self.semantic_tags,
            "creation_timestamp": self.creation_timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TensorSignature':
        """Create signature from dictionary."""
        return cls(
            modality=ModalityType(data["modality"]),
            depth=DepthType(data["depth"]),
            context=ContextType(data["context"]),
            salience=data["salience"],
            autonomy_index=data["autonomy_index"],
            prime_factors=data.get("prime_factors"),
            semantic_tags=data.get("semantic_tags", []),
            creation_timestamp=data.get("creation_timestamp")
        )


class CognitivePrimitiveTensor:
    """
    5-dimensional cognitive primitive tensor for representing agent states and processes.
    
    Tensor shape: [modality, depth, context, salience, autonomy_index]
    
    This class provides the atomic substrate for distributed cognition,
    where each primitive becomes a node in the hypergraph with tensor weights
    representing cognitive state and processing characteristics.
    """
    
    def __init__(
        self,
        signature: TensorSignature,
        data: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, ...]] = None
    ):
        """
        Initialize cognitive primitive tensor.
        
        Args:
            signature: Tensor signature with cognitive metadata
            data: Optional tensor data array
            shape: Optional explicit shape specification
        """
        self.signature = signature
        
        # Default shape based on cognitive primitive dimensions
        if shape is None:
            shape = (4, 3, 3, 100, 100)  # [modality, depth, context, salience_bins, autonomy_bins]
        
        if data is None:
            self.data = np.zeros(shape, dtype=np.float32)
        else:
            if data.shape != shape:
                raise ValueError(f"Data shape {data.shape} doesn't match expected shape {shape}")
            self.data = data.astype(np.float32)
        
        self.shape = shape
        self._validate_tensor()
    
    def _validate_tensor(self):
        """Validate tensor structure and constraints."""
        if len(self.shape) != 5:
            raise ValueError(f"Cognitive primitive tensor must be 5-dimensional, got {len(self.shape)}")
        
        # Validate dimension constraints
        if self.shape[0] != 4:  # modality dimension
            raise ValueError(f"Modality dimension must be 4, got {self.shape[0]}")
        if self.shape[1] != 3:  # depth dimension
            raise ValueError(f"Depth dimension must be 3, got {self.shape[1]}")
        if self.shape[2] != 3:  # context dimension
            raise ValueError(f"Context dimension must be 3, got {self.shape[2]}")
    
    def get_primitive_encoding(self) -> np.ndarray:
        """
        Get the cognitive primitive encoding vector.
        
        Returns a flattened representation suitable for hypergraph node encoding.
        """
        # Extract key tensor values based on signature
        modality_slice = self.data[int(self.signature.modality), :, :, :, :]
        depth_slice = self.data[:, int(self.signature.depth), :, :, :]
        context_slice = self.data[:, :, int(self.signature.context), :, :]
        
        # Combine with signature metadata
        encoding = np.concatenate([
            modality_slice.flatten()[:10],  # Sample from modality slice
            depth_slice.flatten()[:10],     # Sample from depth slice
            context_slice.flatten()[:10],   # Sample from context slice
            [self.signature.salience],      # Salience value
            [self.signature.autonomy_index] # Autonomy value
        ])
        
        return encoding
    
    def update_salience(self, new_salience: float):
        """Update salience value with validation."""
        if not (0.0 <= new_salience <= 1.0):
            raise ValueError(f"Salience must be in [0.0, 1.0], got {new_salience}")
        self.signature.salience = new_salience
    
    def update_autonomy(self, new_autonomy: float):
        """Update autonomy index with validation."""
        if not (0.0 <= new_autonomy <= 1.0):
            raise ValueError(f"Autonomy index must be in [0.0, 1.0], got {new_autonomy}")
        self.signature.autonomy_index = new_autonomy
    
    def compute_degrees_of_freedom(self) -> int:
        """
        Compute degrees of freedom (DOF) for the tensor.
        
        Returns the effective number of independent parameters
        in the cognitive primitive representation.
        """
        # Base DOF from tensor shape
        base_dof = np.prod(self.shape)
        
        # Constraints reduce DOF
        constraints = 2  # salience and autonomy range constraints
        
        # Consider sparsity and structure
        non_zero_elements = np.count_nonzero(self.data)
        effective_dof = min(base_dof - constraints, non_zero_elements)
        
        return max(1, effective_dof)  # At least 1 DOF
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tensor to dictionary for serialization."""
        return {
            "signature": self.signature.to_dict(),
            "data": self.data.tolist(),
            "shape": self.shape,
            "dof": self.compute_degrees_of_freedom()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitivePrimitiveTensor':
        """Create tensor from dictionary."""
        signature = TensorSignature.from_dict(data["signature"])
        tensor_data = np.array(data["data"], dtype=np.float32)
        shape = tuple(data["shape"])
        
        return cls(signature=signature, data=tensor_data, shape=shape)
    
    def __repr__(self) -> str:
        """String representation of cognitive primitive tensor."""
        return (
            f"CognitivePrimitiveTensor("
            f"modality={self.signature.modality.name}, "
            f"depth={self.signature.depth.name}, "
            f"context={self.signature.context.name}, "
            f"salience={self.signature.salience:.3f}, "
            f"autonomy={self.signature.autonomy_index:.3f}, "
            f"shape={self.shape}, "
            f"dof={self.compute_degrees_of_freedom()})"
        )


def create_primitive_tensor(
    modality: Union[ModalityType, str],
    depth: Union[DepthType, str],
    context: Union[ContextType, str],
    salience: float = 0.5,
    autonomy_index: float = 0.5,
    semantic_tags: Optional[List[str]] = None
) -> CognitivePrimitiveTensor:
    """
    Factory function to create cognitive primitive tensors with validation.
    
    Args:
        modality: Sensory/cognitive modality
        depth: Processing depth level
        context: Contextual scope
        salience: Attention salience [0.0, 1.0]
        autonomy_index: Autonomy level [0.0, 1.0]
        semantic_tags: Optional semantic annotations
    
    Returns:
        Configured CognitivePrimitiveTensor
    """
    # Convert string inputs to enums if needed
    if isinstance(modality, str):
        modality = ModalityType[modality.upper()]
    if isinstance(depth, str):
        depth = DepthType[depth.upper()]
    if isinstance(context, str):
        context = ContextType[context.upper()]
    
    signature = TensorSignature(
        modality=modality,
        depth=depth,
        context=context,
        salience=salience,
        autonomy_index=autonomy_index,
        semantic_tags=semantic_tags or []
    )
    
    return CognitivePrimitiveTensor(signature=signature)


# Predefined cognitive primitive patterns for common use cases
VISUAL_SURFACE_LOCAL = create_primitive_tensor(
    ModalityType.VISUAL, DepthType.SURFACE, ContextType.LOCAL,
    semantic_tags=["perception", "immediate", "visual"]
)

SYMBOLIC_PRAGMATIC_GLOBAL = create_primitive_tensor(
    ModalityType.SYMBOLIC, DepthType.PRAGMATIC, ContextType.GLOBAL,
    semantic_tags=["reasoning", "abstract", "global"]
)

TEXTUAL_SEMANTIC_TEMPORAL = create_primitive_tensor(
    ModalityType.TEXTUAL, DepthType.SEMANTIC, ContextType.TEMPORAL,
    semantic_tags=["language", "meaning", "temporal"]
)

AUDITORY_SURFACE_LOCAL = create_primitive_tensor(
    ModalityType.AUDITORY, DepthType.SURFACE, ContextType.LOCAL,
    semantic_tags=["sound", "immediate", "auditory"]
)