"""
CogML - Comprehensive cognitive architecture for artificial general intelligence

This package provides the core components for the CogML cognitive architecture,
including cognitive primitive tensors, hypergraph encoding, AtomSpace integration,
and neural-symbolic reasoning capabilities.

Phase 1 Components:
- Cognitive primitive tensor encoding with 5D shape [modality, depth, context, salience, autonomy_index]
- Bidirectional scheme-to-hypergraph translation
- AtomSpace integration for symbolic reasoning
- Comprehensive validation and testing framework
"""

__version__ = "0.1.0"
__author__ = "OpenCog Community"
__email__ = "info@opencog.org"
__license__ = "Apache-2.0"

# Phase 1: Cognitive Primitives & Foundational Hypergraph Encoding
from .cognitive_primitives import (
    CognitivePrimitiveTensor,
    TensorSignature,
    ModalityType,
    DepthType,
    ContextType,
    create_primitive_tensor,
    VISUAL_SURFACE_LOCAL,
    SYMBOLIC_PRAGMATIC_GLOBAL,
    TEXTUAL_SEMANTIC_TEMPORAL,
    AUDITORY_SURFACE_LOCAL
)

from .hypergraph_encoding import (
    HypergraphEncoder,
    AtomSpaceAdapter,
    SchemeTranslator,
    HypergraphNode,
    HypergraphLink
)

from .validation import (
    TensorValidator,
    PrimitiveValidator,
    EncodingValidator,
    PerformanceBenchmarker,
    ValidationResult,
    run_comprehensive_validation
)

# Core modules for future phases
# from .atomspace import AtomSpace
# from .pln import PLN
# from .learning import LearningEngine
# from .agents import CognitiveAgent

__all__ = [
    # Cognitive Primitives
    "CognitivePrimitiveTensor",
    "TensorSignature", 
    "ModalityType",
    "DepthType",
    "ContextType",
    "create_primitive_tensor",
    
    # Predefined patterns
    "VISUAL_SURFACE_LOCAL",
    "SYMBOLIC_PRAGMATIC_GLOBAL", 
    "TEXTUAL_SEMANTIC_TEMPORAL",
    "AUDITORY_SURFACE_LOCAL",
    
    # Hypergraph Encoding
    "HypergraphEncoder",
    "AtomSpaceAdapter",
    "SchemeTranslator",
    "HypergraphNode",
    "HypergraphLink",
    
    # Validation Framework
    "TensorValidator",
    "PrimitiveValidator",
    "EncodingValidator",
    "PerformanceBenchmarker",
    "ValidationResult",
    "run_comprehensive_validation"
]