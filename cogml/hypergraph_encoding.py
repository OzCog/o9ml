"""
Hypergraph Encoding & AtomSpace Integration

This module provides bidirectional translation between cognitive primitives
and AtomSpace hypergraph patterns, enabling seamless integration between
tensor-based cognitive representations and symbolic reasoning systems.

Core Features:
- Scheme adapter microservices for agentic grammar integration
- Round-trip translation between tensors and hypergraph patterns
- AtomSpace node/link encoding with tensor metadata
- Cognitive grammar pattern generation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import re
from .cognitive_primitives import CognitivePrimitiveTensor, TensorSignature, ModalityType, DepthType, ContextType


@dataclass
class HypergraphNode:
    """
    Represents a node in the cognitive hypergraph with tensor metadata.
    
    Each node encodes cognitive state information and can be translated
    to/from AtomSpace node representations.
    """
    node_id: str
    node_type: str  # ConceptNode, WordNode, etc.
    tensor_signature: TensorSignature
    attributes: Dict[str, Any]
    
    def to_scheme(self) -> str:
        """Convert node to Scheme AtomSpace representation."""
        # Create node with tensor metadata
        scheme_lines = [
            f"({self.node_type} \"{self.node_id}\"",
            f"    (stv {self.tensor_signature.salience:.6f} {self.tensor_signature.autonomy_index:.6f})"
        ]
        
        # Add tensor signature as attributes if needed
        if self.attributes:
            for key, value in self.attributes.items():
                if isinstance(value, str):
                    scheme_lines.append(f"    (attribute \"{key}\" \"{value}\")")
                else:
                    scheme_lines.append(f"    (attribute \"{key}\" {value})")
        
        scheme_lines.append(")")
        return "\n".join(scheme_lines)


@dataclass
class HypergraphLink:
    """
    Represents a link/edge in the cognitive hypergraph.
    
    Links encode relationships between cognitive primitives with
    associated tensor weights and relational metadata.
    """
    link_id: str
    link_type: str  # EvaluationLink, InheritanceLink, etc.
    source_nodes: List[str]
    target_nodes: List[str]
    tensor_weights: np.ndarray
    relationship_type: str
    
    def to_scheme(self) -> str:
        """Convert link to Scheme AtomSpace representation."""
        # Calculate confidence from tensor weights
        confidence = float(np.mean(self.tensor_weights)) if len(self.tensor_weights) > 0 else 0.5
        strength = float(np.max(self.tensor_weights)) if len(self.tensor_weights) > 0 else 0.5
        
        # Create link structure
        scheme_lines = [
            f"({self.link_type}",
            f"    (stv {strength:.6f} {confidence:.6f})"
        ]
        
        # Add predicate if evaluation link
        if self.link_type == "EvaluationLink":
            scheme_lines.append(f"    (PredicateNode \"{self.relationship_type}\")")
            scheme_lines.append("    (ListLink")
            for node_id in self.source_nodes + self.target_nodes:
                scheme_lines.append(f"        (ConceptNode \"{node_id}\")")
            scheme_lines.append("    )")
        else:
            # Simple inheritance or other link types
            for node_id in self.source_nodes + self.target_nodes:
                scheme_lines.append(f"    (ConceptNode \"{node_id}\")")
        
        scheme_lines.append(")")
        return "\n".join(scheme_lines)


class SchemeTranslator:
    """
    Bidirectional translator between cognitive primitives and Scheme AtomSpace patterns.
    
    Provides round-trip translation capabilities ensuring encoding accuracy
    and maintaining tensor metadata throughout transformations.
    """
    
    def __init__(self):
        self.node_registry: Dict[str, HypergraphNode] = {}
        self.link_registry: Dict[str, HypergraphLink] = {}
        self.translation_history: List[Dict[str, Any]] = []
    
    def tensor_to_scheme(self, tensor: CognitivePrimitiveTensor, node_id: str) -> str:
        """
        Convert cognitive primitive tensor to Scheme representation.
        
        Args:
            tensor: Cognitive primitive tensor to convert
            node_id: Unique identifier for the resulting node
            
        Returns:
            Scheme code representing the tensor as AtomSpace patterns
        """
        # Determine node type based on tensor characteristics
        node_type = self._infer_node_type(tensor)
        
        # Create hypergraph node
        node = HypergraphNode(
            node_id=node_id,
            node_type=node_type,
            tensor_signature=tensor.signature,
            attributes={
                "modality": tensor.signature.modality.name,
                "depth": tensor.signature.depth.name,
                "context": tensor.signature.context.name,
                "prime_factors": tensor.signature.prime_factors,
                "dof": tensor.compute_degrees_of_freedom(),
                "tensor_shape": tensor.shape
            }
        )
        
        # Register node
        self.node_registry[node_id] = node
        
        # Record translation
        self.translation_history.append({
            "type": "tensor_to_scheme",
            "tensor_id": id(tensor),
            "node_id": node_id,
            "timestamp": tensor.signature.creation_timestamp
        })
        
        return node.to_scheme()
    
    def scheme_to_tensor(self, scheme_code: str) -> CognitivePrimitiveTensor:
        """
        Convert Scheme AtomSpace patterns back to cognitive primitive tensor.
        
        Args:
            scheme_code: Scheme representation to parse
            
        Returns:
            Reconstructed cognitive primitive tensor
        """
        # Parse scheme code to extract tensor metadata
        parsed_data = self._parse_scheme_code(scheme_code)
        
        # Reconstruct tensor signature
        signature = TensorSignature(
            modality=ModalityType[parsed_data["modality"]],
            depth=DepthType[parsed_data["depth"]],
            context=ContextType[parsed_data["context"]],
            salience=parsed_data.get("salience", 0.5),
            autonomy_index=parsed_data.get("autonomy_index", 0.5)
        )
        
        # Create tensor with reconstructed signature
        tensor = CognitivePrimitiveTensor(signature=signature)
        
        # Record reverse translation
        self.translation_history.append({
            "type": "scheme_to_tensor",
            "scheme_code": scheme_code[:100] + "..." if len(scheme_code) > 100 else scheme_code,
            "tensor_id": id(tensor),
            "timestamp": signature.creation_timestamp
        })
        
        return tensor
    
    def _infer_node_type(self, tensor: CognitivePrimitiveTensor) -> str:
        """Infer appropriate AtomSpace node type from tensor characteristics."""
        if tensor.signature.modality == ModalityType.TEXTUAL:
            if tensor.signature.depth == DepthType.SURFACE:
                return "WordNode"
            else:
                return "ConceptNode"
        elif tensor.signature.modality == ModalityType.SYMBOLIC:
            return "ConceptNode"
        elif tensor.signature.modality in [ModalityType.VISUAL, ModalityType.AUDITORY]:
            return "SensoryNode"
        else:
            return "ConceptNode"
    
    def _parse_scheme_code(self, scheme_code: str) -> Dict[str, Any]:
        """Parse Scheme code to extract tensor metadata."""
        parsed_data = {}
        
        # Extract strength and confidence values
        stv_match = re.search(r'stv\s+([\d.]+)\s+([\d.]+)', scheme_code)
        if stv_match:
            parsed_data["salience"] = float(stv_match.group(1))
            parsed_data["autonomy_index"] = float(stv_match.group(2))
        
        # Extract attributes
        attr_matches = re.findall(r'attribute\s+"([^"]+)"\s+"([^"]+)"', scheme_code)
        for key, value in attr_matches:
            parsed_data[key] = value
        
        # Set defaults if not found
        parsed_data.setdefault("modality", "SYMBOLIC")
        parsed_data.setdefault("depth", "SEMANTIC")
        parsed_data.setdefault("context", "LOCAL")
        
        return parsed_data
    
    def validate_round_trip(self, tensor: CognitivePrimitiveTensor, node_id: str) -> bool:
        """
        Validate round-trip translation accuracy.
        
        Tests tensor -> scheme -> tensor conversion to ensure information preservation.
        """
        try:
            # Forward translation
            scheme_code = self.tensor_to_scheme(tensor, node_id)
            
            # Reverse translation
            reconstructed_tensor = self.scheme_to_tensor(scheme_code)
            
            # Compare signatures
            original_sig = tensor.signature
            reconstructed_sig = reconstructed_tensor.signature
            
            # Check key properties
            modality_match = original_sig.modality == reconstructed_sig.modality
            depth_match = original_sig.depth == reconstructed_sig.depth
            context_match = original_sig.context == reconstructed_sig.context
            salience_match = abs(original_sig.salience - reconstructed_sig.salience) < 1e-6
            autonomy_match = abs(original_sig.autonomy_index - reconstructed_sig.autonomy_index) < 1e-6
            
            return all([modality_match, depth_match, context_match, salience_match, autonomy_match])
            
        except Exception as e:
            print(f"Round-trip validation failed: {e}")
            return False


class AtomSpaceAdapter:
    """
    Adapter for integrating cognitive primitives with AtomSpace operations.
    
    Provides high-level interface for creating, querying, and manipulating
    cognitive primitives within AtomSpace hypergraph structures.
    """
    
    def __init__(self):
        self.scheme_translator = SchemeTranslator()
        self.atomspace_patterns: List[str] = []
        self.cognitive_catalog: Dict[str, CognitivePrimitiveTensor] = {}
    
    def encode_agent_state(
        self,
        agent_id: str,
        state_tensors: List[CognitivePrimitiveTensor],
        relationships: Optional[List[Tuple[str, str, str]]] = None
    ) -> str:
        """
        Encode agent state as hypergraph nodes and links.
        
        Args:
            agent_id: Unique agent identifier
            state_tensors: List of cognitive primitive tensors representing state
            relationships: Optional list of (source, relation, target) tuples
            
        Returns:
            Complete Scheme representation of agent state
        """
        scheme_patterns = []
        
        # Create agent node
        agent_scheme = f"(ConceptNode \"{agent_id}\")"
        scheme_patterns.append(agent_scheme)
        
        # Encode each state tensor
        for i, tensor in enumerate(state_tensors):
            tensor_id = f"{agent_id}_state_{i}"
            tensor_scheme = self.scheme_translator.tensor_to_scheme(tensor, tensor_id)
            scheme_patterns.append(tensor_scheme)
            
            # Create association link between agent and state
            association = self._create_association_link(agent_id, tensor_id, "has_state")
            scheme_patterns.append(association)
            
            # Store in catalog
            self.cognitive_catalog[tensor_id] = tensor
        
        # Add explicit relationships if provided
        if relationships:
            for source, relation, target in relationships:
                rel_link = self._create_evaluation_link(source, relation, target)
                scheme_patterns.append(rel_link)
        
        # Combine all patterns
        complete_scheme = "\n\n".join(scheme_patterns)
        self.atomspace_patterns.append(complete_scheme)
        
        return complete_scheme
    
    def _create_association_link(self, source: str, target: str, relation_type: str) -> str:
        """Create an association link between two nodes."""
        return f"""(EvaluationLink
    (PredicateNode "{relation_type}")
    (ListLink
        (ConceptNode "{source}")
        (ConceptNode "{target}")
    )
)"""
    
    def _create_evaluation_link(self, source: str, relation: str, target: str) -> str:
        """Create an evaluation link with explicit relation."""
        return f"""(EvaluationLink
    (PredicateNode "{relation}")
    (ListLink
        (ConceptNode "{source}")
        (ConceptNode "{target}")
    )
)"""
    
    def query_cognitive_patterns(self, pattern_type: str) -> List[str]:
        """Query stored cognitive patterns by type."""
        matching_patterns = []
        for pattern in self.atomspace_patterns:
            if pattern_type.lower() in pattern.lower():
                matching_patterns.append(pattern)
        return matching_patterns
    
    def get_tensor_by_id(self, tensor_id: str) -> Optional[CognitivePrimitiveTensor]:
        """Retrieve cognitive primitive tensor by ID."""
        return self.cognitive_catalog.get(tensor_id)
    
    def export_atomspace_file(self, filename: str):
        """Export all patterns to a Scheme file for AtomSpace loading."""
        with open(filename, 'w') as f:
            f.write(";; Cognitive Primitives AtomSpace Encoding\n")
            f.write(";; Generated by CogML Hypergraph Encoding\n\n")
            
            for i, pattern in enumerate(self.atomspace_patterns):
                f.write(f";; Pattern {i+1}\n")
                f.write(pattern)
                f.write("\n\n")


class HypergraphEncoder:
    """
    Main encoder for converting cognitive architectures to hypergraph representations.
    
    Orchestrates the encoding process and provides utilities for visualization,
    validation, and performance analysis of cognitive primitive transformations.
    """
    
    def __init__(self):
        self.atomspace_adapter = AtomSpaceAdapter()
        self.encoding_stats: Dict[str, Any] = {
            "total_tensors_encoded": 0,
            "total_schemes_generated": 0,
            "successful_round_trips": 0,
            "failed_round_trips": 0,
            "average_encoding_time": 0.0
        }
    
    def encode_cognitive_system(
        self,
        agents: Dict[str, List[CognitivePrimitiveTensor]],
        system_relationships: Optional[List[Tuple[str, str, str]]] = None
    ) -> str:
        """
        Encode complete cognitive system with multiple agents.
        
        Args:
            agents: Dictionary mapping agent IDs to their cognitive state tensors
            system_relationships: Inter-agent relationships
            
        Returns:
            Complete AtomSpace representation of cognitive system
        """
        import time
        start_time = time.time()
        
        system_schemes = []
        
        # Encode each agent
        for agent_id, state_tensors in agents.items():
            agent_scheme = self.atomspace_adapter.encode_agent_state(
                agent_id, state_tensors
            )
            system_schemes.append(f";; Agent: {agent_id}")
            system_schemes.append(agent_scheme)
            
            self.encoding_stats["total_tensors_encoded"] += len(state_tensors)
        
        # Add system-level relationships
        if system_relationships:
            system_schemes.append(";; System Relationships")
            for source, relation, target in system_relationships:
                rel_scheme = self.atomspace_adapter._create_evaluation_link(
                    source, relation, target
                )
                system_schemes.append(rel_scheme)
        
        # Update statistics
        encoding_time = time.time() - start_time
        self.encoding_stats["total_schemes_generated"] += 1
        self.encoding_stats["average_encoding_time"] = (
            (self.encoding_stats["average_encoding_time"] * 
             (self.encoding_stats["total_schemes_generated"] - 1) + encoding_time) /
            self.encoding_stats["total_schemes_generated"]
        )
        
        return "\n\n".join(system_schemes)
    
    def validate_encoding_accuracy(self, test_tensors: List[CognitivePrimitiveTensor]) -> Dict[str, float]:
        """
        Run comprehensive validation tests on encoding accuracy.
        
        Args:
            test_tensors: List of tensors to test round-trip translation
            
        Returns:
            Validation metrics and accuracy statistics
        """
        results = {
            "total_tests": len(test_tensors),
            "successful_round_trips": 0,
            "failed_round_trips": 0,
            "accuracy_rate": 0.0,
            "average_precision": 0.0
        }
        
        precision_scores = []
        
        for i, tensor in enumerate(test_tensors):
            node_id = f"test_tensor_{i}"
            is_valid = self.atomspace_adapter.scheme_translator.validate_round_trip(
                tensor, node_id
            )
            
            if is_valid:
                results["successful_round_trips"] += 1
                precision_scores.append(1.0)
            else:
                results["failed_round_trips"] += 1
                precision_scores.append(0.0)
        
        # Calculate metrics
        results["accuracy_rate"] = results["successful_round_trips"] / results["total_tests"]
        results["average_precision"] = np.mean(precision_scores) if precision_scores else 0.0
        
        # Update global stats
        self.encoding_stats["successful_round_trips"] += results["successful_round_trips"]
        self.encoding_stats["failed_round_trips"] += results["failed_round_trips"]
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for encoding operations."""
        return {
            **self.encoding_stats,
            "total_patterns_generated": len(self.atomspace_adapter.atomspace_patterns),
            "catalog_size": len(self.atomspace_adapter.cognitive_catalog),
            "memory_usage_estimate": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage of encoded structures."""
        tensor_memory = 0
        scheme_memory = 0
        
        for tensor in self.atomspace_adapter.cognitive_catalog.values():
            tensor_memory += tensor.data.nbytes
        
        for pattern in self.atomspace_adapter.atomspace_patterns:
            scheme_memory += len(pattern.encode('utf-8'))
        
        return {
            "tensor_memory_bytes": tensor_memory,
            "scheme_memory_bytes": scheme_memory,
            "total_memory_bytes": tensor_memory + scheme_memory
        }