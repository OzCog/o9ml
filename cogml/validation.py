"""
Validation Framework for Cognitive Primitives & Hypergraph Encoding

This module provides comprehensive validation, testing, and verification
utilities for cognitive primitive tensors and their hypergraph encodings.

Core Features:
- Tensor shape and constraint validation
- Round-trip translation accuracy testing
- Performance benchmarking for primitive operations
- Memory usage analysis and optimization
- Exhaustive test pattern generation
"""

import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from .cognitive_primitives import (
    CognitivePrimitiveTensor, TensorSignature, ModalityType, DepthType, ContextType,
    create_primitive_tensor
)
from .hypergraph_encoding import HypergraphEncoder, AtomSpaceAdapter, SchemeTranslator


@dataclass
class ValidationResult:
    """
    Comprehensive validation result with metrics and diagnostics.
    """
    test_name: str
    passed: bool
    execution_time: float
    memory_usage: Dict[str, float]
    error_message: Optional[str] = None
    metrics: Dict[str, float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "error_message": self.error_message,
            "metrics": self.metrics or {}
        }


class TensorValidator:
    """
    Validator for cognitive primitive tensor structure and constraints.
    
    Ensures tensor integrity, validates dimensional constraints,
    and checks tensor signature consistency.
    """
    
    def __init__(self):
        self.validation_history: List[ValidationResult] = []
    
    def validate_tensor_structure(self, tensor: CognitivePrimitiveTensor) -> ValidationResult:
        """
        Validate tensor structure and dimensional constraints.
        
        Args:
            tensor: Cognitive primitive tensor to validate
            
        Returns:
            ValidationResult with structure validation details
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Test 1: Shape validation
            if len(tensor.shape) != 5:
                raise ValueError(f"Expected 5D tensor, got {len(tensor.shape)}D")
            
            # Test 2: Dimension constraints
            expected_dims = (4, 3, 3, 100, 100)  # [modality, depth, context, salience_bins, autonomy_bins]
            if tensor.shape[:3] != expected_dims[:3]:
                raise ValueError(f"Invalid categorical dimensions: {tensor.shape[:3]}")
            
            # Test 3: Data type validation
            if tensor.data.dtype != np.float32:
                raise ValueError(f"Expected float32 dtype, got {tensor.data.dtype}")
            
            # Test 4: Signature consistency
            self._validate_signature_consistency(tensor.signature)
            
            # Test 5: Tensor data integrity
            if np.any(np.isnan(tensor.data)) or np.any(np.isinf(tensor.data)):
                raise ValueError("Tensor contains NaN or infinite values")
            
            # Calculate metrics
            metrics = {
                "data_sparsity": 1.0 - (np.count_nonzero(tensor.data) / tensor.data.size),
                "data_range_min": float(np.min(tensor.data)),
                "data_range_max": float(np.max(tensor.data)),
                "degrees_of_freedom": tensor.compute_degrees_of_freedom()
            }
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            result = ValidationResult(
                test_name="tensor_structure_validation",
                passed=True,
                execution_time=end_time - start_time,
                memory_usage={
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": end_memory - start_memory
                },
                metrics=metrics
            )
            
        except Exception as e:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            result = ValidationResult(
                test_name="tensor_structure_validation",
                passed=False,
                execution_time=end_time - start_time,
                memory_usage={
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": end_memory - start_memory
                },
                error_message=str(e)
            )
        
        self.validation_history.append(result)
        return result
    
    def _validate_signature_consistency(self, signature: TensorSignature):
        """Validate tensor signature internal consistency."""
        # Range checks
        if not (0.0 <= signature.salience <= 1.0):
            raise ValueError(f"Salience out of range: {signature.salience}")
        if not (0.0 <= signature.autonomy_index <= 1.0):
            raise ValueError(f"Autonomy index out of range: {signature.autonomy_index}")
        
        # Enum validation
        if not isinstance(signature.modality, ModalityType):
            raise ValueError(f"Invalid modality type: {type(signature.modality)}")
        if not isinstance(signature.depth, DepthType):
            raise ValueError(f"Invalid depth type: {type(signature.depth)}")
        if not isinstance(signature.context, ContextType):
            raise ValueError(f"Invalid context type: {type(signature.context)}")
        
        # Prime factors consistency
        if signature.prime_factors is None or len(signature.prime_factors) == 0:
            raise ValueError("Prime factors not computed")
    
    def validate_tensor_operations(self, tensor: CognitivePrimitiveTensor) -> ValidationResult:
        """
        Validate tensor operations and transformations.
        
        Tests key tensor methods for correctness and performance.
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Test encoding generation
            encoding = tensor.get_primitive_encoding()
            if len(encoding) != 32:  # Expected encoding length
                raise ValueError(f"Invalid encoding length: {len(encoding)}")
            
            # Test salience updates
            original_salience = tensor.signature.salience
            tensor.update_salience(0.8)
            if tensor.signature.salience != 0.8:
                raise ValueError("Salience update failed")
            tensor.update_salience(original_salience)  # Restore
            
            # Test autonomy updates
            original_autonomy = tensor.signature.autonomy_index
            tensor.update_autonomy(0.3)
            if tensor.signature.autonomy_index != 0.3:
                raise ValueError("Autonomy update failed")
            tensor.update_autonomy(original_autonomy)  # Restore
            
            # Test serialization
            tensor_dict = tensor.to_dict()
            reconstructed = CognitivePrimitiveTensor.from_dict(tensor_dict)
            if reconstructed.shape != tensor.shape:
                raise ValueError("Serialization round-trip failed")
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            metrics = {
                "encoding_generation_time": 0.001,  # Placeholder
                "serialization_size_bytes": len(json.dumps(tensor_dict)),
                "operations_per_second": 1000.0 / (end_time - start_time)
            }
            
            result = ValidationResult(
                test_name="tensor_operations_validation",
                passed=True,
                execution_time=end_time - start_time,
                memory_usage={
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": end_memory - start_memory
                },
                metrics=metrics
            )
            
        except Exception as e:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            result = ValidationResult(
                test_name="tensor_operations_validation",
                passed=False,
                execution_time=end_time - start_time,
                memory_usage={
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": end_memory - start_memory
                },
                error_message=str(e)
            )
        
        self.validation_history.append(result)
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class PrimitiveValidator:
    """
    Validator for cognitive primitive patterns and transformations.
    
    Tests primitive creation, manipulation, and cognitive semantics.
    """
    
    def __init__(self):
        self.test_patterns: List[CognitivePrimitiveTensor] = []
        self.validation_history: List[ValidationResult] = []
    
    def generate_exhaustive_test_patterns(self) -> List[CognitivePrimitiveTensor]:
        """
        Generate exhaustive test patterns covering all primitive combinations.
        
        Creates comprehensive test suite for all modality/depth/context combinations
        with various salience and autonomy values.
        """
        test_patterns = []
        
        # Test all enum combinations
        for modality in ModalityType:
            for depth in DepthType:
                for context in ContextType:
                    # Test with different salience/autonomy values
                    for salience in [0.0, 0.25, 0.5, 0.75, 1.0]:
                        for autonomy in [0.0, 0.25, 0.5, 0.75, 1.0]:
                            tensor = create_primitive_tensor(
                                modality=modality,
                                depth=depth,
                                context=context,
                                salience=salience,
                                autonomy_index=autonomy,
                                semantic_tags=[f"test_{modality.name}_{depth.name}_{context.name}"]
                            )
                            test_patterns.append(tensor)
        
        self.test_patterns = test_patterns
        return test_patterns
    
    def validate_primitive_patterns(self) -> ValidationResult:
        """
        Validate all generated primitive patterns.
        
        Tests creation, consistency, and basic operations on all patterns.
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            if not self.test_patterns:
                self.generate_exhaustive_test_patterns()
            
            failed_patterns = []
            tensor_validator = TensorValidator()
            
            for i, pattern in enumerate(self.test_patterns):
                structure_result = tensor_validator.validate_tensor_structure(pattern)
                operations_result = tensor_validator.validate_tensor_operations(pattern)
                
                if not (structure_result.passed and operations_result.passed):
                    failed_patterns.append(i)
            
            if failed_patterns:
                raise ValueError(f"Failed patterns: {failed_patterns}")
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            metrics = {
                "total_patterns_tested": len(self.test_patterns),
                "pattern_density": len(self.test_patterns) / (4 * 3 * 3 * 5 * 5),  # Coverage
                "average_pattern_size_bytes": np.mean([p.data.nbytes for p in self.test_patterns]),
                "validation_rate_patterns_per_second": len(self.test_patterns) / (end_time - start_time)
            }
            
            result = ValidationResult(
                test_name="primitive_patterns_validation",
                passed=True,
                execution_time=end_time - start_time,
                memory_usage={
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": end_memory - start_memory
                },
                metrics=metrics
            )
            
        except Exception as e:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            result = ValidationResult(
                test_name="primitive_patterns_validation",
                passed=False,
                execution_time=end_time - start_time,
                memory_usage={
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": end_memory - start_memory
                },
                error_message=str(e)
            )
        
        self.validation_history.append(result)
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class EncodingValidator:
    """
    Validator for hypergraph encoding and AtomSpace integration.
    
    Tests round-trip translation accuracy, scheme generation,
    and AtomSpace pattern correctness.
    """
    
    def __init__(self):
        self.encoder = HypergraphEncoder()
        self.validation_history: List[ValidationResult] = []
    
    def validate_round_trip_accuracy(self, test_tensors: List[CognitivePrimitiveTensor]) -> ValidationResult:
        """
        Comprehensive round-trip translation accuracy validation.
        
        Tests tensor -> scheme -> tensor translation for fidelity.
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            accuracy_results = self.encoder.validate_encoding_accuracy(test_tensors)
            
            if accuracy_results["accuracy_rate"] < 0.95:  # 95% accuracy threshold
                raise ValueError(f"Round-trip accuracy too low: {accuracy_results['accuracy_rate']}")
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            metrics = {
                **accuracy_results,
                "accuracy_threshold_met": accuracy_results["accuracy_rate"] >= 0.95,
                "translation_rate_per_second": len(test_tensors) / (end_time - start_time)
            }
            
            result = ValidationResult(
                test_name="round_trip_accuracy_validation",
                passed=True,
                execution_time=end_time - start_time,
                memory_usage={
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": end_memory - start_memory
                },
                metrics=metrics
            )
            
        except Exception as e:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            result = ValidationResult(
                test_name="round_trip_accuracy_validation",
                passed=False,
                execution_time=end_time - start_time,
                memory_usage={
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": end_memory - start_memory
                },
                error_message=str(e)
            )
        
        self.validation_history.append(result)
        return result
    
    def validate_scheme_generation(self, test_tensors: List[CognitivePrimitiveTensor]) -> ValidationResult:
        """
        Validate Scheme code generation quality and correctness.
        
        Tests generated Scheme patterns for syntax and semantic correctness.
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            scheme_translator = SchemeTranslator()
            generated_schemes = []
            
            for i, tensor in enumerate(test_tensors):
                node_id = f"validation_tensor_{i}"
                scheme_code = scheme_translator.tensor_to_scheme(tensor, node_id)
                
                # Basic syntax validation
                if not self._validate_scheme_syntax(scheme_code):
                    raise ValueError(f"Invalid Scheme syntax for tensor {i}")
                
                generated_schemes.append(scheme_code)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            metrics = {
                "schemes_generated": len(generated_schemes),
                "average_scheme_length": np.mean([len(s) for s in generated_schemes]),
                "generation_rate_per_second": len(generated_schemes) / (end_time - start_time),
                "total_scheme_size_bytes": sum(len(s.encode('utf-8')) for s in generated_schemes)
            }
            
            result = ValidationResult(
                test_name="scheme_generation_validation",
                passed=True,
                execution_time=end_time - start_time,
                memory_usage={
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": end_memory - start_memory
                },
                metrics=metrics
            )
            
        except Exception as e:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            result = ValidationResult(
                test_name="scheme_generation_validation",
                passed=False,
                execution_time=end_time - start_time,
                memory_usage={
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": end_memory - start_memory
                },
                error_message=str(e)
            )
        
        self.validation_history.append(result)
        return result
    
    def _validate_scheme_syntax(self, scheme_code: str) -> bool:
        """Basic Scheme syntax validation."""
        # Count parentheses
        open_parens = scheme_code.count('(')
        close_parens = scheme_code.count(')')
        
        if open_parens != close_parens:
            return False
        
        # Check for basic AtomSpace patterns
        required_patterns = ['Node', 'Link']
        has_required = any(pattern in scheme_code for pattern in required_patterns)
        
        return has_required
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class PerformanceBenchmarker:
    """
    Performance benchmarking for cognitive primitive operations.
    
    Measures execution times, memory usage, and scalability characteristics
    of cognitive primitive and hypergraph encoding operations.
    """
    
    def __init__(self):
        self.benchmark_results: Dict[str, List[Dict[str, float]]] = {}
    
    def benchmark_tensor_creation(self, num_iterations: int = 1000) -> Dict[str, float]:
        """
        Benchmark tensor creation performance.
        
        Measures time and memory usage for creating cognitive primitive tensors.
        """
        start_memory = self._get_memory_usage()
        
        creation_times = []
        for i in range(num_iterations):
            start_time = time.perf_counter()
            
            tensor = create_primitive_tensor(
                modality=ModalityType.SYMBOLIC,
                depth=DepthType.SEMANTIC,
                context=ContextType.LOCAL,
                salience=0.5,
                autonomy_index=0.5
            )
            
            end_time = time.perf_counter()
            creation_times.append(end_time - start_time)
        
        end_memory = self._get_memory_usage()
        
        results = {
            "average_creation_time": np.mean(creation_times),
            "min_creation_time": np.min(creation_times),
            "max_creation_time": np.max(creation_times),
            "std_creation_time": np.std(creation_times),
            "memory_usage_mb": end_memory - start_memory,
            "tensors_per_second": 1.0 / np.mean(creation_times)
        }
        
        self.benchmark_results["tensor_creation"] = [results]
        return results
    
    def benchmark_encoding_performance(
        self,
        tensor_counts: List[int] = [10, 50, 100, 500, 1000]
    ) -> Dict[str, List[float]]:
        """
        Benchmark hypergraph encoding performance at different scales.
        
        Tests performance scalability for encoding varying numbers of tensors.
        """
        results = {
            "tensor_counts": tensor_counts,
            "encoding_times": [],
            "memory_usage": [],
            "schemes_per_second": []
        }
        
        encoder = HypergraphEncoder()
        
        for count in tensor_counts:
            # Generate test tensors
            test_tensors = []
            for i in range(count):
                tensor = create_primitive_tensor(
                    modality=ModalityType(i % 4),
                    depth=DepthType(i % 3),
                    context=ContextType(i % 3),
                    salience=0.5,
                    autonomy_index=0.5
                )
                test_tensors.append(tensor)
            
            # Benchmark encoding
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage()
            
            agents = {f"agent_{i}": [test_tensors[i]] for i in range(count)}
            encoded_system = encoder.encode_cognitive_system(agents)
            
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            encoding_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            results["encoding_times"].append(encoding_time)
            results["memory_usage"].append(memory_delta)
            results["schemes_per_second"].append(count / encoding_time)
            
            # Cleanup
            del test_tensors, agents, encoded_system
            gc.collect()
        
        self.benchmark_results["encoding_performance"] = [results]
        return results
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance analysis report."""
        report_lines = [
            "# CogML Performance Benchmark Report",
            "",
            "## Summary",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        for benchmark_name, results_list in self.benchmark_results.items():
            report_lines.append(f"## {benchmark_name.replace('_', ' ').title()}")
            
            for results in results_list:
                for key, value in results.items():
                    if isinstance(value, float):
                        report_lines.append(f"- {key}: {value:.6f}")
                    elif isinstance(value, list) and len(value) <= 10:
                        report_lines.append(f"- {key}: {value}")
                    else:
                        report_lines.append(f"- {key}: [array of {len(value)} values]")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


def run_comprehensive_validation() -> Dict[str, ValidationResult]:
    """
    Run comprehensive validation suite for all cognitive primitive components.
    
    Returns:
        Dictionary of validation results for each test category
    """
    results = {}
    
    print("🧬 Running Comprehensive Cognitive Primitives Validation...")
    
    # Generate test data
    primitive_validator = PrimitiveValidator()
    test_patterns = primitive_validator.generate_exhaustive_test_patterns()[:50]  # Limit for speed
    
    print(f"Generated {len(test_patterns)} test patterns")
    
    # 1. Tensor Structure Validation
    print("Testing tensor structure...")
    tensor_validator = TensorValidator()
    if test_patterns:
        results["tensor_structure"] = tensor_validator.validate_tensor_structure(test_patterns[0])
        results["tensor_operations"] = tensor_validator.validate_tensor_operations(test_patterns[0])
    
    # 2. Primitive Patterns Validation
    print("Testing primitive patterns...")
    results["primitive_patterns"] = primitive_validator.validate_primitive_patterns()
    
    # 3. Encoding Validation
    print("Testing hypergraph encoding...")
    encoding_validator = EncodingValidator()
    results["round_trip_accuracy"] = encoding_validator.validate_round_trip_accuracy(test_patterns[:10])
    results["scheme_generation"] = encoding_validator.validate_scheme_generation(test_patterns[:10])
    
    # 4. Performance Benchmarking
    print("Running performance benchmarks...")
    benchmarker = PerformanceBenchmarker()
    creation_perf = benchmarker.benchmark_tensor_creation(100)
    encoding_perf = benchmarker.benchmark_encoding_performance([10, 25, 50])
    
    print("✅ Validation complete!")
    
    # Summary
    passed_tests = sum(1 for result in results.values() if result.passed)
    total_tests = len(results)
    
    print(f"\n📊 Validation Summary: {passed_tests}/{total_tests} tests passed")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status} {test_name}: {result.execution_time:.3f}s")
        if not result.passed:
            print(f"    Error: {result.error_message}")
    
    return results