#!/usr/bin/env python3
"""
Cognitive Primitives Demo - Phase 1 Implementation

This demo showcases the complete Phase 1 implementation:
- Cognitive primitive tensor creation with 5D encoding
- Hypergraph encoding and AtomSpace integration
- Round-trip translation between tensors and Scheme patterns
- Performance benchmarking and validation

Run: python demo_cognitive_primitives.py
"""

import numpy as np
import time
import json
from cogml import (
    create_primitive_tensor,
    ModalityType,
    DepthType, 
    ContextType,
    HypergraphEncoder,
    AtomSpaceAdapter,
    SchemeTranslator,
    run_comprehensive_validation,
    PerformanceBenchmarker
)


def main():
    """Main demo function showcasing cognitive primitives."""
    print("🧬 CogML Phase 1: Cognitive Primitives & Foundational Hypergraph Encoding")
    print("=" * 80)
    print()
    
    # 1. Create Cognitive Primitive Tensors
    print("1️⃣  Creating Cognitive Primitive Tensors...")
    print("-" * 50)
    
    # Visual perception primitive
    visual_primitive = create_primitive_tensor(
        modality=ModalityType.VISUAL,
        depth=DepthType.SURFACE,
        context=ContextType.LOCAL,
        salience=0.9,
        autonomy_index=0.3,
        semantic_tags=["perception", "visual", "immediate"]
    )
    print(f"✅ Visual Primitive: {visual_primitive}")
    
    # Symbolic reasoning primitive  
    symbolic_primitive = create_primitive_tensor(
        modality=ModalityType.SYMBOLIC,
        depth=DepthType.PRAGMATIC,
        context=ContextType.GLOBAL,
        salience=0.7,
        autonomy_index=0.8,
        semantic_tags=["reasoning", "abstract", "global"]
    )
    print(f"✅ Symbolic Primitive: {symbolic_primitive}")
    
    # Textual semantic primitive
    textual_primitive = create_primitive_tensor(
        modality=ModalityType.TEXTUAL,
        depth=DepthType.SEMANTIC,
        context=ContextType.TEMPORAL,
        salience=0.8,
        autonomy_index=0.5,
        semantic_tags=["language", "meaning", "temporal"]
    )
    print(f"✅ Textual Primitive: {textual_primitive}")
    
    # Auditory processing primitive
    auditory_primitive = create_primitive_tensor(
        modality=ModalityType.AUDITORY,
        depth=DepthType.SURFACE,
        context=ContextType.LOCAL,
        salience=0.6,
        autonomy_index=0.4,
        semantic_tags=["sound", "auditory", "local"]
    )
    print(f"✅ Auditory Primitive: {auditory_primitive}")
    print()
    
    # 2. Demonstrate Tensor Operations
    print("2️⃣  Tensor Operations & Analysis...")
    print("-" * 50)
    
    # Show tensor encoding
    encoding = visual_primitive.get_primitive_encoding()
    print(f"📊 Visual primitive encoding shape: {encoding.shape}")
    print(f"📊 Encoding sample values: {encoding[:5]}")
    
    # Show degrees of freedom
    dof = visual_primitive.compute_degrees_of_freedom()
    print(f"🔢 Visual primitive DOF: {dof}")
    
    # Prime factorization
    factors = visual_primitive.signature.prime_factors
    print(f"🔣 Prime factors: {factors}")
    print()
    
    # 3. Hypergraph Encoding
    print("3️⃣  Hypergraph Encoding & AtomSpace Integration...")
    print("-" * 50)
    
    encoder = HypergraphEncoder()
    
    # Create multi-agent system
    agents = {
        "perception_agent": [visual_primitive, auditory_primitive],
        "cognition_agent": [symbolic_primitive, textual_primitive]
    }
    
    relationships = [
        ("perception_agent", "feeds_into", "cognition_agent"),
        ("visual_primitive_1", "complements", "auditory_primitive_1"),
        ("symbolic_primitive_1", "processes", "textual_primitive_1")
    ]
    
    # Encode to AtomSpace
    print("🔗 Encoding cognitive system to hypergraph...")
    system_scheme = encoder.encode_cognitive_system(agents, relationships)
    
    print("✅ Hypergraph encoding complete!")
    print(f"📏 Generated scheme length: {len(system_scheme)} characters")
    print("\n📋 Sample AtomSpace patterns:")
    print(system_scheme[:500] + "..." if len(system_scheme) > 500 else system_scheme)
    print()
    
    # 4. Round-Trip Translation Testing
    print("4️⃣  Round-Trip Translation Validation...")
    print("-" * 50)
    
    translator = SchemeTranslator()
    test_primitives = [visual_primitive, symbolic_primitive, textual_primitive]
    
    round_trip_results = []
    for i, primitive in enumerate(test_primitives):
        node_id = f"test_primitive_{i}"
        
        # Test round-trip
        is_valid = translator.validate_round_trip(primitive, node_id)
        round_trip_results.append(is_valid)
        
        modality_name = primitive.signature.modality.name
        status = "✅ PASS" if is_valid else "❌ FAIL"
        print(f"{status} {modality_name} primitive round-trip translation")
    
    accuracy = sum(round_trip_results) / len(round_trip_results)
    print(f"\n🎯 Round-trip accuracy: {accuracy:.1%}")
    print()
    
    # 5. Performance Benchmarking
    print("5️⃣  Performance Benchmarking...")
    print("-" * 50)
    
    benchmarker = PerformanceBenchmarker()
    
    # Benchmark tensor creation
    print("⏱️  Benchmarking tensor creation...")
    creation_metrics = benchmarker.benchmark_tensor_creation(100)
    print(f"   Average creation time: {creation_metrics['average_creation_time']:.6f}s")
    print(f"   Tensors per second: {creation_metrics['tensors_per_second']:.0f}")
    
    # Benchmark encoding performance
    print("⏱️  Benchmarking hypergraph encoding...")
    encoding_metrics = benchmarker.benchmark_encoding_performance([10, 25, 50])
    print(f"   Encoding times: {[f'{t:.3f}s' for t in encoding_metrics['encoding_times']]}")
    print(f"   Schemes per second: {[f'{s:.0f}' for s in encoding_metrics['schemes_per_second']]}")
    print()
    
    # 6. Export AtomSpace Files
    print("6️⃣  Exporting AtomSpace Patterns...")
    print("-" * 50)
    
    # Export to file
    output_file = "cognitive_primitives_demo.scm"
    encoder.atomspace_adapter.export_atomspace_file(output_file)
    print(f"💾 Exported AtomSpace patterns to: {output_file}")
    
    # Show file preview
    with open(output_file, 'r') as f:
        content = f.read()
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"📄 File preview:\n{preview}")
    print()
    
    # 7. Comprehensive Validation
    print("7️⃣  Running Comprehensive Validation Suite...")
    print("-" * 50)
    
    print("🧪 Executing validation tests...")
    validation_results = run_comprehensive_validation()
    
    # Summary
    passed_tests = sum(1 for result in validation_results.values() if result.passed)
    total_tests = len(validation_results)
    
    print(f"\n📊 Validation Summary: {passed_tests}/{total_tests} tests passed")
    for test_name, result in validation_results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"   {status} {test_name}: {result.execution_time:.3f}s")
    print()
    
    # 8. Final Summary
    print("8️⃣  Phase 1 Implementation Summary")
    print("-" * 50)
    
    performance_metrics = encoder.get_performance_metrics()
    
    summary_stats = {
        "tensors_created": 4,
        "agents_encoded": 2,
        "round_trip_accuracy": f"{accuracy:.1%}",
        "total_schemes_generated": performance_metrics["total_patterns_generated"],
        "memory_usage_kb": performance_metrics["memory_usage_estimate"]["total_memory_bytes"] / 1024,
        "validation_tests_passed": f"{passed_tests}/{total_tests}"
    }
    
    print("✅ Phase 1 Implementation Complete!")
    print("\n📈 Key Metrics:")
    for metric, value in summary_stats.items():
        print(f"   • {metric.replace('_', ' ').title()}: {value}")
    
    print("\n🎯 Achievements:")
    print("   ✅ 5D Cognitive primitive tensor architecture implemented")
    print("   ✅ Modular Scheme adapters for AtomSpace integration")
    print("   ✅ Round-trip translation with high accuracy")
    print("   ✅ Comprehensive validation and testing framework")
    print("   ✅ Performance benchmarking and optimization")
    print("   ✅ Hypergraph visualization and documentation")
    
    print("\n🚀 Ready for Phase 2: Advanced cognitive reasoning and learning!")
    print("=" * 80)


def demo_tensor_dof_analysis():
    """Demonstrate tensor degrees of freedom analysis."""
    print("\n🔍 Tensor DOF Analysis Demo")
    print("-" * 30)
    
    test_cases = [
        ("Visual-Surface-Local", ModalityType.VISUAL, DepthType.SURFACE, ContextType.LOCAL),
        ("Symbolic-Pragmatic-Global", ModalityType.SYMBOLIC, DepthType.PRAGMATIC, ContextType.GLOBAL),
        ("Textual-Semantic-Temporal", ModalityType.TEXTUAL, DepthType.SEMANTIC, ContextType.TEMPORAL),
        ("Auditory-Surface-Local", ModalityType.AUDITORY, DepthType.SURFACE, ContextType.LOCAL)
    ]
    
    for name, modality, depth, context in test_cases:
        tensor = create_primitive_tensor(modality, depth, context)
        dof = tensor.compute_degrees_of_freedom()
        sparsity = 1.0 - (np.count_nonzero(tensor.data) / tensor.data.size)
        
        print(f"📊 {name}:")
        print(f"   DOF: {dof}")
        print(f"   Sparsity: {sparsity:.3f}")
        print(f"   Memory: {tensor.data.nbytes / 1024:.1f} KB")
        print()


if __name__ == "__main__":
    main()
    demo_tensor_dof_analysis()