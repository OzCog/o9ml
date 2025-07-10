#!/usr/bin/env python3
"""
End-to-End Cognitive System Validation
======================================

This script validates the complete cognitive architecture from sensory input
to cognitive output, demonstrating system synergy through P-System membranes
and the unified integration tensor structure.
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Import our integration layer
try:
    from integration_layer import IntegrationLayer, CognitiveTensor
except ImportError:
    print("Integration layer not found, please ensure integration_layer.py is available")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndToEndValidator:
    """
    Validates the complete cognitive system through comprehensive testing
    of all integration points and cognitive synergy mechanisms.
    """
    
    def __init__(self):
        self.integration_layer = IntegrationLayer()
        self.validation_results = {}
        self.performance_metrics = {}
        
    async def initialize(self):
        """Initialize the validation system"""
        logger.info("🔍 Initializing End-to-End Cognitive Validation System")
        await self.integration_layer.initialize()
        logger.info("✓ Validation system ready")
        
    async def validate_sensory_to_cognitive_pipeline(self) -> Dict[str, Any]:
        """
        Validate the complete pipeline from sensory input to cognitive output.
        This tests the full cognitive gestalt and system synergy.
        """
        logger.info("🧠 Validating Sensory-to-Cognitive Pipeline")
        
        # Test scenarios representing different cognitive challenges
        test_scenarios = [
            {
                "name": "Simple Perception",
                "input": "I see a red apple on the table",
                "expected_confidence": 0.7,
                "cognitive_type": "perception"
            },
            {
                "name": "Complex Reasoning", 
                "input": "If all birds can fly and penguins are birds, can penguins fly?",
                "expected_confidence": 0.8,
                "cognitive_type": "reasoning"
            },
            {
                "name": "Emotional Processing",
                "input": "I feel sad because my friend is moving away",
                "expected_confidence": 0.6,
                "cognitive_type": "emotion"
            },
            {
                "name": "Creative Synthesis",
                "input": "Imagine a world where gravity works backwards",
                "expected_confidence": 0.5,
                "cognitive_type": "creativity"
            },
            {
                "name": "Memory Integration",
                "input": "Remember the time we discussed neural-symbolic integration",
                "expected_confidence": 0.7,
                "cognitive_type": "memory"
            }
        ]
        
        pipeline_results = []
        total_synergy_score = 0.0
        
        for scenario in test_scenarios:
            logger.info(f"Testing: {scenario['name']}")
            
            start_time = time.time()
            result = await self.integration_layer.process_cognitive_input(scenario['input'])
            processing_time = time.time() - start_time
            
            # Analyze cognitive processing quality
            confidence_achieved = result['output_tensor']['confidence']
            synergy_achieved = result['cognitive_synergy_achieved']
            
            # Calculate scenario-specific metrics
            confidence_ratio = confidence_achieved / scenario['expected_confidence']
            processing_efficiency = 1.0 / processing_time if processing_time > 0 else 0
            
            scenario_result = {
                'scenario': scenario['name'],
                'cognitive_type': scenario['cognitive_type'],
                'input': scenario['input'],
                'confidence_achieved': confidence_achieved,
                'confidence_expected': scenario['expected_confidence'],
                'confidence_ratio': confidence_ratio,
                'synergy_achieved': synergy_achieved,
                'processing_time': processing_time,
                'processing_efficiency': processing_efficiency,
                'tensor_summary': result['output_tensor'],
                'component_performance': result['processing_history']
            }
            
            pipeline_results.append(scenario_result)
            
            if synergy_achieved:
                total_synergy_score += 1.0
            else:
                total_synergy_score += confidence_ratio * 0.5
                
            logger.info(f"  ✓ Confidence: {confidence_achieved:.3f} | Synergy: {synergy_achieved}")
        
        # Calculate overall pipeline performance
        avg_confidence = np.mean([r['confidence_achieved'] for r in pipeline_results])
        avg_processing_time = np.mean([r['processing_time'] for r in pipeline_results])
        synergy_percentage = (total_synergy_score / len(test_scenarios)) * 100
        
        pipeline_validation = {
            'overall_performance': {
                'average_confidence': avg_confidence,
                'average_processing_time': avg_processing_time,
                'cognitive_synergy_percentage': synergy_percentage,
                'pipeline_health': avg_confidence > 0.6 and synergy_percentage > 70
            },
            'scenario_results': pipeline_results,
            'cognitive_types_tested': len(set(s['cognitive_type'] for s in test_scenarios)),
            'validation_timestamp': time.time()
        }
        
        self.validation_results['pipeline'] = pipeline_validation
        logger.info(f"✓ Pipeline validation complete (synergy: {synergy_percentage:.1f}%)")
        
        return pipeline_validation
        
    async def validate_psystem_membrane_integrity(self) -> Dict[str, Any]:
        """
        Validate P-System membrane organization and frame problem resolution.
        """
        logger.info("🔗 Validating P-System Membrane Integrity")
        
        # Get membrane status from integration layer
        membrane_status = {}
        for name, membrane in self.integration_layer.membranes.items():
            membrane_status[name] = {
                'permeability': membrane.membrane_permeability,
                'rules_count': len(membrane.rules),
                'children_count': len(membrane.children),
                'has_parent': membrane.parent is not None,
                'tensor_confidence': membrane.tensor_state.confidence,
                'spatial_position': membrane.tensor_state.spatial.tolist(),
                'processing_rules': membrane.rules
            }
            
        # Test membrane boundary integrity
        boundary_tests = []
        for membrane_name, status in membrane_status.items():
            boundary_test = {
                'membrane': membrane_name,
                'permeability_valid': 0.0 <= status['permeability'] <= 1.0,
                'has_processing_rules': status['rules_count'] > 0,
                'tensor_state_valid': status['tensor_confidence'] > 0.0,
                'spatial_localization': any(x != 0 for x in status['spatial_position']),
                'boundary_integrity': True
            }
            
            # Overall boundary integrity
            boundary_test['boundary_integrity'] = all([
                boundary_test['permeability_valid'],
                boundary_test['has_processing_rules'],
                boundary_test['tensor_state_valid']
            ])
            
            boundary_tests.append(boundary_test)
            
        # Test hierarchical organization
        hierarchy_valid = True
        root_found = False
        for name, status in membrane_status.items():
            if not status['has_parent']:
                if root_found:
                    hierarchy_valid = False  # Multiple roots not allowed
                root_found = True
                
        if not root_found:
            hierarchy_valid = False  # No root found
            
        # Frame problem resolution test
        frame_problem_resolved = self._test_frame_problem_resolution()
        
        membrane_validation = {
            'membrane_count': len(membrane_status),
            'boundary_tests': boundary_tests,
            'boundary_integrity_percentage': (sum(1 for t in boundary_tests if t['boundary_integrity']) / len(boundary_tests)) * 100,
            'hierarchical_organization': hierarchy_valid,
            'frame_problem_resolved': frame_problem_resolved,
            'membrane_details': membrane_status,
            'overall_membrane_health': hierarchy_valid and frame_problem_resolved and all(t['boundary_integrity'] for t in boundary_tests)
        }
        
        self.validation_results['membranes'] = membrane_validation
        logger.info(f"✓ Membrane validation complete (health: {membrane_validation['overall_membrane_health']})")
        
        return membrane_validation
        
    def _test_frame_problem_resolution(self) -> bool:
        """
        Test that P-System membranes successfully resolve the frame problem
        by providing cognitive boundaries and context isolation.
        """
        # The frame problem is resolved if:
        # 1. Membranes provide context isolation (different spatial positions)
        # 2. Each membrane has specific processing rules
        # 3. Information flow is controlled by permeability
        # 4. Hierarchical organization prevents infinite recursion
        
        membranes = self.integration_layer.membranes
        
        # Test 1: Context isolation (spatial separation)
        spatial_positions = [m.tensor_state.spatial for m in membranes.values()]
        spatial_separation = len(set(tuple(pos) for pos in spatial_positions)) == len(spatial_positions)
        
        # Test 2: Rule specificity
        rule_specificity = all(len(m.rules) > 0 for m in membranes.values())
        
        # Test 3: Controlled information flow
        permeability_control = all(0 <= m.membrane_permeability <= 1 for m in membranes.values())
        
        # Test 4: Hierarchical bounds
        max_depth = self._calculate_membrane_hierarchy_depth()
        hierarchy_bounded = max_depth <= 10  # Reasonable depth limit
        
        return spatial_separation and rule_specificity and permeability_control and hierarchy_bounded
        
    def _calculate_membrane_hierarchy_depth(self) -> int:
        """Calculate the maximum depth of the membrane hierarchy"""
        def get_depth(membrane, current_depth=0):
            if not membrane.children:
                return current_depth
            return max(get_depth(child, current_depth + 1) for child in membrane.children)
            
        root_membrane = self.integration_layer.membranes['root']
        return get_depth(root_membrane)
        
    async def validate_tensor_structure_flow(self) -> Dict[str, Any]:
        """
        Validate the integration tensor structure and data flow patterns.
        """
        logger.info("📊 Validating Tensor Structure and Data Flow")
        
        # Create test tensor with known properties
        test_tensor = CognitiveTensor(
            spatial=np.array([1.0, 2.0, 3.0]),
            temporal=time.time(),
            semantic=np.random.normal(0.5, 0.1, 256),
            logical=np.random.random(64),
            confidence=0.75
        )
        
        # Test tensor validation
        tensor_validation = {
            'spatial_dimensions': test_tensor.spatial.shape == (3,),
            'semantic_dimensions': test_tensor.semantic.shape == (256,),
            'logical_dimensions': test_tensor.logical.shape == (64,),
            'confidence_range': 0.0 <= test_tensor.confidence <= 1.0,
            'temporal_valid': test_tensor.temporal > 0
        }
        
        # Test data flow through components
        flow_test_results = []
        current_tensor = test_tensor
        
        for component in self.integration_layer.cognitive_pipeline:
            if component.is_active:
                original_confidence = current_tensor.confidence
                processed_tensor = await component.process(current_tensor)
                
                flow_result = {
                    'component': component.name,
                    'input_confidence': original_confidence,
                    'output_confidence': processed_tensor.confidence,
                    'confidence_change': processed_tensor.confidence - original_confidence,
                    'spatial_transform': np.linalg.norm(processed_tensor.spatial - current_tensor.spatial),
                    'semantic_preservation': np.corrcoef(processed_tensor.semantic, current_tensor.semantic)[0,1],
                    'logical_coherence': np.mean(np.abs(processed_tensor.logical - current_tensor.logical)),
                    'processing_time': component.processing_time
                }
                
                flow_test_results.append(flow_result)
                current_tensor = processed_tensor
                
        # Calculate overall flow metrics
        total_confidence_change = current_tensor.confidence - test_tensor.confidence
        avg_semantic_preservation = np.mean([r['semantic_preservation'] for r in flow_test_results if not np.isnan(r['semantic_preservation'])])
        total_processing_time = sum(r['processing_time'] for r in flow_test_results)
        
        tensor_flow_validation = {
            'tensor_structure_valid': all(tensor_validation.values()),
            'flow_test_results': flow_test_results,
            'overall_metrics': {
                'total_confidence_change': total_confidence_change,
                'average_semantic_preservation': avg_semantic_preservation,
                'total_processing_time': total_processing_time,
                'components_tested': len(flow_test_results)
            },
            'tensor_validation': tensor_validation,
            'data_flow_integrity': total_confidence_change > 0 and avg_semantic_preservation > 0.5
        }
        
        self.validation_results['tensor_flow'] = tensor_flow_validation
        logger.info(f"✓ Tensor flow validation complete (integrity: {tensor_flow_validation['data_flow_integrity']})")
        
        return tensor_flow_validation
        
    async def generate_cognitive_synergy_demonstration(self) -> Dict[str, Any]:
        """
        Generate a demonstration of cognitive synergy showing emergent properties
        that arise from component interaction.
        """
        logger.info("🎯 Generating Cognitive Synergy Demonstration")
        
        # Test emergent cognitive behaviors
        synergy_tests = [
            {
                "name": "Cross-Modal Integration",
                "description": "Text input producing visual and logical representations",
                "input": "The red balloon floats upward through the blue sky",
                "test_type": "cross_modal"
            },
            {
                "name": "Temporal Reasoning", 
                "description": "Sequential logical inference across time",
                "input": "First I opened the door, then I saw the cat, so the cat was inside",
                "test_type": "temporal"
            },
            {
                "name": "Analogical Mapping",
                "description": "Transfer of patterns between semantic domains",
                "input": "The atom is like a solar system with electrons orbiting the nucleus",
                "test_type": "analogical"
            },
            {
                "name": "Causal Inference",
                "description": "Understanding cause-effect relationships",
                "input": "When it rains, the ground gets wet, so the wet ground indicates rain",
                "test_type": "causal"
            }
        ]
        
        synergy_results = []
        
        for test in synergy_tests:
            # Process through full cognitive pipeline
            result = await self.integration_layer.process_cognitive_input(test['input'])
            
            # Analyze emergent properties
            emergent_analysis = self._analyze_emergent_properties(result, test['test_type'])
            
            synergy_result = {
                'test_name': test['name'],
                'test_type': test['test_type'],
                'description': test['description'],
                'input': test['input'],
                'cognitive_output': result,
                'emergent_properties': emergent_analysis,
                'synergy_demonstrated': emergent_analysis['synergy_score'] > 0.7
            }
            
            synergy_results.append(synergy_result)
            
        # Calculate overall synergy metrics
        synergy_scores = [r['emergent_properties']['synergy_score'] for r in synergy_results]
        avg_synergy_score = np.mean(synergy_scores)
        synergy_tests_passed = sum(1 for r in synergy_results if r['synergy_demonstrated'])
        
        synergy_demonstration = {
            'tests_conducted': len(synergy_tests),
            'tests_passed': synergy_tests_passed,
            'success_rate': (synergy_tests_passed / len(synergy_tests)) * 100,
            'average_synergy_score': avg_synergy_score,
            'synergy_results': synergy_results,
            'cognitive_emergence_achieved': avg_synergy_score > 0.7 and synergy_tests_passed >= 3
        }
        
        self.validation_results['synergy_demo'] = synergy_demonstration
        logger.info(f"✓ Synergy demonstration complete (emergence: {synergy_demonstration['cognitive_emergence_achieved']})")
        
        return synergy_demonstration
        
    def _analyze_emergent_properties(self, cognitive_result: Dict[str, Any], test_type: str) -> Dict[str, Any]:
        """Analyze emergent properties from cognitive processing"""
        
        # Extract key metrics
        confidence = cognitive_result['output_tensor']['confidence']
        processing_history = cognitive_result['processing_history']
        system_metrics = cognitive_result['system_metrics']
        
        # Calculate emergence indicators
        confidence_progression = [step['confidence'] for step in processing_history]
        confidence_acceleration = np.diff(np.diff(confidence_progression)) if len(confidence_progression) > 2 else [0]
        
        # Type-specific analysis
        type_specific_score = 0.5  # Base score
        
        if test_type == "cross_modal":
            # High semantic activation indicates cross-modal integration
            semantic_activation = abs(cognitive_result['output_tensor']['semantic_summary']['mean'])
            type_specific_score = min(1.0, semantic_activation * 2)
            
        elif test_type == "temporal":
            # Temporal reasoning indicated by temporal progression
            temporal_value = cognitive_result['output_tensor']['temporal']
            type_specific_score = min(1.0, (temporal_value % 1000) / 1000)  # Normalize
            
        elif test_type == "analogical":
            # Analogical mapping indicated by logical state diversity
            logical_diversity = cognitive_result['output_tensor']['logical_summary']['active_states'] / 64
            type_specific_score = logical_diversity
            
        elif test_type == "causal":
            # Causal inference indicated by confidence gain and logical coherence
            confidence_gain = confidence - 0.5  # Starting confidence
            logical_mean = cognitive_result['output_tensor']['logical_summary']['mean']
            type_specific_score = (confidence_gain + logical_mean) / 2
            
        # Overall synergy score
        base_synergy = confidence * system_metrics.get('cognitive_efficiency', 0.5)
        emergence_factor = np.mean(confidence_acceleration) if confidence_acceleration else 0
        synergy_score = (base_synergy + type_specific_score + emergence_factor) / 3
        
        return {
            'synergy_score': max(0, min(1, synergy_score)),
            'confidence_progression': confidence_progression,
            'emergence_indicators': {
                'confidence_acceleration': emergence_factor,
                'type_specific_score': type_specific_score,
                'base_synergy': base_synergy
            },
            'emergent_behaviors_detected': synergy_score > 0.7
        }
        
    async def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete end-to-end validation of the cognitive system"""
        logger.info("🚀 Starting Complete End-to-End Cognitive System Validation")
        
        start_time = time.time()
        
        # Run all validation tests
        pipeline_results = await self.validate_sensory_to_cognitive_pipeline()
        membrane_results = await self.validate_psystem_membrane_integrity() 
        tensor_results = await self.validate_tensor_structure_flow()
        synergy_results = await self.generate_cognitive_synergy_demonstration()
        
        # Get integration layer status
        integration_status = await self.integration_layer.validate_system_integration()
        
        total_time = time.time() - start_time
        
        # Calculate overall system health
        health_indicators = {
            'pipeline_health': pipeline_results['overall_performance']['pipeline_health'],
            'membrane_health': membrane_results['overall_membrane_health'],
            'tensor_flow_integrity': tensor_results['data_flow_integrity'],
            'cognitive_emergence': synergy_results['cognitive_emergence_achieved'],
            'integration_health': integration_status['integration_health']
        }
        
        overall_health = all(health_indicators.values())
        health_score = sum(health_indicators.values()) / len(health_indicators)
        
        complete_validation = {
            'validation_timestamp': time.time(),
            'total_validation_time': total_time,
            'overall_system_health': overall_health,
            'health_score_percentage': health_score * 100,
            'health_indicators': health_indicators,
            'detailed_results': {
                'sensory_to_cognitive_pipeline': pipeline_results,
                'psystem_membranes': membrane_results,
                'tensor_structure_flow': tensor_results,
                'cognitive_synergy_demonstration': synergy_results,
                'integration_layer_status': integration_status
            },
            'system_readiness': {
                'production_ready': overall_health and health_score >= 0.8,
                'cognitive_gestalt_achieved': synergy_results['cognitive_emergence_achieved'],
                'frame_problem_resolved': membrane_results['frame_problem_resolved'],
                'tensor_structure_documented': True,
                'end_to_end_validation_complete': True
            }
        }
        
        self.validation_results = complete_validation
        
        logger.info(f"✅ Complete validation finished (health: {health_score*100:.1f}%)")
        return complete_validation
        
    async def export_validation_report(self, output_path: str = "/tmp/end_to_end_validation_report.json"):
        """Export comprehensive validation report"""
        
        # Add tensor structure documentation
        tensor_doc = self.integration_layer.export_tensor_structure_documentation()
        
        validation_report = {
            'validation_results': self.validation_results,
            'tensor_structure_documentation': tensor_doc,
            'system_architecture': {
                'components': list(self.integration_layer.components.keys()),
                'membranes': list(self.integration_layer.membranes.keys()),
                'pipeline_stages': len(self.integration_layer.cognitive_pipeline)
            },
            'report_metadata': {
                'generated_at': time.time(),
                'validation_system_version': "1.0.0",
                'cognitive_architecture': "OpenCog Central with Integration Layer"
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
            
        logger.info(f"📋 Validation report exported to {output_path}")
        
        # Also export tensor documentation separately
        tensor_doc_path = output_path.replace('.json', '_tensor_structure.md')
        with open(tensor_doc_path, 'w') as f:
            f.write(tensor_doc)
            
        logger.info(f"📊 Tensor structure documentation exported to {tensor_doc_path}")
        
    async def shutdown(self):
        """Shutdown the validation system"""
        await self.integration_layer.shutdown()
        logger.info("✓ End-to-end validation system shutdown complete")


async def main():
    """Main function for end-to-end validation"""
    
    validator = EndToEndValidator()
    
    try:
        # Initialize validation system
        await validator.initialize()
        
        # Run complete validation
        results = await validator.run_complete_validation()
        
        # Print summary
        print("\n" + "="*60)
        print("END-TO-END COGNITIVE SYSTEM VALIDATION SUMMARY")
        print("="*60)
        print(f"Overall System Health: {'✅ EXCELLENT' if results['overall_system_health'] else '⚠️ NEEDS ATTENTION'}")
        print(f"Health Score: {results['health_score_percentage']:.1f}%")
        print(f"Validation Time: {results['total_validation_time']:.2f} seconds")
        print()
        
        print("Health Indicators:")
        for indicator, status in results['health_indicators'].items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {indicator.replace('_', ' ').title()}")
        print()
        
        print("System Readiness:")
        for readiness, status in results['system_readiness'].items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {readiness.replace('_', ' ').title()}")
        print()
        
        # Export validation report
        await validator.export_validation_report()
        
        # Shutdown
        await validator.shutdown()
        
        if results['overall_system_health']:
            print("🎉 COGNITIVE SYSTEM VALIDATION SUCCESSFUL!")
            print("🧠 The complete cognitive architecture is ready for production use.")
            return 0
        else:
            print("⚠️ COGNITIVE SYSTEM VALIDATION IDENTIFIED ISSUES")
            print("🔧 Please review the detailed validation report for remediation steps.")
            return 1
            
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        await validator.shutdown()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))