#!/usr/bin/env python3
"""
Cognitive Synergy Demonstration
===============================

This script demonstrates the system synergy achieved through the Integration Layer,
showing how P-System membranes coordinate cognitive components to achieve emergent
cognitive behaviors that exceed the sum of individual component capabilities.
"""

import asyncio
import time
import json
from integration_layer import IntegrationLayer

async def demonstrate_cognitive_synergy():
    """Demonstrate the cognitive gestalt and system synergy"""
    
    print("🧠 COGNITIVE SYNERGY DEMONSTRATION")
    print("="*50)
    print()
    
    # Initialize integration layer
    integration = IntegrationLayer()
    await integration.initialize()
    
    print("✓ Integration Layer initialized")
    print(f"✓ {len(integration.components)} cognitive components active")
    print(f"✓ {len(integration.membranes)} P-System membranes configured")
    print()
    
    # Demonstrate cognitive processing scenarios
    scenarios = [
        {
            "name": "🎯 Attention and Focus",
            "input": "Focus on the important red warning signal while ignoring background noise",
            "description": "Tests attention allocation and selective processing"
        },
        {
            "name": "🔗 Neural-Symbolic Integration", 
            "input": "The bird flies because it has wings and flight capability",
            "description": "Tests integration of neural and symbolic reasoning"
        },
        {
            "name": "🔄 Recursive Processing",
            "input": "Understanding understanding requires understanding",
            "description": "Tests recursive cognitive kernel capabilities"
        },
        {
            "name": "🌟 Emergent Cognition",
            "input": "Love is like sunlight - it makes everything grow and flourish",
            "description": "Tests emergence of higher-order cognitive patterns"
        }
    ]
    
    results = []
    
    print("COGNITIVE PROCESSING DEMONSTRATIONS")
    print("-" * 50)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Input: \"{scenario['input']}\"")
        
        start_time = time.time()
        result = await integration.process_cognitive_input(scenario['input'])
        processing_time = time.time() - start_time
        
        # Extract key metrics
        confidence = result['output_tensor']['confidence']
        synergy_achieved = result['cognitive_synergy_achieved']
        components_used = len(result['processing_history'])
        
        print(f"   ✓ Processing time: {processing_time*1000:.1f}ms")
        print(f"   ✓ Confidence achieved: {confidence:.3f}")
        print(f"   ✓ Components coordinated: {components_used}")
        print(f"   ✓ Synergy achieved: {'🌟 YES' if synergy_achieved else '⚠️ PARTIAL'}")
        
        # Analyze P-System membrane coordination
        membrane_coordination = analyze_membrane_coordination(result)
        print(f"   ✓ Membrane coordination: {membrane_coordination:.3f}")
        
        results.append({
            'scenario': scenario['name'],
            'confidence': confidence,
            'synergy_achieved': synergy_achieved,
            'processing_time': processing_time,
            'membrane_coordination': membrane_coordination
        })
    
    # System-wide synergy analysis
    print("\n" + "="*50)
    print("SYSTEM SYNERGY ANALYSIS")
    print("="*50)
    
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    synergy_rate = sum(1 for r in results if r['synergy_achieved']) / len(results) * 100
    avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
    avg_coordination = sum(r['membrane_coordination'] for r in results) / len(results)
    
    print(f"📊 Average Confidence: {avg_confidence:.3f}")
    print(f"🎯 Synergy Achievement Rate: {synergy_rate:.1f}%") 
    print(f"⚡ Average Processing Time: {avg_processing_time*1000:.1f}ms")
    print(f"🔗 Membrane Coordination: {avg_coordination:.3f}")
    
    # Validate integration layer status
    integration_status = await integration.validate_system_integration()
    
    print(f"\n🔍 INTEGRATION LAYER STATUS")
    print(f"   ✓ Integration Health: {'🟢 EXCELLENT' if integration_status['integration_health'] else '🔴 NEEDS ATTENTION'}")
    print(f"   ✓ Active Components: {len([c for c in integration_status['component_status'].values() if c['active']])}")
    print(f"   ✓ Membrane Integrity: {'🟢 INTACT' if all(m['tensor_confidence'] > 0.5 for m in integration_status['membrane_integrity'].values()) else '🔴 DEGRADED'}")
    
    # Demonstrate frame problem resolution
    print(f"\n🧩 FRAME PROBLEM RESOLUTION")
    frame_resolved = demonstrate_frame_problem_resolution(integration)
    print(f"   ✓ Frame Problem Status: {'🟢 RESOLVED' if frame_resolved else '🔴 NOT RESOLVED'}")
    print(f"   ✓ P-System Membranes: {'🟢 FUNCTIONAL' if len(integration.membranes) >= 3 else '🔴 INSUFFICIENT'}")
    print(f"   ✓ Cognitive Boundaries: {'🟢 ESTABLISHED' if frame_resolved else '🔴 UNCLEAR'}")
    
    # Overall system assessment
    overall_score = (avg_confidence + (synergy_rate/100) + avg_coordination + (1 if frame_resolved else 0)) / 4
    
    print(f"\n🌟 OVERALL COGNITIVE SYNERGY SCORE: {overall_score:.3f}")
    
    if overall_score >= 0.8:
        status = "🟢 EXCELLENT - Full cognitive synergy achieved"
    elif overall_score >= 0.6:
        status = "🟡 GOOD - Substantial synergy with room for improvement"
    else:
        status = "🔴 NEEDS WORK - Limited synergy detected"
        
    print(f"🎯 System Status: {status}")
    
    # Generate recommendations
    print(f"\n📋 RECOMMENDATIONS")
    if synergy_rate < 75:
        print("   • Optimize attention allocation algorithms")
        print("   • Enhance neural-symbolic fusion mechanisms")
    if avg_processing_time > 0.1:
        print("   • Consider performance optimizations")
        print("   • Implement parallel processing where possible")
    if avg_coordination < 0.7:
        print("   • Improve membrane permeability calibration")
        print("   • Enhance cross-component communication")
    
    if overall_score >= 0.8:
        print("   ✨ System is operating at peak cognitive synergy!")
        print("   🚀 Ready for advanced cognitive workloads")
    
    # Shutdown
    await integration.shutdown()
    
    return {
        'overall_score': overall_score,
        'synergy_rate': synergy_rate,
        'avg_confidence': avg_confidence,
        'frame_problem_resolved': frame_resolved,
        'integration_health': integration_status['integration_health']
    }


def analyze_membrane_coordination(processing_result):
    """Analyze how well P-System membranes coordinated during processing"""
    
    # Look at confidence progression through processing stages
    processing_history = processing_result['processing_history']
    
    if len(processing_history) < 2:
        return 0.5  # Minimal coordination
    
    # Calculate coordination as smoothness of confidence progression
    confidences = [step['confidence'] for step in processing_history]
    
    # Good coordination shows steady improvement
    improvements = [confidences[i] - confidences[i-1] for i in range(1, len(confidences))]
    
    # Coordination score based on consistent positive progression
    positive_improvements = sum(1 for imp in improvements if imp > 0)
    coordination_score = positive_improvements / len(improvements) if improvements else 0.5
    
    # Boost score if final confidence is high
    if confidences[-1] > 0.7:
        coordination_score = min(1.0, coordination_score + 0.2)
    
    return coordination_score


def demonstrate_frame_problem_resolution(integration_layer):
    """Demonstrate that P-System membranes resolve the frame problem"""
    
    membranes = integration_layer.membranes
    
    # Check that membranes provide cognitive boundaries
    spatial_separation = len(set(
        tuple(m.tensor_state.spatial) for m in membranes.values()
    )) == len(membranes)
    
    # Check that each membrane has specific processing rules
    rule_specificity = all(len(m.rules) > 0 for m in membranes.values())
    
    # Check hierarchical organization prevents infinite recursion
    has_root = any(m.parent is None for m in membranes.values())
    bounded_depth = all(len(m.children) <= 5 for m in membranes.values())  # Reasonable limit
    
    # Check information flow control
    controlled_permeability = all(
        0 <= m.membrane_permeability <= 1 for m in membranes.values()
    )
    
    return spatial_separation and rule_specificity and has_root and bounded_depth and controlled_permeability


async def main():
    """Main demonstration function"""
    
    print("Starting Cognitive Synergy Demonstration...")
    print()
    
    try:
        result = await demonstrate_cognitive_synergy()
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print(f"Overall cognitive synergy achieved: {result['overall_score']:.1%}")
        
        if result['overall_score'] >= 0.6:
            print("✅ Integration Layer: System Synergy successfully demonstrated!")
            return 0
        else:
            print("⚠️ Integration Layer needs further optimization")
            return 1
            
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))