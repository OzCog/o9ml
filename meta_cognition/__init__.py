"""
Phase 5: Recursive Meta-Cognition & Evolutionary Optimization
Meta-Cognitive Pathways Implementation

This module implements recursive meta-cognitive monitoring and self-analysis
capabilities that enable the system to observe and improve itself.
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

from cogml.cognitive_primitives import CognitivePrimitiveTensor, TensorSignature
from cogml.hypergraph_encoding import HypergraphEncoder, AtomSpaceAdapter
from ecan.attention_kernel import AttentionKernel, ECANAttentionTensor


class MetaCognitiveLevel(Enum):
    """Levels of meta-cognitive processing"""
    OBJECT = "object"  # Direct cognitive processing
    META = "meta"  # Thinking about thinking
    META_META = "meta_meta"  # Thinking about thinking about thinking


class CognitiveState(Enum):
    """States of cognitive processing"""
    EXPLORING = "exploring"
    CONVERGING = "converging"
    OPTIMIZING = "optimizing"
    REFLECTING = "reflecting"


@dataclass
class MetaCognitiveMetrics:
    """Metrics for meta-cognitive performance tracking"""
    self_awareness_level: float = 0.0  # [0.0, 1.0]
    performance_metric: Dict[str, float] = field(default_factory=dict)
    evolutionary_generation: int = 0
    fitness_score: float = 0.0
    adaptation_rate: float = 0.0
    cognitive_complexity: str = "simple"
    meta_level: MetaCognitiveLevel = MetaCognitiveLevel.OBJECT
    reflection_depth: int = 1
    optimization_target: str = "accuracy"
    timestamp: float = field(default_factory=time.time)


@dataclass 
class CognitiveStateSnapshot:
    """Snapshot of current cognitive state for introspection"""
    active_tensors: Dict[str, CognitivePrimitiveTensor]
    attention_focus: List[Tuple[str, float]]
    processing_metrics: Dict[str, float]
    meta_cognitive_metrics: MetaCognitiveMetrics
    timestamp: float = field(default_factory=time.time)


class MetaCognitiveMonitor:
    """
    Core meta-cognitive monitoring system that observes and analyzes
    cognitive processes recursively.
    """
    
    def __init__(self, max_reflection_depth: int = 5, monitoring_interval: float = 1.0):
        self.max_reflection_depth = max_reflection_depth
        self.monitoring_interval = monitoring_interval
        self.cognitive_history: deque = deque(maxlen=1000)
        self.meta_patterns: Dict[str, Any] = {}
        self.self_analysis_results: List[Dict] = []
        self.recursive_depth = 0
        self.logger = logging.getLogger(__name__)
        
    def observe_cognitive_state(self, 
                              attention_kernel: AttentionKernel,
                              active_tensors: Dict[str, CognitivePrimitiveTensor]) -> CognitiveStateSnapshot:
        """Observe and capture current cognitive state"""
        
        # Get attention focus
        attention_focus = attention_kernel.get_attention_focus()
        
        # Collect processing metrics
        processing_metrics = attention_kernel.get_performance_metrics()
        
        # Compute meta-cognitive metrics
        meta_metrics = self._compute_meta_metrics(active_tensors, attention_focus)
        
        # Create snapshot
        snapshot = CognitiveStateSnapshot(
            active_tensors=active_tensors,
            attention_focus=attention_focus,
            processing_metrics=processing_metrics,
            meta_cognitive_metrics=meta_metrics
        )
        
        # Store in history
        self.cognitive_history.append(snapshot)
        
        return snapshot
    
    def recursive_self_analysis(self, current_snapshot: CognitiveStateSnapshot) -> Dict[str, Any]:
        """
        Perform recursive self-analysis at multiple meta-cognitive levels
        """
        if self.recursive_depth >= self.max_reflection_depth:
            return {"depth_limit_reached": True, "final_depth": self.recursive_depth}
        
        self.recursive_depth += 1
        
        try:
            # Level 1: Object-level analysis
            object_analysis = self._analyze_object_level(current_snapshot)
            
            # Level 2: Meta-level analysis (thinking about thinking)
            meta_analysis = self._analyze_meta_level(object_analysis, current_snapshot)
            
            # Level 3: Meta-meta level analysis (recursive reflection)
            if self.recursive_depth < self.max_reflection_depth - 1:
                meta_meta_analysis = self.recursive_self_analysis(current_snapshot)
            else:
                meta_meta_analysis = {"recursion_terminated": True}
            
            analysis_result = {
                "recursion_depth": self.recursive_depth,
                "object_level": object_analysis,
                "meta_level": meta_analysis,
                "meta_meta_level": meta_meta_analysis,
                "convergence_score": self._compute_convergence_score(),
                "timestamp": time.time()
            }
            
            self.self_analysis_results.append(analysis_result)
            
            return analysis_result
            
        finally:
            self.recursive_depth -= 1
    
    def _compute_meta_metrics(self, 
                             active_tensors: Dict[str, CognitivePrimitiveTensor],
                             attention_focus: List[Tuple[str, float]]) -> MetaCognitiveMetrics:
        """Compute meta-cognitive performance metrics"""
        
        # Self-awareness based on introspective capacity
        self_awareness = min(1.0, len(self.cognitive_history) / 100.0)
        
        # Performance metrics
        performance_metrics = {
            "accuracy": np.mean([t.signature.salience for t in active_tensors.values()]) if active_tensors else 0.0,
            "efficiency": len(attention_focus) / max(1, len(active_tensors)),
            "adaptability": self._compute_adaptability_score()
        }
        
        # Cognitive complexity assessment
        complexity = self._assess_cognitive_complexity(active_tensors)
        
        return MetaCognitiveMetrics(
            self_awareness_level=self_awareness,
            performance_metric=performance_metrics,
            evolutionary_generation=len(self.self_analysis_results),
            fitness_score=np.mean(list(performance_metrics.values())),
            adaptation_rate=self._compute_adaptation_rate(),
            cognitive_complexity=complexity,
            meta_level=MetaCognitiveLevel.META,
            reflection_depth=self.recursive_depth + 1,
            optimization_target="accuracy"
        )
    
    def _compute_adaptability_score(self) -> float:
        """Compute system's adaptability based on history"""
        if len(self.cognitive_history) < 5:
            return 0.5  # Default moderate adaptability
        
        recent_snapshots = list(self.cognitive_history)[-5:]
        metric_changes = []
        
        for i in range(1, len(recent_snapshots)):
            prev_metrics = recent_snapshots[i-1].meta_cognitive_metrics
            curr_metrics = recent_snapshots[i].meta_cognitive_metrics
            
            # Measure change in key metrics
            change = abs(curr_metrics.fitness_score - prev_metrics.fitness_score)
            metric_changes.append(change)
        
        # Higher variability suggests higher adaptability
        return min(1.0, np.std(metric_changes) * 2) if metric_changes else 0.5
    
    def _compute_adaptation_rate(self) -> float:
        """Compute rate of cognitive adaptation"""
        if len(self.cognitive_history) < 3:
            return 0.0
        
        recent_snapshots = list(self.cognitive_history)[-3:]
        
        # Measure improvement in fitness scores over time
        fitness_scores = [s.meta_cognitive_metrics.fitness_score for s in recent_snapshots]
        
        if len(fitness_scores) >= 2:
            recent_change = fitness_scores[-1] - fitness_scores[0]
            return max(0.0, min(1.0, recent_change + 0.5))  # Normalize to [0,1]
        
        return 0.0
    
    def _assess_cognitive_complexity(self, active_tensors: Dict[str, CognitivePrimitiveTensor]) -> str:
        """Assess current cognitive complexity level"""
        if not active_tensors:
            return "simple"
        
        avg_dof = np.mean([t.compute_degrees_of_freedom() for t in active_tensors.values()])
        modality_count = len(set(t.signature.modality for t in active_tensors.values()))
        
        if avg_dof > 100 and modality_count > 2:
            return "complex"
        elif avg_dof > 50 or modality_count > 1:
            return "moderate"
        else:
            return "simple"
    
    def _analyze_object_level(self, snapshot: CognitiveStateSnapshot) -> Dict[str, Any]:
        """Analyze direct cognitive processing (object level)"""
        
        # Analyze tensor patterns
        tensor_analysis = {
            "active_tensor_count": len(snapshot.active_tensors),
            "average_salience": np.mean([t.signature.salience for t in snapshot.active_tensors.values()]) if snapshot.active_tensors else 0.0,
            "modality_distribution": self._analyze_modality_distribution(snapshot.active_tensors),
            "complexity_score": np.mean([t.compute_degrees_of_freedom() for t in snapshot.active_tensors.values()]) if snapshot.active_tensors else 0.0
        }
        
        # Analyze attention patterns
        attention_analysis = {
            "focus_strength": np.mean([score for _, score in snapshot.attention_focus]) if snapshot.attention_focus else 0.0,
            "focus_distribution": len(snapshot.attention_focus),
            "attention_entropy": self._compute_attention_entropy(snapshot.attention_focus)
        }
        
        return {
            "tensor_analysis": tensor_analysis,
            "attention_analysis": attention_analysis,
            "cognitive_state": self._infer_cognitive_state(snapshot),
            "processing_efficiency": snapshot.processing_metrics.get('tensor_ops_per_second', 0)
        }
    
    def _analyze_meta_level(self, object_analysis: Dict, snapshot: CognitiveStateSnapshot) -> Dict[str, Any]:
        """Analyze thinking about thinking (meta level)"""
        
        # Assess self-monitoring quality
        monitoring_quality = {
            "introspection_depth": len(self.cognitive_history),
            "pattern_recognition": len(self.meta_patterns),
            "self_awareness_trend": self._compute_self_awareness_trend(),
            "meta_cognitive_stability": self._assess_meta_stability()
        }
        
        # Evaluate cognitive strategy effectiveness
        strategy_effectiveness = {
            "attention_strategy_score": self._evaluate_attention_strategy(),
            "learning_efficiency": self._compute_learning_efficiency(),
            "adaptation_success_rate": self._compute_adaptation_success_rate()
        }
        
        return {
            "monitoring_quality": monitoring_quality,
            "strategy_effectiveness": strategy_effectiveness,
            "meta_cognitive_insights": self._generate_meta_insights(object_analysis),
            "improvement_recommendations": self._generate_improvement_recommendations()
        }
    
    def _compute_convergence_score(self) -> float:
        """Compute how well the recursive analysis is converging"""
        if len(self.self_analysis_results) < 2:
            return 0.0
        
        recent_results = self.self_analysis_results[-5:]
        if len(recent_results) < 2:
            return 0.0
        
        # Compare stability of key metrics across recent analyses
        stability_scores = []
        for key in ["object_level", "meta_level"]:
            values = [result.get(key, {}).get("cognitive_state", "") for result in recent_results]
            if values:
                stability_scores.append(1.0 if len(set(values)) == 1 else 0.5)
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def _analyze_modality_distribution(self, tensors: Dict[str, CognitivePrimitiveTensor]) -> Dict[str, int]:
        """Analyze distribution of tensor modalities"""
        distribution = defaultdict(int)
        for tensor in tensors.values():
            distribution[tensor.signature.modality.name] += 1
        return dict(distribution)
    
    def _compute_attention_entropy(self, attention_focus: List[Tuple[str, float]]) -> float:
        """Compute entropy of attention distribution"""
        if not attention_focus:
            return 0.0
        
        scores = [score for _, score in attention_focus]
        total = sum(scores)
        
        if total == 0:
            return 0.0
        
        probabilities = [score/total for score in scores]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        return entropy
    
    def _infer_cognitive_state(self, snapshot: CognitiveStateSnapshot) -> str:
        """Infer current cognitive state from snapshot"""
        attention_entropy = self._compute_attention_entropy(snapshot.attention_focus)
        avg_salience = np.mean([t.signature.salience for t in snapshot.active_tensors.values()]) if snapshot.active_tensors else 0
        
        if attention_entropy > 2.0:
            return CognitiveState.EXPLORING.value
        elif avg_salience > 0.8:
            return CognitiveState.CONVERGING.value
        elif len(snapshot.attention_focus) < 3:
            return CognitiveState.OPTIMIZING.value
        else:
            return CognitiveState.REFLECTING.value
    
    def _compute_self_awareness_trend(self) -> float:
        """Compute trend in self-awareness over time"""
        if len(self.cognitive_history) < 3:
            return 0.0
        
        recent_awareness = [s.meta_cognitive_metrics.self_awareness_level 
                          for s in list(self.cognitive_history)[-5:]]
        
        if len(recent_awareness) >= 2:
            return (recent_awareness[-1] - recent_awareness[0]) / len(recent_awareness)
        
        return 0.0
    
    def _assess_meta_stability(self) -> float:
        """Assess stability of meta-cognitive processes"""
        if len(self.self_analysis_results) < 3:
            return 0.5
        
        recent_convergence_scores = [r.get("convergence_score", 0) 
                                   for r in self.self_analysis_results[-3:]]
        
        return np.mean(recent_convergence_scores) if recent_convergence_scores else 0.5
    
    def _evaluate_attention_strategy(self) -> float:
        """Evaluate effectiveness of current attention strategy"""
        if len(self.cognitive_history) < 2:
            return 0.5
        
        recent_snapshots = list(self.cognitive_history)[-5:]
        focus_effectiveness = []
        
        for snapshot in recent_snapshots:
            if snapshot.attention_focus:
                avg_focus_strength = np.mean([score for _, score in snapshot.attention_focus])
                focus_effectiveness.append(avg_focus_strength)
        
        return np.mean(focus_effectiveness) if focus_effectiveness else 0.5
    
    def _compute_learning_efficiency(self) -> float:
        """Compute learning efficiency based on cognitive history"""
        if len(self.cognitive_history) < 3:
            return 0.5
        
        # Measure improvement in cognitive metrics over time
        snapshots = list(self.cognitive_history)[-5:]
        fitness_improvements = []
        
        for i in range(1, len(snapshots)):
            prev_fitness = snapshots[i-1].meta_cognitive_metrics.fitness_score
            curr_fitness = snapshots[i].meta_cognitive_metrics.fitness_score
            improvement = curr_fitness - prev_fitness
            fitness_improvements.append(improvement)
        
        if fitness_improvements:
            avg_improvement = np.mean(fitness_improvements)
            return max(0.0, min(1.0, avg_improvement + 0.5))
        
        return 0.5
    
    def _compute_adaptation_success_rate(self) -> float:
        """Compute rate of successful adaptations"""
        if len(self.self_analysis_results) < 2:
            return 0.5
        
        successful_adaptations = sum(1 for result in self.self_analysis_results 
                                   if result.get("convergence_score", 0) > 0.7)
        
        return successful_adaptations / len(self.self_analysis_results)
    
    def _generate_meta_insights(self, object_analysis: Dict) -> List[str]:
        """Generate insights about cognitive processing"""
        insights = []
        
        cognitive_state = object_analysis.get("cognitive_state", "")
        efficiency = object_analysis.get("processing_efficiency", 0)
        
        if cognitive_state == CognitiveState.EXPLORING.value:
            insights.append("System is in exploratory mode, high cognitive diversity detected")
        elif cognitive_state == CognitiveState.CONVERGING.value:
            insights.append("System is converging on solutions, focused attention detected")
        
        if efficiency > 1000:
            insights.append("High processing efficiency indicates optimized cognitive flow")
        elif efficiency < 100:
            insights.append("Low processing efficiency suggests need for optimization")
        
        return insights
    
    def _generate_improvement_recommendations(self) -> List[str]:
        """Generate recommendations for cognitive improvement"""
        recommendations = []
        
        if len(self.cognitive_history) > 0:
            latest_snapshot = self.cognitive_history[-1]
            self_awareness = latest_snapshot.meta_cognitive_metrics.self_awareness_level
            
            if self_awareness < 0.5:
                recommendations.append("Increase introspective monitoring frequency")
            
            if latest_snapshot.meta_cognitive_metrics.adaptation_rate < 0.3:
                recommendations.append("Enhance adaptive mechanisms for faster learning")
            
            if len(latest_snapshot.attention_focus) > 10:
                recommendations.append("Focus attention more selectively to improve efficiency")
        
        return recommendations
    
    def get_meta_cognitive_status(self) -> Dict[str, Any]:
        """Get comprehensive meta-cognitive status report"""
        return {
            "current_reflection_depth": self.recursive_depth,
            "cognitive_history_length": len(self.cognitive_history),
            "total_self_analyses": len(self.self_analysis_results),
            "meta_patterns_discovered": len(self.meta_patterns),
            "latest_convergence_score": self.self_analysis_results[-1].get("convergence_score", 0) if self.self_analysis_results else 0,
            "self_monitoring_active": True,
            "timestamp": time.time()
        }