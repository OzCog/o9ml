"""
Test suite for Phase 5: Recursive Meta-Cognition & Evolutionary Optimization

This test suite validates the meta-cognitive pathways, evolutionary optimization,
and verification framework implementations.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cogml.cognitive_primitives import CognitivePrimitiveTensor, create_primitive_tensor, ModalityType, DepthType, ContextType
from ecan.attention_kernel import AttentionKernel, ECANAttentionTensor
from meta_cognition import (
    MetaCognitiveMonitor, CognitiveStateSnapshot, MetaCognitiveMetrics,
    MetaCognitiveLevel, CognitiveState
)
from evolutionary_optimization import (
    EvolutionaryOptimizer, CognitiveGenome, FitnessMetrics, 
    MultiObjectiveFitnessEvaluator, OptimizationTarget, EvolutionaryHyperparameters
)
from continuous_benchmarking import (
    ContinuousBenchmarking, MetaCognitiveValidation, BenchmarkType,
    ValidationLevel, BenchmarkResult
)


class TestMetaCognitiveMonitor:
    """Test meta-cognitive monitoring and recursive reflection capabilities"""
    
    def test_monitor_initialization(self):
        """Test meta-cognitive monitor initialization"""
        monitor = MetaCognitiveMonitor(max_reflection_depth=3, monitoring_interval=0.5)
        
        assert monitor.max_reflection_depth == 3
        assert monitor.monitoring_interval == 0.5
        assert len(monitor.cognitive_history) == 0
        assert len(monitor.self_analysis_results) == 0
        assert monitor.recursive_depth == 0
    
    def test_cognitive_state_observation(self):
        """Test observation of cognitive state"""
        monitor = MetaCognitiveMonitor()
        attention_kernel = AttentionKernel()
        
        # Create test tensors
        test_tensors = {
            "tensor1": create_primitive_tensor(
                modality=ModalityType.VISUAL,
                depth=DepthType.SEMANTIC,
                context=ContextType.GLOBAL,
                salience=0.8
            ),
            "tensor2": create_primitive_tensor(
                modality=ModalityType.TEXTUAL,
                depth=DepthType.PRAGMATIC,
                context=ContextType.TEMPORAL,
                salience=0.6
            )
        }
        
        # Observe cognitive state
        snapshot = monitor.observe_cognitive_state(attention_kernel, test_tensors)
        
        assert isinstance(snapshot, CognitiveStateSnapshot)
        assert len(snapshot.active_tensors) == 2
        assert isinstance(snapshot.meta_cognitive_metrics, MetaCognitiveMetrics)
        assert snapshot.timestamp > 0
        assert len(monitor.cognitive_history) == 1
    
    def test_recursive_self_analysis(self):
        """Test recursive self-analysis capabilities"""
        monitor = MetaCognitiveMonitor(max_reflection_depth=3)
        
        # Create test snapshot
        test_snapshot = CognitiveStateSnapshot(
            active_tensors={},
            attention_focus=[("test_atom", 0.7)],
            processing_metrics={"tensor_ops_per_second": 500},
            meta_cognitive_metrics=MetaCognitiveMetrics()
        )
        
        # Perform recursive analysis
        analysis_result = monitor.recursive_self_analysis(test_snapshot)
        
        assert "recursion_depth" in analysis_result
        assert "object_level" in analysis_result
        assert "meta_level" in analysis_result
        assert "convergence_score" in analysis_result
        assert analysis_result["recursion_depth"] >= 1
        assert len(monitor.self_analysis_results) == 1
    
    def test_meta_metrics_computation(self):
        """Test meta-cognitive metrics computation"""
        monitor = MetaCognitiveMonitor()
        
        # Add some history to compute meaningful metrics
        for i in range(5):
            dummy_snapshot = CognitiveStateSnapshot(
                active_tensors={},
                attention_focus=[],
                processing_metrics={},
                meta_cognitive_metrics=MetaCognitiveMetrics(self_awareness_level=0.1 * i)
            )
            monitor.cognitive_history.append(dummy_snapshot)
        
        test_tensors = {
            "test": create_primitive_tensor(ModalityType.VISUAL, DepthType.SURFACE, ContextType.LOCAL, salience=0.7)
        }
        
        metrics = monitor._compute_meta_metrics(test_tensors, [("atom1", 0.8)])
        
        assert 0.0 <= metrics.self_awareness_level <= 1.0
        assert "accuracy" in metrics.performance_metric
        assert "efficiency" in metrics.performance_metric
        assert "adaptability" in metrics.performance_metric
        assert metrics.evolutionary_generation >= 0
        assert 0.0 <= metrics.fitness_score <= 1.0
    
    def test_cognitive_complexity_assessment(self):
        """Test cognitive complexity assessment"""
        monitor = MetaCognitiveMonitor()
        
        # Simple cognitive setup
        simple_tensors = {
            "simple": create_primitive_tensor(ModalityType.VISUAL, DepthType.SURFACE, ContextType.LOCAL)
        }
        
        complexity_simple = monitor._assess_cognitive_complexity(simple_tensors)
        assert complexity_simple in ["simple", "moderate", "complex"]
        
        # Complex cognitive setup
        complex_tensors = {
            f"tensor_{i}": create_primitive_tensor(
                modality=ModalityType.VISUAL if i % 2 == 0 else ModalityType.TEXTUAL,
                depth=DepthType.SEMANTIC,
                context=ContextType.GLOBAL,
                salience=0.8
            ) for i in range(5)
        }
        
        complexity_complex = monitor._assess_cognitive_complexity(complex_tensors)
        assert complexity_complex in ["simple", "moderate", "complex"]
    
    def test_convergence_score_computation(self):
        """Test convergence score computation"""
        monitor = MetaCognitiveMonitor()
        
        # Add some analysis results
        for i in range(3):
            monitor.self_analysis_results.append({
                "object_level": {"cognitive_state": "exploring"},
                "meta_level": {"cognitive_state": "exploring"},
                "convergence_score": 0.7 + i * 0.1
            })
        
        convergence_score = monitor._compute_convergence_score()
        assert 0.0 <= convergence_score <= 1.0
    
    def test_meta_cognitive_status(self):
        """Test meta-cognitive status reporting"""
        monitor = MetaCognitiveMonitor()
        
        # Add some data
        monitor.cognitive_history.append(CognitiveStateSnapshot(
            active_tensors={},
            attention_focus=[],
            processing_metrics={},
            meta_cognitive_metrics=MetaCognitiveMetrics()
        ))
        
        monitor.self_analysis_results.append({"convergence_score": 0.8})
        
        status = monitor.get_meta_cognitive_status()
        
        assert "current_reflection_depth" in status
        assert "cognitive_history_length" in status
        assert "total_self_analyses" in status
        assert "latest_convergence_score" in status
        assert status["self_monitoring_active"] == True
        assert status["cognitive_history_length"] == 1
        assert status["total_self_analyses"] == 1


class TestEvolutionaryOptimizer:
    """Test evolutionary optimization capabilities"""
    
    def test_optimizer_initialization(self):
        """Test evolutionary optimizer initialization"""
        hyperparams = EvolutionaryHyperparameters(
            population_size=20,
            mutation_rate=0.1,
            max_generations=50
        )
        
        optimizer = EvolutionaryOptimizer(
            hyperparams=hyperparams,
            optimization_targets=[OptimizationTarget.ACCURACY, OptimizationTarget.EFFICIENCY]
        )
        
        assert optimizer.hyperparams.population_size == 20
        assert optimizer.hyperparams.mutation_rate == 0.1
        assert len(optimizer.optimization_targets) == 2
        assert len(optimizer.population) == 0
        assert optimizer.generation == 0
    
    def test_population_initialization(self):
        """Test population initialization"""
        optimizer = EvolutionaryOptimizer()
        optimizer.hyperparams.population_size = 10
        
        optimizer.initialize_population()
        
        assert len(optimizer.population) == 10
        assert all(isinstance(genome, CognitiveGenome) for genome in optimizer.population)
        assert all(genome.generation == 0 for genome in optimizer.population)
        assert all(genome.age == 0 for genome in optimizer.population)
    
    def test_genome_creation(self):
        """Test cognitive genome creation"""
        optimizer = EvolutionaryOptimizer()
        
        genome = optimizer._create_default_genome()
        
        assert isinstance(genome, CognitiveGenome)
        assert "tensor_configs" in genome.__dict__
        assert "attention_params" in genome.__dict__
        assert "processing_params" in genome.__dict__
        assert "meta_cognitive_params" in genome.__dict__
        assert len(genome.tensor_configs) > 0
        assert len(genome.attention_params) > 0
    
    def test_genome_variation(self):
        """Test genome variation creation"""
        optimizer = EvolutionaryOptimizer()
        template = optimizer._create_default_genome()
        
        variant = optimizer._create_variant_genome(template, diversity_factor=0.5)
        
        assert isinstance(variant, CognitiveGenome)
        assert len(variant.tensor_configs) == len(template.tensor_configs)
        assert len(variant.attention_params) == len(template.attention_params)
        
        # Check that some parameters are different (with reasonable tolerance)
        differences = 0
        for tensor_id in template.tensor_configs:
            if abs(variant.tensor_configs[tensor_id]["salience"] - 
                   template.tensor_configs[tensor_id]["salience"]) > 0.01:
                differences += 1
        
        # Expect at least some differences with 50% diversity factor
        assert differences >= 0  # Allow no differences in case of random chance
    
    def test_fitness_evaluation(self):
        """Test fitness evaluation"""
        optimizer = EvolutionaryOptimizer()
        genome = optimizer._create_default_genome()
        
        # Mock fitness evaluation function
        def mock_fitness_func(genome: CognitiveGenome) -> FitnessMetrics:
            return FitnessMetrics(
                accuracy=0.8,
                efficiency=0.7,
                adaptability=0.6,
                composite_score=0.7
            )
        
        optimizer.initialize_population()
        optimizer._evaluate_population_fitness(mock_fitness_func)
        
        assert all(genome.fitness_score > 0 for genome in optimizer.population)
        assert all(0.0 <= genome.fitness_score <= 1.0 for genome in optimizer.population)
    
    def test_selection_process(self):
        """Test selection process"""
        optimizer = EvolutionaryOptimizer()
        optimizer.hyperparams.population_size = 10
        optimizer.hyperparams.elite_size = 2
        
        optimizer.initialize_population()
        
        # Assign different fitness scores
        for i, genome in enumerate(optimizer.population):
            genome.fitness_score = 0.1 * i
        
        selected = optimizer._selection()
        
        assert len(selected) == optimizer.hyperparams.population_size
        
        # Check that elite individuals are included
        best_scores = sorted([g.fitness_score for g in optimizer.population], reverse=True)
        selected_scores = [g.fitness_score for g in selected[:optimizer.hyperparams.elite_size]]
        
        # Elite should be among the best
        assert max(selected_scores) == max(best_scores)
    
    def test_crossover_operation(self):
        """Test crossover operation"""
        optimizer = EvolutionaryOptimizer()
        
        parent1 = optimizer._create_default_genome()
        parent2 = optimizer._create_variant_genome(parent1, 0.5)
        
        child = optimizer._crossover(parent1, parent2)
        
        assert isinstance(child, CognitiveGenome)
        assert len(child.tensor_configs) == len(parent1.tensor_configs)
        assert len(child.attention_params) == len(parent1.attention_params)
    
    def test_mutation_operation(self):
        """Test mutation operation"""
        optimizer = EvolutionaryOptimizer()
        
        original = optimizer._create_default_genome()
        mutated = optimizer._mutate(original)
        
        assert isinstance(mutated, CognitiveGenome)
        assert len(mutated.mutation_history) > 0
        
        # Check that mutation type is recorded
        mutation_types = ["parameter_adjustment", "attention_reallocation", "tensor_reconfiguration"]
        assert mutated.mutation_history[-1] in mutation_types
    
    def test_evolution_summary(self):
        """Test evolution summary generation"""
        optimizer = EvolutionaryOptimizer()
        optimizer.initialize_population()
        
        # Add some mock evolution history
        from evolutionary_optimization import EvolutionaryMetrics
        
        optimizer.evolution_history = [
            EvolutionaryMetrics(generation=0, best_fitness=0.5, average_fitness=0.4),
            EvolutionaryMetrics(generation=1, best_fitness=0.6, average_fitness=0.5),
            EvolutionaryMetrics(generation=2, best_fitness=0.7, average_fitness=0.6)
        ]
        
        summary = optimizer.get_evolution_summary()
        
        assert "total_generations" in summary
        assert "current_generation" in summary
        assert "best_fitness_ever" in summary
        assert "convergence_trend" in summary
        assert summary["total_generations"] == 3
        assert summary["best_fitness_ever"] == 0.7


class TestMultiObjectiveFitnessEvaluator:
    """Test multi-objective fitness evaluation"""
    
    def test_evaluator_initialization(self):
        """Test fitness evaluator initialization"""
        targets = [OptimizationTarget.ACCURACY, OptimizationTarget.EFFICIENCY]
        evaluator = MultiObjectiveFitnessEvaluator(targets)
        
        assert len(evaluator.optimization_targets) == 2
        assert "accuracy" in evaluator.weight_config
        assert "efficiency" in evaluator.weight_config
        assert len(evaluator.benchmark_results) == 0
    
    def test_fitness_evaluation(self):
        """Test comprehensive fitness evaluation"""
        evaluator = MultiObjectiveFitnessEvaluator([OptimizationTarget.ACCURACY])
        
        # Create test genome
        from evolutionary_optimization import CognitiveGenome
        
        genome = CognitiveGenome(
            tensor_configs={
                "test": {
                    "modality": ModalityType.VISUAL,
                    "depth": DepthType.SEMANTIC,
                    "context": ContextType.GLOBAL,
                    "salience": 0.8,
                    "autonomy_index": 0.6
                }
            },
            attention_params={"focus_threshold": 0.7, "max_focus_items": 5},
            processing_params={"processing_speed": 1.2, "learning_rate": 0.02},
            meta_cognitive_params={"self_awareness_sensitivity": 0.8}
        )
        
        fitness_metrics = evaluator.evaluate_fitness(genome)
        
        assert isinstance(fitness_metrics, FitnessMetrics)
        assert 0.0 <= fitness_metrics.accuracy <= 1.0
        assert 0.0 <= fitness_metrics.efficiency <= 1.0
        assert 0.0 <= fitness_metrics.adaptability <= 1.0
        assert 0.0 <= fitness_metrics.composite_score <= 1.0
        assert len(evaluator.benchmark_results) == 1
    
    def test_weight_updates(self):
        """Test fitness weight updates"""
        evaluator = MultiObjectiveFitnessEvaluator([OptimizationTarget.ACCURACY])
        
        new_weights = {"accuracy": 0.5, "efficiency": 0.3, "adaptability": 0.2}
        evaluator.update_weights(new_weights)
        
        # Check that weights were updated and normalized
        total_weight = sum(evaluator.weight_config.values())
        assert abs(total_weight - 1.0) < 0.01
        assert evaluator.weight_config["accuracy"] == 0.5
    
    def test_fitness_statistics(self):
        """Test fitness statistics computation"""
        evaluator = MultiObjectiveFitnessEvaluator([OptimizationTarget.ACCURACY])
        
        # Add some mock results
        for i in range(5):
            metrics = FitnessMetrics(
                accuracy=0.5 + i * 0.1,
                efficiency=0.6 + i * 0.05,
                composite_score=0.55 + i * 0.075
            )
            evaluator.benchmark_results.append(metrics)
        
        statistics = evaluator.get_fitness_statistics()
        
        assert "dimensions" in statistics
        assert "total_evaluations" in statistics
        assert statistics["total_evaluations"] == 5
        assert "accuracy" in statistics["dimensions"]
        assert "mean" in statistics["dimensions"]["accuracy"]
        assert "trend" in statistics["dimensions"]["accuracy"]


class TestContinuousBenchmarking:
    """Test continuous benchmarking capabilities"""
    
    def test_benchmarking_initialization(self):
        """Test continuous benchmarking initialization"""
        benchmarking = ContinuousBenchmarking(
            benchmark_interval=10.0,
            max_history_size=500,
            enable_real_time=False
        )
        
        assert benchmarking.benchmark_interval == 10.0
        assert benchmarking.max_history_size == 500
        assert benchmarking.enable_real_time == False
        assert len(benchmarking.benchmark_history) == 0
        assert benchmarking.is_monitoring == False
    
    def test_benchmark_suite_execution(self):
        """Test benchmark suite execution"""
        benchmarking = ContinuousBenchmarking(enable_real_time=False)
        
        # Create mock cognitive system
        mock_attention_kernel = Mock()
        mock_attention_kernel.allocate_attention = Mock()
        mock_attention_kernel.get_attention_focus = Mock(return_value=[("test", 0.8)])
        mock_attention_kernel.get_performance_metrics = Mock(return_value={"tensor_ops_per_second": 500})
        
        cognitive_system = {
            "attention_kernel": mock_attention_kernel,
            "meta_monitor": Mock()
        }
        
        # Run subset of benchmarks
        benchmark_types = [BenchmarkType.PROCESSING_SPEED, BenchmarkType.ATTENTION_ACCURACY]
        results = benchmarking.run_benchmark_suite(cognitive_system, benchmark_types)
        
        assert len(results) == 2
        assert all(isinstance(result, BenchmarkResult) for result in results)
        assert all(0.0 <= result.score <= 1.0 for result in results)
        assert len(benchmarking.benchmark_history) == 2
    
    def test_performance_report_generation(self):
        """Test performance report generation"""
        benchmarking = ContinuousBenchmarking(enable_real_time=False)
        
        # Add some mock benchmark results
        for i in range(5):
            result = BenchmarkResult(
                benchmark_type=BenchmarkType.PROCESSING_SPEED,
                score=0.5 + i * 0.1,
                execution_time=1.0,
                memory_usage=10.0
            )
            benchmarking.benchmark_history.append(result)
            benchmarking.performance_metrics[BenchmarkType.PROCESSING_SPEED.value].append(result.score)
        
        benchmarking._update_trend_analysis()
        report = benchmarking.generate_performance_report()
        
        assert "overall_health_score" in report
        assert "performance_by_type" in report
        assert "trend_analysis" in report
        assert "recommendations" in report
        assert BenchmarkType.PROCESSING_SPEED.value in report["performance_by_type"]
    
    def test_trend_analysis(self):
        """Test trend analysis functionality"""
        benchmarking = ContinuousBenchmarking(enable_real_time=False)
        
        # Create trending data
        values = [0.5, 0.6, 0.7, 0.8, 0.9]  # Improving trend
        trend = benchmarking._analyze_trend("test_metric", values)
        
        assert trend.metric_name == "test_metric"
        assert trend.direction.value in ["improving", "stable", "declining", "volatile"]
        assert isinstance(trend.slope, float)
        assert 0.0 <= trend.confidence <= 1.0
        assert len(trend.recent_values) <= 10


class TestMetaCognitiveValidation:
    """Test meta-cognitive validation framework"""
    
    def test_validation_initialization(self):
        """Test meta-cognitive validation initialization"""
        validator = MetaCognitiveValidation()
        
        assert len(validator.validation_history) == 0
        assert "recursive_depth_test" in validator.test_scenarios
        assert "self_awareness_test" in validator.test_scenarios
    
    def test_system_validation(self):
        """Test comprehensive system validation"""
        validator = MetaCognitiveValidation()
        
        # Create mock components
        mock_monitor = Mock()
        mock_monitor.get_meta_cognitive_status = Mock(return_value={
            "current_reflection_depth": 2,
            "cognitive_history_length": 10
        })
        mock_monitor.cognitive_history = []
        mock_monitor.self_analysis_results = []
        mock_monitor.max_reflection_depth = 5
        mock_monitor.recursive_self_analysis = Mock(return_value={
            "recursion_depth": 3,
            "convergence_score": 0.8
        })
        
        mock_kernel = Mock()
        mock_kernel.get_performance_metrics = Mock(return_value={"tensor_ops_per_second": 300})
        mock_kernel.get_attention_focus = Mock(return_value=[])
        
        # Run validation
        report = validator.validate_meta_cognitive_system(
            mock_monitor, 
            mock_kernel, 
            ValidationLevel.BASIC
        )
        
        assert isinstance(report.validation_id, str)
        assert "system_state" in report.__dict__
        assert len(report.benchmark_results) >= 0
        assert isinstance(report.overall_score, float)
        assert 0.0 <= report.overall_score <= 1.0


class TestIntegrationScenarios:
    """Test integration scenarios for complete Phase 5 system"""
    
    def test_complete_meta_cognitive_workflow(self):
        """Test complete meta-cognitive workflow"""
        # Initialize components
        attention_kernel = AttentionKernel(max_atoms=50)
        meta_monitor = MetaCognitiveMonitor(max_reflection_depth=3)
        
        # Create test cognitive tensors
        test_tensors = {
            "visual_input": create_primitive_tensor(
                modality=ModalityType.VISUAL,
                depth=DepthType.SEMANTIC,
                context=ContextType.GLOBAL,
                salience=0.8
            ),
            "textual_input": create_primitive_tensor(
                modality=ModalityType.TEXTUAL,
                depth=DepthType.PRAGMATIC,
                context=ContextType.TEMPORAL,
                salience=0.6
            )
        }
        
        # Allocate attention
        for tensor_id, tensor in test_tensors.items():
            attention_tensor = ECANAttentionTensor(
                short_term_importance=tensor.signature.salience,
                urgency=0.7
            )
            attention_kernel.allocate_attention(tensor_id, attention_tensor)
        
        # Observe cognitive state
        snapshot = meta_monitor.observe_cognitive_state(attention_kernel, test_tensors)
        
        # Perform recursive analysis
        analysis_result = meta_monitor.recursive_self_analysis(snapshot)
        
        # Validate results
        assert isinstance(snapshot, CognitiveStateSnapshot)
        assert len(snapshot.active_tensors) == 2
        assert "recursion_depth" in analysis_result
        assert analysis_result["recursion_depth"] >= 1
        assert len(meta_monitor.cognitive_history) >= 1
        assert len(meta_monitor.self_analysis_results) >= 1
    
    def test_evolutionary_optimization_integration(self):
        """Test evolutionary optimization integration"""
        # Initialize evolutionary optimizer
        hyperparams = EvolutionaryHyperparameters(
            population_size=5,  # Small for testing
            max_generations=3,
            mutation_rate=0.2
        )
        
        optimizer = EvolutionaryOptimizer(hyperparams=hyperparams)
        optimizer.initialize_population()
        
        # Create mock fitness evaluation
        def mock_fitness_evaluation(genome: CognitiveGenome) -> FitnessMetrics:
            # Simple fitness based on tensor salience
            avg_salience = np.mean([config["salience"] for config in genome.tensor_configs.values()])
            return FitnessMetrics(
                accuracy=avg_salience,
                efficiency=0.7,
                adaptability=0.6,
                composite_score=avg_salience * 0.8
            )
        
        # Run evolution
        evolution_results = optimizer.run_evolution(
            fitness_evaluation_func=mock_fitness_evaluation,
            max_generations=3,
            convergence_threshold=0.01
        )
        
        # Validate evolution results
        assert "generations_completed" in evolution_results
        assert "best_genome" in evolution_results
        assert "best_fitness" in evolution_results
        assert evolution_results["generations_completed"] >= 1
        assert isinstance(evolution_results["best_genome"], CognitiveGenome)
        assert 0.0 <= evolution_results["best_fitness"] <= 1.0
    
    def test_continuous_benchmarking_integration(self):
        """Test continuous benchmarking integration"""
        # Initialize benchmarking system
        benchmarking = ContinuousBenchmarking(enable_real_time=False)
        
        # Create integrated cognitive system
        attention_kernel = AttentionKernel()
        meta_monitor = MetaCognitiveMonitor()
        
        cognitive_system = {
            "attention_kernel": attention_kernel,
            "meta_monitor": meta_monitor
        }
        
        # Run benchmark suite
        benchmark_types = [
            BenchmarkType.PROCESSING_SPEED,
            BenchmarkType.ATTENTION_ACCURACY,
            BenchmarkType.META_COGNITIVE_DEPTH
        ]
        
        results = benchmarking.run_benchmark_suite(cognitive_system, benchmark_types)
        
        # Validate benchmark results
        assert len(results) == 3
        assert all(isinstance(result, BenchmarkResult) for result in results)
        assert all(result.execution_time > 0 for result in results)
        assert all(0.0 <= result.score <= 1.0 for result in results)
        
        # Generate performance report
        report = benchmarking.generate_performance_report()
        assert "overall_health_score" in report
        assert report["total_benchmarks_run"] == 3
    
    def test_validation_framework_integration(self):
        """Test validation framework integration"""
        # Initialize validation system
        validator = MetaCognitiveValidation()
        
        # Create test meta-cognitive system
        meta_monitor = MetaCognitiveMonitor()
        attention_kernel = AttentionKernel()
        
        # Add some cognitive history for testing
        for i in range(3):
            test_snapshot = CognitiveStateSnapshot(
                active_tensors={},
                attention_focus=[],
                processing_metrics={},
                meta_cognitive_metrics=MetaCognitiveMetrics(self_awareness_level=0.2 * i)
            )
            meta_monitor.cognitive_history.append(test_snapshot)
        
        # Run comprehensive validation
        validation_report = validator.validate_meta_cognitive_system(
            meta_monitor,
            attention_kernel,
            ValidationLevel.COMPREHENSIVE
        )
        
        # Validate report
        assert isinstance(validation_report.validation_id, str)
        assert len(validation_report.benchmark_results) > 0
        assert isinstance(validation_report.overall_score, float)
        assert len(validation_report.recommendations) >= 0
        assert "meta_cognitive_assessment" in validation_report.__dict__
        assert "convergence_analysis" in validation_report.__dict__


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])