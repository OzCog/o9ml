"""
Phase 5: Evolutionary Optimization Framework
Adaptive Optimization Implementation

This module implements evolutionary algorithms for cognitive architecture optimization,
continuous benchmarking, and adaptive hyperparameter tuning.
"""

import numpy as np
import random
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import copy

from cogml.cognitive_primitives import CognitivePrimitiveTensor, TensorSignature, ModalityType, DepthType, ContextType
from cogml.hypergraph_encoding import HypergraphEncoder
from ecan.attention_kernel import AttentionKernel, ECANAttentionTensor
from meta_cognition import MetaCognitiveMonitor, CognitiveStateSnapshot, MetaCognitiveMetrics


class OptimizationTarget(Enum):
    """Optimization targets for evolutionary algorithms"""
    SPEED = "speed"
    ACCURACY = "accuracy"
    GENERALIZATION = "generalization"
    EFFICIENCY = "efficiency"
    ADAPTABILITY = "adaptability"


class SelectionMethod(Enum):
    """Selection methods for evolutionary algorithms"""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    ELITE = "elite"


class MutationType(Enum):
    """Types of mutations for cognitive architectures"""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    STRUCTURE_MODIFICATION = "structure_modification"
    ATTENTION_REALLOCATION = "attention_reallocation"
    TENSOR_RECONFIGURATION = "tensor_reconfiguration"


@dataclass
class EvolutionaryHyperparameters:
    """Hyperparameters for evolutionary optimization"""
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    selection_pressure: float = 2.0
    elite_size: int = 5
    max_generations: int = 100
    convergence_threshold: float = 0.001
    diversity_maintenance: float = 0.2


@dataclass
class CognitiveGenome:
    """Represents a cognitive architecture configuration as a genome"""
    tensor_configs: Dict[str, Dict[str, Any]]
    attention_params: Dict[str, float]
    processing_params: Dict[str, float]
    meta_cognitive_params: Dict[str, float]
    fitness_score: float = 0.0
    generation: int = 0
    age: int = 0
    mutation_history: List[str] = field(default_factory=list)


@dataclass
class FitnessMetrics:
    """Multi-objective fitness metrics for cognitive performance"""
    accuracy: float = 0.0
    efficiency: float = 0.0
    adaptability: float = 0.0
    generalization: float = 0.0
    speed: float = 0.0
    robustness: float = 0.0
    novelty: float = 0.0
    stability: float = 0.0
    composite_score: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvolutionaryMetrics:
    """Metrics tracking evolutionary progress"""
    generation: int = 0
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    population_diversity: float = 0.0
    convergence_rate: float = 0.0
    mutation_success_rate: float = 0.0
    crossover_success_rate: float = 0.0
    elapsed_time: float = 0.0
    evaluations_performed: int = 0


class EvolutionaryOptimizer:
    """
    Core evolutionary optimization engine for cognitive architectures.
    Uses genetic algorithms to evolve and optimize cognitive performance.
    """
    
    def __init__(self, 
                 hyperparams: EvolutionaryHyperparameters = None,
                 optimization_targets: List[OptimizationTarget] = None):
        
        self.hyperparams = hyperparams or EvolutionaryHyperparameters()
        self.optimization_targets = optimization_targets or [OptimizationTarget.ACCURACY, OptimizationTarget.EFFICIENCY]
        
        self.population: List[CognitiveGenome] = []
        self.generation = 0
        self.evolution_history: List[EvolutionaryMetrics] = []
        self.fitness_evaluator = MultiObjectiveFitnessEvaluator(self.optimization_targets)
        
        self.logger = logging.getLogger(__name__)
        
    def initialize_population(self, 
                            template_genome: CognitiveGenome = None,
                            diversity_factor: float = 0.5) -> None:
        """Initialize the evolutionary population"""
        
        self.population = []
        
        # Create base genome if none provided
        if template_genome is None:
            template_genome = self._create_default_genome()
        
        # Generate diverse population
        for i in range(self.hyperparams.population_size):
            if i == 0:
                # Keep one exact copy of template
                genome = copy.deepcopy(template_genome)
            else:
                # Create variations
                genome = self._create_variant_genome(template_genome, diversity_factor)
            
            genome.generation = 0
            genome.age = 0
            self.population.append(genome)
        
        self.logger.info(f"Initialized population of {len(self.population)} cognitive genomes")
    
    def evolve_generation(self, 
                         fitness_evaluation_func: Callable[[CognitiveGenome], FitnessMetrics]) -> EvolutionaryMetrics:
        """Evolve one generation of the population"""
        
        start_time = time.time()
        
        # Evaluate fitness for all individuals
        self._evaluate_population_fitness(fitness_evaluation_func)
        
        # Selection
        selected_genomes = self._selection()
        
        # Reproduction (crossover and mutation)
        new_population = self._reproduction(selected_genomes)
        
        # Replacement
        self.population = new_population
        self.generation += 1
        
        # Update ages
        for genome in self.population:
            genome.age += 1
        
        # Compute evolutionary metrics
        metrics = self._compute_evolutionary_metrics(time.time() - start_time)
        self.evolution_history.append(metrics)
        
        self.logger.info(f"Generation {self.generation}: Best fitness = {metrics.best_fitness:.4f}, "
                        f"Avg fitness = {metrics.average_fitness:.4f}, "
                        f"Diversity = {metrics.population_diversity:.4f}")
        
        return metrics
    
    def run_evolution(self, 
                     fitness_evaluation_func: Callable[[CognitiveGenome], FitnessMetrics],
                     max_generations: int = None,
                     convergence_threshold: float = None) -> Dict[str, Any]:
        """Run complete evolutionary optimization process"""
        
        max_gens = max_generations or self.hyperparams.max_generations
        convergence_thresh = convergence_threshold or self.hyperparams.convergence_threshold
        
        evolution_start_time = time.time()
        convergence_counter = 0
        
        evolution_results = {
            "generations_completed": 0,
            "best_genome": None,
            "best_fitness": 0.0,
            "convergence_achieved": False,
            "total_time": 0.0,
            "evolution_trajectory": []
        }
        
        for generation in range(max_gens):
            metrics = self.evolve_generation(fitness_evaluation_func)
            evolution_results["evolution_trajectory"].append(metrics)
            
            # Check for convergence
            if generation > 0:
                prev_best = self.evolution_history[-2].best_fitness
                current_best = metrics.best_fitness
                improvement = current_best - prev_best
                
                if improvement < convergence_thresh:
                    convergence_counter += 1
                else:
                    convergence_counter = 0
                
                # Converged if no improvement for several generations
                if convergence_counter >= 5:
                    evolution_results["convergence_achieved"] = True
                    break
            
            # Early stopping if fitness plateaus
            if metrics.best_fitness > 0.99:
                break
        
        # Finalize results
        best_genome = max(self.population, key=lambda g: g.fitness_score)
        evolution_results.update({
            "generations_completed": self.generation,
            "best_genome": best_genome,
            "best_fitness": best_genome.fitness_score,
            "total_time": time.time() - evolution_start_time
        })
        
        return evolution_results
    
    def _create_default_genome(self) -> CognitiveGenome:
        """Create a default cognitive genome"""
        
        tensor_configs = {
            "primary_visual": {
                "modality": ModalityType.VISUAL,
                "depth": DepthType.SEMANTIC,
                "context": ContextType.GLOBAL,
                "salience": 0.7,
                "autonomy_index": 0.5
            },
            "primary_textual": {
                "modality": ModalityType.TEXTUAL,
                "depth": DepthType.PRAGMATIC,
                "context": ContextType.TEMPORAL,
                "salience": 0.8,
                "autonomy_index": 0.6
            },
            "primary_symbolic": {
                "modality": ModalityType.SYMBOLIC,
                "depth": DepthType.SURFACE,
                "context": ContextType.LOCAL,
                "salience": 0.6,
                "autonomy_index": 0.4
            }
        }
        
        attention_params = {
            "focus_threshold": 0.5,
            "spreading_rate": 0.8,
            "decay_rate": 0.1,
            "max_focus_items": 7
        }
        
        processing_params = {
            "processing_speed": 1.0,
            "memory_capacity": 1000.0,
            "learning_rate": 0.01,
            "adaptation_rate": 0.1
        }
        
        meta_cognitive_params = {
            "self_awareness_sensitivity": 0.7,
            "reflection_depth": 3,
            "introspection_frequency": 1.0,
            "meta_learning_rate": 0.05
        }
        
        return CognitiveGenome(
            tensor_configs=tensor_configs,
            attention_params=attention_params,
            processing_params=processing_params,
            meta_cognitive_params=meta_cognitive_params
        )
    
    def _create_variant_genome(self, template: CognitiveGenome, diversity_factor: float) -> CognitiveGenome:
        """Create a variant of a template genome with random variations"""
        
        variant = copy.deepcopy(template)
        
        # Vary tensor configurations
        for tensor_id, config in variant.tensor_configs.items():
            if random.random() < diversity_factor:
                config["salience"] = np.clip(config["salience"] + np.random.normal(0, 0.1), 0.0, 1.0)
                config["autonomy_index"] = np.clip(config["autonomy_index"] + np.random.normal(0, 0.1), 0.0, 1.0)
        
        # Vary attention parameters
        for param, value in variant.attention_params.items():
            if random.random() < diversity_factor:
                if param in ["focus_threshold", "spreading_rate"]:
                    variant.attention_params[param] = np.clip(value + np.random.normal(0, 0.1), 0.0, 1.0)
                elif param == "max_focus_items":
                    variant.attention_params[param] = max(1, int(value + np.random.normal(0, 2)))
        
        # Vary processing parameters
        for param, value in variant.processing_params.items():
            if random.random() < diversity_factor:
                multiplier = 1.0 + np.random.normal(0, 0.2)
                variant.processing_params[param] = max(0.01, value * multiplier)
        
        # Vary meta-cognitive parameters
        for param, value in variant.meta_cognitive_params.items():
            if random.random() < diversity_factor:
                if isinstance(value, float):
                    if param in ["self_awareness_sensitivity", "introspection_frequency"]:
                        variant.meta_cognitive_params[param] = np.clip(value + np.random.normal(0, 0.1), 0.0, 1.0)
                    else:
                        variant.meta_cognitive_params[param] = max(0.001, value + np.random.normal(0, value * 0.2))
                elif isinstance(value, int):
                    variant.meta_cognitive_params[param] = max(1, int(value + np.random.normal(0, 1)))
        
        return variant
    
    def _evaluate_population_fitness(self, fitness_func: Callable[[CognitiveGenome], FitnessMetrics]) -> None:
        """Evaluate fitness for entire population"""
        
        for genome in self.population:
            try:
                fitness_metrics = fitness_func(genome)
                genome.fitness_score = fitness_metrics.composite_score
            except Exception as e:
                self.logger.warning(f"Fitness evaluation failed for genome: {e}")
                genome.fitness_score = 0.0
    
    def _selection(self) -> List[CognitiveGenome]:
        """Select individuals for reproduction using tournament selection"""
        
        selected = []
        tournament_size = max(2, int(self.hyperparams.population_size * 0.1))
        
        # Elite selection - always keep best performers
        sorted_pop = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
        elite_count = min(self.hyperparams.elite_size, len(sorted_pop))
        selected.extend(sorted_pop[:elite_count])
        
        # Tournament selection for remaining slots
        remaining_slots = self.hyperparams.population_size - elite_count
        
        for _ in range(remaining_slots):
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda g: g.fitness_score)
            selected.append(copy.deepcopy(winner))
        
        return selected
    
    def _reproduction(self, selected_genomes: List[CognitiveGenome]) -> List[CognitiveGenome]:
        """Create new generation through crossover and mutation"""
        
        new_population = []
        
        # Keep elite individuals unchanged
        elite_count = min(self.hyperparams.elite_size, len(selected_genomes))
        new_population.extend(selected_genomes[:elite_count])
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.hyperparams.population_size:
            parent1 = random.choice(selected_genomes)
            parent2 = random.choice(selected_genomes)
            
            # Crossover
            if random.random() < self.hyperparams.crossover_rate and parent1 != parent2:
                child = self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(random.choice([parent1, parent2]))
            
            # Mutation
            if random.random() < self.hyperparams.mutation_rate:
                child = self._mutate(child)
            
            child.generation = self.generation + 1
            child.age = 0
            new_population.append(child)
        
        return new_population[:self.hyperparams.population_size]
    
    def _crossover(self, parent1: CognitiveGenome, parent2: CognitiveGenome) -> CognitiveGenome:
        """Perform crossover between two genomes"""
        
        child = copy.deepcopy(parent1)
        
        # Crossover tensor configurations
        for tensor_id in child.tensor_configs.keys():
            if tensor_id in parent2.tensor_configs and random.random() < 0.5:
                for param, value in parent2.tensor_configs[tensor_id].items():
                    if random.random() < 0.5:
                        child.tensor_configs[tensor_id][param] = value
        
        # Crossover attention parameters
        for param in child.attention_params.keys():
            if param in parent2.attention_params and random.random() < 0.5:
                child.attention_params[param] = parent2.attention_params[param]
        
        # Crossover processing parameters
        for param in child.processing_params.keys():
            if param in parent2.processing_params and random.random() < 0.5:
                child.processing_params[param] = parent2.processing_params[param]
        
        # Crossover meta-cognitive parameters
        for param in child.meta_cognitive_params.keys():
            if param in parent2.meta_cognitive_params and random.random() < 0.5:
                child.meta_cognitive_params[param] = parent2.meta_cognitive_params[param]
        
        return child
    
    def _mutate(self, genome: CognitiveGenome) -> CognitiveGenome:
        """Apply mutations to a genome"""
        
        mutation_types = [
            MutationType.PARAMETER_ADJUSTMENT,
            MutationType.ATTENTION_REALLOCATION,
            MutationType.TENSOR_RECONFIGURATION
        ]
        
        mutation_type = random.choice(mutation_types)
        genome.mutation_history.append(mutation_type.value)
        
        if mutation_type == MutationType.PARAMETER_ADJUSTMENT:
            self._mutate_parameters(genome)
        elif mutation_type == MutationType.ATTENTION_REALLOCATION:
            self._mutate_attention(genome)
        elif mutation_type == MutationType.TENSOR_RECONFIGURATION:
            self._mutate_tensors(genome)
        
        return genome
    
    def _mutate_parameters(self, genome: CognitiveGenome) -> None:
        """Mutate processing and meta-cognitive parameters"""
        
        # Mutate processing parameters
        for param, value in genome.processing_params.items():
            if random.random() < 0.3:  # 30% chance per parameter
                mutation_strength = np.random.normal(1.0, 0.1)
                genome.processing_params[param] = max(0.001, value * mutation_strength)
        
        # Mutate meta-cognitive parameters
        for param, value in genome.meta_cognitive_params.items():
            if random.random() < 0.3:
                if isinstance(value, float):
                    if param in ["self_awareness_sensitivity", "introspection_frequency"]:
                        genome.meta_cognitive_params[param] = np.clip(value + np.random.normal(0, 0.05), 0.0, 1.0)
                    else:
                        genome.meta_cognitive_params[param] = max(0.001, value + np.random.normal(0, value * 0.1))
                elif isinstance(value, int):
                    genome.meta_cognitive_params[param] = max(1, int(value + np.random.normal(0, 0.5)))
    
    def _mutate_attention(self, genome: CognitiveGenome) -> None:
        """Mutate attention parameters"""
        
        for param, value in genome.attention_params.items():
            if random.random() < 0.4:  # 40% chance per parameter
                if param in ["focus_threshold", "spreading_rate", "decay_rate"]:
                    genome.attention_params[param] = np.clip(value + np.random.normal(0, 0.05), 0.0, 1.0)
                elif param == "max_focus_items":
                    genome.attention_params[param] = max(1, int(value + np.random.normal(0, 1)))
    
    def _mutate_tensors(self, genome: CognitiveGenome) -> None:
        """Mutate tensor configurations"""
        
        for tensor_id, config in genome.tensor_configs.items():
            if random.random() < 0.3:  # 30% chance per tensor
                # Mutate salience and autonomy
                config["salience"] = np.clip(config["salience"] + np.random.normal(0, 0.05), 0.0, 1.0)
                config["autonomy_index"] = np.clip(config["autonomy_index"] + np.random.normal(0, 0.05), 0.0, 1.0)
                
                # Occasionally change modality, depth, or context
                if random.random() < 0.1:  # 10% chance for structural change
                    if random.random() < 0.33:
                        config["modality"] = random.choice(list(ModalityType))
                    elif random.random() < 0.5:
                        config["depth"] = random.choice(list(DepthType))
                    else:
                        config["context"] = random.choice(list(ContextType))
    
    def _compute_evolutionary_metrics(self, generation_time: float) -> EvolutionaryMetrics:
        """Compute metrics for current generation"""
        
        fitness_scores = [genome.fitness_score for genome in self.population]
        
        best_fitness = max(fitness_scores)
        average_fitness = np.mean(fitness_scores)
        population_diversity = self._compute_population_diversity()
        
        # Compute convergence rate
        convergence_rate = 0.0
        if len(self.evolution_history) > 0:
            prev_best = self.evolution_history[-1].best_fitness
            convergence_rate = (best_fitness - prev_best) / max(prev_best, 0.001)
        
        return EvolutionaryMetrics(
            generation=self.generation,
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            population_diversity=population_diversity,
            convergence_rate=convergence_rate,
            mutation_success_rate=self._compute_mutation_success_rate(),
            crossover_success_rate=self._compute_crossover_success_rate(),
            elapsed_time=generation_time,
            evaluations_performed=len(self.population)
        )
    
    def _compute_population_diversity(self) -> float:
        """Compute diversity of current population"""
        
        # Compute diversity based on parameter variations
        diversity_measures = []
        
        # Diversity in attention parameters
        focus_thresholds = [g.attention_params.get("focus_threshold", 0.5) for g in self.population]
        diversity_measures.append(np.std(focus_thresholds))
        
        # Diversity in processing parameters
        learning_rates = [g.processing_params.get("learning_rate", 0.01) for g in self.population]
        diversity_measures.append(np.std(learning_rates) * 10)  # Scale for comparison
        
        # Diversity in meta-cognitive parameters
        awareness_levels = [g.meta_cognitive_params.get("self_awareness_sensitivity", 0.7) for g in self.population]
        diversity_measures.append(np.std(awareness_levels))
        
        return np.mean(diversity_measures)
    
    def _compute_mutation_success_rate(self) -> float:
        """Compute success rate of mutations (placeholder)"""
        # This would track which mutations led to fitness improvements
        return 0.3  # Placeholder value
    
    def _compute_crossover_success_rate(self) -> float:
        """Compute success rate of crossovers (placeholder)"""
        # This would track which crossovers led to fitness improvements
        return 0.4  # Placeholder value
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of evolutionary progress"""
        
        if not self.evolution_history:
            return {"status": "No evolution history available"}
        
        best_ever = max(self.evolution_history, key=lambda m: m.best_fitness)
        latest = self.evolution_history[-1]
        
        return {
            "total_generations": len(self.evolution_history),
            "current_generation": self.generation,
            "best_fitness_ever": best_ever.best_fitness,
            "current_best_fitness": latest.best_fitness,
            "current_average_fitness": latest.average_fitness,
            "current_diversity": latest.population_diversity,
            "population_size": len(self.population),
            "convergence_trend": self._analyze_convergence_trend(),
            "optimization_targets": [target.value for target in self.optimization_targets],
            "evolution_efficiency": self._compute_evolution_efficiency()
        }
    
    def _analyze_convergence_trend(self) -> str:
        """Analyze whether evolution is converging"""
        if len(self.evolution_history) < 5:
            return "insufficient_data"
        
        recent_fitness = [m.best_fitness for m in self.evolution_history[-5:]]
        trend = np.polyfit(range(len(recent_fitness)), recent_fitness, 1)[0]
        
        if trend > 0.01:
            return "improving"
        elif trend > -0.01:
            return "stable"
        else:
            return "declining"
    
    def _compute_evolution_efficiency(self) -> float:
        """Compute efficiency of evolutionary process"""
        if not self.evolution_history:
            return 0.0
        
        # Efficiency = improvement per generation
        initial_fitness = self.evolution_history[0].best_fitness
        current_fitness = self.evolution_history[-1].best_fitness
        generations = len(self.evolution_history)
        
        return (current_fitness - initial_fitness) / generations if generations > 0 else 0.0


class MultiObjectiveFitnessEvaluator:
    """
    Multi-objective fitness evaluator for cognitive architectures.
    Evaluates multiple performance dimensions and combines them into composite scores.
    """
    
    def __init__(self, optimization_targets: List[OptimizationTarget]):
        self.optimization_targets = optimization_targets
        self.weight_config = self._create_default_weights()
        self.benchmark_results: List[FitnessMetrics] = []
        
    def evaluate_fitness(self, 
                        genome: CognitiveGenome,
                        test_scenarios: List[Dict[str, Any]] = None) -> FitnessMetrics:
        """Evaluate comprehensive fitness of a cognitive genome"""
        
        fitness_metrics = FitnessMetrics()
        
        # Evaluate each fitness dimension
        fitness_metrics.accuracy = self._evaluate_accuracy(genome, test_scenarios)
        fitness_metrics.efficiency = self._evaluate_efficiency(genome)
        fitness_metrics.adaptability = self._evaluate_adaptability(genome)
        fitness_metrics.generalization = self._evaluate_generalization(genome, test_scenarios)
        fitness_metrics.speed = self._evaluate_speed(genome)
        fitness_metrics.robustness = self._evaluate_robustness(genome)
        fitness_metrics.novelty = self._evaluate_novelty(genome)
        fitness_metrics.stability = self._evaluate_stability(genome)
        
        # Compute composite score
        fitness_metrics.composite_score = self._compute_composite_score(fitness_metrics)
        
        self.benchmark_results.append(fitness_metrics)
        
        return fitness_metrics
    
    def _create_default_weights(self) -> Dict[str, float]:
        """Create default weights for fitness dimensions"""
        return {
            "accuracy": 0.25,
            "efficiency": 0.20,
            "adaptability": 0.15,
            "generalization": 0.15,
            "speed": 0.10,
            "robustness": 0.10,
            "novelty": 0.03,
            "stability": 0.02
        }
    
    def _evaluate_accuracy(self, genome: CognitiveGenome, test_scenarios: List[Dict] = None) -> float:
        """Evaluate accuracy of cognitive processing"""
        
        # Simulate accuracy based on genome configuration
        base_accuracy = 0.7
        
        # Accuracy improves with higher salience values
        avg_salience = np.mean([config["salience"] for config in genome.tensor_configs.values()])
        salience_bonus = avg_salience * 0.2
        
        # Accuracy improves with focused attention
        focus_threshold = genome.attention_params.get("focus_threshold", 0.5)
        focus_bonus = focus_threshold * 0.1
        
        # Meta-cognitive awareness contributes to accuracy
        awareness = genome.meta_cognitive_params.get("self_awareness_sensitivity", 0.7)
        awareness_bonus = awareness * 0.1
        
        total_accuracy = base_accuracy + salience_bonus + focus_bonus + awareness_bonus
        return min(1.0, total_accuracy)
    
    def _evaluate_efficiency(self, genome: CognitiveGenome) -> float:
        """Evaluate processing efficiency"""
        
        base_efficiency = 0.6
        
        # Efficiency improves with optimized attention parameters
        max_focus = genome.attention_params.get("max_focus_items", 7)
        focus_efficiency = 1.0 / (1.0 + max_focus / 10.0)  # Fewer focus items = more efficient
        
        # Processing speed contributes to efficiency
        processing_speed = genome.processing_params.get("processing_speed", 1.0)
        speed_bonus = min(0.3, processing_speed * 0.1)
        
        # Learning rate affects efficiency (moderate is best)
        learning_rate = genome.processing_params.get("learning_rate", 0.01)
        lr_efficiency = 1.0 - abs(learning_rate - 0.02)  # Optimal around 0.02
        
        total_efficiency = base_efficiency + focus_efficiency * 0.2 + speed_bonus + lr_efficiency * 0.1
        return min(1.0, max(0.0, total_efficiency))
    
    def _evaluate_adaptability(self, genome: CognitiveGenome) -> float:
        """Evaluate adaptability and learning capability"""
        
        base_adaptability = 0.5
        
        # Higher learning rates improve adaptability
        learning_rate = genome.processing_params.get("learning_rate", 0.01)
        adaptation_rate = genome.processing_params.get("adaptation_rate", 0.1)
        
        learning_bonus = min(0.3, learning_rate * 10)  # Scale learning rate
        adaptation_bonus = min(0.3, adaptation_rate * 2)
        
        # Meta-learning contributes to adaptability
        meta_learning_rate = genome.meta_cognitive_params.get("meta_learning_rate", 0.05)
        meta_bonus = min(0.2, meta_learning_rate * 4)
        
        total_adaptability = base_adaptability + learning_bonus + adaptation_bonus + meta_bonus
        return min(1.0, total_adaptability)
    
    def _evaluate_generalization(self, genome: CognitiveGenome, test_scenarios: List[Dict] = None) -> float:
        """Evaluate generalization capability"""
        
        base_generalization = 0.6
        
        # Diversity in tensor modalities improves generalization
        modalities = set(config["modality"] for config in genome.tensor_configs.values())
        modality_diversity = len(modalities) / len(ModalityType)
        
        # Balanced autonomy indices help generalization
        autonomy_values = [config["autonomy_index"] for config in genome.tensor_configs.values()]
        autonomy_balance = 1.0 - np.std(autonomy_values)  # Low variance is good
        
        total_generalization = base_generalization + modality_diversity * 0.3 + autonomy_balance * 0.1
        return min(1.0, total_generalization)
    
    def _evaluate_speed(self, genome: CognitiveGenome) -> float:
        """Evaluate processing speed"""
        
        processing_speed = genome.processing_params.get("processing_speed", 1.0)
        
        # Speed is directly related to processing speed parameter
        # But with diminishing returns for very high speeds
        speed_score = processing_speed / (1.0 + processing_speed * 0.2)
        
        # Attention efficiency affects speed
        decay_rate = genome.attention_params.get("decay_rate", 0.1)
        decay_bonus = decay_rate * 0.5  # Faster decay = faster processing
        
        total_speed = speed_score + decay_bonus
        return min(1.0, total_speed)
    
    def _evaluate_robustness(self, genome: CognitiveGenome) -> float:
        """Evaluate robustness to perturbations"""
        
        base_robustness = 0.7
        
        # Higher memory capacity improves robustness
        memory_capacity = genome.processing_params.get("memory_capacity", 1000.0)
        memory_bonus = min(0.2, memory_capacity / 5000.0)
        
        # Stability in attention parameters
        spreading_rate = genome.attention_params.get("spreading_rate", 0.8)
        stability_bonus = min(0.1, spreading_rate * 0.1)
        
        total_robustness = base_robustness + memory_bonus + stability_bonus
        return min(1.0, total_robustness)
    
    def _evaluate_novelty(self, genome: CognitiveGenome) -> float:
        """Evaluate novelty compared to previous genomes"""
        
        if len(self.benchmark_results) < 2:
            return 0.5  # Default novelty for early genomes
        
        # Compare current genome configuration to recent ones
        # This is a simplified novelty measure
        recent_results = self.benchmark_results[-10:]  # Compare to last 10
        
        # Measure diversity in configuration space
        novelty_score = 0.5  # Base novelty
        
        # Add bonus for unique parameter combinations
        if hasattr(genome, 'mutation_history') and genome.mutation_history:
            unique_mutations = len(set(genome.mutation_history))
            novelty_score += min(0.3, unique_mutations * 0.1)
        
        return min(1.0, novelty_score)
    
    def _evaluate_stability(self, genome: CognitiveGenome) -> float:
        """Evaluate stability of cognitive processing"""
        
        base_stability = 0.8
        
        # Moderate parameters tend to be more stable
        salience_values = [config["salience"] for config in genome.tensor_configs.values()]
        salience_stability = 1.0 - np.std(salience_values)  # Low variance is stable
        
        # Consistent attention parameters
        focus_threshold = genome.attention_params.get("focus_threshold", 0.5)
        threshold_stability = 1.0 - abs(focus_threshold - 0.5) * 2  # Optimal around 0.5
        
        total_stability = base_stability * 0.7 + salience_stability * 0.2 + threshold_stability * 0.1
        return min(1.0, max(0.0, total_stability))
    
    def _compute_composite_score(self, metrics: FitnessMetrics) -> float:
        """Compute weighted composite fitness score"""
        
        weighted_sum = (
            metrics.accuracy * self.weight_config["accuracy"] +
            metrics.efficiency * self.weight_config["efficiency"] +
            metrics.adaptability * self.weight_config["adaptability"] +
            metrics.generalization * self.weight_config["generalization"] +
            metrics.speed * self.weight_config["speed"] +
            metrics.robustness * self.weight_config["robustness"] +
            metrics.novelty * self.weight_config["novelty"] +
            metrics.stability * self.weight_config["stability"]
        )
        
        return weighted_sum
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update fitness dimension weights"""
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            for key in new_weights:
                new_weights[key] /= total_weight
        
        self.weight_config.update(new_weights)
    
    def get_fitness_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fitness statistics"""
        
        if not self.benchmark_results:
            return {"status": "No fitness data available"}
        
        # Compute statistics for each dimension
        dimensions = ["accuracy", "efficiency", "adaptability", "generalization", 
                     "speed", "robustness", "novelty", "stability", "composite_score"]
        
        statistics = {}
        
        for dim in dimensions:
            values = [getattr(result, dim) for result in self.benchmark_results]
            statistics[dim] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "trend": self._compute_trend(values)
            }
        
        return {
            "dimensions": statistics,
            "total_evaluations": len(self.benchmark_results),
            "weight_configuration": self.weight_config,
            "optimization_targets": [target.value for target in self.optimization_targets]
        }
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction for a series of values"""
        if len(values) < 3:
            return "insufficient_data"
        
        recent_values = values[-5:]  # Look at last 5 values
        trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        if trend_slope > 0.01:
            return "improving"
        elif trend_slope > -0.01:
            return "stable"
        else:
            return "declining"