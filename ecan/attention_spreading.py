"""
Attention Spreading

Implements activation spreading mechanisms integrated with AtomSpace for 
dynamic attention propagation across the hypergraph cognitive network.

Key features:
- Activation spreading with AtomSpace integration
- Hypergraph-based attention flow
- Cross-modal attention synchronization  
- Attention conflict resolution protocols
- Dynamic attention topology management
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from .attention_kernel import ECANAttentionTensor, AttentionKernel

logger = logging.getLogger(__name__)


@dataclass
class AttentionLink:
    """Represents a connection for attention spreading"""
    source_atom: str
    target_atom: str
    link_strength: float  # [0.0, 1.0]
    link_type: str = "generic"
    bidirectional: bool = True
    decay_factor: float = 0.95
    
    def compute_spread_amount(self, source_activation: float) -> float:
        """Compute amount of attention to spread through this link"""
        return source_activation * self.link_strength * self.decay_factor


@dataclass
class SpreadingResult:
    """Result of an attention spreading operation"""
    atoms_affected: int
    total_spread: float
    spread_iterations: int
    convergence_achieved: bool
    execution_time: float
    attention_distribution: Dict[str, float]


class AttentionSpreading:
    """
    Attention spreading engine for hypergraph-based activation propagation.
    
    Implements sophisticated spreading algorithms that respect the hypergraph
    structure while maintaining attention conservation and stability.
    """
    
    def __init__(self, 
                 spreading_rate: float = 0.8,
                 decay_rate: float = 0.1,
                 convergence_threshold: float = 0.001,
                 max_iterations: int = 100,
                 preserve_total_attention: bool = True):
        """
        Initialize attention spreading engine.
        
        Args:
            spreading_rate: Base rate of attention spreading [0.0, 1.0]
            decay_rate: Rate of attention decay during spreading [0.0, 1.0]
            convergence_threshold: Threshold for convergence detection
            max_iterations: Maximum spreading iterations
            preserve_total_attention: Whether to preserve total attention amount
        """
        self.spreading_rate = spreading_rate
        self.decay_rate = decay_rate
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.preserve_total_attention = preserve_total_attention
        
        # Attention topology (graph structure)
        self.attention_links: Dict[str, List[AttentionLink]] = defaultdict(list)
        self.reverse_links: Dict[str, List[AttentionLink]] = defaultdict(list)
        
        # Spreading metrics
        self.metrics = {
            'spreads_performed': 0,
            'total_atoms_affected': 0,
            'average_convergence_iterations': 0.0,
            'convergence_success_rate': 0.0,
            'total_spreading_time': 0.0
        }
        
        logger.info(f"AttentionSpreading initialized: rate={spreading_rate}, "
                   f"decay={decay_rate}, convergence_threshold={convergence_threshold}")
    
    def add_attention_link(self, link: AttentionLink) -> bool:
        """
        Add a link for attention spreading.
        
        Args:
            link: Attention link to add
            
        Returns:
            True if link was added successfully
        """
        # Add forward link
        self.attention_links[link.source_atom].append(link)
        
        # Add reverse link if bidirectional
        if link.bidirectional:
            reverse_link = AttentionLink(
                source_atom=link.target_atom,
                target_atom=link.source_atom,
                link_strength=link.link_strength,
                link_type=link.link_type,
                bidirectional=False,  # Avoid infinite recursion
                decay_factor=link.decay_factor
            )
            self.attention_links[link.target_atom].append(reverse_link)
        
        # Update reverse lookup
        self.reverse_links[link.target_atom].append(link)
        
        logger.debug(f"Added attention link: {link.source_atom} -> {link.target_atom} "
                    f"(strength={link.link_strength:.3f})")
        return True
    
    def spread_attention(self, 
                        attention_kernel: AttentionKernel,
                        source_atoms: Optional[List[str]] = None,
                        spread_focus_only: bool = False) -> SpreadingResult:
        """
        Perform attention spreading across the network.
        
        Args:
            attention_kernel: Attention kernel containing current attention state
            source_atoms: Specific atoms to spread from (None for all focus atoms)
            spread_focus_only: Only spread from atoms in attention focus
            
        Returns:
            SpreadingResult with detailed metrics
        """
        start_time = time.time()
        
        # Determine source atoms for spreading
        if source_atoms is None:
            if spread_focus_only:
                focus_atoms = attention_kernel.get_attention_focus()
                source_atoms = [atom_id for atom_id, _ in focus_atoms]
            else:
                # Get all atoms with attention
                source_atoms = list(attention_kernel.attention_tensors.keys())
        
        if not source_atoms:
            logger.warning("No source atoms for attention spreading")
            return SpreadingResult(0, 0.0, 0, True, 0.0, {})
        
        # Initialize spreading state
        current_attention = {}
        for atom_id in source_atoms:
            tensor = attention_kernel.get_attention(atom_id)
            if tensor:
                current_attention[atom_id] = tensor.compute_activation_strength()
        
        # Perform iterative spreading
        iteration = 0
        converged = False
        total_spread = 0.0
        atoms_affected = set(source_atoms)
        
        for iteration in range(self.max_iterations):
            new_attention = current_attention.copy()
            iteration_spread = 0.0
            
            # Spread from each source atom
            for source_atom in current_attention:
                if source_atom not in self.attention_links:
                    continue
                
                source_activation = current_attention[source_atom]
                
                # Spread to connected atoms
                for link in self.attention_links[source_atom]:
                    spread_amount = link.compute_spread_amount(source_activation)
                    
                    # Apply spreading
                    if link.target_atom not in new_attention:
                        new_attention[link.target_atom] = 0.0
                    
                    new_attention[link.target_atom] += spread_amount * self.spreading_rate
                    iteration_spread += spread_amount
                    atoms_affected.add(link.target_atom)
            
            # Apply decay to all attention values
            for atom_id in new_attention:
                new_attention[atom_id] *= (1.0 - self.decay_rate)
            
            # Check for convergence
            if self._check_convergence(current_attention, new_attention):
                converged = True
                break
            
            current_attention = new_attention
            total_spread += iteration_spread
        
        # Apply attention conservation if enabled
        if self.preserve_total_attention:
            current_attention = self._normalize_attention(current_attention, source_atoms, attention_kernel)
        
        # Update attention kernel with new values
        updated_count = self._apply_spread_results(current_attention, attention_kernel)
        
        # Create result
        execution_time = time.time() - start_time
        result = SpreadingResult(
            atoms_affected=len(atoms_affected),
            total_spread=total_spread,
            spread_iterations=iteration + 1,
            convergence_achieved=converged,
            execution_time=execution_time,
            attention_distribution=current_attention
        )
        
        # Update metrics
        self._update_spreading_metrics(result)
        
        logger.info(f"Attention spreading complete: {len(atoms_affected)} atoms affected, "
                   f"{iteration + 1} iterations, converged={converged}")
        
        return result
    
    def _check_convergence(self, 
                          old_attention: Dict[str, float], 
                          new_attention: Dict[str, float]) -> bool:
        """Check if attention spreading has converged"""
        all_atoms = set(old_attention.keys()) | set(new_attention.keys())
        
        total_change = 0.0
        for atom_id in all_atoms:
            old_val = old_attention.get(atom_id, 0.0)
            new_val = new_attention.get(atom_id, 0.0)
            total_change += abs(new_val - old_val)
        
        return total_change < self.convergence_threshold
    
    def _normalize_attention(self, 
                           attention: Dict[str, float],
                           original_atoms: List[str],
                           attention_kernel: AttentionKernel) -> Dict[str, float]:
        """Normalize attention to preserve total attention amount"""
        # Calculate original total attention
        original_total = 0.0
        for atom_id in original_atoms:
            tensor = attention_kernel.get_attention(atom_id)
            if tensor:
                original_total += tensor.compute_activation_strength()
        
        # Calculate current total
        current_total = sum(attention.values())
        
        if current_total > 0 and original_total > 0:
            # Scale to preserve total
            scale_factor = original_total / current_total
            attention = {atom_id: val * scale_factor for atom_id, val in attention.items()}
        
        return attention
    
    def _apply_spread_results(self, 
                            attention_distribution: Dict[str, float],
                            attention_kernel: AttentionKernel) -> int:
        """Apply spreading results to attention kernel"""
        updated_count = 0
        
        for atom_id, activation_strength in attention_distribution.items():
            # Get existing tensor or create new one
            existing_tensor = attention_kernel.get_attention(atom_id)
            
            if existing_tensor:
                # Update existing tensor based on new activation strength
                # Adjust short-term importance proportionally
                if existing_tensor.compute_activation_strength() > 0:
                    scale_factor = activation_strength / existing_tensor.compute_activation_strength()
                    sti_delta = (existing_tensor.short_term_importance * scale_factor) - existing_tensor.short_term_importance
                    
                    success = attention_kernel.update_attention(
                        atom_id, 
                        short_term_delta=np.clip(sti_delta, -1.0, 1.0)
                    )
                    if success:
                        updated_count += 1
            else:
                # Create new tensor for atoms that received spread attention
                if activation_strength > 0.01:  # Threshold for creating new tensors
                    new_tensor = ECANAttentionTensor(
                        short_term_importance=min(activation_strength, 1.0),
                        long_term_importance=0.0,
                        urgency=0.0,
                        confidence=0.5,
                        spreading_factor=0.7,
                        decay_rate=self.decay_rate
                    )
                    
                    success = attention_kernel.allocate_attention(atom_id, new_tensor)
                    if success:
                        updated_count += 1
        
        return updated_count
    
    def create_semantic_topology(self, 
                                atom_similarities: Dict[Tuple[str, str], float],
                                min_similarity: float = 0.3) -> int:
        """
        Create attention topology based on semantic similarities.
        
        Args:
            atom_similarities: Dictionary mapping (atom1, atom2) to similarity score
            min_similarity: Minimum similarity threshold for creating links
            
        Returns:
            Number of links created
        """
        links_created = 0
        
        for (atom1, atom2), similarity in atom_similarities.items():
            if similarity >= min_similarity:
                # Create bidirectional link with strength based on similarity
                link = AttentionLink(
                    source_atom=atom1,
                    target_atom=atom2,
                    link_strength=similarity,
                    link_type="semantic",
                    bidirectional=True,
                    decay_factor=0.95
                )
                
                self.add_attention_link(link)
                links_created += 1
        
        logger.info(f"Created semantic topology: {links_created} links from {len(atom_similarities)} similarities")
        return links_created
    
    def create_temporal_topology(self, 
                               atom_sequence: List[str],
                               temporal_strength: float = 0.6,
                               window_size: int = 3) -> int:
        """
        Create attention topology based on temporal sequences.
        
        Args:
            atom_sequence: Sequence of atoms in temporal order
            temporal_strength: Base strength for temporal links
            window_size: Size of temporal attention window
            
        Returns:
            Number of links created
        """
        links_created = 0
        
        for i, source_atom in enumerate(atom_sequence):
            # Create links to atoms within temporal window
            for j in range(1, min(window_size + 1, len(atom_sequence) - i)):
                target_atom = atom_sequence[i + j]
                
                # Strength decreases with temporal distance
                strength = temporal_strength * (1.0 - (j - 1) / window_size)
                
                link = AttentionLink(
                    source_atom=source_atom,
                    target_atom=target_atom,
                    link_strength=strength,
                    link_type="temporal",
                    bidirectional=False,  # Temporal links are directional
                    decay_factor=0.9
                )
                
                self.add_attention_link(link)
                links_created += 1
        
        logger.info(f"Created temporal topology: {links_created} links from sequence of {len(atom_sequence)} atoms")
        return links_created
    
    def analyze_attention_flow(self, 
                              attention_kernel: AttentionKernel) -> Dict[str, Any]:
        """
        Analyze current attention flow patterns.
        
        Args:
            attention_kernel: Attention kernel to analyze
            
        Returns:
            Analysis results including flow metrics and patterns
        """
        focus_atoms = attention_kernel.get_attention_focus()
        if not focus_atoms:
            return {'flow_patterns': [], 'connectivity_metrics': {}}
        
        # Analyze connectivity patterns
        connectivity_analysis = self._analyze_connectivity()
        
        # Predict attention flow paths
        flow_predictions = self._predict_attention_flow(focus_atoms)
        
        # Calculate flow efficiency metrics
        efficiency_metrics = self._calculate_flow_efficiency(focus_atoms)
        
        return {
            'connectivity_metrics': connectivity_analysis,
            'flow_predictions': flow_predictions,
            'efficiency_metrics': efficiency_metrics,
            'focus_atoms_count': len(focus_atoms),
            'total_links': sum(len(links) for links in self.attention_links.values()),
            'network_density': self._calculate_network_density()
        }
    
    def _analyze_connectivity(self) -> Dict[str, Any]:
        """Analyze attention network connectivity"""
        total_atoms = set()
        total_links = 0
        
        for source_atom, links in self.attention_links.items():
            total_atoms.add(source_atom)
            total_links += len(links)
            for link in links:
                total_atoms.add(link.target_atom)
        
        # Calculate connectivity metrics
        if total_atoms:
            avg_degree = total_links / len(total_atoms)
            max_degree = max(len(links) for links in self.attention_links.values()) if self.attention_links else 0
        else:
            avg_degree = 0
            max_degree = 0
        
        return {
            'total_atoms': len(total_atoms),
            'total_links': total_links,
            'average_degree': avg_degree,
            'max_degree': max_degree,
            'connectivity_ratio': avg_degree / max(len(total_atoms), 1)
        }
    
    def _predict_attention_flow(self, 
                               focus_atoms: List[Tuple[str, ECANAttentionTensor]]) -> List[Dict[str, Any]]:
        """Predict attention flow paths from focus atoms"""
        predictions = []
        
        for atom_id, tensor in focus_atoms[:5]:  # Limit to top 5 focus atoms
            if atom_id in self.attention_links:
                # Find strongest outgoing links
                outgoing_links = sorted(
                    self.attention_links[atom_id],
                    key=lambda x: x.link_strength,
                    reverse=True
                )[:3]  # Top 3 links
                
                flow_path = {
                    'source_atom': atom_id,
                    'source_activation': tensor.compute_activation_strength(),
                    'predicted_targets': [
                        {
                            'target_atom': link.target_atom,
                            'predicted_flow': link.compute_spread_amount(tensor.compute_activation_strength()),
                            'link_strength': link.link_strength,
                            'link_type': link.link_type
                        }
                        for link in outgoing_links
                    ]
                }
                predictions.append(flow_path)
        
        return predictions
    
    def _calculate_flow_efficiency(self, 
                                  focus_atoms: List[Tuple[str, ECANAttentionTensor]]) -> Dict[str, float]:
        """Calculate attention flow efficiency metrics"""
        if not focus_atoms:
            return {'flow_efficiency': 0.0, 'spread_potential': 0.0}
        
        total_activation = sum(tensor.compute_activation_strength() for _, tensor in focus_atoms)
        
        # Calculate potential spread based on link strengths
        potential_spread = 0.0
        for atom_id, tensor in focus_atoms:
            if atom_id in self.attention_links:
                activation = tensor.compute_activation_strength()
                for link in self.attention_links[atom_id]:
                    potential_spread += link.compute_spread_amount(activation)
        
        flow_efficiency = potential_spread / max(total_activation, 0.001)
        
        return {
            'flow_efficiency': flow_efficiency,
            'spread_potential': potential_spread,
            'total_focus_activation': total_activation
        }
    
    def _calculate_network_density(self) -> float:
        """Calculate attention network density"""
        total_atoms = set()
        for source_atom, links in self.attention_links.items():
            total_atoms.add(source_atom)
            for link in links:
                total_atoms.add(link.target_atom)
        
        n = len(total_atoms)
        if n < 2:
            return 0.0
        
        total_links = sum(len(links) for links in self.attention_links.values())
        max_possible_links = n * (n - 1)  # Directed graph
        
        return total_links / max_possible_links if max_possible_links > 0 else 0.0
    
    def _update_spreading_metrics(self, result: SpreadingResult):
        """Update spreading performance metrics"""
        self.metrics['spreads_performed'] += 1
        self.metrics['total_atoms_affected'] += result.atoms_affected
        self.metrics['total_spreading_time'] += result.execution_time
        
        # Update convergence metrics
        total_spreads = self.metrics['spreads_performed']
        if result.convergence_achieved:
            success_count = self.metrics['convergence_success_rate'] * (total_spreads - 1) + 1
            self.metrics['convergence_success_rate'] = success_count / total_spreads
        else:
            success_count = self.metrics['convergence_success_rate'] * (total_spreads - 1)
            self.metrics['convergence_success_rate'] = success_count / total_spreads
        
        # Update average iterations
        old_avg = self.metrics['average_convergence_iterations']
        self.metrics['average_convergence_iterations'] = (
            (old_avg * (total_spreads - 1) + result.spread_iterations) / total_spreads
        )
    
    def get_spreading_metrics(self) -> Dict[str, Any]:
        """Get comprehensive spreading metrics"""
        return {
            **self.metrics,
            'atoms_per_spread': self.metrics['total_atoms_affected'] / max(self.metrics['spreads_performed'], 1),
            'average_spread_time': self.metrics['total_spreading_time'] / max(self.metrics['spreads_performed'], 1),
            'network_size': len(set().union(*[
                {source} | {link.target_atom for link in links}
                for source, links in self.attention_links.items()
            ])) if self.attention_links else 0,
            'total_topology_links': sum(len(links) for links in self.attention_links.values())
        }
    
    def reset_topology(self):
        """Reset attention topology and metrics"""
        self.attention_links.clear()
        self.reverse_links.clear()
        
        self.metrics = {
            'spreads_performed': 0,
            'total_atoms_affected': 0,
            'average_convergence_iterations': 0.0,
            'convergence_success_rate': 0.0,
            'total_spreading_time': 0.0
        }
        
        logger.info("Reset attention spreading topology and metrics")