#!/usr/bin/env python3
"""
Quick visualization of the cognitive primitives architecture
"""

import matplotlib.pyplot as plt
import numpy as np
from cogml import create_primitive_tensor, ModalityType, DepthType, ContextType

def create_architecture_visualization():
    """Create a visual representation of the cognitive primitives architecture."""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('🧬 CogML Phase 1: Cognitive Primitives Architecture', fontsize=16, fontweight='bold')
    
    # 1. Tensor Dimensions Visualization
    ax1.set_title('5D Tensor Structure')
    dimensions = ['Modality\n(4)', 'Depth\n(3)', 'Context\n(3)', 'Salience\n(100)', 'Autonomy\n(100)']
    sizes = [4, 3, 3, 100, 100]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    bars = ax1.bar(dimensions, sizes, color=colors, alpha=0.7)
    ax1.set_ylabel('Dimension Size')
    ax1.set_yscale('log')
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{size}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Modality Distribution
    ax2.set_title('Cognitive Modalities')
    modalities = ['Visual', 'Auditory', 'Textual', 'Symbolic']
    modality_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    wedges, texts, autotexts = ax2.pie([1, 1, 1, 1], labels=modalities, colors=modality_colors, 
                                       autopct='%1.0f%%', startangle=90)
    
    # 3. Performance Metrics
    ax3.set_title('Performance Benchmarks')
    metrics = ['Tensor\nCreation\n(26K/s)', 'Encoding\n(2.5K/s)', 'Round-trip\nAccuracy\n(100%)', 'Memory\nEfficiency\n(95%+)']
    values = [26482, 2500, 100, 95]
    colors_perf = ['#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
    
    bars_perf = ax3.bar(metrics, values, color=colors_perf, alpha=0.7)
    ax3.set_ylabel('Performance Value')
    ax3.set_yscale('log')
    
    # 4. Architecture Components
    ax4.set_title('System Architecture')
    ax4.axis('off')
    
    # Create architecture diagram
    components = [
        {'name': 'Cognitive\nPrimitives', 'pos': (0.2, 0.8), 'color': '#FF6B6B'},
        {'name': 'Hypergraph\nEncoding', 'pos': (0.8, 0.8), 'color': '#4ECDC4'},
        {'name': 'Scheme\nTranslation', 'pos': (0.2, 0.4), 'color': '#45B7D1'},
        {'name': 'AtomSpace\nIntegration', 'pos': (0.8, 0.4), 'color': '#96CEB4'},
        {'name': 'Validation\nFramework', 'pos': (0.5, 0.1), 'color': '#FECA57'}
    ]
    
    # Draw components
    for comp in components:
        circle = plt.Circle(comp['pos'], 0.12, color=comp['color'], alpha=0.7)
        ax4.add_patch(circle)
        ax4.text(comp['pos'][0], comp['pos'][1], comp['name'], 
                ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Draw connections
    connections = [
        ((0.2, 0.8), (0.8, 0.8)),  # Primitives to Encoding
        ((0.2, 0.8), (0.2, 0.4)),  # Primitives to Translation
        ((0.8, 0.8), (0.8, 0.4)),  # Encoding to AtomSpace
        ((0.2, 0.4), (0.8, 0.4)),  # Translation to AtomSpace
        ((0.5, 0.3), (0.5, 0.22))  # Center to Validation
    ]
    
    for start, end in connections:
        ax4.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.5, linewidth=2)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('cognitive_primitives_architecture.png', dpi=300, bbox_inches='tight')
    print("📊 Architecture visualization saved as 'cognitive_primitives_architecture.png'")
    plt.show()

def create_tensor_example_visualization():
    """Create visualization of example tensor encodings."""
    
    # Create example tensors
    examples = [
        ("Visual Perception", ModalityType.VISUAL, DepthType.SURFACE, ContextType.LOCAL, 0.9, 0.3),
        ("Symbolic Reasoning", ModalityType.SYMBOLIC, DepthType.PRAGMATIC, ContextType.GLOBAL, 0.7, 0.8),
        ("Language Processing", ModalityType.TEXTUAL, DepthType.SEMANTIC, ContextType.TEMPORAL, 0.8, 0.5),
        ("Audio Analysis", ModalityType.AUDITORY, DepthType.SURFACE, ContextType.LOCAL, 0.6, 0.4)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('🧠 Example Cognitive Primitive Tensors', fontsize=16, fontweight='bold')
    
    for i, (name, modality, depth, context, salience, autonomy) in enumerate(examples):
        ax = axes[i//2, i%2]
        
        # Create tensor
        tensor = create_primitive_tensor(modality, depth, context, salience, autonomy)
        
        # Get encoding
        encoding = tensor.get_primitive_encoding()
        
        # Create visualization
        ax.bar(range(len(encoding)), encoding, alpha=0.7, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][i])
        ax.set_title(f'{name}\n{modality.name}-{depth.name}-{context.name}')
        ax.set_xlabel('Encoding Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add tensor info
        info_text = f'Salience: {salience}\nAutonomy: {autonomy}\nDOF: {tensor.compute_degrees_of_freedom()}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('cognitive_tensor_examples.png', dpi=300, bbox_inches='tight')
    print("📊 Tensor examples visualization saved as 'cognitive_tensor_examples.png'")
    plt.show()

if __name__ == "__main__":
    print("🎨 Creating CogML Phase 1 visualizations...")
    create_architecture_visualization()
    create_tensor_example_visualization()
    print("✅ Visualizations complete!")