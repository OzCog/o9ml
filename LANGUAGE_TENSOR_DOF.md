# Language Layer: Tensor Dimensions Documentation

## Language Tensor DOF Structure

The Language Layer extends the foundation tensor framework with language-specific
cognitive representations for natural language understanding and generation.

### Core Tensor Dimensions (from FOUNDATION_TENSOR_DOF.md)

**Spatial (3D)**: Syntactic positioning in parse structures
- Dimension: 3 (x, y, z coordinates in parse tree)  
- Use cases: Syntactic dependency trees, grammatical hierarchy
- Implementation: `float spatial[3]` - [tree_depth, child_index, phrase_level]

**Temporal (1D)**: Sequential position and timing in language
- Dimension: 1 (position in sentence/discourse sequence)
- Use cases: Word order, sentence flow, discourse coherence
- Implementation: `float temporal[1]` - [sequence_position]

**Semantic (256D)**: Conceptual meaning and semantic relationships  
- Dimension: 256 (semantic embedding space)
- Use cases: Word meanings, concept similarity, semantic roles
- Implementation: `float semantic[256]` - dense concept embeddings

**Logical (64D)**: Grammatical and logical relationships
- Dimension: 64 (grammatical feature space)
- Use cases: Parts of speech, grammatical roles, logical connectives
- Implementation: `float logical[64]` - grammatical feature vectors

### Language-Specific Tensor Operations

#### Syntactic Parsing Operations
```cpp
// Parse tree spatial positioning
result.spatial[0] = parse_depth;      // depth in parse tree
result.spatial[1] = child_index;      // position among siblings  
result.spatial[2] = phrase_level;     // phrasal vs lexical level
```

#### Semantic Composition Operations  
```cpp
// Semantic vector composition (simplified)
for(int i = 0; i < 256; i++) {
    result.semantic[i] = alpha * word1.semantic[i] + 
                        beta * word2.semantic[i] + 
                        gamma * context.semantic[i];
}
```

#### Grammatical Feature Integration
```cpp
// Grammatical feature inheritance in parse structures
for(int i = 0; i < 64; i++) {
    result.logical[i] = std::max(parent.logical[i], 
                                child.logical[i] * inheritance_weight);
}
```

### Integration with Link Grammar

Link Grammar structures map to tensor dimensions as follows:

- **Spatial**: Link positions in sentence (left/right connections)
- **Temporal**: Word sequence order preservation  
- **Semantic**: Disjunct semantic roles and meanings
- **Logical**: Link types and grammatical constraints

### Integration with AtomSpace

Language tensors integrate with AtomSpace atoms:

- **ConceptNode**: Primarily semantic[256D] representation
- **PredicateNode**: Logical[64D] + semantic[256D] features
- **EvaluationLink**: Spatial[3D] relationship + logical[64D] structure
- **BindLink**: Pattern matching across all tensor dimensions

### Cognitive Grammar Encoding

The language layer forms the interface for neural-symbolic convergence:

```
Natural Language Input
       ↓
Link Grammar Parsing → Spatial[3D] + Temporal[1D] structure
       ↓  
Semantic Analysis → Semantic[256D] embeddings
       ↓
Grammatical Analysis → Logical[64D] features  
       ↓
AtomSpace Integration → Unified tensor representation
       ↓
PLN Reasoning → Cognitive inference over language tensors
```

### Tensor Shape Consistency

All language processing maintains consistent tensor shapes:
- **Spatial**: Always [3] for parse tree coordinates
- **Temporal**: Always [1] for sequence position
- **Semantic**: Always [256] for concept embeddings  
- **Logical**: Always [64] for grammatical features

This ensures seamless integration with other cognitive layers while
preserving the rich structure needed for natural language cognition.