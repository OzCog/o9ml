# Language Tensor Dimensions and Architecture

This document describes the tensor dimensions and vector spaces used in the OpenCog Natural Language Cognition layer, demonstrating how linguistic information flows through the cognitive architecture.

## Overview

The language processing pipeline transforms natural language through multiple representational spaces, each with specific tensor dimensions optimized for different cognitive operations.

## Tensor Architecture

### 1. Input Text Representation
- **Format**: Raw UTF-8 text strings
- **Tokenization**: Word-level and character-level tokens
- **Vocabulary Size**: ~50,000 common English words
- **Encoding**: Variable-length sequences

### 2. Link Grammar Parse Space
- **Dimension**: Variable connectivity graph
- **Link Types**: ~200 grammatical link types (Ss, Os, Ds, etc.)
- **Parse Trees**: Hierarchical structures with confidence scores
- **Complexity**: O(n³) parsing time for n words

### 3. RelEx Dependency Space
- **Dependency Relations**: ~40 core relation types
  - `_subj`, `_obj`, `_iobj` (grammatical relations)
  - `_amod`, `_det`, `_prep` (modifier relations)
  - `_conj`, `_appos` (coordination relations)
- **Attributes**: ~15 linguistic feature dimensions
  - Part-of-speech (12 categories)
  - Tense (6 values: present, past, future, etc.)
  - Number (3 values: singular, plural, uncountable)
  - Gender (4 values: masculine, feminine, neuter, unknown)
  - Definiteness (2 values: definite, indefinite)
  - Person (3 values: first, second, third)

### 4. AtomSpace Representation
- **Node Types**: 
  - WordNode: Individual lexical items
  - ConceptNode: Semantic concepts
  - PredicateNode: Relations and properties
  - SentenceNode: Complete utterances
- **Link Types**:
  - EvaluationLink: Predicate assertions
  - ListLink: Argument structures
  - InheritanceLink: Taxonomic relations
  - SimilarityLink: Semantic similarity

### 5. Semantic Vector Space
- **Dimensions**: 256D semantic embedding space
- **Word Vectors**: Dense real-valued representations
- **Similarity Metrics**: Cosine similarity, Euclidean distance
- **Coverage**: ~95% of common vocabulary

### 6. PLN Logical Space
- **Truth Values**: (strength, confidence) pairs in [0,1]²
- **Logical Forms**: First-order predicate logic
- **Inference Rules**: ~20 core PLN rules
  - Modus Ponens: P→Q, P ⊢ Q
  - Deduction: P→Q, Q→R ⊢ P→R
  - Abduction: P→Q, Q ⊢ P (probabilistic)
  - Induction: P∧Q observed ⊢ P→Q (probabilistic)

## Tensor Flow Pipeline

```
Raw Text → Tokens → Parse Graph → Dependencies → AtomSpace → Vectors → Logic
  |         |          |            |            |          |        |
 UTF-8   List[str]  Graph(V,E)  Relations   Hypergraph   ℝ²⁵⁶    PLN(TV)
  |         |          |            |            |          |        |
Variable  Variable   O(n³)      O(n²)      O(atoms)    O(vocab)   O(rules)
```

## Dimensional Analysis

### Memory Complexity
- **Text Storage**: O(sequence_length)
- **Parse Representation**: O(n² × link_types) 
- **Dependency Graph**: O(n × relations × attributes)
- **AtomSpace**: O(atoms × average_arity)
- **Semantic Vectors**: O(vocabulary × 256)
- **Logic Database**: O(facts × 2) for truth values

### Processing Complexity
- **Parsing**: O(n³) for Link Grammar
- **Dependency Extraction**: O(n²) for relation finding
- **AtomSpace Operations**: O(log atoms) for indexed access
- **Vector Operations**: O(256) for similarity computation
- **PLN Inference**: O(rules × facts) for forward chaining

## Tensor Transformations

### 1. Parse → Dependency Transformation
```python
# Input: Parse tree with link structure
parse_graph: Graph[Word, LinkType] 
# Output: Dependency relations
dependencies: List[Tuple[Relation, Word, Word]]
```

### 2. Dependency → AtomSpace Transformation
```scheme
;; Input: _subj(loves, John)
;; Output: AtomSpace representation
(EvaluationLink
    (PredicateNode "subject")
    (ListLink
        (WordNode "loves")
        (WordNode "John")
    )
)
```

### 3. AtomSpace → Vector Transformation
```python
# Input: AtomSpace atoms
atoms: List[Atom]
# Output: Embedded vectors
vectors: numpy.ndarray[atoms, 256]
```

### 4. Vector → Logic Transformation
```python
# Input: Semantic vectors
embeddings: numpy.ndarray[256]
# Output: Logical predicates with truth values
predicates: List[Tuple[Predicate, TruthValue]]
```

## Performance Characteristics

### Throughput Metrics
- **Parsing Speed**: ~100 sentences/second
- **Dependency Extraction**: ~500 sentences/second
- **AtomSpace Storage**: ~1000 operations/second
- **Vector Computation**: ~10,000 operations/second
- **PLN Inference**: ~100 inferences/second

### Accuracy Metrics
- **Parse Accuracy**: ~95% for grammatical sentences
- **Dependency Accuracy**: ~90% for core relations
- **Semantic Similarity**: ~85% correlation with human judgments
- **Logical Consistency**: ~98% for well-formed inputs

## Integration with Cognitive Architecture

### Attention Allocation
- **Parse Confidence**: Higher confidence parses receive more attention
- **Semantic Importance**: Content words weighted over function words
- **Logical Relevance**: Goal-relevant inferences prioritized

### Memory Formation
- **Working Memory**: Active parse trees and dependencies
- **Episodic Memory**: Complete sentence representations
- **Semantic Memory**: Abstract concept relationships
- **Procedural Memory**: Parsing and inference rules

### Learning and Adaptation
- **Parse Learning**: Update grammar weights from experience
- **Semantic Learning**: Refine word embeddings
- **Logical Learning**: Strengthen successful inference patterns

## Example Tensor Flow

For the sentence "Alice ate the mushroom.":

1. **Input**: `"Alice ate the mushroom."`
2. **Tokens**: `["Alice", "ate", "the", "mushroom", "."]`
3. **Parse**: Link structure with confidence 0.98
4. **Dependencies**: `[("_subj", "ate", "Alice"), ("_obj", "ate", "mushroom")]`
5. **AtomSpace**: 15 atoms (nodes + links)
6. **Vectors**: 5 × 256D embeddings
7. **Logic**: 3 facts with truth values: (0.95, 0.9), (0.9, 0.8), (0.8, 0.7)

## Future Extensions

### Planned Tensor Enhancements
- **Contextual Embeddings**: 512D context-aware vectors
- **Multimodal Integration**: Image-text joint embeddings
- **Temporal Sequences**: LSTM/Transformer attention patterns
- **Cross-lingual**: Multilingual tensor spaces

### Scalability Considerations
- **Distributed Processing**: Parallel parse processing
- **Hierarchical Storage**: Multi-level memory organization
- **Approximate Inference**: Probabilistic reasoning shortcuts
- **Dynamic Compression**: Adaptive tensor dimension reduction

This tensor architecture provides the mathematical foundation for sophisticated natural language understanding and reasoning in the OpenCog cognitive framework.