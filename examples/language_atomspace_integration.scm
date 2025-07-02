;; Language Layer: AtomSpace Integration Example
;; Demonstrates natural language cognition using lg-atomese

;; Load required modules
(use-modules (opencog))
(use-modules (opencog nlp))
(use-modules (opencog nlp lg-parse))
(use-modules (opencog nlp lg-dict))

;; Example 1: Basic sentence parsing to AtomSpace
(define example-sentence "The cat sits on the mat.")

;; Example structure after parsing (conceptual representation)
;; This shows how link grammar structures map to AtomSpace

;; Basic atoms for sentence elements
(ConceptNode "cat" (stv 0.9 0.8))
(ConceptNode "mat" (stv 0.9 0.8)) 
(ConceptNode "sitting" (stv 0.8 0.7))

;; Relational structure from link grammar parsing
(EvaluationLink
    (PredicateNode "sits-on")
    (ListLink
        (ConceptNode "cat")
        (ConceptNode "mat")))

;; Link Grammar structure preservation  
(EvaluationLink
    (PredicateNode "lg-subject")
    (ListLink
        (ConceptNode "sitting")
        (ConceptNode "cat")))

(EvaluationLink
    (PredicateNode "lg-object") 
    (ListLink
        (ConceptNode "sitting")
        (ConceptNode "mat")))

;; Example 2: More complex sentence with modifiers
(define complex-sentence "The quick brown fox jumps over the lazy dog.")

;; Modifier relationships
(EvaluationLink
    (PredicateNode "modifier")
    (ListLink
        (ConceptNode "fox")
        (ConceptNode "quick")))

(EvaluationLink
    (PredicateNode "modifier")
    (ListLink
        (ConceptNode "fox") 
        (ConceptNode "brown")))

;; Main action relationship
(EvaluationLink
    (PredicateNode "jumps-over")
    (ListLink
        (ConceptNode "fox")
        (ConceptNode "dog")))

;; Example 3: Pattern matching for semantic understanding
;; Pattern to match subject-verb-object structures
(define svo-pattern
    (BindLink
        (VariableList
            (VariableNode "$subject")
            (VariableNode "$action") 
            (VariableNode "$object"))
        (AndLink
            (EvaluationLink
                (PredicateNode "lg-subject")
                (ListLink
                    (VariableNode "$action")
                    (VariableNode "$subject")))
            (EvaluationLink
                (PredicateNode "lg-object")
                (ListLink
                    (VariableNode "$action")
                    (VariableNode "$object"))))
        (EvaluationLink
            (PredicateNode "semantic-svo")
            (ListLink
                (VariableNode "$subject")
                (VariableNode "$action")
                (VariableNode "$object")))))

;; Example tensor structure for language cognition
;; Language tensors follow the foundation layer specification:
;; - Spatial (3D): Syntactic position in parse tree
;; - Temporal (1D): Sequence position in sentence  
;; - Semantic (256D): Word/concept embeddings
;; - Logical (64D): Grammatical relationship strength

;; Conceptual tensor for "cat" concept
(define cat-tensor
    (TensorNode "cat-tensor"
        (stv 0.9 0.8)
        ;; Spatial: [x, y, z] position in parse tree
        (FloatValue 2.0 1.0 0.0)  ; leaf node, level 1, position 2
        ;; Temporal: [t] sequence position  
        (FloatValue 1.0)          ; second word in sentence
        ;; Semantic: 256D embedding (simplified to key dimensions)
        (FloatValue 0.8 0.9 0.7 0.6)  ; animal, mammal, pet, domestic...
        ;; Logical: 64D grammatical features (simplified)
        (FloatValue 0.9 0.0 0.8)))    ; noun, not-verb, definite...

;; Print integration status
(display "Language Layer AtomSpace Integration Loaded\n")
(display "- Link Grammar parsing structures defined\n")
(display "- Semantic pattern matching ready\n") 
(display "- Language tensor framework established\n")