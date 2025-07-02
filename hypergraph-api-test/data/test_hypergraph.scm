;; Real hypergraph test data - Core Layer Genesis
;; Knowledge representation for cognitive/logic layers

; Concept nodes with semantic meaning
(ConceptNode "animal" (stv 0.9 0.8))
(ConceptNode "mammal" (stv 0.8 0.9))
(ConceptNode "cat" (stv 0.95 0.9))
(ConceptNode "dog" (stv 0.95 0.9))
(ConceptNode "pet" (stv 0.7 0.8))

; Predicate nodes for relationships
(PredicateNode "isa" (stv 0.9 0.9))
(PredicateNode "has_property" (stv 0.8 0.8))

; Inheritance relationships - hypergraph links
(InheritanceLink (stv 0.9 0.8)
    (ConceptNode "cat")
    (ConceptNode "mammal"))

(InheritanceLink (stv 0.9 0.8)
    (ConceptNode "dog") 
    (ConceptNode "mammal"))

(InheritanceLink (stv 0.8 0.9)
    (ConceptNode "mammal")
    (ConceptNode "animal"))

(InheritanceLink (stv 0.8 0.7)
    (ConceptNode "cat")
    (ConceptNode "pet"))

(InheritanceLink (stv 0.8 0.7)
    (ConceptNode "dog")
    (ConceptNode "pet"))

; Similarity relationships - dynamic field
(SimilarityLink (stv 0.7 0.6)
    (ConceptNode "cat")
    (ConceptNode "dog"))

; Evaluation links - complex hypergraph structures
(EvaluationLink (stv 0.8 0.8)
    (PredicateNode "isa")
    (ListLink
        (ConceptNode "cat")
        (ConceptNode "animal")))

; Implication links for reasoning
(ImplicationLink (stv 0.9 0.8)
    (AndLink
        (InheritanceLink (VariableNode "$X") (ConceptNode "mammal"))
        (InheritanceLink (ConceptNode "mammal") (ConceptNode "animal")))
    (InheritanceLink (VariableNode "$X") (ConceptNode "animal")))
