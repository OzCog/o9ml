;; Core Layer: Hypergraph Genesis Knowledge Base
;; Real cognitive/reasoning data for testing

;; Basic concept hierarchy
(ConceptNode "Entity" (stv 1.0 0.9))
(ConceptNode "PhysicalEntity" (stv 0.9 0.8))
(ConceptNode "AbstractEntity" (stv 0.8 0.8))

;; Living entities
(ConceptNode "LivingThing" (stv 0.9 0.9))
(ConceptNode "Animal" (stv 0.9 0.8))
(ConceptNode "Plant" (stv 0.8 0.8))
(ConceptNode "Mammal" (stv 0.8 0.9))
(ConceptNode "Bird" (stv 0.8 0.8))
(ConceptNode "Fish" (stv 0.8 0.8))

;; Specific animals
(ConceptNode "Cat" (stv 0.95 0.9))
(ConceptNode "Dog" (stv 0.95 0.9))
(ConceptNode "Eagle" (stv 0.9 0.8))
(ConceptNode "Salmon" (stv 0.85 0.8))

;; Properties and attributes
(ConceptNode "HasProperty" (stv 0.8 0.8))
(ConceptNode "Color" (stv 0.7 0.8))
(ConceptNode "Size" (stv 0.7 0.8))
(ConceptNode "Behavior" (stv 0.8 0.7))

;; Inheritance hierarchy - hypergraph structure
(InheritanceLink (stv 0.9 0.9)
    (ConceptNode "PhysicalEntity")
    (ConceptNode "Entity"))

(InheritanceLink (stv 0.9 0.9)
    (ConceptNode "AbstractEntity")
    (ConceptNode "Entity"))

(InheritanceLink (stv 0.9 0.8)
    (ConceptNode "LivingThing")
    (ConceptNode "PhysicalEntity"))

(InheritanceLink (stv 0.9 0.8)
    (ConceptNode "Animal")
    (ConceptNode "LivingThing"))

(InheritanceLink (stv 0.9 0.8)
    (ConceptNode "Plant")
    (ConceptNode "LivingThing"))

(InheritanceLink (stv 0.8 0.9)
    (ConceptNode "Mammal")
    (ConceptNode "Animal"))

(InheritanceLink (stv 0.8 0.8)
    (ConceptNode "Bird")
    (ConceptNode "Animal"))

(InheritanceLink (stv 0.8 0.8)
    (ConceptNode "Fish")
    (ConceptNode "Animal"))

(InheritanceLink (stv 0.9 0.8)
    (ConceptNode "Cat")
    (ConceptNode "Mammal"))

(InheritanceLink (stv 0.9 0.8)
    (ConceptNode "Dog")
    (ConceptNode "Mammal"))

(InheritanceLink (stv 0.9 0.8)
    (ConceptNode "Eagle")
    (ConceptNode "Bird"))

(InheritanceLink (stv 0.85 0.8)
    (ConceptNode "Salmon")
    (ConceptNode "Fish"))

;; Similarity relationships - dynamic field
(SimilarityLink (stv 0.7 0.6)
    (ConceptNode "Cat")
    (ConceptNode "Dog"))

(SimilarityLink (stv 0.5 0.5)
    (ConceptNode "Eagle")
    (ConceptNode "Salmon"))

;; Complex reasoning structures
(ImplicationLink (stv 0.9 0.8)
    (InheritanceLink (VariableNode "$X") (ConceptNode "Mammal"))
    (InheritanceLink (VariableNode "$X") (ConceptNode "Animal")))

(ImplicationLink (stv 0.9 0.8)
    (InheritanceLink (VariableNode "$X") (ConceptNode "Animal"))
    (InheritanceLink (VariableNode "$X") (ConceptNode "LivingThing")))

;; Evaluation structures for property attribution
(EvaluationLink (stv 0.8 0.7)
    (PredicateNode "HasColor")
    (ListLink
        (ConceptNode "Cat")
        (ConceptNode "Orange")))

(EvaluationLink (stv 0.8 0.7)
    (PredicateNode "HasBehavior")
    (ListLink
        (ConceptNode "Cat")
        (ConceptNode "Hunting")))
