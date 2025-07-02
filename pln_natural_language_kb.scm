;; PLN Knowledge Base generated from Natural Language
;; Using RelEx parsing and AtomSpace representation

;; Load PLN modules
(use-modules (opencog))
(use-modules (opencog nlp))
(use-modules (opencog pln))

;; Initialize AtomSpace
(define as (cog-new-atomspace))
(cog-set-atomspace! as)

;; =================================================================
;; PARSED NATURAL LANGUAGE FACTS
;; =================================================================

;; Sentence 1: Alice ate the mushroom.
(SentenceNode "sentence-1" "Alice ate the mushroom.")

;; Eating action facts
(EvaluationLink (stv 0.95 0.9)
    (PredicateNode "ate")
    (ListLink
        (ConceptNode "Alice")
        (ConceptNode "mushroom")
    )
)

;; Implication: If X ate Y, then Y was consumed
(ImplicationLink (stv 0.9 0.8)
    (EvaluationLink
        (PredicateNode "ate")
        (ListLink
            (VariableNode "X")
            (VariableNode "Y")
        )
    )
    (EvaluationLink
        (PredicateNode "consumed")
        (ListLink
            (VariableNode "Y")
        )
    )
)

;; Sentence 2: John loves Mary.
(SentenceNode "sentence-2" "John loves Mary.")

;; Love relationship facts
(EvaluationLink (stv 0.9 0.8)
    (PredicateNode "loves")
    (ListLink
        (ConceptNode "John")
        (ConceptNode "Mary")
    )
)

;; Implication: If X loves Y, then X cares about Y
(ImplicationLink (stv 0.8 0.7)
    (EvaluationLink
        (PredicateNode "loves")
        (ListLink
            (VariableNode "X")
            (VariableNode "Y")
        )
    )
    (EvaluationLink
        (PredicateNode "cares_about")
        (ListLink
            (VariableNode "X")
            (VariableNode "Y")
        )
    )
)

;; Sentence 3: The cat sat on the mat.
(SentenceNode "sentence-3" "The cat sat on the mat.")

;; Spatial relationship facts
(EvaluationLink (stv 0.9 0.8)
    (PredicateNode "sat_on")
    (ListLink
        (ConceptNode "cat")
        (ConceptNode "mat")
    )
)

;; Implication: If X sat on Y, then X was above Y
(ImplicationLink (stv 0.8 0.7)
    (EvaluationLink
        (PredicateNode "sat_on")
        (ListLink
            (VariableNode "X")
            (VariableNode "Y")
        )
    )
    (EvaluationLink
        (PredicateNode "above")
        (ListLink
            (VariableNode "X")
            (VariableNode "Y")
        )
    )
)

;; =================================================================
;; PLN REASONING QUERIES
;; =================================================================

;; Query 1: What did Alice consume?
(define query-1
    (GetLink
        (VariableNode "what")
        (EvaluationLink
            (PredicateNode "consumed")
            (ListLink
                (VariableNode "what")
            )
        )
    )
)

;; Query 2: Who loves whom?
(define query-2
    (GetLink
        (ListLink
            (VariableNode "lover")
            (VariableNode "beloved")
        )
        (EvaluationLink
            (PredicateNode "loves")
            (ListLink
                (VariableNode "lover")
                (VariableNode "beloved")
            )
        )
    )
)

;; Query 3: What spatial relationships exist?
(define query-3
    (GetLink
        (ListLink
            (VariableNode "above-entity")
            (VariableNode "below-entity")
        )
        (EvaluationLink
            (PredicateNode "above")
            (ListLink
                (VariableNode "above-entity")
                (VariableNode "below-entity")
            )
        )
    )
)

;; Example PLN inference execution
;; (cog-execute! query-1)
;; (cog-execute! query-2)
;; (cog-execute! query-3)

;; Enable PLN reasoning
;; (pln-load)
;; (pln-add-rule 'modus-ponens)
;; (pln-add-rule 'deduction)

(display "PLN Knowledge Base loaded successfully!")
(newline)