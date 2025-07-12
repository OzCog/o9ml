/**
 * @file NeuralInferenceKernel.h
 * @brief Neural inference hooks for AtomSpace integration
 * 
 * The NeuralInferenceKernel provides neural inference capabilities that
 * seamlessly integrate with AtomSpace hypergraph structures, enabling
 * cognitive reasoning through neural-symbolic pathways.
 */

#pragma once

#include "../core/AgenticKernel.h"
#include "SymbolicTensorKernel.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

namespace orchestral {

/**
 * @brief AtomSpace atom representation for neural inference
 */
struct AtomSpaceAtom {
    std::string atom_type;           // "ConceptNode", "LinkNode", etc.
    std::string atom_name;           // Unique identifier
    std::vector<std::string> outgoing_set;  // Connected atoms
    std::unordered_map<std::string, float> truth_values; // Strength, confidence, etc.
    NeuralSymbolicTensor neural_representation;  // Neural embedding
    
    AtomSpaceAtom(const std::string& type = "", const std::string& name = "") 
        : atom_type(type), atom_name(name) {}
};

/**
 * @brief Hypergraph pattern for neural-symbolic reasoning
 */
struct HypergraphPattern {
    std::vector<AtomSpaceAtom> nodes;
    std::vector<std::pair<size_t, size_t>> edges;  // Node index pairs
    std::unordered_map<std::string, float> pattern_weights;
    float pattern_confidence;
    int recursion_depth;
    
    HypergraphPattern() : pattern_confidence(0.0f), recursion_depth(0) {}
};

/**
 * @brief Neural inference result with cognitive reasoning trace
 */
struct NeuralInferenceResult {
    std::vector<AtomSpaceAtom> inferred_atoms;
    std::vector<HypergraphPattern> reasoning_patterns;
    float inference_confidence;
    std::string reasoning_trace;
    std::chrono::milliseconds inference_time;
    float cognitive_load;
    bool recursive_depth_exceeded;
    
    NeuralInferenceResult() 
        : inference_confidence(0.0f),
          inference_time(0),
          cognitive_load(0.0f),
          recursive_depth_exceeded(false) {}
};

/**
 * @brief Inference strategy for neural-symbolic reasoning
 */
enum class InferenceStrategy {
    FORWARD_CHAINING,
    BACKWARD_CHAINING,
    RECURSIVE_PATTERN_MATCHING,
    NEURAL_ATTENTION_GUIDED,
    HYPERGRAPH_TRAVERSAL,
    PROBABILISTIC_REASONING
};

/**
 * @brief Neural inference kernel for AtomSpace integration
 */
class NeuralInferenceKernel : public AgenticKernel {
public:
    /**
     * @brief Construct a new Neural Inference Kernel
     * @param name Unique name for this kernel instance
     * @param symbolic_kernel Shared pointer to symbolic tensor kernel
     * @param max_recursion_depth Maximum depth for recursive inference
     */
    explicit NeuralInferenceKernel(const std::string& name = "neural_inference_kernel",
                                  std::shared_ptr<SymbolicTensorKernel> symbolic_kernel = nullptr,
                                  int max_recursion_depth = 20);
    
    ~NeuralInferenceKernel() override = default;
    
    // AgenticKernel interface
    bool initialize() override;
    void shutdown() override;
    CognitiveResult process(const CognitiveInput& input) override;
    void handleEvent(const KernelEvent& event) override;
    std::vector<std::string> getCapabilities() const override;
    bool canProcess(const std::string& inputType) const override;
    double getCurrentLoad() const override;
    
    /**
     * @brief Perform neural inference on AtomSpace pattern
     * @param pattern Input hypergraph pattern
     * @param strategy Inference strategy to use
     * @param max_depth Maximum recursion depth for this inference
     * @return Neural inference result with reasoning trace
     */
    NeuralInferenceResult performInference(const HypergraphPattern& pattern,
                                         InferenceStrategy strategy = InferenceStrategy::NEURAL_ATTENTION_GUIDED,
                                         int max_depth = -1);
    
    /**
     * @brief Create AtomSpace atom from symbolic expression
     * @param atom_type Type of atom (ConceptNode, LinkNode, etc.)
     * @param atom_name Name/identifier for the atom
     * @param truth_values Truth values for the atom
     * @return Newly created AtomSpace atom with neural representation
     */
    AtomSpaceAtom createAtom(const std::string& atom_type,
                           const std::string& atom_name,
                           const std::unordered_map<std::string, float>& truth_values = {});
    
    /**
     * @brief Create hypergraph pattern from atoms and connections
     * @param atoms Vector of atoms to include in pattern
     * @param connections Connections between atoms (index pairs)
     * @return Hypergraph pattern ready for inference
     */
    HypergraphPattern createPattern(const std::vector<AtomSpaceAtom>& atoms,
                                  const std::vector<std::pair<size_t, size_t>>& connections);
    
    /**
     * @brief Perform recursive pattern matching with neural guidance
     * @param query_pattern Pattern to match
     * @param knowledge_base Vector of patterns to search
     * @param depth Current recursion depth
     * @return Matched patterns with confidence scores
     */
    std::vector<std::pair<HypergraphPattern, float>> recursivePatternMatch(
        const HypergraphPattern& query_pattern,
        const std::vector<HypergraphPattern>& knowledge_base,
        int depth = 0);
    
    /**
     * @brief Apply neural attention to hypergraph traversal
     * @param pattern Input pattern
     * @param attention_weights Attention weights for different atom types
     * @return Attention-guided traversal path
     */
    std::vector<AtomSpaceAtom> neuralAttentionTraversal(
        const HypergraphPattern& pattern,
        const std::unordered_map<std::string, float>& attention_weights);
    
    /**
     * @brief Integrate with external AtomSpace instance
     * @param atomspace_interface Function to query external AtomSpace
     */
    void setAtomSpaceInterface(std::function<std::vector<AtomSpaceAtom>(const std::string&)> atomspace_interface);
    
    /**
     * @brief Set symbolic tensor kernel for neural-symbolic operations
     * @param symbolic_kernel Shared pointer to symbolic tensor kernel
     */
    void setSymbolicKernel(std::shared_ptr<SymbolicTensorKernel> symbolic_kernel);
    
    /**
     * @brief Configure inference parameters
     * @param confidence_threshold Minimum confidence for inference results
     * @param attention_decay Rate of attention decay during traversal
     * @param pattern_complexity_limit Maximum pattern complexity to consider
     */
    void setInferenceParameters(float confidence_threshold = 0.3f,
                              float attention_decay = 0.9f,
                              int pattern_complexity_limit = 100);

private:
    /**
     * @brief Forward chaining inference implementation
     * @param pattern Input pattern
     * @param depth Current recursion depth
     * @return Inference result
     */
    NeuralInferenceResult forwardChaining(const HypergraphPattern& pattern, int depth);
    
    /**
     * @brief Backward chaining inference implementation
     * @param pattern Input pattern
     * @param depth Current recursion depth
     * @return Inference result
     */
    NeuralInferenceResult backwardChaining(const HypergraphPattern& pattern, int depth);
    
    /**
     * @brief Neural attention guided inference
     * @param pattern Input pattern
     * @param depth Current recursion depth
     * @return Inference result
     */
    NeuralInferenceResult neuralAttentionInference(const HypergraphPattern& pattern, int depth);
    
    /**
     * @brief Compute neural representation for AtomSpace atom
     * @param atom Atom to compute representation for
     * @return Neural representation as tensor
     */
    NeuralSymbolicTensor computeAtomNeuralRepresentation(const AtomSpaceAtom& atom);
    
    /**
     * @brief Compute pattern similarity using neural embeddings
     * @param pattern1 First pattern
     * @param pattern2 Second pattern
     * @return Similarity score [0.0, 1.0]
     */
    float computePatternSimilarity(const HypergraphPattern& pattern1, const HypergraphPattern& pattern2);
    
    /**
     * @brief Apply cognitive reasoning constraints
     * @param result Inference result to validate
     * @return true if result passes cognitive constraints
     */
    bool applyCognitiveConstraints(const NeuralInferenceResult& result);
    
    /**
     * @brief Generate reasoning trace for inference process
     * @param patterns Patterns used in reasoning
     * @param strategy Strategy employed
     * @param depth Recursion depth reached
     * @return Human-readable reasoning trace
     */
    std::string generateReasoningTrace(const std::vector<HypergraphPattern>& patterns,
                                     InferenceStrategy strategy,
                                     int depth);
    
    /**
     * @brief Update cognitive load based on inference complexity
     * @param pattern_count Number of patterns processed
     * @param recursion_depth Depth of recursion
     * @param inference_time Time taken for inference
     */
    void updateCognitiveLoad(size_t pattern_count, int recursion_depth, 
                           std::chrono::milliseconds inference_time);
    
    // Configuration
    std::shared_ptr<SymbolicTensorKernel> symbolic_kernel_;
    int max_recursion_depth_;
    float confidence_threshold_;
    float attention_decay_;
    int pattern_complexity_limit_;
    
    // AtomSpace integration
    std::function<std::vector<AtomSpaceAtom>(const std::string&)> atomspace_interface_;
    std::unordered_map<std::string, AtomSpaceAtom> atom_cache_;
    
    // Processing state
    std::atomic<double> current_load_{0.0};
    std::atomic<size_t> active_inferences_{0};
    std::atomic<float> cognitive_load_{0.0f};
    
    // Knowledge base and patterns
    std::vector<HypergraphPattern> pattern_knowledge_base_;
    std::unordered_map<std::string, float> atom_type_weights_;
    
    // Performance tracking
    std::atomic<uint64_t> total_inferences_{0};
    std::atomic<uint64_t> successful_inferences_{0};
    std::chrono::milliseconds total_inference_time_{0};
    
    // Recursion tracking
    std::unordered_map<std::string, int> pattern_recursion_counts_;
    
    // Neural attention weights
    std::unordered_map<std::string, float> global_attention_weights_;
};

} // namespace orchestral