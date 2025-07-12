/**
 * @file HypergraphKernel.h
 * @brief Custom operators for hypergraph computation with neural-symbolic integration
 * 
 * The HypergraphKernel implements custom ggml operators for efficient hypergraph
 * computation, supporting cognitive workloads with recursive processing and
 * attention-based traversal strategies.
 */

#pragma once

#include "../core/AgenticKernel.h"
#include "SymbolicTensorKernel.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <functional>

namespace orchestral {

/**
 * @brief Hypergraph node with neural-symbolic representation
 */
struct HypergraphNode {
    std::string node_id;
    std::string node_type;
    NeuralSymbolicTensor representation;
    std::unordered_map<std::string, float> properties;
    std::vector<std::string> connected_edges;
    float activation_level;
    int processing_depth;
    
    HypergraphNode(const std::string& id = "", const std::string& type = "")
        : node_id(id), node_type(type), activation_level(0.0f), processing_depth(0) {}
};

/**
 * @brief Hypergraph edge (hyperedge) connecting multiple nodes
 */
struct HypergraphEdge {
    std::string edge_id;
    std::string edge_type;
    std::vector<std::string> connected_nodes;  // Can connect multiple nodes
    NeuralSymbolicTensor edge_representation;
    float edge_weight;
    std::unordered_map<std::string, float> edge_properties;
    bool is_directed;
    
    HypergraphEdge(const std::string& id = "", const std::string& type = "")
        : edge_id(id), edge_type(type), edge_weight(1.0f), is_directed(false) {}
};

/**
 * @brief Complete hypergraph structure
 */
struct Hypergraph {
    std::unordered_map<std::string, HypergraphNode> nodes;
    std::unordered_map<std::string, HypergraphEdge> edges;
    std::unordered_map<std::string, float> global_properties;
    std::string graph_id;
    int recursion_level;
    
    Hypergraph(const std::string& id = "") : graph_id(id), recursion_level(0) {}
};

/**
 * @brief Hypergraph operation types for custom ggml operators
 */
enum class HypergraphOp {
    NODE_ACTIVATION,
    EDGE_PROPAGATION,
    ATTENTION_FLOW,
    RECURSIVE_TRAVERSAL,
    PATTERN_MATCHING,
    SUBGRAPH_EXTRACTION,
    GRAPH_FUSION,
    NEURAL_EMBEDDING,
    SYMBOLIC_REASONING,
    COGNITIVE_PROCESSING
};

/**
 * @brief Result of hypergraph computation
 */
struct HypergraphResult {
    Hypergraph result_graph;
    std::vector<std::string> activated_nodes;
    std::vector<std::string> traversal_path;
    std::unordered_map<std::string, float> attention_weights;
    float computation_confidence;
    std::chrono::milliseconds processing_time;
    float memory_usage_mb;
    bool recursion_limit_reached;
    std::string operation_trace;
    
    HypergraphResult() 
        : computation_confidence(0.0f),
          processing_time(0),
          memory_usage_mb(0.0f),
          recursion_limit_reached(false) {}
};

/**
 * @brief Traversal strategy for hypergraph processing
 */
enum class TraversalStrategy {
    BREADTH_FIRST,
    DEPTH_FIRST,
    ATTENTION_GUIDED,
    NEURAL_FLOW,
    SYMBOLIC_REASONING,
    COGNITIVE_PRIORITY
};

/**
 * @brief Custom hypergraph computation kernel
 */
class HypergraphKernel : public AgenticKernel {
public:
    /**
     * @brief Construct a new Hypergraph Kernel
     * @param name Unique name for this kernel instance
     * @param symbolic_kernel Shared pointer to symbolic tensor kernel
     * @param max_recursion Maximum recursion depth for hypergraph operations
     */
    explicit HypergraphKernel(const std::string& name = "hypergraph_kernel",
                             std::shared_ptr<SymbolicTensorKernel> symbolic_kernel = nullptr,
                             int max_recursion = 15);
    
    ~HypergraphKernel() override = default;
    
    // AgenticKernel interface
    bool initialize() override;
    void shutdown() override;
    CognitiveResult process(const CognitiveInput& input) override;
    void handleEvent(const KernelEvent& event) override;
    std::vector<std::string> getCapabilities() const override;
    bool canProcess(const std::string& inputType) const override;
    double getCurrentLoad() const override;
    
    /**
     * @brief Execute hypergraph operation with custom ggml operator
     * @param op Operation type to perform
     * @param graph Input hypergraph
     * @param parameters Operation-specific parameters
     * @return Result of hypergraph computation
     */
    HypergraphResult executeHypergraphOp(HypergraphOp op,
                                       const Hypergraph& graph,
                                       const std::unordered_map<std::string, float>& parameters = {});
    
    /**
     * @brief Create hypergraph from symbolic expressions
     * @param expressions Vector of symbolic expressions
     * @param connections Connection patterns between expressions
     * @return Newly created hypergraph
     */
    Hypergraph createFromSymbolic(const std::vector<std::string>& expressions,
                                const std::vector<std::pair<std::string, std::string>>& connections = {});
    
    /**
     * @brief Perform recursive hypergraph traversal
     * @param graph Input hypergraph
     * @param start_node Starting node ID
     * @param strategy Traversal strategy
     * @param max_depth Maximum traversal depth
     * @return Traversal result with path and activations
     */
    HypergraphResult recursiveTraversal(const Hypergraph& graph,
                                      const std::string& start_node,
                                      TraversalStrategy strategy = TraversalStrategy::ATTENTION_GUIDED,
                                      int max_depth = -1);
    
    /**
     * @brief Apply neural attention flow through hypergraph
     * @param graph Input hypergraph
     * @param attention_sources Nodes that generate attention
     * @param flow_strength Strength of attention flow
     * @return Graph with updated attention weights
     */
    HypergraphResult applyAttentionFlow(const Hypergraph& graph,
                                      const std::vector<std::string>& attention_sources,
                                      float flow_strength = 1.0f);
    
    /**
     * @brief Perform hypergraph pattern matching
     * @param query_graph Pattern to match
     * @param target_graph Graph to search in
     * @param similarity_threshold Minimum similarity for matches
     * @return Matched subgraphs with confidence scores
     */
    std::vector<std::pair<Hypergraph, float>> patternMatch(
        const Hypergraph& query_graph,
        const Hypergraph& target_graph,
        float similarity_threshold = 0.7f);
    
    /**
     * @brief Extract subgraph based on cognitive criteria
     * @param graph Input hypergraph
     * @param extraction_criteria Criteria for subgraph extraction
     * @return Extracted subgraph
     */
    Hypergraph extractSubgraph(const Hypergraph& graph,
                             const std::unordered_map<std::string, float>& extraction_criteria);
    
    /**
     * @brief Fuse multiple hypergraphs with neural-symbolic integration
     * @param graphs Vector of hypergraphs to fuse
     * @param fusion_weights Weights for each graph in fusion
     * @return Fused hypergraph
     */
    Hypergraph fuseHypergraphs(const std::vector<Hypergraph>& graphs,
                             const std::vector<float>& fusion_weights = {});
    
    /**
     * @brief Optimize hypergraph structure for cognitive workloads
     * @param graph Input hypergraph
     * @param optimization_targets Targets for optimization
     * @return Optimized hypergraph
     */
    Hypergraph optimizeForCognition(const Hypergraph& graph,
                                  const std::unordered_map<std::string, float>& optimization_targets);
    
    /**
     * @brief Set symbolic tensor kernel for neural-symbolic operations
     * @param symbolic_kernel Shared pointer to symbolic tensor kernel
     */
    void setSymbolicKernel(std::shared_ptr<SymbolicTensorKernel> symbolic_kernel);
    
    /**
     * @brief Configure hypergraph processing parameters
     * @param max_nodes Maximum nodes to process in one operation
     * @param attention_threshold Minimum attention level for activation
     * @param memory_limit Memory limit for hypergraph operations (MB)
     */
    void setProcessingParameters(int max_nodes = 1000,
                               float attention_threshold = 0.1f,
                               float memory_limit = 512.0f);

private:
    /**
     * @brief Initialize custom ggml operators for hypergraph computation
     */
    bool initializeHypergraphOperators();
    
    /**
     * @brief Compute neural representation for hypergraph node
     * @param node Node to compute representation for
     * @return Neural-symbolic tensor representation
     */
    NeuralSymbolicTensor computeNodeRepresentation(const HypergraphNode& node);
    
    /**
     * @brief Compute neural representation for hypergraph edge
     * @param edge Edge to compute representation for
     * @param connected_nodes Nodes connected by this edge
     * @return Neural-symbolic tensor representation
     */
    NeuralSymbolicTensor computeEdgeRepresentation(const HypergraphEdge& edge,
                                                 const std::vector<HypergraphNode>& connected_nodes);
    
    /**
     * @brief Activate nodes based on neural flow
     * @param graph Hypergraph to process
     * @param activation_sources Source nodes for activation
     * @param activation_strength Strength of activation signal
     * @return Graph with updated node activations
     */
    Hypergraph activateNodes(const Hypergraph& graph,
                           const std::vector<std::string>& activation_sources,
                           float activation_strength);
    
    /**
     * @brief Propagate signals through hypergraph edges
     * @param graph Hypergraph to process
     * @param propagation_decay Signal decay factor
     * @return Graph with propagated signals
     */
    Hypergraph propagateEdges(const Hypergraph& graph, float propagation_decay = 0.9f);
    
    /**
     * @brief Compute hypergraph similarity for pattern matching
     * @param graph1 First hypergraph
     * @param graph2 Second hypergraph
     * @return Similarity score [0.0, 1.0]
     */
    float computeGraphSimilarity(const Hypergraph& graph1, const Hypergraph& graph2);
    
    /**
     * @brief Validate hypergraph structure for cognitive processing
     * @param graph Hypergraph to validate
     * @return true if graph is valid for cognitive operations
     */
    bool validateCognitiveStructure(const Hypergraph& graph);
    
    /**
     * @brief Estimate memory usage for hypergraph operation
     * @param graph Hypergraph to analyze
     * @param operation Operation to be performed
     * @return Estimated memory usage in MB
     */
    float estimateMemoryUsage(const Hypergraph& graph, HypergraphOp operation);
    
    /**
     * @brief Apply cognitive constraints to hypergraph processing
     * @param graph Input hypergraph
     * @return Graph with applied cognitive constraints
     */
    Hypergraph applyCognitiveConstraints(const Hypergraph& graph);
    
    /**
     * @brief Generate processing trace for hypergraph operations
     * @param operation Operation performed
     * @param graph_size Number of nodes in graph
     * @param processing_time Time taken for processing
     * @return Human-readable processing trace
     */
    std::string generateProcessingTrace(HypergraphOp operation,
                                      size_t graph_size,
                                      std::chrono::milliseconds processing_time);
    
    // Configuration
    std::shared_ptr<SymbolicTensorKernel> symbolic_kernel_;
    int max_recursion_;
    int max_nodes_;
    float attention_threshold_;
    float memory_limit_mb_;
    
    // Processing state
    std::atomic<double> current_load_{0.0};
    std::atomic<size_t> active_operations_{0};
    
    // Hypergraph caching and optimization
    std::unordered_map<std::string, Hypergraph> graph_cache_;
    std::unordered_map<std::string, float> operation_costs_;
    
    // Cognitive processing parameters
    std::unordered_map<std::string, float> node_type_priorities_;
    std::unordered_map<std::string, float> edge_type_weights_;
    
    // Performance tracking
    std::atomic<uint64_t> total_operations_{0};
    std::atomic<uint64_t> successful_operations_{0};
    std::atomic<float> total_memory_used_mb_{0.0f};
    std::chrono::milliseconds total_processing_time_{0};
    
    // Custom ggml operators (placeholder for actual ggml integration)
    void* hypergraph_ggml_context_;
    std::unordered_map<std::string, void*> custom_operators_;
};

} // namespace orchestral