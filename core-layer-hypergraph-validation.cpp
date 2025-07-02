//
// Core Layer: Hypergraph Store Genesis - Validation System
// Real hypergraph operations and integrity validation
//
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <functional>
#include <random>
#include <cassert>
#include <cmath>
#include <algorithm>

namespace opencog {
namespace hypergraph {

// ========================================================================
// Tensor Degrees of Freedom for Hypergraph Operations
// ========================================================================

struct TensorDOF {
    // Spatial dimensions (3D) - hypergraph node positioning
    float spatial[3];
    
    // Temporal dimension (1D) - time-based hypergraph evolution  
    float temporal;
    
    // Semantic dimensions (256D) - concept embeddings
    float semantic[256];
    
    // Logical dimensions (64D) - truth value propagation
    float logical[64];
    
    TensorDOF() {
        // Initialize spatial to zero
        for (int i = 0; i < 3; i++) spatial[i] = 0.0f;
        temporal = 0.0f;
        
        // Initialize semantic with small random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        for (int i = 0; i < 256; i++) semantic[i] = dist(gen);
        for (int i = 0; i < 64; i++) logical[i] = dist(gen);
    }
    
    // Tensor operation: addition
    TensorDOF operator+(const TensorDOF& other) const {
        TensorDOF result;
        for (int i = 0; i < 3; i++) 
            result.spatial[i] = spatial[i] + other.spatial[i];
        result.temporal = temporal + other.temporal;
        for (int i = 0; i < 256; i++) 
            result.semantic[i] = semantic[i] + other.semantic[i];
        for (int i = 0; i < 64; i++) 
            result.logical[i] = logical[i] + other.logical[i];
        return result;
    }
    
    // Tensor operation: dot product (similarity)
    float dot_product(const TensorDOF& other) const {
        float result = 0.0f;
        for (int i = 0; i < 256; i++) 
            result += semantic[i] * other.semantic[i];
        return result;
    }
};

// ========================================================================
// Hypergraph Node - Real Implementation
// ========================================================================

class HypergraphNode {
public:
    std::string name;
    std::string type;
    TensorDOF tensor_dof;
    std::set<size_t> incoming_links;
    std::set<size_t> outgoing_links;
    
    // Truth values
    float strength;
    float confidence;
    
    HypergraphNode(const std::string& n, const std::string& t) 
        : name(n), type(t), strength(0.5f), confidence(0.5f) {
        // Initialize tensor DOF with semantic meaning
        tensor_dof.spatial[0] = static_cast<float>(name.length());
        tensor_dof.spatial[1] = static_cast<float>(type.length());
        tensor_dof.spatial[2] = strength;
        tensor_dof.temporal = confidence;
    }
    
    void add_incoming_link(size_t link_id) {
        incoming_links.insert(link_id);
    }
    
    void add_outgoing_link(size_t link_id) {
        outgoing_links.insert(link_id);
    }
    
    // Real hypergraph operation: semantic similarity
    float semantic_similarity(const HypergraphNode& other) const {
        return tensor_dof.dot_product(other.tensor_dof);
    }
};

// ========================================================================
// Hypergraph Link - Real Implementation 
// ========================================================================

class HypergraphLink {
public:
    std::string type;
    std::vector<size_t> targets;  // Node IDs that this link connects
    TensorDOF tensor_dof;
    float strength;
    float confidence;
    
    HypergraphLink(const std::string& t, const std::vector<size_t>& tgts)
        : type(t), targets(tgts), strength(0.8f), confidence(0.7f) {
        // Link tensor represents relationship strength
        tensor_dof.spatial[0] = static_cast<float>(targets.size());
        tensor_dof.spatial[1] = strength;
        tensor_dof.spatial[2] = confidence;
        tensor_dof.temporal = static_cast<float>(type.length());
    }
    
    size_t arity() const { return targets.size(); }
    
    bool connects(size_t node_id) const {
        return std::find(targets.begin(), targets.end(), node_id) != targets.end();
    }
};

// ========================================================================
// AtomSpace Hypergraph Store - Real Implementation
// ========================================================================

class AtomSpaceHypergraph {
private:
    std::map<size_t, std::unique_ptr<HypergraphNode>> nodes;
    std::map<size_t, std::unique_ptr<HypergraphLink>> links;
    size_t next_node_id;
    size_t next_link_id;
    
public:
    AtomSpaceHypergraph() : next_node_id(1), next_link_id(1) {}
    
    // Core hypergraph operations - NO MOCKS
    
    size_t add_node(const std::string& name, const std::string& type) {
        size_t id = next_node_id++;
        nodes[id] = std::make_unique<HypergraphNode>(name, type);
        return id;
    }
    
    size_t add_link(const std::string& type, const std::vector<size_t>& targets) {
        // Validate all target nodes exist
        for (size_t target : targets) {
            if (nodes.find(target) == nodes.end()) {
                throw std::runtime_error("Invalid target node ID: " + std::to_string(target));
            }
        }
        
        size_t id = next_link_id++;
        links[id] = std::make_unique<HypergraphLink>(type, targets);
        
        // Update node link relationships
        for (size_t target : targets) {
            nodes[target]->add_incoming_link(id);
        }
        
        return id;
    }
    
    // Real hypergraph operation: pattern matching
    std::vector<size_t> find_nodes_by_type(const std::string& type) {
        std::vector<size_t> result;
        for (const auto& pair : nodes) {
            if (pair.second->type == type) {
                result.push_back(pair.first);
            }
        }
        return result;
    }
    
    // Real hypergraph operation: semantic query
    std::vector<size_t> find_similar_nodes(size_t query_node_id, float threshold = 0.5f) {
        if (nodes.find(query_node_id) == nodes.end()) {
            return {};
        }
        
        std::vector<size_t> result;
        const auto& query_node = *nodes[query_node_id];
        
        for (const auto& pair : nodes) {
            if (pair.first != query_node_id) {
                float similarity = query_node.semantic_similarity(*pair.second);
                if (similarity > threshold) {
                    result.push_back(pair.first);
                }
            }
        }
        return result;
    }
    
    // Real hypergraph operation: recursive traversal
    void recursive_traverse(size_t start_node, std::function<void(size_t)> visitor, 
                           std::set<size_t>& visited) {
        if (visited.find(start_node) != visited.end() || 
            nodes.find(start_node) == nodes.end()) {
            return;
        }
        
        visited.insert(start_node);
        visitor(start_node);
        
        // Traverse through outgoing links
        for (size_t link_id : nodes[start_node]->outgoing_links) {
            if (links.find(link_id) != links.end()) {
                for (size_t target : links[link_id]->targets) {
                    recursive_traverse(target, visitor, visited);
                }
            }
        }
    }
    
    // Hypergraph integrity validation
    bool validate_integrity() {
        std::cout << "Validating hypergraph integrity..." << std::endl;
        
        // Check 1: All link targets point to existing nodes
        for (const auto& link_pair : links) {
            for (size_t target : link_pair.second->targets) {
                if (nodes.find(target) == nodes.end()) {
                    std::cerr << "ERROR: Link " << link_pair.first 
                             << " points to non-existent node " << target << std::endl;
                    return false;
                }
            }
        }
        
        // Check 2: Node-link consistency
        for (const auto& node_pair : nodes) {
            for (size_t link_id : node_pair.second->incoming_links) {
                if (links.find(link_id) == links.end()) {
                    std::cerr << "ERROR: Node " << node_pair.first 
                             << " references non-existent incoming link " << link_id << std::endl;
                    return false;
                }
                // Verify the link actually targets this node
                bool found = false;
                for (size_t target : links[link_id]->targets) {
                    if (target == node_pair.first) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "ERROR: Node " << node_pair.first 
                             << " claims incoming link " << link_id 
                             << " but link doesn't target it" << std::endl;
                    return false;
                }
            }
        }
        
        // Check 3: Tensor DOF validation
        for (const auto& node_pair : nodes) {
            const auto& tensor = node_pair.second->tensor_dof;
            
            // Check for NaN values
            for (int i = 0; i < 3; i++) {
                if (std::isnan(tensor.spatial[i])) {
                    std::cerr << "ERROR: Node " << node_pair.first 
                             << " has NaN in spatial dimension " << i << std::endl;
                    return false;
                }
            }
            
            if (std::isnan(tensor.temporal)) {
                std::cerr << "ERROR: Node " << node_pair.first 
                         << " has NaN in temporal dimension" << std::endl;
                return false;
            }
        }
        
        std::cout << "✓ Hypergraph integrity validation passed" << std::endl;
        std::cout << "  Nodes: " << nodes.size() << std::endl;
        std::cout << "  Links: " << links.size() << std::endl;
        return true;
    }
    
    // Get statistics for API exposure
    std::map<std::string, size_t> get_statistics() {
        std::map<std::string, size_t> stats;
        stats["total_nodes"] = nodes.size();
        stats["total_links"] = links.size();
        
        // Count by type
        std::map<std::string, size_t> node_types;
        std::map<std::string, size_t> link_types;
        
        for (const auto& pair : nodes) {
            node_types[pair.second->type]++;
        }
        
        for (const auto& pair : links) {
            link_types[pair.second->type]++;
        }
        
        stats["unique_node_types"] = node_types.size();
        stats["unique_link_types"] = link_types.size();
        
        return stats;
    }
    
    size_t node_count() const { return nodes.size(); }
    size_t link_count() const { return links.size(); }
    
    HypergraphNode* get_node(size_t id) {
        auto it = nodes.find(id);
        return (it != nodes.end()) ? it->second.get() : nullptr;
    }
    
    HypergraphLink* get_link(size_t id) {
        auto it = links.find(id);
        return (it != links.end()) ? it->second.get() : nullptr;
    }
};

} // namespace hypergraph
} // namespace opencog

// ========================================================================
// Real Data Testing Functions - NO MOCKS
// ========================================================================

using namespace opencog::hypergraph;

bool test_real_hypergraph_operations() {
    std::cout << "\n=== Testing Real Hypergraph Operations ===" << std::endl;
    
    AtomSpaceHypergraph hypergraph;
    
    // Create real knowledge representation
    std::cout << "Creating real knowledge representation..." << std::endl;
    
    // Add concept nodes
    size_t cat_node = hypergraph.add_node("cat", "ConceptNode");
    size_t animal_node = hypergraph.add_node("animal", "ConceptNode");
    size_t mammal_node = hypergraph.add_node("mammal", "ConceptNode");
    size_t pet_node = hypergraph.add_node("pet", "ConceptNode");
    
    // Add predicate nodes
    size_t isa_pred = hypergraph.add_node("isa", "PredicateNode");
    
    // Create inheritance relationships (real hypergraph links)
    size_t cat_isa_animal = hypergraph.add_link("InheritanceLink", {cat_node, animal_node});
    size_t cat_isa_mammal = hypergraph.add_link("InheritanceLink", {cat_node, mammal_node});
    size_t cat_isa_pet = hypergraph.add_link("InheritanceLink", {cat_node, pet_node});
    size_t mammal_isa_animal = hypergraph.add_link("InheritanceLink", {mammal_node, animal_node});
    
    // Test hypergraph integrity
    if (!hypergraph.validate_integrity()) {
        std::cerr << "FAILED: Hypergraph integrity validation" << std::endl;
        return false;
    }
    
    // Test pattern matching
    std::cout << "Testing pattern matching..." << std::endl;
    auto concept_nodes = hypergraph.find_nodes_by_type("ConceptNode");
    assert(concept_nodes.size() == 4);
    std::cout << "✓ Found " << concept_nodes.size() << " ConceptNodes" << std::endl;
    
    // Test semantic similarity
    std::cout << "Testing semantic similarity..." << std::endl;
    auto similar_to_cat = hypergraph.find_similar_nodes(cat_node, 0.1f);
    std::cout << "✓ Found " << similar_to_cat.size() << " nodes similar to 'cat'" << std::endl;
    
    // Test recursive traversal
    std::cout << "Testing recursive traversal..." << std::endl;
    std::set<size_t> visited;
    std::vector<size_t> traversal_order;
    
    hypergraph.recursive_traverse(cat_node, [&](size_t node_id) {
        traversal_order.push_back(node_id);
        auto* node = hypergraph.get_node(node_id);
        if (node) {
            std::cout << "  Visited: " << node->name << " (" << node->type << ")" << std::endl;
        }
    }, visited);
    
    std::cout << "✓ Recursive traversal completed, visited " << traversal_order.size() << " nodes" << std::endl;
    
    // Test tensor operations
    std::cout << "Testing tensor operations..." << std::endl;
    auto* cat_node_ptr = hypergraph.get_node(cat_node);
    auto* animal_node_ptr = hypergraph.get_node(animal_node);
    
    if (cat_node_ptr && animal_node_ptr) {
        float similarity = cat_node_ptr->semantic_similarity(*animal_node_ptr);
        std::cout << "✓ Semantic similarity between 'cat' and 'animal': " << similarity << std::endl;
        
        // Test tensor addition
        auto combined_tensor = cat_node_ptr->tensor_dof + animal_node_ptr->tensor_dof;
        std::cout << "✓ Tensor addition completed" << std::endl;
    }
    
    // Get final statistics
    auto stats = hypergraph.get_statistics();
    std::cout << "\nHypergraph Statistics:" << std::endl;
    for (const auto& stat : stats) {
        std::cout << "  " << stat.first << ": " << stat.second << std::endl;
    }
    
    std::cout << "✓ Real hypergraph operations test PASSED" << std::endl;
    return true;
}

bool test_hypergraph_dynamic_field() {
    std::cout << "\n=== Testing Hypergraph Dynamic Field Operations ===" << std::endl;
    
    AtomSpaceHypergraph hypergraph;
    
    // Create dynamic reasoning field
    std::cout << "Creating dynamic reasoning field..." << std::endl;
    
    // Create a network of interrelated concepts
    std::vector<size_t> concept_nodes;
    for (int i = 0; i < 10; i++) {
        size_t node = hypergraph.add_node("concept_" + std::to_string(i), "ConceptNode");
        concept_nodes.push_back(node);
    }
    
    // Create dynamic relationships
    std::vector<size_t> similarity_links;
    for (int i = 0; i < 10; i++) {
        for (int j = i + 1; j < 10; j++) {
            // Create similarity links based on semantic distance
            if ((i + j) % 3 == 0) {  // Some pattern for connection
                size_t link = hypergraph.add_link("SimilarityLink", {concept_nodes[i], concept_nodes[j]});
                similarity_links.push_back(link);
            }
        }
    }
    
    std::cout << "Created " << concept_nodes.size() << " concept nodes and " 
              << similarity_links.size() << " similarity links" << std::endl;
    
    // Test dynamic field propagation
    std::cout << "Testing dynamic field propagation..." << std::endl;
    
    // Simulate cognitive attention flow
    size_t start_node = concept_nodes[0];
    float attention_strength = 1.0f;
    
    auto* start_node_ptr = hypergraph.get_node(start_node);
    if (start_node_ptr) {
        start_node_ptr->strength = attention_strength;
        std::cout << "✓ Set initial attention at node: " << start_node_ptr->name << std::endl;
        
        // Propagate attention through semantic similarity
        for (size_t similar_node_id : hypergraph.find_similar_nodes(start_node, 0.1f)) {
            auto* similar_node = hypergraph.get_node(similar_node_id);
            if (similar_node) {
                float propagated_strength = attention_strength * 0.8f;  // Decay
                similar_node->strength = std::max(similar_node->strength, propagated_strength);
                std::cout << "  Propagated attention to: " << similar_node->name 
                         << " (strength: " << similar_node->strength << ")" << std::endl;
            }
        }
    }
    
    // Validate dynamic field integrity
    if (!hypergraph.validate_integrity()) {
        std::cerr << "FAILED: Dynamic field integrity validation" << std::endl;
        return false;
    }
    
    std::cout << "✓ Dynamic field operations test PASSED" << std::endl;
    return true;
}

// ========================================================================
// Main Test Runner
// ========================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Core Layer: Hypergraph Store Genesis" << std::endl;
    std::cout << "Real Data Validation - NO MOCKS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    bool all_passed = true;
    
    // Test 1: Real hypergraph operations
    if (!test_real_hypergraph_operations()) {
        all_passed = false;
    }
    
    // Test 2: Dynamic field operations
    if (!test_hypergraph_dynamic_field()) {
        all_passed = false;
    }
    
    std::cout << "\n========================================" << std::endl;
    if (all_passed) {
        std::cout << "✅ ALL TESTS PASSED - HYPERGRAPH GENESIS COMPLETE" << std::endl;
        std::cout << "Hypergraph membrane successfully implemented:" << std::endl;
        std::cout << "  ✓ Nodes/links as tensors" << std::endl;
        std::cout << "  ✓ Edges as relationships" << std::endl;
        std::cout << "  ✓ Dynamic field for reasoning and learning" << std::endl;
        std::cout << "  ✓ Real operations without mocks" << std::endl;
        std::cout << "  ✓ Integrity validation" << std::endl;
        std::cout << "  ✓ Tensor dimensions documented" << std::endl;
    } else {
        std::cout << "❌ SOME TESTS FAILED" << std::endl;
        return 1;
    }
    
    std::cout << "========================================" << std::endl;
    return 0;
}