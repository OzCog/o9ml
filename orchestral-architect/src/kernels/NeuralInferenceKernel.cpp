/**
 * @file NeuralInferenceKernel.cpp
 * @brief Implementation of neural inference hooks for AtomSpace integration
 */

#include "NeuralInferenceKernel.h"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <numeric>

namespace orchestral {

NeuralInferenceKernel::NeuralInferenceKernel(const std::string& name,
                                           std::shared_ptr<SymbolicTensorKernel> symbolic_kernel,
                                           int max_recursion_depth)
    : AgenticKernel(name, "neural_inference"),
      symbolic_kernel_(symbolic_kernel),
      max_recursion_depth_(max_recursion_depth),
      confidence_threshold_(0.3f),
      attention_decay_(0.9f),
      pattern_complexity_limit_(100) {
    
    // Initialize atom type weights for cognitive reasoning
    atom_type_weights_["ConceptNode"] = 1.0f;
    atom_type_weights_["PredicateNode"] = 1.2f;
    atom_type_weights_["LinkNode"] = 0.8f;
    atom_type_weights_["InheritanceLink"] = 1.1f;
    atom_type_weights_["SimilarityLink"] = 0.9f;
    atom_type_weights_["EvaluationLink"] = 1.3f;
    atom_type_weights_["ImplicationLink"] = 1.4f;
    
    // Initialize global attention weights
    global_attention_weights_["high_confidence"] = 1.5f;
    global_attention_weights_["recursive_pattern"] = 1.3f;
    global_attention_weights_["novel_inference"] = 1.2f;
    global_attention_weights_["cached_result"] = 0.7f;
}

bool NeuralInferenceKernel::initialize() {
    // Initialize basic reasoning patterns in knowledge base
    std::vector<AtomSpaceAtom> basic_atoms = {
        createAtom("ConceptNode", "Animal", {{"strength", 0.9f}, {"confidence", 0.8f}}),
        createAtom("ConceptNode", "Dog", {{"strength", 0.95f}, {"confidence", 0.9f}}),
        createAtom("ConceptNode", "Mammal", {{"strength", 0.85f}, {"confidence", 0.8f}})
    };
    
    std::vector<std::pair<size_t, size_t>> basic_connections = {{0, 1}, {1, 2}};
    auto basic_pattern = createPattern(basic_atoms, basic_connections);
    pattern_knowledge_base_.push_back(basic_pattern);
    
    setActive(true);
    return true;
}

void NeuralInferenceKernel::shutdown() {
    pattern_knowledge_base_.clear();
    atom_cache_.clear();
    pattern_recursion_counts_.clear();
    setActive(false);
}

CognitiveResult NeuralInferenceKernel::process(const CognitiveInput& input) {
    auto start = std::chrono::high_resolution_clock::now();
    
    active_inferences_.fetch_add(1);
    current_load_ = std::min(1.0, static_cast<double>(active_inferences_) / 3.0);
    
    CognitiveResult result;
    
    try {
        if (input.type == "atomspace_query") {
            // Parse atomspace query and perform inference
            auto query_atoms = parseAtomSpaceQuery(input.data);
            auto query_pattern = createPattern(query_atoms, {});
            
            InferenceStrategy strategy = InferenceStrategy::NEURAL_ATTENTION_GUIDED;
            
            // Check for strategy hint in context
            auto strategy_it = input.contextWeights.find("inference_strategy");
            if (strategy_it != input.contextWeights.end()) {
                if (strategy_it->second > 0.8f) {
                    strategy = InferenceStrategy::RECURSIVE_PATTERN_MATCHING;
                } else if (strategy_it->second > 0.6f) {
                    strategy = InferenceStrategy::FORWARD_CHAINING;
                } else if (strategy_it->second > 0.4f) {
                    strategy = InferenceStrategy::BACKWARD_CHAINING;
                }
            }
            
            auto inference_result = performInference(query_pattern, strategy);
            
            // Convert to cognitive result
            std::ostringstream oss;
            oss << "Neural Inference Result: ";
            oss << inference_result.inferred_atoms.size() << " atoms inferred, ";
            oss << "confidence: " << inference_result.inference_confidence << ", ";
            oss << "patterns: " << inference_result.reasoning_patterns.size();
            
            result.processedData = oss.str();
            result.success = true;
            result.estimatedValue = inference_result.inference_confidence;
            
            // Extract attention weights from reasoning patterns
            for (const auto& pattern : inference_result.reasoning_patterns) {
                for (const auto& weight : pattern.pattern_weights) {
                    result.attentionWeights[weight.first] = weight.second;
                }
            }
            
        } else if (input.type == "hypergraph_pattern") {
            // Process hypergraph pattern directly
            auto pattern = parseHypergraphPattern(input.data);
            auto inference_result = performInference(pattern);
            
            result.processedData = "Hypergraph inference completed: " + 
                                 std::to_string(inference_result.reasoning_patterns.size()) + " patterns matched";
            result.success = true;
            result.estimatedValue = inference_result.inference_confidence;
            
        } else if (input.type == "cognitive_reasoning") {
            // Perform general cognitive reasoning
            std::vector<AtomSpaceAtom> reasoning_atoms = {
                createAtom("ConceptNode", "problem", {{"urgency", input.urgency}})
            };
            auto reasoning_pattern = createPattern(reasoning_atoms, {});
            
            auto inference_result = performInference(reasoning_pattern, 
                                                   InferenceStrategy::RECURSIVE_PATTERN_MATCHING);
            
            result.processedData = "Cognitive reasoning trace: " + inference_result.reasoning_trace;
            result.success = true;
            result.estimatedValue = inference_result.inference_confidence;
            
        } else {
            result.success = false;
            result.errorMessage = "NeuralInferenceKernel can only process atomspace_query, hypergraph_pattern, or cognitive_reasoning input";
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        result.processingTime = duration;
        
        total_inferences_.fetch_add(1);
        if (result.success) {
            successful_inferences_.fetch_add(1);
        }
        total_inference_time_ += duration;
        
        updateMetrics(result, duration);
        
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Neural inference error: ") + e.what();
    }
    
    active_inferences_.fetch_sub(1);
    current_load_ = std::min(1.0, static_cast<double>(active_inferences_) / 3.0);
    
    return result;
}

void NeuralInferenceKernel::handleEvent(const KernelEvent& event) {
    if (event.eventType == "atomspace_update") {
        // Clear atom cache when AtomSpace is updated
        atom_cache_.clear();
        
    } else if (event.eventType == "inference_request") {
        auto strategy_it = event.payload.find("strategy");
        auto pattern_it = event.payload.find("pattern");
        
        if (strategy_it != event.payload.end() && pattern_it != event.payload.end()) {
            // Process inference request from another kernel
            InferenceStrategy strategy = InferenceStrategy::NEURAL_ATTENTION_GUIDED;
            
            if (strategy_it->second == "forward_chaining") {
                strategy = InferenceStrategy::FORWARD_CHAINING;
            } else if (strategy_it->second == "backward_chaining") {
                strategy = InferenceStrategy::BACKWARD_CHAINING;
            } else if (strategy_it->second == "recursive_pattern") {
                strategy = InferenceStrategy::RECURSIVE_PATTERN_MATCHING;
            }
            
            // Parse pattern and perform inference
            auto pattern = parseHypergraphPattern(pattern_it->second);
            auto result = performInference(pattern, strategy);
            
            // Send response event
            KernelEvent response;
            response.eventType = "inference_response";
            response.sourceKernel = getName();
            response.targetKernel = event.sourceKernel;
            response.payload["confidence"] = std::to_string(result.inference_confidence);
            response.payload["atoms_count"] = std::to_string(result.inferred_atoms.size());
            emitEvent(response);
        }
        
    } else if (event.eventType == "attention_update") {
        // Update global attention weights
        for (const auto& payload_item : event.payload) {
            if (payload_item.first.find("attention_") == 0) {
                std::string attention_key = payload_item.first.substr(10); // Remove "attention_" prefix
                global_attention_weights_[attention_key] = std::stof(payload_item.second);
            }
        }
    }
}

std::vector<std::string> NeuralInferenceKernel::getCapabilities() const {
    return {
        "atomspace_integration",
        "neural_symbolic_inference",
        "hypergraph_reasoning",
        "recursive_pattern_matching",
        "cognitive_reasoning",
        "attention_guided_inference",
        "forward_backward_chaining",
        "probabilistic_reasoning"
    };
}

bool NeuralInferenceKernel::canProcess(const std::string& inputType) const {
    return inputType == "atomspace_query" || 
           inputType == "hypergraph_pattern" ||
           inputType == "cognitive_reasoning" ||
           inputType == "inference_request";
}

double NeuralInferenceKernel::getCurrentLoad() const {
    return current_load_.load();
}

NeuralInferenceResult NeuralInferenceKernel::performInference(const HypergraphPattern& pattern,
                                                            InferenceStrategy strategy,
                                                            int max_depth) {
    if (max_depth == -1) {
        max_depth = max_recursion_depth_;
    }
    
    NeuralInferenceResult result;
    
    // Check pattern complexity limit
    if (pattern.nodes.size() > static_cast<size_t>(pattern_complexity_limit_)) {
        result.inference_confidence = 0.0f;
        result.reasoning_trace = "Pattern too complex for inference";
        return result;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        switch (strategy) {
            case InferenceStrategy::FORWARD_CHAINING:
                result = forwardChaining(pattern, 0);
                break;
                
            case InferenceStrategy::BACKWARD_CHAINING:
                result = backwardChaining(pattern, 0);
                break;
                
            case InferenceStrategy::RECURSIVE_PATTERN_MATCHING: {
                auto matches = recursivePatternMatch(pattern, pattern_knowledge_base_, 0);
                for (const auto& match : matches) {
                    result.reasoning_patterns.push_back(match.first);
                    for (const auto& atom : match.first.nodes) {
                        result.inferred_atoms.push_back(atom);
                    }
                }
                if (!matches.empty()) {
                    result.inference_confidence = matches[0].second;
                }
                break;
            }
            
            case InferenceStrategy::NEURAL_ATTENTION_GUIDED:
                result = neuralAttentionInference(pattern, 0);
                break;
                
            case InferenceStrategy::HYPERGRAPH_TRAVERSAL: {
                auto traversal_path = neuralAttentionTraversal(pattern, global_attention_weights_);
                result.inferred_atoms = traversal_path;
                result.inference_confidence = traversal_path.empty() ? 0.0f : 0.7f;
                break;
            }
            
            case InferenceStrategy::PROBABILISTIC_REASONING:
                // Implement probabilistic reasoning using truth values
                result = neuralAttentionInference(pattern, 0);
                // Adjust confidence based on probabilistic calculations
                result.inference_confidence *= 0.8f;
                break;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Apply cognitive constraints
        if (!applyCognitiveConstraints(result)) {
            result.inference_confidence *= 0.5f;
        }
        
        // Generate reasoning trace
        result.reasoning_trace = generateReasoningTrace(result.reasoning_patterns, strategy, max_depth);
        
        // Update cognitive load
        updateCognitiveLoad(result.reasoning_patterns.size(), max_depth, result.inference_time);
        
    } catch (const std::exception& e) {
        result.inference_confidence = 0.0f;
        result.reasoning_trace = "Inference failed: " + std::string(e.what());
    }
    
    return result;
}

AtomSpaceAtom NeuralInferenceKernel::createAtom(const std::string& atom_type,
                                              const std::string& atom_name,
                                              const std::unordered_map<std::string, float>& truth_values) {
    AtomSpaceAtom atom(atom_type, atom_name);
    atom.truth_values = truth_values;
    
    // Compute neural representation
    if (symbolic_kernel_) {
        std::string symbolic_expr = "(" + atom_type + " " + atom_name + ")";
        atom.neural_representation = symbolic_kernel_->createFromSymbolic(symbolic_expr, truth_values);
    } else {
        // Fallback: create basic neural representation
        atom.neural_representation = computeAtomNeuralRepresentation(atom);
    }
    
    // Cache the atom
    std::string cache_key = atom_type + ":" + atom_name;
    atom_cache_[cache_key] = atom;
    
    return atom;
}

HypergraphPattern NeuralInferenceKernel::createPattern(const std::vector<AtomSpaceAtom>& atoms,
                                                     const std::vector<std::pair<size_t, size_t>>& connections) {
    HypergraphPattern pattern;
    pattern.nodes = atoms;
    pattern.edges = connections;
    pattern.recursion_depth = 0;
    
    // Compute pattern weights based on atom types and truth values
    for (const auto& atom : atoms) {
        auto type_weight_it = atom_type_weights_.find(atom.atom_type);
        float type_weight = type_weight_it != atom_type_weights_.end() ? type_weight_it->second : 1.0f;
        
        float truth_weight = 1.0f;
        auto strength_it = atom.truth_values.find("strength");
        if (strength_it != atom.truth_values.end()) {
            truth_weight = strength_it->second;
        }
        
        pattern.pattern_weights[atom.atom_name] = type_weight * truth_weight;
    }
    
    // Compute overall pattern confidence
    if (!pattern.pattern_weights.empty()) {
        float total_weight = 0.0f;
        for (const auto& weight : pattern.pattern_weights) {
            total_weight += weight.second;
        }
        pattern.pattern_confidence = total_weight / pattern.pattern_weights.size();
    } else {
        pattern.pattern_confidence = 0.5f;
    }
    
    return pattern;
}

std::vector<std::pair<HypergraphPattern, float>> NeuralInferenceKernel::recursivePatternMatch(
    const HypergraphPattern& query_pattern,
    const std::vector<HypergraphPattern>& knowledge_base,
    int depth) {
    
    std::vector<std::pair<HypergraphPattern, float>> matches;
    
    if (depth >= max_recursion_depth_) {
        return matches;
    }
    
    for (const auto& kb_pattern : knowledge_base) {
        float similarity = computePatternSimilarity(query_pattern, kb_pattern);
        
        if (similarity > confidence_threshold_) {
            matches.emplace_back(kb_pattern, similarity);
            
            // Recursive exploration of similar patterns
            if (depth < max_recursion_depth_ - 1) {
                auto recursive_matches = recursivePatternMatch(kb_pattern, knowledge_base, depth + 1);
                for (auto& recursive_match : recursive_matches) {
                    recursive_match.second *= 0.9f; // Decay confidence for recursive matches
                    matches.push_back(recursive_match);
                }
            }
        }
    }
    
    // Sort by confidence
    std::sort(matches.begin(), matches.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return matches;
}

std::vector<AtomSpaceAtom> NeuralInferenceKernel::neuralAttentionTraversal(
    const HypergraphPattern& pattern,
    const std::unordered_map<std::string, float>& attention_weights) {
    
    std::vector<AtomSpaceAtom> traversal_path;
    
    if (pattern.nodes.empty()) {
        return traversal_path;
    }
    
    // Start with highest attention atom
    size_t current_atom_idx = 0;
    float max_attention = 0.0f;
    
    for (size_t i = 0; i < pattern.nodes.size(); ++i) {
        const auto& atom = pattern.nodes[i];
        float attention = 0.0f;
        
        // Compute attention based on atom type and truth values
        auto type_weight_it = atom_type_weights_.find(atom.atom_type);
        if (type_weight_it != atom_type_weights_.end()) {
            attention += type_weight_it->second;
        }
        
        auto strength_it = atom.truth_values.find("strength");
        if (strength_it != atom.truth_values.end()) {
            attention *= strength_it->second;
        }
        
        if (attention > max_attention) {
            max_attention = attention;
            current_atom_idx = i;
        }
    }
    
    // Traverse the pattern following attention-weighted edges
    std::vector<bool> visited(pattern.nodes.size(), false);
    traversal_path.push_back(pattern.nodes[current_atom_idx]);
    visited[current_atom_idx] = true;
    
    // Continue traversal following edges
    while (traversal_path.size() < pattern.nodes.size()) {
        size_t next_atom_idx = current_atom_idx;
        float max_edge_attention = 0.0f;
        
        for (const auto& edge : pattern.edges) {
            size_t target_idx = (edge.first == current_atom_idx) ? edge.second : 
                               (edge.second == current_atom_idx) ? edge.first : SIZE_MAX;
            
            if (target_idx != SIZE_MAX && !visited[target_idx]) {
                float edge_attention = 1.0f;
                
                // Apply attention decay
                edge_attention *= attention_decay_;
                
                if (edge_attention > max_edge_attention) {
                    max_edge_attention = edge_attention;
                    next_atom_idx = target_idx;
                }
            }
        }
        
        if (next_atom_idx == current_atom_idx) {
            // No more connected atoms, find next unvisited with highest attention
            max_attention = 0.0f;
            for (size_t i = 0; i < pattern.nodes.size(); ++i) {
                if (!visited[i]) {
                    float attention = atom_type_weights_[pattern.nodes[i].atom_type];
                    if (attention > max_attention) {
                        max_attention = attention;
                        next_atom_idx = i;
                    }
                }
            }
        }
        
        if (next_atom_idx == current_atom_idx) {
            break; // No more atoms to visit
        }
        
        traversal_path.push_back(pattern.nodes[next_atom_idx]);
        visited[next_atom_idx] = true;
        current_atom_idx = next_atom_idx;
    }
    
    return traversal_path;
}

void NeuralInferenceKernel::setAtomSpaceInterface(
    std::function<std::vector<AtomSpaceAtom>(const std::string&)> atomspace_interface) {
    atomspace_interface_ = atomspace_interface;
}

void NeuralInferenceKernel::setSymbolicKernel(std::shared_ptr<SymbolicTensorKernel> symbolic_kernel) {
    symbolic_kernel_ = symbolic_kernel;
}

void NeuralInferenceKernel::setInferenceParameters(float confidence_threshold,
                                                  float attention_decay,
                                                  int pattern_complexity_limit) {
    confidence_threshold_ = confidence_threshold;
    attention_decay_ = attention_decay;
    pattern_complexity_limit_ = pattern_complexity_limit;
}

// Private method implementations

NeuralInferenceResult NeuralInferenceKernel::forwardChaining(const HypergraphPattern& pattern, int depth) {
    NeuralInferenceResult result;
    
    if (depth >= max_recursion_depth_) {
        result.recursive_depth_exceeded = true;
        return result;
    }
    
    // Forward chaining: start from facts and derive new conclusions
    for (const auto& atom : pattern.nodes) {
        if (atom.atom_type == "ConceptNode" || atom.atom_type == "PredicateNode") {
            // This is a fact, try to find rules that can use it
            for (const auto& kb_pattern : pattern_knowledge_base_) {
                for (const auto& kb_atom : kb_pattern.nodes) {
                    if (kb_atom.atom_name == atom.atom_name) {
                        result.inferred_atoms.push_back(kb_atom);
                        result.reasoning_patterns.push_back(kb_pattern);
                    }
                }
            }
        }
    }
    
    result.inference_confidence = result.inferred_atoms.empty() ? 0.0f : 0.8f;
    return result;
}

NeuralInferenceResult NeuralInferenceKernel::backwardChaining(const HypergraphPattern& pattern, int depth) {
    NeuralInferenceResult result;
    
    if (depth >= max_recursion_depth_) {
        result.recursive_depth_exceeded = true;
        return result;
    }
    
    // Backward chaining: start from goal and work backwards to find supporting evidence
    for (const auto& atom : pattern.nodes) {
        if (atom.atom_type == "ImplicationLink" || atom.atom_type == "EvaluationLink") {
            // This is a goal, try to find supporting facts
            for (const auto& kb_pattern : pattern_knowledge_base_) {
                float similarity = computePatternSimilarity(pattern, kb_pattern);
                if (similarity > confidence_threshold_) {
                    result.inferred_atoms.insert(result.inferred_atoms.end(), 
                                               kb_pattern.nodes.begin(), kb_pattern.nodes.end());
                    result.reasoning_patterns.push_back(kb_pattern);
                }
            }
        }
    }
    
    result.inference_confidence = result.inferred_atoms.empty() ? 0.0f : 0.7f;
    return result;
}

NeuralInferenceResult NeuralInferenceKernel::neuralAttentionInference(const HypergraphPattern& pattern, int depth) {
    NeuralInferenceResult result;
    
    if (depth >= max_recursion_depth_) {
        result.recursive_depth_exceeded = true;
        return result;
    }
    
    // Use neural attention to guide inference
    auto attention_path = neuralAttentionTraversal(pattern, global_attention_weights_);
    result.inferred_atoms = attention_path;
    
    // For each atom in the attention path, try to find related patterns
    for (const auto& atom : attention_path) {
        for (const auto& kb_pattern : pattern_knowledge_base_) {
            for (const auto& kb_atom : kb_pattern.nodes) {
                if (kb_atom.atom_type == atom.atom_type) {
                    float similarity = computeAtomSimilarity(atom, kb_atom);
                    if (similarity > confidence_threshold_) {
                        result.reasoning_patterns.push_back(kb_pattern);
                    }
                }
            }
        }
    }
    
    result.inference_confidence = attention_path.empty() ? 0.0f : 0.75f;
    return result;
}

NeuralSymbolicTensor NeuralInferenceKernel::computeAtomNeuralRepresentation(const AtomSpaceAtom& atom) {
    // Create a basic neural representation for the atom
    NeuralSymbolicTensor tensor(256); // Default embedding dimension
    
    // Hash the atom name and type to create a deterministic embedding
    std::hash<std::string> hasher;
    auto name_hash = hasher(atom.atom_name);
    auto type_hash = hasher(atom.atom_type);
    
    for (size_t i = 0; i < tensor.neural_embedding.size(); ++i) {
        float val = static_cast<float>((name_hash + type_hash + i) % 1000) / 1000.0f - 0.5f;
        tensor.neural_embedding[i] = val;
    }
    
    // Set confidence based on truth values
    auto strength_it = atom.truth_values.find("strength");
    if (strength_it != atom.truth_values.end()) {
        tensor.confidence_score = strength_it->second;
    }
    
    auto confidence_it = atom.truth_values.find("confidence");
    if (confidence_it != atom.truth_values.end()) {
        tensor.confidence_score = std::min(tensor.confidence_score, confidence_it->second);
    }
    
    tensor.symbolic_expression = "(" + atom.atom_type + " " + atom.atom_name + ")";
    
    return tensor;
}

float NeuralInferenceKernel::computePatternSimilarity(const HypergraphPattern& pattern1, 
                                                    const HypergraphPattern& pattern2) {
    if (pattern1.nodes.empty() || pattern2.nodes.empty()) {
        return 0.0f;
    }
    
    // Compare patterns based on node types and structure
    float node_similarity = 0.0f;
    float edge_similarity = 0.0f;
    
    // Node similarity
    std::unordered_map<std::string, int> type_count1, type_count2;
    for (const auto& atom : pattern1.nodes) {
        type_count1[atom.atom_type]++;
    }
    for (const auto& atom : pattern2.nodes) {
        type_count2[atom.atom_type]++;
    }
    
    int common_types = 0;
    int total_types = 0;
    for (const auto& type_count : type_count1) {
        total_types += type_count.second;
        auto it = type_count2.find(type_count.first);
        if (it != type_count2.end()) {
            common_types += std::min(type_count.second, it->second);
        }
    }
    
    node_similarity = total_types > 0 ? static_cast<float>(common_types) / total_types : 0.0f;
    
    // Edge similarity (simplified)
    if (pattern1.edges.size() == pattern2.edges.size()) {
        edge_similarity = 0.5f; // Simplified edge comparison
    }
    
    return 0.7f * node_similarity + 0.3f * edge_similarity;
}

float NeuralInferenceKernel::computeAtomSimilarity(const AtomSpaceAtom& atom1, const AtomSpaceAtom& atom2) {
    if (atom1.atom_type != atom2.atom_type) {
        return 0.0f;
    }
    
    if (atom1.atom_name == atom2.atom_name) {
        return 1.0f;
    }
    
    // Compare neural representations if using symbolic kernel
    if (symbolic_kernel_) {
        const auto& embedding1 = atom1.neural_representation.neural_embedding;
        const auto& embedding2 = atom2.neural_representation.neural_embedding;
        
        if (embedding1.size() == embedding2.size()) {
            float dot_product = 0.0f;
            float norm1 = 0.0f, norm2 = 0.0f;
            
            for (size_t i = 0; i < embedding1.size(); ++i) {
                dot_product += embedding1[i] * embedding2[i];
                norm1 += embedding1[i] * embedding1[i];
                norm2 += embedding2[i] * embedding2[i];
            }
            
            if (norm1 > 0.0f && norm2 > 0.0f) {
                return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
            }
        }
    }
    
    return 0.3f; // Default similarity for same-type atoms
}

bool NeuralInferenceKernel::applyCognitiveConstraints(const NeuralInferenceResult& result) {
    // Apply cognitive plausibility constraints
    
    // Check if inference confidence is reasonable
    if (result.inference_confidence > 0.99f) {
        return false; // Too certain, likely overfitting
    }
    
    // Check if too many atoms were inferred (cognitive overload)
    if (result.inferred_atoms.size() > 50) {
        return false;
    }
    
    // Check if reasoning patterns are coherent
    if (result.reasoning_patterns.size() > 10) {
        return false; // Too many patterns, likely incoherent
    }
    
    return true;
}

std::string NeuralInferenceKernel::generateReasoningTrace(const std::vector<HypergraphPattern>& patterns,
                                                        InferenceStrategy strategy,
                                                        int depth) {
    std::ostringstream trace;
    
    trace << "Neural Inference Trace [depth=" << depth << "]:\n";
    
    switch (strategy) {
        case InferenceStrategy::FORWARD_CHAINING:
            trace << "Strategy: Forward Chaining\n";
            break;
        case InferenceStrategy::BACKWARD_CHAINING:
            trace << "Strategy: Backward Chaining\n";
            break;
        case InferenceStrategy::NEURAL_ATTENTION_GUIDED:
            trace << "Strategy: Neural Attention Guided\n";
            break;
        case InferenceStrategy::RECURSIVE_PATTERN_MATCHING:
            trace << "Strategy: Recursive Pattern Matching\n";
            break;
        default:
            trace << "Strategy: Unknown\n";
    }
    
    trace << "Patterns involved: " << patterns.size() << "\n";
    
    for (size_t i = 0; i < patterns.size() && i < 3; ++i) {
        trace << "  Pattern " << i << ": " << patterns[i].nodes.size() 
              << " nodes, confidence=" << patterns[i].pattern_confidence << "\n";
    }
    
    return trace.str();
}

void NeuralInferenceKernel::updateCognitiveLoad(size_t pattern_count, int recursion_depth, 
                                               std::chrono::milliseconds inference_time) {
    // Compute cognitive load based on complexity factors
    float load = 0.0f;
    
    // Pattern count factor
    load += static_cast<float>(pattern_count) * 0.1f;
    
    // Recursion depth factor
    load += static_cast<float>(recursion_depth) * 0.2f;
    
    // Time factor
    load += static_cast<float>(inference_time.count()) * 0.001f;
    
    // Update with exponential moving average
    float current_cognitive_load = cognitive_load_.load();
    float new_load = 0.9f * current_cognitive_load + 0.1f * load;
    cognitive_load_.store(new_load);
}

// Helper methods for parsing (simplified implementations)

std::vector<AtomSpaceAtom> NeuralInferenceKernel::parseAtomSpaceQuery(const std::string& query) {
    std::vector<AtomSpaceAtom> atoms;
    
    // Simple parsing for demonstration
    if (query.find("Dog") != std::string::npos) {
        atoms.push_back(createAtom("ConceptNode", "Dog"));
    }
    if (query.find("Animal") != std::string::npos) {
        atoms.push_back(createAtom("ConceptNode", "Animal"));
    }
    
    return atoms;
}

HypergraphPattern NeuralInferenceKernel::parseHypergraphPattern(const std::string& pattern_str) {
    // Simple parsing for demonstration
    HypergraphPattern pattern;
    
    auto atoms = parseAtomSpaceQuery(pattern_str);
    std::vector<std::pair<size_t, size_t>> connections;
    
    if (atoms.size() > 1) {
        connections.push_back({0, 1});
    }
    
    return createPattern(atoms, connections);
}

} // namespace orchestral