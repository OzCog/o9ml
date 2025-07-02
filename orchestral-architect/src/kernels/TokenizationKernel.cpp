/**
 * @file TokenizationKernel.cpp
 * @brief Implementation of the TokenizationKernel class
 */

#include "TokenizationKernel.h"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace orchestral {

// WordTokenizer implementation

std::vector<Token> WordTokenizer::tokenize(const std::string& text) {
    std::vector<Token> tokens;
    std::istringstream iss(text);
    std::string word;
    size_t position = 0;
    
    while (iss >> word) {
        // Find actual position in text
        position = text.find(word, position);
        
        Token token(word, "word");
        token.position = position;
        tokens.push_back(token);
        
        position += word.length();
    }
    
    return tokens;
}

// SubwordTokenizer implementation

SubwordTokenizer::SubwordTokenizer() {
    buildVocabulary();
}

std::vector<Token> SubwordTokenizer::tokenize(const std::string& text) {
    std::vector<Token> tokens;
    
    // Simple BPE-like tokenization
    std::string current = text;
    size_t position = 0;
    
    while (!current.empty()) {
        std::string bestMatch;
        size_t bestLength = 1;
        
        // Find longest vocabulary match
        for (const auto& vocab : vocabulary_) {
            if (current.substr(0, vocab.first.length()) == vocab.first &&
                vocab.first.length() > bestLength) {
                bestMatch = vocab.first;
                bestLength = vocab.first.length();
            }
        }
        
        if (bestMatch.empty()) {
            bestMatch = current.substr(0, 1);
            bestLength = 1;
        }
        
        Token token(bestMatch, "subword");
        token.position = position;
        tokens.push_back(token);
        
        current = current.substr(bestLength);
        position += bestLength;
    }
    
    return tokens;
}

void SubwordTokenizer::buildVocabulary() {
    // Basic vocabulary for demonstration
    vocabulary_["the"] = 1000;
    vocabulary_["and"] = 900;
    vocabulary_["ing"] = 800;
    vocabulary_["tion"] = 700;
    vocabulary_["er"] = 600;
    vocabulary_["ly"] = 500;
    vocabulary_["ed"] = 400;
    vocabulary_["is"] = 300;
    vocabulary_["it"] = 200;
    vocabulary_["of"] = 100;
}

// LinguisticTokenizer implementation

std::vector<Token> LinguisticTokenizer::tokenize(const std::string& text) {
    std::vector<Token> tokens;
    
    // Initialize regex patterns
    sentenceBoundary_ = std::regex(R"([.!?]+)");
    wordBoundary_ = std::regex(R"(\W+)");
    
    // Initialize stop words
    stopWords_ = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"};
    
    // Tokenize by words
    std::sregex_token_iterator iter(text.begin(), text.end(), wordBoundary_, -1);
    std::sregex_token_iterator end;
    
    size_t position = 0;
    
    for (; iter != end; ++iter) {
        std::string word = *iter;
        if (word.empty()) continue;
        
        // Find position in original text
        position = text.find(word, position);
        
        Token token(word, "word");
        token.position = position;
        
        // Mark stop words
        if (stopWords_.find(word) != stopWords_.end()) {
            token.type = "stopword";
        }
        
        tokens.push_back(token);
        position += word.length();
    }
    
    return tokens;
}

// TokenizationKernel implementation

TokenizationKernel::TokenizationKernel(const std::string& name)
    : AgenticKernel(name, "tokenization"),
      attentionEnabled_(true),
      multiStrategyEnabled_(false),
      attentionThreshold_(0.1),
      maxTokens_(10000) {
    
    // Initialize default attention vocabulary
    attentionVocabulary_ = {
        {"important", 0.9},
        {"critical", 0.95},
        {"urgent", 0.8},
        {"simple", 0.7},
        {"complex", 0.85},
        {"test", 0.6},
        {"hello", 0.75},
        {"world", 0.7}
    };
    
    // Initialize salience weights
    salienceWeights_ = {
        {"noun", 0.8},
        {"verb", 0.75},
        {"adjective", 0.6},
        {"adverb", 0.5},
        {"stopword", 0.1}
    };
}

bool TokenizationKernel::initialize() {
    // Add default tokenization strategies
    addStrategy(std::make_unique<WordTokenizer>());
    addStrategy(std::make_unique<SubwordTokenizer>());
    addStrategy(std::make_unique<LinguisticTokenizer>());
    
    // Set default active strategy
    setActiveStrategy("word_tokenizer");
    
    // Mark as active
    setActive(true);
    
    return true;
}

void TokenizationKernel::shutdown() {
    strategies_.clear();
    activeStrategy_.clear();
    setActive(false);
}

CognitiveResult TokenizationKernel::process(const CognitiveInput& input) {
    auto start = std::chrono::high_resolution_clock::now();
    
    processingCount_.fetch_add(1);
    currentLoad_ = std::min(1.0, static_cast<double>(processingCount_) / 10.0);
    
    CognitiveResult result;
    
    try {
        if (input.type != "text" && input.type != "string") {
            result.success = false;
            result.errorMessage = "TokenizationKernel can only process text input";
            processingCount_.fetch_sub(1);
            return result;
        }
        
        // Get active strategy
        auto it = strategies_.find(activeStrategy_);
        if (it == strategies_.end()) {
            result.success = false;
            result.errorMessage = "No active tokenization strategy";
            processingCount_.fetch_sub(1);
            return result;
        }
        
        // Tokenize the input
        std::vector<Token> tokens = it->second->tokenize(input.data);
        
        // Apply attention weighting
        if (attentionEnabled_) {
            calculateAttentionWeights(tokens, input);
        }
        
        // Calculate salience
        calculateSalience(tokens);
        
        // Extract features
        extractFeatures(tokens);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Create result
        result = createResult(tokens, duration);
        
        totalTokensProcessed_ += tokens.size();
        totalProcessingTime_ += duration;
        
        updateMetrics(result, duration);
        
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Processing error: ") + e.what();
    }
    
    processingCount_.fetch_sub(1);
    currentLoad_ = std::min(1.0, static_cast<double>(processingCount_) / 10.0);
    
    return result;
}

void TokenizationKernel::handleEvent(const KernelEvent& event) {
    if (event.eventType == "set_attention_vocab") {
        auto it = event.payload.find("vocabulary");
        if (it != event.payload.end()) {
            // Parse vocabulary from payload (simplified)
            // In practice, you'd use JSON or another structured format
        }
    } else if (event.eventType == "set_strategy") {
        auto it = event.payload.find("strategy");
        if (it != event.payload.end()) {
            setActiveStrategy(it->second);
        }
    }
}

std::vector<std::string> TokenizationKernel::getCapabilities() const {
    return {
        "text_tokenization",
        "attention_weighting", 
        "linguistic_analysis",
        "multi_strategy_processing"
    };
}

bool TokenizationKernel::canProcess(const std::string& inputType) const {
    return inputType == "text" || inputType == "string";
}

double TokenizationKernel::getCurrentLoad() const {
    return currentLoad_.load();
}

void TokenizationKernel::addStrategy(std::unique_ptr<TokenizationStrategy> strategy) {
    if (strategy) {
        std::string name = strategy->getName();
        strategies_[name] = std::move(strategy);
        
        if (activeStrategy_.empty()) {
            activeStrategy_ = name;
        }
    }
}

bool TokenizationKernel::setActiveStrategy(const std::string& strategyName) {
    auto it = strategies_.find(strategyName);
    if (it != strategies_.end()) {
        activeStrategy_ = strategyName;
        return true;
    }
    return false;
}

std::vector<std::string> TokenizationKernel::getAvailableStrategies() const {
    std::vector<std::string> strategies;
    strategies.reserve(strategies_.size());
    
    for (const auto& pair : strategies_) {
        strategies.push_back(pair.first);
    }
    
    return strategies;
}

void TokenizationKernel::setAttentionVocabulary(const std::unordered_map<std::string, double>& vocab) {
    attentionVocabulary_ = vocab;
}

CognitiveResult TokenizationKernel::processText(const std::string& text, 
                                               const std::string& strategyName) {
    CognitiveInput input(text, "text");
    
    if (!strategyName.empty()) {
        std::string originalStrategy = activeStrategy_;
        setActiveStrategy(strategyName);
        
        CognitiveResult result = process(input);
        
        setActiveStrategy(originalStrategy);
        return result;
    }
    
    return process(input);
}

void TokenizationKernel::calculateAttentionWeights(std::vector<Token>& tokens, 
                                                   const CognitiveInput& context) {
    for (Token& token : tokens) {
        double weight = 0.5;  // Default attention weight
        
        // Check attention vocabulary
        auto it = attentionVocabulary_.find(token.text);
        if (it != attentionVocabulary_.end()) {
            weight = it->second;
        }
        
        // Apply context weighting
        auto contextIt = context.contextWeights.find(token.text);
        if (contextIt != context.contextWeights.end()) {
            weight = std::max(weight, contextIt->second);
        }
        
        // Apply urgency factor
        weight *= (0.5 + 0.5 * context.urgency);
        
        // Position-based attention (beginning and end get more attention)
        double positionFactor = 1.0;
        if (tokens.size() > 1) {
            size_t idx = &token - &tokens[0];
            if (idx == 0 || idx == tokens.size() - 1) {
                positionFactor = 1.2;
            } else if (idx < 3 || idx >= tokens.size() - 3) {
                positionFactor = 1.1;
            }
        }
        
        weight *= positionFactor;
        token.attentionWeight = std::min(1.0, weight);
    }
}

void TokenizationKernel::calculateSalience(std::vector<Token>& tokens) {
    for (Token& token : tokens) {
        double salience = 0.5;  // Base salience
        
        // Type-based salience
        auto it = salienceWeights_.find(token.type);
        if (it != salienceWeights_.end()) {
            salience = it->second;
        }
        
        // Length-based salience (longer words more salient)
        if (token.text.length() > 6) {
            salience *= 1.2;
        } else if (token.text.length() < 3) {
            salience *= 0.8;
        }
        
        // Frequency-based salience (rare words more salient)
        // This would typically use a frequency corpus
        if (token.text.length() > 8) {
            salience *= 1.3;  // Assume long words are rare
        }
        
        token.salience = std::min(1.0, salience);
    }
}

void TokenizationKernel::extractFeatures(std::vector<Token>& tokens) {
    for (Token& token : tokens) {
        // Length feature
        token.features["length"] = static_cast<double>(token.text.length());
        
        // Character type features
        bool hasUpper = std::any_of(token.text.begin(), token.text.end(), ::isupper);
        bool hasLower = std::any_of(token.text.begin(), token.text.end(), ::islower);
        bool hasDigit = std::any_of(token.text.begin(), token.text.end(), ::isdigit);
        
        token.features["has_upper"] = hasUpper ? 1.0 : 0.0;
        token.features["has_lower"] = hasLower ? 1.0 : 0.0;
        token.features["has_digit"] = hasDigit ? 1.0 : 0.0;
        
        // Position features
        token.features["position"] = static_cast<double>(token.position);
        
        // Combined attention-salience score
        token.features["cognitive_score"] = 
            0.6 * token.attentionWeight + 0.4 * token.salience;
    }
}

CognitiveResult TokenizationKernel::createResult(const std::vector<Token>& tokens,
                                                 std::chrono::milliseconds processingTime) {
    CognitiveResult result;
    
    // Build processed data representation
    std::ostringstream oss;
    oss << "Tokenized: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "{\"" << tokens[i].text << "\": " << tokens[i].attentionWeight << "}";
    }
    oss << "]";
    
    result.processedData = oss.str();
    
    // Extract attention weights
    for (const Token& token : tokens) {
        if (token.attentionWeight > attentionThreshold_) {
            result.attentionWeights[token.text] = token.attentionWeight;
        }
    }
    
    // Calculate costs and values
    std::string inputText = "";  // Would need to pass this through
    result.processingCost = calculateProcessingCost(inputText, tokens);
    result.estimatedValue = estimateCognitiveValue(tokens);
    result.processingTime = processingTime;
    result.success = true;
    
    return result;
}

double TokenizationKernel::calculateProcessingCost(const std::string& text, 
                                                  const std::vector<Token>& tokens) {
    // Base cost proportional to text length and token count
    double baseCost = 0.1 + 0.001 * text.length() + 0.01 * tokens.size();
    
    // Strategy complexity factor
    double strategyFactor = 1.0;
    if (activeStrategy_ == "linguistic_tokenizer") {
        strategyFactor = 1.5;  // More complex processing
    } else if (activeStrategy_ == "subword_tokenizer") {
        strategyFactor = 1.2;
    }
    
    // Attention processing overhead
    double attentionFactor = attentionEnabled_ ? 1.3 : 1.0;
    
    return baseCost * strategyFactor * attentionFactor;
}

double TokenizationKernel::estimateCognitiveValue(const std::vector<Token>& tokens) {
    double totalValue = 0.0;
    
    for (const Token& token : tokens) {
        // Value based on attention weight and salience
        double tokenValue = 0.5 * token.attentionWeight + 0.3 * token.salience;
        
        // Bonus for high-attention tokens
        if (token.attentionWeight > 0.8) {
            tokenValue *= 1.5;
        }
        
        // Feature richness bonus
        if (token.features.size() > 4) {
            tokenValue *= 1.1;
        }
        
        totalValue += tokenValue;
    }
    
    // Diminishing returns for large token sets
    double sizeFactor = 1.0 / (1.0 + 0.001 * tokens.size());
    
    return totalValue * sizeFactor;
}

} // namespace orchestral