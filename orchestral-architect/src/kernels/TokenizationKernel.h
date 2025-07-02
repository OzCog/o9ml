/**
 * @file TokenizationKernel.h
 * @brief Multi-strategy tokenization and parsing kernel
 * 
 * The TokenizationKernel provides sophisticated text processing capabilities
 * with multiple tokenization strategies, attention-weighted output, and
 * integration with the neural-symbolic bridge.
 */

#pragma once

#include "../core/AgenticKernel.h"
#include <regex>
#include <unordered_map>
#include <set>

namespace orchestral {

/**
 * @brief Token representation with attention and symbolic metadata
 */
struct Token {
    std::string text;
    std::string type;           // "word", "punctuation", "number", etc.
    size_t position;           // Character position in original text
    double attentionWeight;    // Computed attention score
    double salience;          // Cognitive salience
    std::map<std::string, double> features;  // Linguistic features
    
    Token(const std::string& t = "", const std::string& type = "word")
        : text(t), type(type), position(0), attentionWeight(0.0), salience(0.0) {}
};

/**
 * @brief Tokenization strategy interface
 */
class TokenizationStrategy {
public:
    virtual ~TokenizationStrategy() = default;
    
    /**
     * @brief Tokenize input text
     * @param text Input text to tokenize
     * @return Vector of tokens
     */
    virtual std::vector<Token> tokenize(const std::string& text) = 0;
    
    /**
     * @brief Get strategy name
     * @return Name of this tokenization strategy
     */
    virtual std::string getName() const = 0;
};

/**
 * @brief Word-boundary tokenization strategy
 */
class WordTokenizer : public TokenizationStrategy {
public:
    std::vector<Token> tokenize(const std::string& text) override;
    std::string getName() const override { return "word_tokenizer"; }
};

/**
 * @brief Subword tokenization strategy (BPE-like)
 */
class SubwordTokenizer : public TokenizationStrategy {
public:
    SubwordTokenizer();
    std::vector<Token> tokenize(const std::string& text) override;
    std::string getName() const override { return "subword_tokenizer"; }
    
private:
    std::unordered_map<std::string, int> vocabulary_;
    void buildVocabulary();
};

/**
 * @brief Linguistic structure tokenization
 */
class LinguisticTokenizer : public TokenizationStrategy {
public:
    std::vector<Token> tokenize(const std::string& text) override;
    std::string getName() const override { return "linguistic_tokenizer"; }
    
private:
    std::regex sentenceBoundary_;
    std::regex wordBoundary_;
    std::set<std::string> stopWords_;
};

/**
 * @brief Multi-strategy tokenization kernel
 * 
 * Implements sophisticated text processing with attention allocation,
 * multiple tokenization strategies, and neural-symbolic integration.
 */
class TokenizationKernel : public AgenticKernel {
public:
    /**
     * @brief Construct a new Tokenization Kernel
     * @param name Unique name for this kernel instance
     */
    explicit TokenizationKernel(const std::string& name = "tokenization_kernel");
    
    ~TokenizationKernel() override = default;
    
    // AgenticKernel interface
    bool initialize() override;
    void shutdown() override;
    CognitiveResult process(const CognitiveInput& input) override;
    void handleEvent(const KernelEvent& event) override;
    std::vector<std::string> getCapabilities() const override;
    bool canProcess(const std::string& inputType) const override;
    double getCurrentLoad() const override;
    
    /**
     * @brief Add a tokenization strategy
     * @param strategy Unique pointer to strategy
     */
    void addStrategy(std::unique_ptr<TokenizationStrategy> strategy);
    
    /**
     * @brief Set active tokenization strategy
     * @param strategyName Name of strategy to use
     * @return true if strategy exists and was set
     */
    bool setActiveStrategy(const std::string& strategyName);
    
    /**
     * @brief Get available tokenization strategies
     * @return Vector of strategy names
     */
    std::vector<std::string> getAvailableStrategies() const;
    
    /**
     * @brief Enable/disable attention weighting
     * @param enabled Whether to compute attention weights
     */
    void setAttentionWeighting(bool enabled) { attentionEnabled_ = enabled; }
    
    /**
     * @brief Set attention vocabulary for weighting
     * @param vocab Map of words to base attention weights
     */
    void setAttentionVocabulary(const std::unordered_map<std::string, double>& vocab);
    
    /**
     * @brief Process text with specific strategy
     * @param text Input text
     * @param strategyName Strategy to use (empty for active)
     * @return Tokenization result with attention weights
     */
    CognitiveResult processText(const std::string& text, 
                               const std::string& strategyName = "");
    
private:
    /**
     * @brief Calculate attention weights for tokens
     * @param tokens Tokens to weight
     * @param context Processing context
     */
    void calculateAttentionWeights(std::vector<Token>& tokens, 
                                  const CognitiveInput& context);
    
    /**
     * @brief Calculate salience scores for tokens
     * @param tokens Tokens to analyze
     */
    void calculateSalience(std::vector<Token>& tokens);
    
    /**
     * @brief Extract linguistic features from tokens
     * @param tokens Tokens to analyze
     */
    void extractFeatures(std::vector<Token>& tokens);
    
    /**
     * @brief Convert tokens to structured result
     * @param tokens Processed tokens
     * @param processingTime Time taken for processing
     * @return Cognitive result with attention weights
     */
    CognitiveResult createResult(const std::vector<Token>& tokens,
                                std::chrono::milliseconds processingTime);
    
    /**
     * @brief Calculate processing cost based on text complexity
     * @param text Input text
     * @param tokens Generated tokens
     * @return Estimated processing cost
     */
    double calculateProcessingCost(const std::string& text, 
                                  const std::vector<Token>& tokens);
    
    /**
     * @brief Estimate cognitive value of tokenization
     * @param tokens Generated tokens
     * @return Estimated cognitive value
     */
    double estimateCognitiveValue(const std::vector<Token>& tokens);
    
    // Tokenization strategies
    std::unordered_map<std::string, std::unique_ptr<TokenizationStrategy>> strategies_;
    std::string activeStrategy_;
    
    // Attention and salience
    bool attentionEnabled_;
    std::unordered_map<std::string, double> attentionVocabulary_;
    std::unordered_map<std::string, double> salienceWeights_;
    
    // Processing state
    std::atomic<double> currentLoad_{0.0};
    std::atomic<size_t> processingCount_{0};
    
    // Performance tracking
    std::chrono::milliseconds totalProcessingTime_{0};
    size_t totalTokensProcessed_{0};
    
    // Configuration
    bool multiStrategyEnabled_;
    double attentionThreshold_;
    size_t maxTokens_;
};

} // namespace orchestral