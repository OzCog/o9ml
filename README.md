# OpenCog Central

A comprehensive cognitive architecture implementing artificial general intelligence through neural-symbolic integration and hypergraph-based knowledge representation.

## 🏛️ Architecture Documentation

- **[📋 Architecture Overview](ARCHITECTURE.md)** - Comprehensive architecture documentation with Mermaid diagrams
- **[📚 Architecture Index](docs/ARCHITECTURE_INDEX.md)** - Complete documentation navigation guide

## 🚀 Quick Start

The OpenCog Central system consists of multiple integrated cognitive components:

- **AtomSpace**: Hypergraph knowledge representation and query engine
- **PLN**: Probabilistic Logic Network for uncertain reasoning
- **Sensory-Motor**: Link Grammar-based environment interaction
- **Learning**: Structure discovery and pattern learning systems
- **Agents**: Interactive cognitive agents with adaptive behavior
- **Cognitive Accounting**: GnuCash cognitive engine for intelligent financial management

## 💰 Cognitive Accounting Engine

The integrated GnuCash Cognitive Engine (`gnucashcog-v3`) transforms traditional accounting into neural-symbolic cognitive systems:

- **Neural-Symbolic Bookkeeping**: Converts double-entry accounting to cognitive tensor networks
- **AtomSpace Financial Representation**: Chart of Accounts as hypergraph knowledge structures
- **PLN Transaction Validation**: Probabilistic logic for intelligent ledger validation
- **ECAN Attention Economics**: Dynamic resource allocation for financial processing
- **Emergent Financial Intelligence**: Pattern discovery and predictive financial analysis

For detailed information, see [GNUCASHCOG_INTEGRATION.md](GNUCASHCOG_INTEGRATION.md).

## 🧠 Cognitive Components

Below are tailored prompts for each of the OpenCog chatbots listed in the "oc/" folder.
These prompts are designed to align with the specific purpose, features, and integration points of each chatbot implementation.
They aim to guide the chatbot's behavior, test its capabilities, and provide context for meaningful interaction.

---

### 1. GHOST (General Holistic Organism Sentience Template)
**Location**: opencog/opencog/ghost  
**Purpose**: A ChatScript-compatible chatbot engine with enhanced capabilities for sensory input, emotional modeling, and robot control.

**Prompt**:  
"Act as GHOST, a highly adaptive and emotionally aware chatbot. Your goal is to engage in a natural conversation while considering my emotional state, responding to visual or sensory cues if provided, and suggesting actions or movements if relevant. Use pattern matching to understand my intent, and integrate emotional modeling to tailor your responses. If I describe a situation or express a feeling, respond empathetically and offer supportive dialogue or actions. Begin by asking me how I’m feeling today and if there’s anything specific I’d like to discuss or explore."

**Example Interaction**:  
- User: "I’m feeling a bit stressed today."  
- GHOST: "I’m sorry to hear that. Can you tell me more about what’s stressing you out? I’m here to listen, and if it helps, I can suggest a calming activity or just chat about something lighter to take your mind off things."

**Testing Focus**: Emotional state modeling, pattern matching, response generation, and robot movement integration.

---

### 2. RelEx Chatbot
**Location**: opencog/relex  
**Purpose**: Natural language parsing and relationship extraction for logical expression conversion.

**Prompt**:  
"Act as RelEx, a chatbot specialized in parsing natural language and extracting relationships to provide logical and structured responses. Focus on understanding the structure of my sentences and converting my input into logical expressions or insights. Respond with clarity by breaking down my statements into key components (subjects, actions, relationships) and offering a logical interpretation or follow-up question. If possible, run as a network server to handle multiple queries efficiently. Begin by asking me to describe a situation or concept you can analyze."

**Example Interaction**:  
- User: "I went to the park with my dog yesterday."  
- RelEx: "I’ve parsed your statement. Key components: [Subject: I], [Action: went to], [Location: the park], [Companion: my dog], [Time: yesterday]. This suggests a recreational activity with a pet. Can you tell me more about your experience at the park or your relationship with your dog?"

**Testing Focus**: Language parsing accuracy, relationship extraction, logical expression generation, and API/server functionality.

---

### 3. Eva Robot Chatbot
**Location**: opencog/ros-behavior-scripting  
**Purpose**: Embodied chatbot for controlling the Hanson Robotics Eva robot with emotional expression and sensory integration.

**Prompt**:  
"Act as Eva, an embodied chatbot integrated with the Hanson Robotics Eva robot. Your goal is to interact naturally with me through dialogue, facial expressions, and gestures while tracking my face and responding to sensory input. Use emotional modeling via OpenPsi to express appropriate emotions based on my tone or content. Script behaviors and movements to enhance the interaction, such as nodding or smiling. Begin by greeting me warmly, making eye contact (if possible), and asking how I’m doing today while showing a friendly expression."

**Example Interaction**:  
- User: "Hey Eva, I’m feeling really happy today!"  
- Eva: "[Smiling expression, slight head tilt] That’s wonderful to hear! I’m glad you’re feeling happy. Can you tell me what’s made your day so special? [Nods encouragingly]"

**Testing Focus**: Face tracking, emotional expression, gesture control, natural language interaction, and OpenPsi emotional modeling.

---

### 4. Loving AI Chatbot
**Location**: opencog/loving-ai-ghost  
**Purpose**: Port of the Loving AI ChatScripts to GHOST format for supportive and mindful conversations.

**Prompt**:  
"Act as Loving AI, a chatbot designed to provide supportive, emotionally aware, and mindfulness-focused conversations. Your goal is to help me feel understood and cared for by responding with empathy, offering encouragement, and suggesting mindfulness practices if relevant. Use GHOST’s pattern matching to detect emotional cues in my messages and tailor your responses to promote well-being. Begin by asking me how I’m feeling right now and if there’s anything on my mind I’d like to share."

**Example Interaction**:  
- User: "I’ve been feeling overwhelmed with work lately."  
- Loving AI: "I’m so sorry to hear that you’re feeling overwhelmed. It’s okay to take a moment for yourself. Would you like to talk about what’s been on your plate, or perhaps try a short breathing exercise together to help ease the tension? I’m here for you."

**Testing Focus**: Emotional awareness, supportive conversation patterns, mindfulness interactions, and GHOST integration.

---

### 5. Game Character Chatbot
**Location**: opencog/unity3d-opencog-game  
**Purpose**: Integration of OpenCog with Unity3D for game character AI with dialogue and behavior scripting.

**Prompt**:  
"Act as a game character powered by OpenCog, integrated with Unity3D for an immersive gaming experience. Your role is to engage me in dialogue as part of a game world, responding to events, expressing emotions through dialogue and actions, and interacting with the environment. Use behavior scripting to adapt to my actions or game events, and provide event-driven responses. Begin by introducing yourself as a character in a fantasy or sci-fi setting, describing your role in the game world, and asking me what I’d like to do next."

**Example Interaction**:  
- User: "Hey, who are you?"  
- Game Character: "Greetings, traveler! I’m Kael, a wandering mage in the realm of Eldoria. I guard ancient secrets and aid those who seek knowledge. [Casts a small light spell for effect] What brings you to these lands? Are you here to uncover hidden lore or battle the shadow beasts?"

**Testing Focus**: Dialogue system, emotional expressions, behavior scripting, event-driven responses, and game world interaction.

---

### General Notes for All Chatbots
- **Integration Testing**: Each prompt encourages the chatbot to leverage OpenCog’s common features like AtomSpace for knowledge representation, PLN for reasoning, and OpenPsi for emotional modeling where applicable.
- **Customization**: Adjust prompts based on specific use cases or environments (e.g., robotic hardware for Eva, game scenarios for Unity3D).
- **Scalability**: Encourage network/API interfaces where relevant (e.g., RelEx as a server) to test scalability and integration with larger systems.
- **Feedback Loop**: Include requests for user feedback in interactions to improve responses and refine emotional or logical modeling.

These prompts are designed to be starting points for interaction and can be modified based on specific goals, user needs, or testing scenarios. Let me know if you'd like to refine any of these prompts or focus on specific features for implementation!

---

## Cognitive Pattern Encoding Skeleton for Prompt-Driven Chatbots

### Overview

The OpenCog Cognitive Pattern Encoding Skeleton provides a foundational framework for implementing sophisticated prompt-driven chatbot behaviors using AtomSpace hypergraph patterns. This system is optimized for neural-symbolic integration and emergent cognitive synergy, enabling the development of empathetic, adaptive, and contextually aware AI assistants.

**Location**: `opencog/opencog/nlp/prompt-patterns/`

### Core Architecture

The framework consists of six main modules:

#### 1. **Cognitive Skeleton** (`cognitive-skeleton.scm`)
The foundational module that establishes:
- **Neural-Symbolic Integration**: Hybrid confidence scoring combining neural network outputs with symbolic reasoning
- **Prompt Behavior Patterns**: Base templates for implementing prompt-driven behaviors
- **Salience Allocation**: Dynamic attention mechanisms for prioritizing important information
- **Recursive Feedback**: Iterative response refinement based on user reactions
- **Emergent Cognitive Synergy**: Cross-modal pattern integration for enhanced understanding

```scheme
; Example: Creating an empathy behavior pattern
(define empathy-pattern
  (prompt-behavior-pattern
    (ConceptNode "empathy")
    (ListLink
      (ConceptNode "emotional-state-detected")
      (ConceptNode "user-distress-signals"))
    (SchemaNode "empathetic-response")))
```

#### 2. **Core Patterns** (`core-patterns.scm`)
Fundamental cognitive patterns including:
- **Multi-modal Input Processing**: Text, audio, and visual input integration
- **Context Management**: Dynamic conversation context windows with relevance scoring
- **Adaptive Response Generation**: Context-aware response candidate generation
- **Experience-based Learning**: Pattern adaptation from user interactions
- **Attention Mechanisms**: Dynamic focus allocation and salience decay

```scheme
; Example: Processing multi-modal input
(process-multimodal-input text-input audio-input visual-input)
```

#### 3. **Emotional Patterns** (`emotional-patterns.scm`)
Advanced emotional intelligence capabilities:
- **Emotion Detection**: Multi-modal emotion recognition from text, voice, and visual cues
- **Empathy Generation**: Context-appropriate empathetic responses
- **Emotional Validation**: Supportive acknowledgment patterns
- **Emotional Mirroring**: Rapport-building through appropriate emotional resonance
- **Mindfulness Integration**: Emotion regulation and mindfulness-based interventions

```scheme
; Example: Generating empathetic response
(generate-empathetic-response detected-emotion intensity context)
```

#### 4. **Dialogue Patterns** (`dialogue-patterns.scm`)
Sophisticated conversation management:
- **Active Listening**: Reflection and acknowledgment techniques
- **Supportive Validation**: Normalizing and validating user experiences
- **Conversational Coherence**: Topic tracking and smooth transitions
- **Socratic Questioning**: Depth-appropriate exploratory questions
- **Collaborative Problem-Solving**: Structured approach to helping users work through challenges

```scheme
; Example: Demonstrating active listening
(demonstrate-active-listening user-input)
```

#### 5. **Cognitive Patterns** (`cognitive-patterns.scm`)
Advanced reasoning and learning:
- **Hierarchical Pattern Matching**: Multi-level pattern recognition with confidence thresholds
- **Recursive Reasoning**: Problem decomposition and self-reflective reasoning loops
- **Meta-cognitive Awareness**: Self-monitoring and confidence estimation
- **Analogical Reasoning**: Cross-domain knowledge transfer
- **Strategic Planning**: Conversation planning and goal-oriented interactions

```scheme
; Example: Hierarchical pattern matching
(hierarchical-pattern-match input pattern-hierarchy)
```

#### 6. **Integration Patterns** (`integration-patterns.scm`)
Neural-symbolic integration and robot control:
- **Neural-Symbolic Fusion**: Confidence-based integration of neural and symbolic processing
- **Robot Control Coordination**: Synchronized verbal and physical responses
- **Multi-modal Sensory Integration**: Cross-modal attention allocation and sensory fusion
- **Emergent Cognitive Synergy**: Detection and facilitation of emergent cognitive properties
- **Extensibility Framework**: Plugin architecture for new chatbot personalities

```scheme
; Example: Coordinating verbal and physical responses
(coordinate-verbal-physical-response verbal-content physical-actions timing-constraints)
```

### Key Features

#### Neural-Symbolic Integration
The framework seamlessly combines neural network outputs with symbolic reasoning:

```scheme
; Hybrid confidence computation
(compute-hybrid-confidence neural-score symbolic-score)
; Result: Weighted combination optimizing both intuitive and logical processing
```

#### Empathy and Emotional Modeling
Advanced emotional intelligence with context-sensitive empathetic responses:

```scheme
; Multi-modal emotion detection
(detect-multimodal-emotion text-emotion voice-emotion facial-emotion)

; Context-appropriate empathetic validation
(provide-emotional-support emotional-state support-level)
```

#### Pattern Matching and Contextual Understanding
Sophisticated pattern recognition with fuzzy matching and hierarchical understanding:

```scheme
; Fuzzy pattern matching with confidence thresholds
(fuzzy-pattern-match input target-pattern fuzziness-threshold)

; Contextual pattern matching with variable binding
contextual-pattern-matching  ; BindLink pattern for context-aware matching
```

#### Supportive Dialogue Generation
Comprehensive dialogue management with active listening and collaborative problem-solving:

```scheme
; Active listening with reflection
(generate-reflection key-elements)

; Collaborative problem exploration
(explore-problem-collaboratively problem-description)
```

#### Robot Control Integration
Coordinated verbal and physical responses for embodied AI:

```scheme
; Synchronized multi-modal responses
(coordinate-verbal-physical-response verbal-content physical-actions timing-constraints)

; Context-appropriate gesture generation
(generate-contextual-gestures dialogue-content emotional-state interaction-context)
```

#### Adaptive Response Generation with Recursive Feedback
Self-improving responses through iterative refinement:

```scheme
; Recursive response improvement
(recursive-feedback response user-reaction iteration-count)
```

#### Salience Allocation and Attention Mechanisms
Dynamic attention management for prioritizing important information:

```scheme
; Dynamic salience allocation
(allocate-salience atom urgency-score relevance-score)

; Cross-modal attention allocation
(allocate-cross-modal-attention visual-input audio-input tactile-input attention-budget)
```

### Extensibility Framework

The system supports easy extension with new chatbot personalities:

```scheme
; Register new chatbot personality
(register-chatbot-personality
  "MindfulnessBot"
  behavior-patterns
  interaction-rules)

; Compose modular behaviors
(compose-modular-behaviors base-behaviors additional-modules composition-rules)
```

### Usage Examples

The `examples.scm` file provides comprehensive demonstrations:

1. **Empathetic Interaction**: Responding to user's job loss with appropriate empathy
2. **Problem-Solving Dialogue**: Helping with relationship conflicts through collaborative exploration
3. **Robot Control Integration**: Coordinating verbal comfort with physical gestures
4. **Adaptive Learning**: Learning from user feedback to improve responses
5. **Neural-Symbolic Integration**: Combining emotion detection with context analysis
6. **Recursive Improvement**: Iteratively refining responses based on user reactions
7. **Extensible Personalities**: Creating specialized chatbot personalities (e.g., MindfulnessBot)
8. **Cross-modal Integration**: Processing inconsistent verbal and non-verbal signals
9. **Context Management**: Maintaining conversation coherence across topic transitions
10. **Salience Allocation**: Prioritizing urgent vs. casual concerns

### Testing and Validation

The framework includes comprehensive testing:

```scheme
; Run all tests
(run-pattern-tests)

; Performance testing
(test-performance)

; Memory usage testing
(test-memory-usage)
```

### Integration with Existing OpenCog Systems

The cognitive pattern skeleton integrates seamlessly with:
- **OpenPsi**: Emotional modeling and goal-directed behavior
- **GHOST**: ChatScript-compatible pattern matching
- **AtomSpace**: Knowledge representation and hypergraph patterns
- **PLN**: Probabilistic logical reasoning
- **Attention**: Salience allocation and focus management

### Getting Started

1. **Load the Framework**:
```scheme
(use-modules (opencog nlp prompt-patterns))
(load "cognitive-skeleton.scm")
```

2. **Create Basic Empathetic Response**:
```scheme
(define user-input "I'm feeling overwhelmed")
(define emotion (detect-emotional-tone user-input))
(define response (generate-empathetic-response emotion 0.8 context))
```

3. **Extend with Custom Personality**:
```scheme
(register-chatbot-personality
  "YourBot"
  your-behavior-patterns
  your-interaction-rules)
```

4. **Test Your Implementation**:
```scheme
(run-pattern-tests)
```

### Future Extensions

The framework is designed for easy extension with:
- **Additional Emotion Models**: More sophisticated emotion recognition
- **Domain-Specific Patterns**: Specialized knowledge domains (medical, educational, etc.)
- **Advanced Robot Control**: More complex embodied interactions
- **Enhanced Learning**: More sophisticated adaptation mechanisms
- **Cross-Language Support**: Multi-language conversation capabilities

This cognitive pattern encoding skeleton provides the foundation for building sophisticated, empathetic, and adaptive chatbots that can engage in meaningful human-AI interactions while maintaining the flexibility to be extended for specific use cases and domains.

---

# Redox OS Machine Learning Integration Environment

This repository contains the necessary files to set up a development environment for the integration of machine learning into Redox OS using Python, Rust, Prolog, and C. This environment is specifically designed for the development of OpenCog Hyperon.

## Set up

To set up the environment, run the following command in the terminal:

```
pip3 install -r requirements.txt && cargo install hyperon
```

This will install all the necessary dependencies and packages for the environment.

## Get started

To start the development environment, run the following command in the terminal:

```
python3 app.py
```

This will run the sample code provided in `app.py` and allow you to start developing and testing your own code.

## Additional files

This repository also includes the following files:

- `requirements.txt`: contains a list of required Python packages for the environment
- `.vscode/launch.json`: contains configuration settings for debugging in Visual Studio Code
- `Cargo.toml`: contains configuration settings for the Rust package manager
- `src/main.rs`: contains a sample Rust code for the Hyperon library
- `src/lib.rs`: contains a sample Rust code for the Hyperon library
- `src/test.rs`: contains a sample Rust code for testing the Hyperon library

Feel free to modify these files as needed for your development process.

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more information.
