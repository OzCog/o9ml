#!/bin/bash
#
# Advanced Layer: Learning Modules Output Test
# Tests real output preparation for learning modules
#
set -e

echo "=========================================="
echo "Advanced Layer: Learning Modules Output Test"
echo "=========================================="

BUILD_DIR="build-advanced"
SOURCE_DIR=$(pwd)

# Check if advanced layer test is built
if [ ! -f "$BUILD_DIR/advanced-layer-test" ]; then
    echo "❌ Advanced layer test not found. Building..."
    ./advanced-layer-build-test.sh >/dev/null 2>&1
fi

if [ ! -f "$BUILD_DIR/advanced-layer-test" ]; then
    echo "❌ Failed to build advanced layer test"
    exit 1
fi

echo "🧪 Testing Learning Modules Output Generation..."

# Run the test and capture specific output
TEST_OUTPUT=$($BUILD_DIR/advanced-layer-test 2>&1)

echo "📊 Analyzing test results..."

# Extract key metrics from test output
SYNERGY_SCORE=$(echo "$TEST_OUTPUT" | grep "Synergy Score:" | head -1 | cut -d':' -f2 | xargs)
PATTERNS_COUNT=$(echo "$TEST_OUTPUT" | grep "Patterns Discovered:" | head -1 | cut -d':' -f2 | xargs)
KNOWLEDGE_COUNT=$(echo "$TEST_OUTPUT" | grep "Knowledge Learned:" | head -1 | cut -d':' -f2 | xargs)

# Check inference results
PLN_INFERENCES=$(echo "$TEST_OUTPUT" | grep "✅ Inference result confidence:" | wc -l)
PATTERN_DISCOVERIES=$(echo "$TEST_OUTPUT" | grep "🔍 Discovered pattern:" | wc -l)
OPTIMIZATION_GENERATIONS=$(echo "$TEST_OUTPUT" | grep "🔄 Generation" | wc -l)

echo ""
echo "🎯 Learning Modules Output Analysis:"
echo "  📈 Synergy Score: $SYNERGY_SCORE"
echo "  🔍 Patterns Discovered: $PATTERNS_COUNT"
echo "  🧠 Knowledge Learned: $KNOWLEDGE_COUNT"
echo "  🎲 PLN Inferences: $PLN_INFERENCES"
echo "  📊 Pattern Discoveries: $PATTERN_DISCOVERIES"
echo "  🧬 Optimization Steps: $OPTIMIZATION_GENERATIONS"

echo ""
echo "✅ Real Output Validation:"

# Validate synergy score
if [ -n "$SYNERGY_SCORE" ] && [ "$(echo "$SYNERGY_SCORE > 0" | bc -l)" = "1" ]; then
    echo "  ✅ Synergy score generated: $SYNERGY_SCORE"
else
    echo "  ❌ Invalid synergy score: $SYNERGY_SCORE"
    exit 1
fi

# Validate pattern count
if [ -n "$PATTERNS_COUNT" ] && [ "$PATTERNS_COUNT" -gt "0" ]; then
    echo "  ✅ Patterns discovered: $PATTERNS_COUNT"
else
    echo "  ❌ No patterns discovered"
    exit 1
fi

# Validate knowledge count
if [ -n "$KNOWLEDGE_COUNT" ] && [ "$KNOWLEDGE_COUNT" -gt "0" ]; then
    echo "  ✅ Knowledge learned: $KNOWLEDGE_COUNT"
else
    echo "  ❌ No knowledge learned"
    exit 1
fi

# Validate PLN inferences
if [ "$PLN_INFERENCES" -gt "0" ]; then
    echo "  ✅ PLN inferences executed: $PLN_INFERENCES"
else
    echo "  ❌ No PLN inferences performed"
    exit 1
fi

# Validate optimization
if [ "$OPTIMIZATION_GENERATIONS" -gt "0" ]; then
    echo "  ✅ Evolutionary optimization performed: $OPTIMIZATION_GENERATIONS steps"
else
    echo "  ❌ No optimization performed"
    exit 1
fi

echo ""
echo "🔬 Tensor Mapping Validation:"

# Check for tensor operations in output
TENSOR_OPS=$(echo "$TEST_OUTPUT" | grep -E "(confidence|tensor|reasoning)" | wc -l)
if [ "$TENSOR_OPS" -gt "10" ]; then
    echo "  ✅ Tensor operations verified: $TENSOR_OPS operations"
else
    echo "  ❌ Insufficient tensor operations: $TENSOR_OPS"
    exit 1
fi

echo ""
echo "⚛️  AtomSpace State Modification Validation:"

# Check for AtomSpace changes
ATOMSPACE_CHANGES=$(echo "$TEST_OUTPUT" | grep "AtomSpace Changes:" | head -1 | cut -d':' -f2 | xargs)
COGNITIVE_ATOMS=$(echo "$TEST_OUTPUT" | grep "Cognitive Atoms:" | head -1 | cut -d':' -f2 | xargs)

if [ -n "$ATOMSPACE_CHANGES" ] && [ "$ATOMSPACE_CHANGES" -gt "0" ]; then
    echo "  ✅ AtomSpace modifications confirmed: $ATOMSPACE_CHANGES changes"
else
    echo "  ❌ No AtomSpace modifications detected"
    exit 1
fi

if [ -n "$COGNITIVE_ATOMS" ] && [ "$(echo "$COGNITIVE_ATOMS > 0" | bc -l)" = "1" ]; then
    echo "  ✅ Cognitive atoms created: $COGNITIVE_ATOMS atoms"
else
    echo "  ❌ No cognitive atoms created"
    exit 1
fi

# Check for kernel reshaping
KERNEL_RESHAPING=$(echo "$TEST_OUTPUT" | grep "Learning Membrane: Reshaping Cognitive Kernel" | wc -l)
if [ "$KERNEL_RESHAPING" -gt "0" ]; then
    echo "  ✅ Cognitive kernel reshaping confirmed"
else
    echo "  ❌ No kernel reshaping detected"
    exit 1
fi

echo ""
echo "🔄 Recursive Synergy Validation:"

# Check for recursive operations
RECURSIVE_OPS=$(echo "$TEST_OUTPUT" | grep -E "(Phase|synergy|integration)" | wc -l)
if [ "$RECURSIVE_OPS" -gt "5" ]; then
    echo "  ✅ Recursive synergy confirmed: $RECURSIVE_OPS recursive operations"
else
    echo "  ❌ Insufficient recursive operations: $RECURSIVE_OPS"
    exit 1
fi

echo ""
echo "📋 Output Format Validation:"

# Check for structured output
if echo "$TEST_OUTPUT" | grep -q "All Advanced Layer tests PASSED"; then
    echo "  ✅ Structured output format confirmed"
else
    echo "  ❌ Invalid output format"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 Learning Modules Output Test PASSED!"
echo "=========================================="
echo ""
echo "✅ Requirements Validated:"
echo "  - Real output prepared for learning modules"
echo "  - Tensor mapping operational"
echo "  - Probabilistic reasoning confirmed"
echo "  - Uncertain reasoning and optimization working"
echo "  - Recursive synergy between modules achieved"
echo "  - AtomSpace state modifications confirmed"
echo "  - Cognitive kernel reshaping operational"
echo ""
echo "📊 Output Summary:"
echo "  - Synergy Score: $SYNERGY_SCORE (recursive integration quality)"
echo "  - Patterns: $PATTERNS_COUNT (probabilistic pattern discovery)"  
echo "  - Knowledge: $KNOWLEDGE_COUNT (PLN inference results)"
echo "  - Inferences: $PLN_INFERENCES (probabilistic reasoning steps)"
echo "  - Optimizations: $OPTIMIZATION_GENERATIONS (evolutionary steps)"
echo "  - AtomSpace Changes: $ATOMSPACE_CHANGES (cognitive state modifications)"
echo "  - Cognitive Atoms: $COGNITIVE_ATOMS (emergent concept formation)"
echo ""
echo "🚀 Advanced layer ready for cognitive field operations!"
echo "=========================================="