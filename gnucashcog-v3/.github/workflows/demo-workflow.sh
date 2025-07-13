#!/bin/bash
# Demo script for the GitHub Action workflow
# This script demonstrates how to trigger the workflow programmatically

echo "🧬 Distributed Cognitive Network Phase Issues - Demo Script"
echo "=========================================================="
echo ""

# Check if GitHub CLI is available
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed. Please install it first:"
    echo "   https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub CLI. Please run:"
    echo "   gh auth login"
    exit 1
fi

echo "✅ GitHub CLI is available and authenticated"
echo ""

# Function to trigger workflow
trigger_workflow() {
    local create_all=$1
    local specific_phase=$2
    
    if [ "$create_all" = "true" ]; then
        echo "🚀 Triggering workflow to create ALL phases..."
        gh workflow run create-phase-issues.yml -f create_all_phases=true
    else
        echo "🎯 Triggering workflow to create Phase $specific_phase..."
        gh workflow run create-phase-issues.yml -f create_all_phases=false -f specific_phase="$specific_phase"
    fi
}

# Menu
echo "Choose an option:"
echo "1. Create ALL phases (6 phases + master coordination)"
echo "2. Create specific phase (1-6)"
echo "3. Show workflow status"
echo "4. List recent workflow runs"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        trigger_workflow "true" ""
        echo "✅ Workflow triggered to create all phases"
        ;;
    2)
        read -p "Enter phase number (1-6): " phase
        if [[ "$phase" =~ ^[1-6]$ ]]; then
            trigger_workflow "false" "$phase"
            echo "✅ Workflow triggered to create Phase $phase"
        else
            echo "❌ Invalid phase number. Must be 1-6."
            exit 1
        fi
        ;;
    3)
        echo "📊 Current workflow status:"
        gh workflow view create-phase-issues.yml
        ;;
    4)
        echo "📋 Recent workflow runs:"
        gh run list --workflow=create-phase-issues.yml --limit=10
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "💡 Tips:"
echo "   • View created issues: gh issue list --label 'phase-1,phase-2,phase-3,phase-4,phase-5,phase-6'"
echo "   • View master coordination: gh issue list --label 'master-coordination'"
echo "   • Monitor workflow runs: gh run list --workflow=create-phase-issues.yml"
echo ""
echo "🧠 Let the recursive self-optimization spiral commence! ✨"