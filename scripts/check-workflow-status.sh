#!/bin/bash
# GitHub Actions Workflow Status Checker
# This script checks the status of the GitHub Actions workflows

set -e

echo "🔍 GitHub Actions Workflow Status Checker"
echo "=========================================="

REPO_OWNER="OzCog"
REPO_NAME="cogml"
BRANCH="copilot/fix-25"

echo "Repository: $REPO_OWNER/$REPO_NAME"
echo "Branch: $BRANCH"
echo ""

# Check if gh CLI is available
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo "💡 Install it with: curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg"
    echo "   echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main\" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null"
    echo "   sudo apt update && sudo apt install gh"
    echo ""
    echo "🌐 Alternative: Check manually at https://github.com/$REPO_OWNER/$REPO_NAME/actions"
    exit 1
fi

echo "📊 Recent workflow runs:"
echo ""

# List recent workflow runs
gh run list --repo "$REPO_OWNER/$REPO_NAME" --limit 10 --json displayTitle,status,conclusion,createdAt,workflowName --template '
{{- range . -}}
{{- printf "🔄 %s" .workflowName -}}
{{- if eq .status "completed" -}}
  {{- if eq .conclusion "success" }} ✅{{- else if eq .conclusion "failure" }} ❌{{- else }} ⚠️{{- end -}}
{{- else }} 🟡{{- end }}
{{ printf " - %s" .displayTitle }}
{{ printf "   Created: %s" .createdAt }}
{{ printf "\n" }}
{{- end -}}'

echo ""
echo "🎯 Workflow files in repository:"
echo ""

# List workflow files
for workflow in .github/workflows/*.yml .github/workflows/*.yaml; do
    if [ -f "$workflow" ]; then
        echo "📄 $(basename "$workflow")"
        echo "   $(grep '^name:' "$workflow" | sed 's/name: /   Name: /')"
        echo "   Triggers: $(grep -A 10 '^on:' "$workflow" | grep -E '(push|pull_request)' | tr '\n' ' ' | sed 's/  */ /g')"
        echo ""
    fi
done

echo "💡 Tips:"
echo "   • View detailed run: gh run view <run-id> --repo $REPO_OWNER/$REPO_NAME"
echo "   • Watch logs: gh run watch --repo $REPO_OWNER/$REPO_NAME"
echo "   • Re-run failed: gh run rerun <run-id> --repo $REPO_OWNER/$REPO_NAME"