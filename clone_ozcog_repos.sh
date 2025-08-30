#!/bin/bash

# clone_ozcog_repos.sh
# Script to clone specific OzCog repositories and prepare them for monorepo integration

set -e  # Exit on error

# Function to print informational messages
echo_info() {
    echo -e "\e[34m[INFO]\e[0m $1"
}

# Function to print success messages
echo_success() {
    echo -e "\e[32m[SUCCESS]\e[0m $1"
}

# Function to print error messages
echo_error() {
    echo -e "\e[31m[ERROR]\e[0m $1" >&2
}

# Define the base GitHub URL for OzCog organization
BASE_URL="https://github.com/OzCog"

# Define the repositories to clone with their purposes
declare -A REPOS
REPOS["mlpn"]="to extend ECAN model with Cognitive MLRP toward ERPCAN"
REPOS["ko6ml"]="to extend OpenCog Core with Narrative-Driven Local+API Inference Engine"
REPOS["m0"]="to bridge Frontend & Backend Memory Systems"

# Create main repos directory
REPOS_DIR="ozcog-repos"
mkdir -p "$REPOS_DIR"

echo_info "Starting OzCog repository cloning process..."
echo_info "Target directory: $REPOS_DIR"

# Clone each repository
for repo in "${!REPOS[@]}"; do
    echo_info "Processing repository: $repo"
    echo_info "Purpose: ${REPOS[$repo]}"
    
    REPO_DIR="$REPOS_DIR/$repo"
    
    # Check if repository already exists
    if [ -d "$REPO_DIR" ]; then
        echo_info "Repository '$repo' already exists at $REPO_DIR. Skipping clone."
    else
        echo_info "Cloning repository '$repo' from $BASE_URL/$repo.git..."
        if git clone "$BASE_URL/$repo.git" "$REPO_DIR"; then
            echo_success "Successfully cloned '$repo'"
        else
            echo_error "Failed to clone repository '$repo'"
            exit 1
        fi
    fi
    
    # Remove .git directory to prepare for monorepo integration
    if [ -d "$REPO_DIR/.git" ]; then
        echo_info "Removing .git directory from '$repo' for monorepo integration..."
        rm -rf "$REPO_DIR/.git"
        echo_success "Removed .git directory from '$repo'"
    else
        echo_info "No .git directory found in '$repo' (already removed or not cloned with git)"
    fi
    
    echo_info "Repository '$repo' is ready for monorepo integration"
    echo ""
done

echo_success "All OzCog repositories have been cloned and prepared for monorepo integration!"
echo_info "Cloned repositories are located in: $REPOS_DIR/"
echo_info "Summary:"
for repo in "${!REPOS[@]}"; do
    echo_info "  - $repo: ${REPOS[$repo]}"
done