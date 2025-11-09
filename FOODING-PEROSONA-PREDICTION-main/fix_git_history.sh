#!/bin/bash

# Fix Git History - Remove API Key from Commits
# This script creates a clean repository without the leaked API key

echo "🔒 Fixing Git history to remove API key..."
echo ""

cd "/Users/sushantkumarpal/Desktop/snu project"

# Method 1: Use git filter-repo (if installed)
if command -v git-filter-repo &> /dev/null; then
    echo "Using git-filter-repo to clean history..."
    git filter-repo --path .streamlit/secrets.toml --invert-paths --force
    git push origin main --force
    echo "✅ History cleaned with git-filter-repo"
else
    echo "⚠️  git-filter-repo not found. Using alternative method..."
    echo ""
    echo "Please run these commands manually:"
    echo ""
    echo "# 1. Remove the old commit history"
    echo "git checkout --orphan temp_branch"
    echo ""
    echo "# 2. Add all files"
    echo "git add ."
    echo ""
    echo "# 3. Create new initial commit"
    echo "git commit -m 'Initial commit - SNU Wellness Personas App'"
    echo ""
    echo "# 4. Delete old main branch"
    echo "git branch -D main"
    echo ""
    echo "# 5. Rename temp branch to main"
    echo "git branch -m main"
    echo ""
    echo "# 6. Force push to GitHub"
    echo "git push -f origin main"
    echo ""
fi
