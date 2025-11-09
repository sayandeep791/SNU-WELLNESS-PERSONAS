#!/bin/bash

# SNU Wellness Personas - Deployment Script
# This script will push your code to GitHub and prepare for Streamlit Cloud deployment

echo "🚀 Preparing SNU Wellness Personas for deployment..."
echo ""

# Navigate to project directory
cd "/Users/sushantkumarpal/Desktop/snu project"

# Stage all changes
echo "📦 Staging files..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Prepare for Streamlit Cloud deployment

- Add professional README with project overview
- Include comprehensive PDF project report
- Add deployment guide
- Update requirements.txt
- Clean up configuration files
- Ready for Streamlit Cloud hosting"

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Successfully pushed to GitHub!"
echo ""
echo "📋 Next Steps:"
echo "1. Visit: https://share.streamlit.io/"
echo "2. Sign in with your GitHub account"
echo "3. Click 'New app'"
echo "4. Select repository: Sushant-kumar-pal/FOODING-PEROSONA-PREDICTION"
echo "5. Branch: main"
echo "6. Main file: project.py"
echo "7. Click 'Deploy'"
echo ""
echo "⏱️  Deployment takes 2-5 minutes"
echo "🔗 You'll get a shareable URL like: https://snu-wellness-personas.streamlit.app"
echo ""
echo "Repository: https://github.com/Sushant-kumar-pal/FOODING-PEROSONA-PREDICTION"
