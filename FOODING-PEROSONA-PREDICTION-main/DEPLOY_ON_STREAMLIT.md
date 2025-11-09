# Deploy SNU Wellness Personas App on Streamlit Cloud

## Quick Deployment Steps

### 1. Push Code to GitHub
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Create New App:**
   - Click "New app" button
   - Select your repository: `Sushant-kumar-pal/FOODING-PEROSONA-PREDICTION`
   - Branch: `main`
   - Main file path: `project.py`
   - Click "Deploy"

3. **Wait for Deployment:**
   - Streamlit will install dependencies from `requirements.txt`
   - Build process takes 2-5 minutes
   - You'll get a shareable URL like: `https://your-app-name.streamlit.app`

### 3. Share Your App
Once deployed, you'll receive a permanent URL that you can share with anyone!

## Important Files for Deployment

✅ **Already Configured:**
- `project.py` - Main Streamlit app
- `requirements.txt` - All dependencies listed
- `.streamlit/config.toml` - App configuration
- `outputs/wellness_labeled.csv` - Dataset
- Model files: `kmeans.pkl`, `scaler.pkl`, `mms.pkl`
- Images folder with campus backgrounds

## Troubleshooting

### If deployment fails:
1. Check that all files are pushed to GitHub
2. Verify `requirements.txt` has all packages
3. Check Streamlit Cloud logs for errors
4. Ensure Python version compatibility (3.8-3.11 recommended)

### If data files are missing:
Make sure these files are in the repository:
- `outputs/wellness_labeled.csv`
- `kmeans.pkl`, `scaler.pkl`, `mms.pkl`
- All images in `/images` folder

## Your Repository
https://github.com/Sushant-kumar-pal/FOODING-PEROSONA-PREDICTION

## Support
- Streamlit Docs: https://docs.streamlit.io/
- Community Forum: https://discuss.streamlit.io/
