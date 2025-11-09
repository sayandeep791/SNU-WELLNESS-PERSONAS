# SNU Wellness Personas - Student Lifestyle Clustering

## 🎯 Project Overview
Machine learning application that identifies distinct wellness and nutrition personas among university students using K-Means clustering.

## 🌐 Live Demo
🔗 **[View Live App](https://your-app-will-be-here.streamlit.app)** _(Update after deployment)_

## 📊 Features
- **Interactive 3D/2D PCA Visualizations** - Explore cluster separation
- **Radar Charts** - Compare personas across features
- **Persona Predictions** - Get personalized wellness recommendations
- **AI-Powered Insights** - Detailed analysis for each cluster
- **Professional Reports** - Downloadable PDF documentation

## 🚀 Quick Start

### Local Installation
```bash
# Clone repository
git clone https://github.com/Sushant-kumar-pal/FOODING-PEROSONA-PREDICTION.git
cd FOODING-PEROSONA-PREDICTION

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run project.py
```

### Deploy to Streamlit Cloud
1. Fork this repository
2. Visit [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select this repository and `project.py` as main file
5. Click Deploy!

## 📁 Project Structure
```
├── project.py                          # Main Streamlit application
├── data.csv                            # Student survey data
├── outputs/
│   ├── kmeans.pkl                      # Trained K-Means model
│   ├── scaler.pkl                      # StandardScaler object
│   ├── mms.pkl                         # MinMaxScaler for radar charts
│   └── wellness_labeled.csv            # Labeled dataset
├── SNU_Wellness_Personas_Report.pdf    # Comprehensive project report
├── requirements.txt                    # Python dependencies
└── .streamlit/
    └── config.toml                     # Streamlit configuration
```

## 🛠️ Technologies Used
- **Python 3.13**
- **Streamlit** - Web framework
- **Scikit-learn** - Machine learning (K-Means, PCA)
- **Plotly** - Interactive visualizations
- **Pandas & NumPy** - Data processing

## 📈 Methodology
1. **Data Collection** - Student wellness survey (4 features)
2. **Preprocessing** - Cleaning, imputation, standardization
3. **Clustering** - K-Means (k=4) with PCA visualization
4. **Persona Development** - AI-powered labeling and insights
5. **Deployment** - Interactive Streamlit dashboard

## 🎓 Key Features Analyzed
- `eating_out_per_week` - Frequency of dining out
- `food_budget_per_meal_inr` - Meal spending (₹)
- `sweet_tooth_level` - Sweet preference (1-10)
- `weekly_hobby_hours` - Activity engagement

## 📄 Documentation
- **Full Report**: [SNU_Wellness_Personas_Report.pdf](SNU_Wellness_Personas_Report.pdf)
- **Deployment Guide**: [DEPLOY_ON_STREAMLIT.md](DEPLOY_ON_STREAMLIT.md)

## 👥 Authors
**Sushant Kumar Pal**  
Shiv Nadar University

## 📧 Contact
For questions or collaboration:
- GitHub: [@Sushant-kumar-pal](https://github.com/Sushant-kumar-pal)
- Repository: [FOODING-PEROSONA-PREDICTION](https://github.com/Sushant-kumar-pal/FOODING-PEROSONA-PREDICTION)

## 🎉 Acknowledgments
- Shiv Nadar University Data Science Team
- Student participants in wellness survey
- Streamlit community

---

**License**: MIT  
**Last Updated**: November 2025
