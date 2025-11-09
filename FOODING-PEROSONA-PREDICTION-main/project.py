import os
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import streamlit as st
from io import BytesIO
import base64
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib

warnings.filterwarnings("ignore")

# Configure Streamlit for better performance
st.set_page_config(
    page_title="SNU Wellness Personas",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add mobile/viewport meta and a small responsive CSS layer to improve behavior
# on narrow screens (phones/tablets). This complements existing styles.
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
/* Ensure images and plot containers scale on small screens */
img, .js-plotly-plot, .stImage > img {
    max-width: 100% !important;
    height: auto !important;
}

/* Make sidebar use full width on very small screens */
@media (max-width: 600px) {
    [data-testid="stSidebar"] {
        position: relative !important;
        width: 100% !important;
        transform: none !important;
    }
    .block-container, .stApp, .main {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
    .flex-grid {
        grid-template-columns: 1fr !important;
    }
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_default_csv(path):
    """Cached CSV loading."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def clean_and_impute(df, required_cols):
    """Clean and impute data with caching."""
    df_work = df.copy()
    col_map = {}
    for rc in required_cols:
        matches = [c for c in df_work.columns if c.lower().strip() == rc.lower().strip()]
        if matches:
            col_map[rc] = matches[0]
            continue
        parts = rc.split("_")
        matches = [c for c in df_work.columns if parts[0] in c.lower()]
        if matches:
            col_map[rc] = matches[0]
    if len(col_map) < len(required_cols):
        missing = [c for c in required_cols if c not in col_map]
        for m in missing:
            if m in df_work.columns:
                col_map[m] = m

    df_sel = df_work[[col_map[c] for c in required_cols if c in col_map]].copy()
    df_sel.columns = [c for c in required_cols if c in col_map]

    for col in df_sel.columns:
        df_sel[col] = df_sel[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        df_sel[col] = pd.to_numeric(df_sel[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    df_imp = pd.DataFrame(imputer.fit_transform(df_sel), columns=df_sel.columns)
    return df_imp, imputer


@st.cache_data(ttl=3600, show_spinner=False)
def train_kmeans(df_imp, n_clusters=4):
    """Cached KMeans training."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_imp)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, scaler, X_scaled, labels


@st.cache_data(ttl=3600, show_spinner=False)
def compute_pca(X_scaled, n_components=3):
    """Cached PCA computation."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X_scaled), pca


def assign_distinct_personas(centers_df, feature_names):
    """Assign distinct personas to clusters based on relative characteristics."""
    persona_options = [
        "Health-conscious",
        "Fast-food Lovers", 
        "Premium Eaters",
        "Sweet-tooth Enthusiasts",
        "Budget-conscious",
        "Social Eaters",
        "Balanced Lifestyle",
        "Active Foodies"
    ]
    
    # Calculate scores for each persona type for each cluster
    scores = []
    for idx, row in centers_df.iterrows():
        d = dict(zip(feature_names, row.values))
        eating_out = d.get("eating_out_per_week", 0)
        budget = d.get("food_budget_per_meal_inr", 0)
        sweet_tooth = d.get("sweet_tooth_level", 0)
        hobby_hours = d.get("weekly_hobby_hours", 0)
        
        cluster_scores = {}
        # Health-conscious: Low eating out, low sweet tooth, high hobby hours
        cluster_scores["Health-conscious"] = (6 - eating_out) * 0.3 + (3 - sweet_tooth) * 0.3 + hobby_hours * 0.4
        
        # Fast-food Lovers: High eating out, moderate-high sweet tooth
        cluster_scores["Fast-food Lovers"] = eating_out * 0.4 + sweet_tooth * 0.4 + (10 - hobby_hours) * 0.2
        
        # Premium Eaters: High budget, moderate eating out
        cluster_scores["Premium Eaters"] = (budget / 200) * 0.5 + eating_out * 0.3 + (budget / 200) * 0.2
        
        # Sweet-tooth Enthusiasts: Very high sweet tooth
        cluster_scores["Sweet-tooth Enthusiasts"] = sweet_tooth * 0.6 + eating_out * 0.2 + (10 - hobby_hours) * 0.2
        
        # Budget-conscious: Low budget, low eating out
        cluster_scores["Budget-conscious"] = (120 / max(budget, 1)) * 0.5 + (3 - eating_out) * 0.3 + (10 - sweet_tooth) * 0.2
        
        # Social Eaters: High eating out, moderate budget
        cluster_scores["Social Eaters"] = eating_out * 0.4 + (200 - abs(budget - 150)) / 200 * 0.4 + (10 - hobby_hours) * 0.2
        
        # Balanced Lifestyle: Moderate everything
        cluster_scores["Balanced Lifestyle"] = 10 - abs(eating_out - 3) - abs(budget - 150) / 50 - abs(sweet_tooth - 5) - abs(hobby_hours - 5)
        
        # Active Foodies: High hobby hours, moderate eating out
        cluster_scores["Active Foodies"] = hobby_hours * 0.4 + eating_out * 0.3 + (budget / 200) * 0.3
        
        scores.append(cluster_scores)
    
    # Assign personas to ensure uniqueness
    assigned = []
    used_personas = set()
    
    for cluster_idx, cluster_scores in enumerate(scores):
        # Sort personas by score for this cluster
        sorted_personas = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Find first unused persona
        for persona_name, score in sorted_personas:
            if persona_name not in used_personas:
                assigned.append(persona_name)
                used_personas.add(persona_name)
                break
        else:
            # If all are used, assign the highest scoring one
            assigned.append(sorted_personas[0][0])
    
    return assigned


def persona_label_from_center(center, feature_names):
    """Create distinct lifestyle personas based on cluster centers."""
    d = dict(zip(feature_names, center))
    eating_out = d.get("eating_out_per_week", 0)
    budget = d.get("food_budget_per_meal_inr", 0)
    sweet_tooth = d.get("sweet_tooth_level", 0)
    hobby_hours = d.get("weekly_hobby_hours", 0)
    
    # Health-conscious: Low eating out, low sweet tooth, high hobby hours
    if eating_out <= 2 and sweet_tooth <= 3 and hobby_hours >= 6:
        name = "Health-conscious"
    # Fast-food Lovers: High eating out, moderate-high sweet tooth
    elif eating_out >= 5 and sweet_tooth >= 5:
        name = "Fast-food Lovers"
    # Premium Eaters: High budget, moderate eating out
    elif budget >= 200 and eating_out >= 3:
        name = "Premium Eaters"
    # Sweet-tooth: Very high sweet tooth preference
    elif sweet_tooth >= 7:
        name = "Sweet-tooth Enthusiasts"
    # Budget-conscious: Low budget, low eating out
    elif budget <= 120 and eating_out <= 2:
        name = "Budget-conscious"
    # Social Eaters: High eating out, moderate budget
    elif eating_out >= 4 and 100 <= budget <= 200:
        name = "Social Eaters"
    # Balanced: Everything else
    else:
        name = "Balanced Lifestyle"
    return name


def create_radar_image(centers_norm_df, feature_labels):
    num_vars = len(feature_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, row in centers_norm_df.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f"Cluster {i}", linewidth=2)
        ax.fill(angles, values, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), feature_labels)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def get_video_recommendations(persona_name):
    """Get video recommendations based on persona type - Using popular cooking/nutrition channels."""
    video_recommendations = {
        "Health-conscious": [
            {"title": "Meal Prep 101 - Tasty", "url": "https://www.youtube.com/watch?v=Ul7ZO5S54nI", "platform": "YouTube"},
            {"title": "Easy Healthy Recipes - BuzzFeed", "url": "https://www.youtube.com/watch?v=gTGSbPYfulY", "platform": "YouTube"},
            {"title": "Nutrition Tips - TED-Ed", "url": "https://www.youtube.com/watch?v=Z7x1KuRahMY", "platform": "YouTube"}
        ],
        "Fast-food Lovers": [
            {"title": "Fast Food Secrets - Insider", "url": "https://www.youtube.com/watch?v=nGvzuf9cJNg", "platform": "YouTube"},
            {"title": "Making Popular Fast Foods - Tasty", "url": "https://www.youtube.com/watch?v=gTGSbPYfulY", "platform": "YouTube"},
            {"title": "Quick Meals Guide", "url": "https://www.youtube.com/watch?v=Ul7ZO5S54nI", "platform": "YouTube"}
        ],
        "Premium Eaters": [
            {"title": "Gordon Ramsay Cooking Tips", "url": "https://www.youtube.com/watch?v=qyL_cYxV6QA", "platform": "YouTube"},
            {"title": "Fine Dining Techniques - Bon Appetit", "url": "https://www.youtube.com/watch?v=1L7q5xxDp8M", "platform": "YouTube"},
            {"title": "Professional Cooking", "url": "https://www.youtube.com/watch?v=qyL_cYxV6QA", "platform": "YouTube"}
        ],
        "Sweet-tooth Enthusiasts": [
            {"title": "Dessert Recipes - Tasty", "url": "https://www.youtube.com/watch?v=gTGSbPYfulY", "platform": "YouTube"},
            {"title": "Easy Baking Ideas", "url": "https://www.youtube.com/watch?v=Ul7ZO5S54nI", "platform": "YouTube"},
            {"title": "Sweet Treats Guide", "url": "https://www.youtube.com/watch?v=nGvzuf9cJNg", "platform": "YouTube"}
        ],
        "Budget-conscious": [
            {"title": "Budget Meal Prep - BuzzFeed", "url": "https://www.youtube.com/watch?v=gTGSbPYfulY", "platform": "YouTube"},
            {"title": "Cheap & Healthy Meals", "url": "https://www.youtube.com/watch?v=Ul7ZO5S54nI", "platform": "YouTube"},
            {"title": "Student Cooking Tips", "url": "https://www.youtube.com/watch?v=Z7x1KuRahMY", "platform": "YouTube"}
        ],
        "Social Eaters": [
            {"title": "Party Food Ideas - Tasty", "url": "https://www.youtube.com/watch?v=gTGSbPYfulY", "platform": "YouTube"},
            {"title": "Group Meal Recipes", "url": "https://www.youtube.com/watch?v=Ul7ZO5S54nI", "platform": "YouTube"},
            {"title": "Entertaining Tips", "url": "https://www.youtube.com/watch?v=nGvzuf9cJNg", "platform": "YouTube"}
        ],
        "Balanced Lifestyle": [
            {"title": "Balanced Diet Guide - TED-Ed", "url": "https://www.youtube.com/watch?v=Z7x1KuRahMY", "platform": "YouTube"},
            {"title": "Healthy Lifestyle Tips", "url": "https://www.youtube.com/watch?v=Ul7ZO5S54nI", "platform": "YouTube"},
            {"title": "Nutrition Basics", "url": "https://www.youtube.com/watch?v=gTGSbPYfulY", "platform": "YouTube"}
        ],
        "Active Foodies": [
            {"title": "Fitness Nutrition - Tasty", "url": "https://www.youtube.com/watch?v=gTGSbPYfulY", "platform": "YouTube"},
            {"title": "Pre-Workout Meals", "url": "https://www.youtube.com/watch?v=Ul7ZO5S54nI", "platform": "YouTube"},
            {"title": "Athlete Meal Prep", "url": "https://www.youtube.com/watch?v=Z7x1KuRahMY", "platform": "YouTube"}
        ]
    }
    
    # Find matching persona
    for key in video_recommendations:
        if key.lower() in persona_name.lower():
            return video_recommendations[key]
    
    # Default recommendations
    return video_recommendations["Balanced Lifestyle"]


@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
def scrape_images_from_unsplash(persona_name, food_items=None, num_images=20):
    """Cached image URL retrieval - returns multiple images from Unsplash and food sites."""
    # Massive collection of food and campus images - 50+ URLs
    all_images = [
        # Food variety images
        "https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1571091718767-18b5b1457add?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1551024506-0bccd828d307?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1495521821757-a1efb6729352?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1512058564366-18510be2db19?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1509440159596-0249088772ff?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1498837167922-ddd27525d352?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1482049016688-2d3e1b311543?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1476224203421-9ac39bcb3327?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1455619452474-d2be8b1e70cd?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1484723091739-30a097e8f929?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1606787366850-de6330128bfc?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1529042410759-befb1204b468?w=800&h=600&fit=crop",
        # More diverse food
        "https://images.unsplash.com/photo-1555939594-58d7cb561ad1?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1546793665-c74683f339c1?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1563379926898-05f4575a45d8?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1547573854-74d2a71d0826?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1603073777339-f0ebfe2523ea?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1608039829572-78524f79c4c7?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1540914124281-342587941389?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1550547660-d9450f859349?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1506354666786-959d6d497f1a?w=800&h=600&fit=crop",
        # Healthy foods
        "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1511690656952-34342bb7c2f2?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1505576399279-565b52d4ac71?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1600850056064-a8b380df8395?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1559181567-c3190ca9959b?w=800&h=600&fit=crop",
        # Fast food
        "https://images.unsplash.com/photo-1571091655789-405eb7a3a3a8?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1603360946369-dc9bb6258143?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1594212699903-ec8a3eca50f5?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1585238341710-4412f4c3b3f7?w=800&h=600&fit=crop",
        # Desserts
        "https://images.unsplash.com/photo-1551024601-bec78aea704b?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1563805042-7684c019e1cb?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1488477181946-6428a0291777?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1587314168485-3236d6710814?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1571115177098-24ec42ed204d?w=800&h=600&fit=crop",
        # Campus/Student life
        "https://images.unsplash.com/photo-1523050854058-8df90110c9f1?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1541339907198-e08756dedf3f?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1498243691581-b145c3f54a5a?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1562774053-701939374585?w=800&h=600&fit=crop",
        "https://images.unsplash.com/photo-1524178232363-1fb2b075b655?w=800&h=600&fit=crop"
    ]
    
    # Return requested number of images (shuffle for variety)
    import random
    selected = random.sample(all_images, min(num_images, len(all_images)))
    return selected


@st.cache_data(ttl=86400, show_spinner=False)
def get_internet_image_url(persona_name, food_items=None):
    """Cached single image URL retrieval."""
    images = scrape_images_from_unsplash(persona_name, food_items, num_images=1)
    return images[0] if images else "https://images.unsplash.com/photo-1495521821757-a1efb6729352?w=800&h=600&fit=crop"


def process_persona_parallel(args):
    """Helper function for parallel persona processing."""
    center_row, required_cols, name, n_images, regenerate = args
    try:
        result = openai_summary_and_image(center_row, required_cols, name, n_images=n_images, regenerate=regenerate, debug=False)
        return name, result
    except Exception as e:
        return name, (None, None)


def generate_all_personas_parallel(centers_orig, required_cols, persona_names, n_images=2):
    """Generate all persona summaries and images in parallel."""
    # Convert centers_orig to list if it's a numpy array for hashing
    if hasattr(centers_orig, 'tolist'):
        centers_list = centers_orig.tolist()
    else:
        centers_list = centers_orig
    
    args_list = [(centers_list[i], required_cols, persona_names[i], n_images, False) 
                 for i in range(len(persona_names))]
    
    results = {}
    try:
        with ThreadPoolExecutor(max_workers=min(4, len(persona_names))) as executor:
            futures = {executor.submit(process_persona_parallel, args): args[2] for args in args_list}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    persona_name, result = future.result(timeout=30)
                    results[persona_name] = result
                except Exception as e:
                    results[name] = (None, None)
    except Exception as e:
        # Fallback to sequential processing if parallel fails
        for i, name in enumerate(persona_names):
            try:
                result = openai_summary_and_image(centers_list[i], required_cols, name, n_images=n_images, regenerate=False, debug=False)
                results[name] = result
            except Exception:
                results[name] = (None, None)
    
    return results


def create_pca_charts(X_scaled, labels, persona_names, colors_list):
    """Create PCA charts (caching handled at PCA computation level)."""
    # Compute PCA (these are cached)
    pca_data_3d, pca_3d = compute_pca(X_scaled, n_components=3)
    pca_data_2d, pca_2d = compute_pca(X_scaled, n_components=2)
    
    # Create 3D plot
    fig_3d = go.Figure()
    df_pca_3d = pd.DataFrame(pca_data_3d, columns=['pca1', 'pca2', 'pca3'])
    df_pca_3d['cluster'] = labels
    
    for cluster_id in sorted(df_pca_3d['cluster'].unique()):
        cluster_data = df_pca_3d[df_pca_3d['cluster'] == cluster_id]
        persona_name = persona_names[cluster_id] if cluster_id < len(persona_names) else f"Cluster {cluster_id}"
        
        fig_3d.add_trace(go.Scatter3d(
            x=cluster_data['pca1'],
            y=cluster_data['pca2'],
            z=cluster_data['pca3'],
            mode='markers',
            name=f'Cluster {cluster_id}: {persona_name}',
            marker=dict(
                size=8,
                color=colors_list[cluster_id % len(colors_list)],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hovertemplate=f'<b>{persona_name}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}<extra></extra>'
        ))
    
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            zaxis_title='Third Principal Component',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        title="3D Cluster Visualization",
        height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Create 2D plot
    df_pca_2d = pd.DataFrame(pca_data_2d, columns=['pca1_2d', 'pca2_2d'])
    df_pca_2d['cluster'] = labels
    
    fig_2d = px.scatter(
        df_pca_2d, 
        x='pca1_2d', 
        y='pca2_2d',
        color='cluster',
        hover_data=['cluster'],
        labels={'pca1_2d': 'First Principal Component', 'pca2_2d': 'Second Principal Component'},
        title="2D PCA Visualization",
        color_continuous_scale='Viridis'
    )
    
    fig_2d.update_traces(marker=dict(size=10, opacity=0.7))
    fig_2d.update_layout(height=500, showlegend=True)
    
    return fig_3d, fig_2d, df_pca_3d, df_pca_2d


def create_radar_chart_cached(centers_norm_df, required_cols, persona_names):
    """Create radar chart (caching not needed as it's fast)."""
    fig_radar = go.Figure()
    colors_radar = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    
    for idx, (cluster_id, row) in enumerate(centers_norm_df.iterrows()):
        persona_name = persona_names[cluster_id] if cluster_id < len(persona_names) else f"Cluster {cluster_id}"
        values = row.tolist()
        values += values[:1]  # Close the loop
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=required_cols + [required_cols[0]],
            fill='toself',
            name=f'Cluster {cluster_id}: {persona_name}',
            line_color=colors_radar[cluster_id % len(colors_radar)],
            opacity=0.7
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Interactive Radar Chart - Feature Comparison",
        height=500
    )
    
    return fig_radar


def create_persona_preview_image(center_vals, feature_names, persona_name, foods=None):
    """Create a simple preview image for a persona when OpenAI image generation is not available.
    Returns a BytesIO PNG buffer."""
    try:
        vals = np.array(center_vals, dtype=float)
        # Normalize to 0-1 for plotting
        if vals.max() == vals.min():
            vals_norm = np.zeros_like(vals)
        else:
            vals_norm = (vals - vals.min()) / (vals.max() - vals.min())

        num_vars = len(feature_names)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig = plt.figure(figsize=(7, 4), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

        # Radar subplot
        ax = fig.add_subplot(gs[0], polar=True)
        vals_plot = vals_norm.tolist()
        vals_plot += vals_plot[:1]
        ax.plot(angles, vals_plot, color="#D55E00", linewidth=2)
        ax.fill(angles, vals_plot, color="#D55E00", alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), feature_names)
        ax.set_ylim(0, 1)
        ax.set_title(persona_name, pad=12)

        # Text / foods subplot
        ax2 = fig.add_subplot(gs[1])
        ax2.axis("off")
        lines = [f"Persona: {persona_name}", ""]
        if foods:
            lines.append("Representative foods:")
            for f in foods[:6]:
                lines.append(f"- {f}")
        else:
            lines.append("Representative foods: (not available)")

        ax2.text(0, 1, "\n".join(lines), va="top", fontsize=10)

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None


def openai_summary_and_image(center_vals, feature_names, persona_name, n_images=1, regenerate=False, debug=False):
    """
    Returns (summary_dict_or_None, image_bytes_io_or_None).
    summary_dict has keys: title, bullets (list), program_ideas (list), food_items (list), diet_tips, advice
    """
    # Cost-tolerant mode: always use fallback if set in session state
    cost_tolerant = False
    try:
        cost_tolerant = st.session_state.get('cost_tolerant', False)
    except Exception:
        cost_tolerant = False

    api_key = None
    if not cost_tolerant:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

    # If cost-tolerant or no API key, return a structured fallback summary and a generated preview image.
    if cost_tolerant or not api_key:
        # Build a simple fallback summary dict with persona-specific content
        d = dict(zip(feature_names, center_vals))
        eating_out = d.get("eating_out_per_week", 0)
        budget = d.get("food_budget_per_meal_inr", 0)
        sweet_tooth = d.get("sweet_tooth_level", 0)
        hobby_hours = d.get("weekly_hobby_hours", 0)
        
        if "Health-conscious" in persona_name:
            foods = ["quinoa salad bowls", "grilled chicken wraps", "green smoothies", "protein bowls"]
            bullets = ["Prioritizes nutritious meals", "Low eating-out frequency", "High activity levels", "Prefers whole foods", "Values meal prep"]
            program_ideas = ["Wellness Meal Plan", "Fitness Nutrition Program"]
        elif "Fast-food Lovers" in persona_name:
            foods = ["burger combos", "pizza slices", "fried chicken", "loaded fries"]
            bullets = ["Frequently eats out (5+ times/week)", "Prefers convenience over nutrition", "High sweet tooth preference", "Values speed and taste", "Budget-conscious fast food choices"]
            program_ideas = ["Quick Meal Deals", "Fast-Food Rewards Program"]
        elif "Premium Eaters" in persona_name:
            foods = ["gourmet sandwiches", "artisan pizzas", "premium salads", "specialty coffees"]
            bullets = ["Higher food budget (₹200+)", "Values quality ingredients", "Moderate eating frequency", "Prefers premium options", "Willing to pay for quality"]
            program_ideas = ["Premium Dining Experience", "Gourmet Meal Subscription"]
        elif "Sweet-tooth" in persona_name:
            foods = ["dessert bowls", "sweet shakes", "pastries", "ice cream sundaes"]
            bullets = ["Very high sweet preference (7+)", "Loves desserts and treats", "May need healthier alternatives", "Sweet-focused meal choices", "Dessert-first mindset"]
            program_ideas = ["Sweet Treats Club", "Healthy Dessert Options"]
        elif "Budget-conscious" in persona_name:
            foods = ["economy meals", "value combos", "simple rice bowls", "affordable snacks"]
            bullets = ["Low food budget (≤₹120)", "Minimal eating out", "Value-focused choices", "Price-sensitive decisions", "Home-cooked meal preference"]
            program_ideas = ["Budget Meal Plans", "Student Discount Program"]
        elif "Social Eaters" in persona_name:
            foods = ["sharing platters", "group meals", "social dining options", "communal dishes"]
            bullets = ["High eating-out frequency", "Social dining preference", "Moderate budget", "Group meal activities", "Food as social experience"]
            program_ideas = ["Group Dining Deals", "Social Eating Events"]
        elif "Active Foodies" in persona_name:
            foods = ["protein-rich meals", "energy bowls", "post-workout snacks", "nutritious wraps"]
            bullets = ["High activity levels", "Focus on nutrition and energy", "Moderate eating frequency", "Performance-oriented food choices", "Balanced active lifestyle"]
            program_ideas = ["Active Lifestyle Meal Plans", "Fitness Nutrition Program"]
        else:  # Balanced
            foods = ["balanced meal combos", "variety plates", "mixed cuisines", "flexible options"]
            bullets = ["Moderate eating patterns", "Balanced preferences", "Flexible food choices", "Varied lifestyle", "Adaptable to different options"]
            program_ideas = ["Flexible Meal Plans", "Variety Program"]
        
        summary_dict = {
            "title": persona_name,
            "bullets": bullets,
            "program_ideas": program_ideas,
            "food_items": foods,
            "diet_tips": [
                "Focus on balanced nutrition",
                "Stay hydrated throughout the day",
                "Plan meals ahead when possible"
            ],
            "advice": [
                "Tailor messaging to lifestyle preferences",
                "Offer flexible options for different needs"
            ]
        }
        # create a simple preview image
        img_buf = create_persona_preview_image(center_vals, feature_names, persona_name, foods=foods)
        return summary_dict, img_buf

    raw_text = None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Ask the model to return strict JSON for easy parsing and include diet tips/advice
        prompt = (
            f"You are an expert in campus consumer segmentation and student nutrition.\n"
            f"Produce a JSON object with these keys:"
            f"title (short persona title), bullets (3-5 short actionable bullets), program_ideas (array of 2 short program names),"
            f"food_items (array of 4 short food item phrases), diet_tips (3 short evidence-based tips), advice (2 concise communication lines).")
        prompt += f"\nOnly output valid JSON — no extra commentary.\nPersona name: {persona_name}.\nFeatures (means):\n"
        for f, v in zip(feature_names, center_vals):
            prompt += f"- {f}: {v:.2f}\n"

        # Allow the UI to request multiple images; return only one for backwards compat unless changed

        chat_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        raw = chat_resp.choices[0].message.content.strip()
        raw_text = raw

        # Try to extract JSON from the model output
        summary_dict = None
        try:
            # sometimes the model returns code fences — strip them
            if raw.startswith("```"):
                # remove first and last fence
                raw = raw.strip().strip("`\n")
            summary_dict = json.loads(raw)
        except Exception:
            # fallback: try to locate JSON substring
            import re

            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    summary_dict = json.loads(m.group(0))
                except Exception:
                    summary_dict = None

        # If parsing failed, create a simple fallback textual summary
        if not summary_dict:
            summary_text = raw.replace("\n", " ")[:800]
            summary_dict = {
                "title": persona_name,
                "bullets": [summary_text],
                "program_ideas": [],
                "food_items": [],
            }
        
        # Build an image prompt from food items
        foods = summary_dict.get("food_items") or []
        if not foods:
            # fallback heuristics
            d = dict(zip(feature_names, center_vals))
            if d.get("sweet_tooth_level", 0) >= 6:
                foods = ["dessert bowls", "sweet shakes", "fruit yoghurt"]
            elif d.get("eating_out_per_week", 0) >= 4:
                foods = ["burger wraps", "quick noodles", "street-style bowls"]
            else:
                foods = ["salad bowls", "grilled sandwiches", "smoothie"]

        image_prompt = (
            f"Photorealistic top-down photo of a university food spread representing the persona '{persona_name}': "
            f"include {', '.join(foods[:3])}. Bright, appetizing, diverse plating, natural lighting. Use shallow depth-of-field and vibrant colors."
        )

        # Image caching directory
        images_dir = os.path.join(os.getcwd(), "outputs", "images")
        os.makedirs(images_dir, exist_ok=True)
        safe_name = "".join([c if c.isalnum() or c in (' ', '-', '_') else '_' for c in persona_name]).strip().replace(' ', '_')

        # If not regenerating and cached images exist, load them
        cached_imgs = []
        if not regenerate:
            for i in range(max(1, n_images)):
                p = os.path.join(images_dir, f"{safe_name}_{i}.png")
                if os.path.exists(p):
                    try:
                        cached_imgs.append(BytesIO(open(p, "rb").read()))
                    except Exception:
                        pass
        if cached_imgs:
            # return first image for backward-compat and let caller access more by reading files
            if debug:
                return summary_dict, cached_imgs[0], raw_text
            return summary_dict, cached_imgs[0]

        # Attempt image generation (n_images variations)
        try:
            img_resp = client.images.generate(
                prompt=image_prompt, 
                n=n_images, 
                size="512x512",
                response_format="b64_json"
            )
            imgs = []
            for idx, data0 in enumerate(img_resp.data):
                if not data0:
                    continue
                b64_json = data0.b64_json if hasattr(data0, 'b64_json') else None
                img_bytes = None
                if b64_json:
                    try:
                        img_bytes = base64.b64decode(b64_json)
                    except Exception:
                        img_bytes = None

                url = data0.url if hasattr(data0, 'url') else None
                if img_bytes is None and url:
                    try:
                        r = requests.get(url, timeout=10)
                        if r.status_code == 200:
                            img_bytes = r.content
                    except Exception:
                        img_bytes = None

                if img_bytes:
                    # write separate cached files per variation
                    img_path = os.path.join(images_dir, f"{safe_name}_{idx}.png")
                    try:
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        imgs.append(BytesIO(img_bytes))
                    except Exception:
                        try:
                            imgs.append(BytesIO(img_bytes))
                        except Exception:
                            pass

            if imgs:
                if debug:
                    return summary_dict, imgs[0], raw_text
                return summary_dict, imgs[0]
            if debug:
                return summary_dict, None, raw_text
            return summary_dict, None
        except Exception:
            if debug:
                return summary_dict, None, raw_text
            return summary_dict, None

    except Exception:
        if debug:
            return None, None, raw_text
        return None, None


def main():

    # Sidebar: App info and persona predictor only (no API key input)

    st.sidebar.image("https://img.icons8.com/color/96/healthy-eating.png", width=80)
    st.sidebar.markdown("""
    ## SNU Wellness Personas
    <span style='color:#4F8A8B;font-size:16px;'>Discover student lifestyle clusters and wellness insights.</span>
    """, unsafe_allow_html=True)
    
    # The app always uses the bundled `data.csv` that lives next to this script.
    # We intentionally remove the upload option so predictions always come from
    # the repository-provided data.csv (persistent, consistent behavior for users).
    default_csv_path = os.path.join(os.path.dirname(__file__), "data.csv")
    st.sidebar.markdown(f"**Default CSV path:** `{default_csv_path}`")
    
    cost_tolerant = st.sidebar.checkbox("Cost-tolerant mode", value=True)
    st.session_state['cost_tolerant'] = cost_tolerant

    # Add comprehensive CSS for entire interface - LIGHT THEME ENFORCED
    st.markdown("""
    <style>
    /* FORCE LIGHT THEME - MAXIMUM VISIBILITY */
    * {
        color: #000000 !important;
    }
    
    .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #ffffff !important;
        background-image: 
            linear-gradient(rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.95)),
            url('https://images.unsplash.com/photo-1523050854058-8df90110c9f1?w=1920&h=1080&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
        background-image: 
            linear-gradient(rgba(240, 242, 246, 0.98), rgba(240, 242, 246, 0.98)),
            url('https://images.unsplash.com/photo-1541339907198-e08756dedf3f?w=800&h=1200&fit=crop');
        background-size: cover;
        background-position: center;
    }
    
    /* Ensure all text is black */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, td, th {
        color: #000000 !important;
    }
    
    /* White background for all containers */
    .element-container, .stMarkdown, div[data-testid="column"] {
        background-color: transparent !important;
    }
    
    /* 3D Buttons - All Streamlit buttons */
    button, .stButton > button, [data-testid="stButton"] button {
        transform-style: preserve-3d;
        transform: translateZ(0);
        position: relative;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 4px 8px rgba(0,0,0,0.2),
            0 8px 16px rgba(0,0,0,0.1),
            inset 0 1px 0 rgba(255,255,255,0.3),
            inset 0 -2px 0 rgba(0,0,0,0.2) !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    button:hover, .stButton > button:hover, [data-testid="stButton"] button:hover {
        transform: translateY(-3px) translateZ(20px) rotateX(5deg) !important;
        box-shadow: 
            0 8px 16px rgba(102, 126, 234, 0.4),
            0 16px 32px rgba(118, 75, 162, 0.3),
            inset 0 1px 0 rgba(255,255,255,0.4),
            inset 0 -3px 0 rgba(0,0,0,0.2) !important;
    }
    
    button:active, .stButton > button:active, [data-testid="stButton"] button:active {
        transform: translateY(0px) translateZ(5px) rotateX(-2deg) !important;
        box-shadow: 
            0 2px 4px rgba(0,0,0,0.3),
            inset 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    /* 3D Input Fields */
    input[type="text"], input[type="number"], textarea, select,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        transform-style: preserve-3d;
        transform: translateZ(10px);
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        background: rgba(255,255,255,0.95) !important;
        box-shadow: 
            0 4px 8px rgba(102, 126, 234, 0.2),
            inset 0 2px 4px rgba(0,0,0,0.05),
            0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    input:focus, textarea:focus, select:focus,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        transform: translateZ(20px) scale(1.02) !important;
        box-shadow: 
            0 8px 16px rgba(102, 126, 234, 0.4),
            inset 0 2px 4px rgba(0,0,0,0.05),
            0 0 0 5px rgba(102, 126, 234, 0.2) !important;
        border-color: #764ba2 !important;
    }
    
    /* 3D Sliders */
    .stSlider > div > div {
        transform-style: preserve-3d;
        transform: translateZ(5px);
    }
    
    .stSlider > div > div > div {
        box-shadow: 
            0 4px 8px rgba(0,0,0,0.2),
            inset 0 1px 0 rgba(255,255,255,0.3) !important;
    }
    
    /* 3D Checkboxes */
    .stCheckbox > label {
        transform-style: preserve-3d;
        transform: translateZ(5px);
    }
    
    .stCheckbox > label > div {
        box-shadow: 
            0 3px 6px rgba(0,0,0,0.2),
            inset 0 1px 0 rgba(255,255,255,0.3) !important;
        border-radius: 6px !important;
    }
    
    /* 3D File Uploader */
    .stFileUploader > div {
        transform-style: preserve-3d;
        transform: translateZ(10px);
        border: 3px dashed #667eea !important;
        border-radius: 12px !important;
        padding: 20px !important;
        background: rgba(102, 126, 234, 0.05) !important;
        box-shadow: 
            0 6px 12px rgba(102, 126, 234, 0.2),
            inset 0 2px 4px rgba(255,255,255,0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div:hover {
        transform: translateZ(20px) scale(1.02) !important;
        box-shadow: 
            0 10px 20px rgba(102, 126, 234, 0.3),
            inset 0 2px 4px rgba(255,255,255,0.4) !important;
    }
    
    /* 3D Cards and Containers */
    .resizable-container, .card-3d, div[style*="background"] {
        transform-style: preserve-3d;
        transform: translateZ(0);
        position: relative;
    }
    
    /* Flexible resizable containers with 3D effect */
    .resizable-container {
        position: relative;
        resize: both;
        overflow: auto;
        min-width: 400px;
        min-height: 300px;
        max-width: 100%;
        max-height: 800px;
        border: 3px solid #667eea !important;
        border-radius: 15px !important;
        padding: 20px;
        margin: 15px 0;
        background: rgba(255,255,255,0.98) !important;
        box-shadow: 
            0 8px 16px rgba(102, 126, 234, 0.3),
            0 16px 32px rgba(118, 75, 162, 0.2),
            inset 0 1px 0 rgba(255,255,255,0.5),
            inset 0 -2px 0 rgba(0,0,0,0.1) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        transform: perspective(1000px) rotateX(2deg) translateZ(0) !important;
    }
    
    .resizable-container:hover {
        transform: perspective(1000px) rotateX(0deg) translateZ(30px) scale(1.02) !important;
        box-shadow: 
            0 12px 24px rgba(102, 126, 234, 0.4),
            0 24px 48px rgba(118, 75, 162, 0.3),
            inset 0 1px 0 rgba(255,255,255,0.6),
            inset 0 -3px 0 rgba(0,0,0,0.1) !important;
        border-color: #764ba2 !important;
    }
    
    /* 3D Text Effects */
    h1, h2, h3, h4, h5, h6 {
        transform-style: preserve-3d;
        text-shadow: 
            2px 2px 4px rgba(0,0,0,0.3),
            4px 4px 8px rgba(0,0,0,0.2),
            0 0 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* 3D Dataframes */
    .dataframe {
        transform-style: preserve-3d;
        transform: perspective(800px) rotateX(1deg) translateZ(10px);
        box-shadow: 
            0 8px 16px rgba(0,0,0,0.2),
            0 16px 32px rgba(102, 126, 234, 0.1),
            inset 0 1px 0 rgba(255,255,255,0.3) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* 3D Expanders */
    .streamlit-expanderHeader {
        transform-style: preserve-3d;
        transform: translateZ(5px);
        box-shadow: 
            0 4px 8px rgba(0,0,0,0.15),
            inset 0 1px 0 rgba(255,255,255,0.3) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        transform: translateZ(15px) scale(1.02) !important;
        box-shadow: 
            0 6px 12px rgba(102, 126, 234, 0.3),
            inset 0 1px 0 rgba(255,255,255,0.4) !important;
    }
    
    /* 3D Sidebar */
    [data-testid="stSidebar"] {
        transform-style: preserve-3d;
        transform: perspective(1500px) rotateY(-2deg) translateZ(0);
        box-shadow: 
            -10px 0 30px rgba(0,0,0,0.2),
            inset 2px 0 0 rgba(102, 126, 234, 0.1) !important;
    }
    
    /* 3D Main Content */
    .main .block-container {
        transform-style: preserve-3d;
        transform: perspective(1500px) rotateY(1deg) translateZ(0);
    }
    
    /* 3D Images */
    img {
        transform-style: preserve-3d;
        transition: all 0.3s ease;
        border-radius: 12px;
        box-shadow: 
            0 6px 12px rgba(0,0,0,0.2),
            0 12px 24px rgba(102, 126, 234, 0.15) !important;
    }
    
    img:hover {
        transform: translateZ(30px) rotateY(5deg) rotateX(5deg) scale(1.05) !important;
        box-shadow: 
            0 12px 24px rgba(0,0,0,0.3),
            0 24px 48px rgba(102, 126, 234, 0.25) !important;
    }
    
    /* 3D Success/Info/Warning Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        transform-style: preserve-3d;
        transform: perspective(800px) rotateX(2deg) translateZ(15px);
        box-shadow: 
            0 6px 12px rgba(0,0,0,0.2),
            0 12px 24px rgba(102, 126, 234, 0.15),
            inset 0 1px 0 rgba(255,255,255,0.3) !important;
        border-radius: 12px !important;
        border: 2px solid !important;
    }
    
    /* 3D Plotly Charts Container */
    .js-plotly-plot {
        transform-style: preserve-3d;
        transform: perspective(1000px) rotateX(1deg) translateZ(10px);
        box-shadow: 
            0 8px 16px rgba(0,0,0,0.2),
            0 16px 32px rgba(102, 126, 234, 0.1) !important;
        border-radius: 15px !important;
        padding: 10px;
        background: rgba(255,255,255,0.95) !important;
    }
    
    /* 3D Download Buttons */
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%) !important;
    }
    
    [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-3px) translateZ(20px) rotateX(5deg) !important;
        box-shadow: 
            0 8px 16px rgba(76, 175, 80, 0.4),
            0 16px 32px rgba(69, 160, 73, 0.3) !important;
    }
    
    /* 3D Background Effect */
    .main {
        background: #ffffff !important;
        position: relative;
    }
    
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: #f5f7fa !important;
        pointer-events: none;
        z-index: -1;
    }
    
    /* 3D Parallax Effect on Scroll */
    @keyframes float3D {
        0%, 100% {
            transform: translateY(0px) translateZ(0px) rotateX(0deg);
        }
        50% {
            transform: translateY(-10px) translateZ(20px) rotateX(2deg);
        }
    }
    
    .float-3d {
        animation: float3D 6s ease-in-out infinite;
    }
    
    /* 3D Card Hover Effects */
    .card-3d, div[style*="background:linear-gradient"] {
        transform-style: preserve-3d;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .card-3d:hover, div[style*="background:linear-gradient"]:hover {
        transform: translateZ(40px) rotateY(5deg) rotateX(5deg) scale(1.02) !important;
        box-shadow: 
            0 20px 40px rgba(0,0,0,0.3),
            0 40px 80px rgba(102, 126, 234, 0.2) !important;
    }
    
    .resize-handle {
        position: absolute;
        bottom: 0;
        right: 0;
        width: 20px;
        height: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        cursor: nwse-resize;
        border-radius: 12px 0 12px 0;
        opacity: 0.7;
        transition: opacity 0.3s;
    }
    
    .resize-handle:hover {
        opacity: 1;
    }
    
    .resize-handle::after {
        content: '⤡';
        position: absolute;
        bottom: 2px;
        right: 2px;
        color: white;
        font-size: 12px;
    }
    
    /* Chart container wrapper */
    .chart-wrapper {
        width: 100%;
        height: 100%;
        min-height: 400px;
    }
    
    /* Flexible grid layout */
    .flex-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    .flex-item {
        min-width: 400px;
        resize: both;
        overflow: auto;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Responsive containers */
    .responsive-container {
        width: 100%;
        height: auto;
        min-height: 300px;
        position: relative;
    }
    
    @media (max-width: 768px) {
        .flex-grid {
            grid-template-columns: 1fr;
        }
        .flex-item {
            min-width: 100%;
        }
        .resizable-container {
            min-width: 100%;
        }
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    @keyframes rotate {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    
    .animated-card {
        animation: slideIn 0.6s ease-out;
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    
    /* Gradient animations */
    @keyframes gradientShift {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    .animated-gradient {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    /* Scroll animations */
    .fade-in {
        opacity: 0;
        animation: fadeInUp 0.8s ease forwards;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with SNU branding and animations
    st.markdown("""
    <div class="animated-gradient float-3d" style='padding:2em;border-radius:20px;margin-bottom:1.5em;transform-style:preserve-3d;transform:perspective(1500px) rotateX(5deg) translateZ(50px);box-shadow:0 20px 40px rgba(0,0,0,0.3), 0 40px 80px rgba(102, 126, 234, 0.2), inset 0 1px 0 rgba(255,255,255,0.3);background:linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;'>
        <div style='display:flex;justify-content:space-between;align-items:center;gap:20px;transform-style:preserve-3d;flex-wrap:wrap;'>
            <div style='display:flex;align-items:center;gap:20px;'>
                <img src='https://images.unsplash.com/photo-1498243691581-b145c3f54a5a?w=120&h=120&fit=crop' width='100' style='border-radius:50%;border:4px solid white;filter:drop-shadow(0 8px 16px rgba(0,0,0,0.3));animation: pulse 3s infinite;transform:translateZ(30px);'/>
                <div style='transform:translateZ(20px);'>
                    <h1 style='font-size:3rem;font-weight:900;color:#ffffff !important;margin:0;text-shadow:0 4px 8px rgba(0,0,0,0.4), 0 8px 16px rgba(0,0,0,0.3), 0 0 30px rgba(102, 126, 234, 0.5);animation: slideIn 0.8s ease-out;transform:translateZ(10px);'>🍽️ SNU Wellness Personas</h1>
                    <p style='color:#f0f0f0 !important;font-size:1.4rem;font-weight:600;margin:0.3em 0 0 0;text-shadow:0 2px 4px rgba(0,0,0,0.3), 0 0 20px rgba(255,255,255,0.2);animation: slideIn 1s ease-out;transform:translateZ(5px);'>Discover Student Lifestyle Clusters & Nutrition Insights</p>
                </div>
            </div>
            <div style='background:rgba(255,255,255,0.2);padding:1em;border-radius:12px;backdrop-filter:blur(10px);transform:translateZ(25px);'>
                <img src='https://images.unsplash.com/photo-1523050854058-8df90110c9f1?w=200&h=80&fit=crop' width='150' style='border-radius:8px;'/>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data (from sidebar upload or default) - with progress indicator
    start_time = time.time()
    # Always load the default CSV from the app folder
    df_raw = load_default_csv(default_csv_path)
    if df_raw is None:
        st.sidebar.warning(f"No default CSV found at `{default_csv_path}` — please add `data.csv` to the app folder.")
    else:
        st.sidebar.success(f"Loaded default CSV: `{os.path.basename(default_csv_path)}`")

    required_cols = [
        "eating_out_per_week",
        "food_budget_per_meal_inr",
        "sweet_tooth_level",
        "weekly_hobby_hours",
    ]


    if df_raw is not None:
        # Clean and impute data (cached)
        with st.spinner("Processing data (using cached results for faster loading)..."):
            df_imp, imputer = clean_and_impute(df_raw, required_cols)
        
        # Make data preview collapsible
        with st.expander("📋 Preview Data (Click to view)", expanded=False):
            st.dataframe(df_imp.head(10), use_container_width=True, hide_index=True)

        # Train KMeans (cached)
        with st.spinner("Training clustering model (using cached results for faster loading)..."):
            kmeans, scaler, X_scaled, labels = train_kmeans(df_imp, n_clusters=4)
        df_imp["cluster"] = labels
        
        # Calculate cluster centers and persona names first
        centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers_orig, columns=required_cols)
        # Use distinct persona assignment to ensure unique personas
        persona_names = assign_distinct_personas(centers_df, required_cols)
        
        # Enhanced cluster counts display
        cluster_counts = df_imp["cluster"].value_counts().sort_index()
        st.markdown("""
        <div style='margin-top:1.5em;margin-bottom:1em;'>
            <h3 style='color:#1a2636;font-size:1.5rem;font-weight:600;'>📊 Cluster Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a visual representation of cluster counts
        count_cols = st.columns(len(cluster_counts))
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        for idx, (cluster_id, count) in enumerate(cluster_counts.items()):
            with count_cols[idx]:
                persona_name = persona_names[cluster_id] if cluster_id < len(persona_names) else f"Cluster {cluster_id}"
                percentage = (count / len(df_imp)) * 100
                st.markdown(f"""
                <div style='background:linear-gradient(135deg, {colors[cluster_id % len(colors)]} 0%, {colors[(cluster_id+1) % len(colors)]} 100%);padding:1.5em;border-radius:12px;text-align:center;box-shadow:0 4px 8px rgba(0,0,0,0.1);'>
                    <h2 style='color:#fff;font-size:2.5rem;margin:0;font-weight:700;'>{count}</h2>
                    <p style='color:#fff;font-size:1rem;margin:0.3em 0 0 0;font-weight:500;'>{persona_name}</p>
                    <p style='color:#fff;font-size:0.9rem;margin:0.2em 0 0 0;opacity:0.9;'>{percentage:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

        # Comprehensive Analysis section
        # Provide a robust set of static analyses (summary stats, distributions, correlations,
        # PCA variance, clustering metrics and cross-tabs) that are less likely to behave
        # inconsistently across deployments than fully interactive Plotly charts.
        with st.expander("🔎 Data Analysis & Diagnostics", expanded=False):
            st.markdown("""
            <div style='margin-bottom:0.5em;'>
                <h3 style='color:#1a2636;font-size:1.2rem;font-weight:600;'>Dataset summary & diagnostics</h3>
                <p style='color:#555;margin-top:0.2em;'>Overview statistics, missing values, distributions and clustering diagnostics derived from the loaded dataset.</p>
            </div>
            """, unsafe_allow_html=True)

            # Basic summary
            st.subheader("Basic dataset info")
            st.write(f"Rows: {len(df_raw)} — Features: {len(df_raw.columns)}")
            try:
                st.write("**Missing values (per column)**")
                mv = df_raw.isnull().sum()
                st.dataframe(mv[mv > 0].sort_values(ascending=False))
            except Exception:
                pass

            # Numeric summary from imputed dataframe
            st.subheader("Descriptive statistics (numeric features)")
            numeric_cols = df_imp.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(df_imp[numeric_cols].describe().T.round(3))

            # Plots helper
            def fig_to_bytes(fig):
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                return buf

            # Distributions for required columns
            st.subheader("Feature distributions")
            for col in required_cols:
                try:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.histplot(df_imp[col].dropna(), kde=True, ax=ax, color='#667eea')
                    ax.set_title(f"Distribution: {col}")
                    ax.set_ylabel("count")
                    buf = fig_to_bytes(fig)
                    st.image(buf, use_column_width=True)
                except Exception:
                    st.write(f"Could not plot distribution for {col}")

            # Boxplots
            st.subheader("Boxplots (outliers)")
            try:
                fig, axes = plt.subplots(1, len(required_cols), figsize=(4 * len(required_cols), 3))
                if len(required_cols) == 1:
                    axes = [axes]
                for ax, col in zip(axes, required_cols):
                    sns.boxplot(y=df_imp[col], ax=ax, color='#764ba2')
                    ax.set_title(col)
                buf = fig_to_bytes(fig)
                st.image(buf, use_column_width=True)
            except Exception:
                pass

            # Correlation heatmap
            st.subheader("Correlation matrix")
            try:
                corr = df_imp.corr()
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', ax=ax)
                ax.set_title('Feature correlation (imputed data)')
                buf = fig_to_bytes(fig)
                st.image(buf, use_column_width=True)
            except Exception:
                st.write("Correlation matrix could not be computed.")

            # PCA explained variance (stable scalar summary)
            st.subheader("PCA (variance explained)")
            try:
                pca_n = min(4, X_scaled.shape[1])
                from sklearn.decomposition import PCA as SKPCA
                pca_model = SKPCA(n_components=pca_n, random_state=42)
                pca_model.fit(X_scaled)
                evr = pca_model.explained_variance_ratio_
                evr_df = pd.DataFrame({'component': [f'PC{i+1}' for i in range(len(evr))], 'explained_variance_ratio': evr})
                st.dataframe(evr_df.round(3))
                # Scree plot
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(range(1, len(evr)+1), evr, marker='o')
                ax.set_xlabel('PC')
                ax.set_ylabel('Explained variance ratio')
                ax.set_title('Scree plot')
                buf = fig_to_bytes(fig)
                st.image(buf, use_column_width=True)
            except Exception:
                st.write("PCA could not be computed")

            # Clustering diagnostics
            st.subheader("Clustering diagnostics")
            try:
                if len(set(labels)) > 1 and len(X_scaled) > len(set(labels)):
                    sil = silhouette_score(X_scaled, labels)
                    db = davies_bouldin_score(X_scaled, labels)
                    st.write(f"Silhouette score: **{sil:.3f}**")
                    st.write(f"Davies-Bouldin score: **{db:.3f}**")
                else:
                    st.write("Not enough clusters/samples to compute clustering diagnostics.")
            except Exception:
                st.write("Clustering diagnostics unavailable")

            # Cluster counts and simple crosstabs
            st.subheader("Cluster counts & budget cross-tab")
            try:
                st.bar_chart(cluster_counts)
                # budget buckets
                df_imp['budget_bucket'] = pd.cut(df_imp['food_budget_per_meal_inr'], bins=[-1,99,149,199,499,10000], labels=['<=99','100-149','150-199','200-499','500+'])
                ct = pd.crosstab(df_imp['cluster'], df_imp['budget_bucket'])
                st.dataframe(ct)
            except Exception:
                st.write("Could not compute cluster cross-tabs")

            # Save a copy of analysis outputs for download
            try:
                analysis_out = os.path.join(os.getcwd(), 'outputs')
                os.makedirs(analysis_out, exist_ok=True)
                evr_df.to_csv(os.path.join(analysis_out, 'pca_explained_variance.csv'), index=False)
                ct.to_csv(os.path.join(analysis_out, 'cluster_budget_crosstab.csv'))
                st.success(f"Analysis artifacts saved to: {analysis_out}")
            except Exception:
                pass

        # Enhanced cluster centers display with persona names - make it collapsible
        with st.expander("📈 Cluster Centers (Average Values)", expanded=False):
            # Create an enhanced dataframe with persona names
            centers_display = centers_df.copy()
            centers_display.insert(0, 'Persona', persona_names)
            centers_display.insert(1, 'Cluster ID', range(len(persona_names)))
            st.dataframe(centers_display.round(2), use_container_width=True, hide_index=True)
        
        # Add a detailed breakdown for each cluster
        st.markdown("""
        <div style='margin-top:1.5em;margin-bottom:1em;'>
            <h3 style='color:#1a2636;font-size:1.5rem;font-weight:600;'>📋 Detailed Cluster Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display each cluster's information in cards
        info_cols = st.columns(2)
        for idx, (cluster_id, row) in enumerate(centers_df.iterrows()):
            with info_cols[idx % 2]:
                persona_name = persona_names[cluster_id]
                count = cluster_counts.get(cluster_id, 0)
                percentage = (count / len(df_imp)) * 100 if len(df_imp) > 0 else 0
                
                st.markdown(f"""
                <div class="card-3d float-3d" style='background:linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);border-radius:15px;padding:1.5em;margin-bottom:1em;border:3px solid #667eea;transform-style:preserve-3d;transform:perspective(1000px) rotateX(2deg) translateZ(20px);box-shadow:0 8px 16px rgba(0,0,0,0.15), 0 16px 32px rgba(102, 126, 234, 0.1), inset 0 1px 0 rgba(255,255,255,0.4), inset 0 -2px 0 rgba(0,0,0,0.1);'>
                    <h4 style='color:#667eea;font-size:1.3rem;margin:0 0 0.5em 0;font-weight:700;text-shadow:2px 2px 4px rgba(0,0,0,0.2), 0 0 10px rgba(102, 126, 234, 0.3);transform:translateZ(10px);'>Cluster {cluster_id}: {persona_name}</h4>
                    <div style='background:#f0f4f8;padding:0.8em;border-radius:10px;margin:0.5em 0;transform:translateZ(5px);box-shadow:inset 0 2px 4px rgba(0,0,0,0.05), 0 2px 4px rgba(102, 126, 234, 0.1);'>
                        <p style='margin:0.3em 0;color:#333;text-shadow:1px 1px 2px rgba(255,255,255,0.8);'><strong>Students:</strong> {count} ({percentage:.1f}%)</p>
                        <p style='margin:0.3em 0;color:#333;text-shadow:1px 1px 2px rgba(255,255,255,0.8);'><strong>Eating Out:</strong> {row['eating_out_per_week']:.1f} times/week</p>
                        <p style='margin:0.3em 0;color:#333;text-shadow:1px 1px 2px rgba(255,255,255,0.8);'><strong>Budget:</strong> ₹{row['food_budget_per_meal_inr']:.0f} per meal</p>
                        <p style='margin:0.3em 0;color:#333;text-shadow:1px 1px 2px rgba(255,255,255,0.8);'><strong>Sweet Tooth:</strong> {row['sweet_tooth_level']:.1f}/10</p>
                        <p style='margin:0.3em 0;color:#333;text-shadow:1px 1px 2px rgba(255,255,255,0.8);'><strong>Hobby Hours:</strong> {row['weekly_hobby_hours']:.1f} hours/week</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        mms = MinMaxScaler()
        centers_norm = mms.fit_transform(centers_orig)
        centers_norm_df = pd.DataFrame(centers_norm, columns=required_cols)

        # Interactive Radar Chart disabled
        st.markdown("""
        <div style='margin-top:2em;margin-bottom:1em;'>
            <h3 style='color:#1a2636;font-size:1.5rem;font-weight:600;'>📊 Feature Comparison - Interactive Radar Chart (disabled)</h3>
            <p style='color:#555;'>Radar chart feature comparison has been disabled because it caused inconsistent UI behavior. The cluster centers table and persona cards remain available.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Traditional radar chart removed to save space - interactive chart above is sufficient
        
        # Persona cards with LLM summary and images
        st.markdown("""
        <div style='margin-top:2em;margin-bottom:1em;'>
            <h2 style='color:#1a2636;font-size:2rem;font-weight:700;border-bottom:3px solid #667eea;padding-bottom:0.5em;'>🌟 Lifestyle Personas</h2>
            <p style='color:#555;font-size:1.1rem;'>Explore distinct student lifestyle clusters with detailed insights and food recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Pre-generate all persona summaries and images in parallel
        with st.spinner("Loading persona insights..."):
            try:
                persona_results = generate_all_personas_parallel(centers_orig, required_cols, persona_names, n_images=2)
            except Exception as e:
                st.warning(f"Parallel processing had issues, using sequential fallback: {str(e)}")
                persona_results = {}
                for i, name in enumerate(persona_names):
                    try:
                        result = openai_summary_and_image(centers_orig[i], required_cols, name, n_images=2, regenerate=False, debug=False)
                        persona_results[name] = result
                    except Exception as ex:
                        persona_results[name] = (None, None)
        
        # Create persona cards with cluster statistics
        for i, (name, center_row) in enumerate(zip(persona_names, centers_orig)):
            # Get cluster statistics
            cluster_data = df_imp[df_imp['cluster'] == i]
            cluster_count = len(cluster_data)
            cluster_pct = (cluster_count / len(df_imp)) * 100 if len(df_imp) > 0 else 0
            
            # Enhanced persona card header with 3D statistics
            st.markdown(f"""
            <div class="card-3d float-3d" style='background:linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);border-radius:20px;padding:1.5em;margin-bottom:1em;transform-style:preserve-3d;transform:perspective(1200px) rotateX(3deg) translateZ(30px);box-shadow:0 12px 24px rgba(0,0,0,0.2), 0 24px 48px rgba(102, 126, 234, 0.15), inset 0 1px 0 rgba(255,255,255,0.5), inset 0 -3px 0 rgba(0,0,0,0.1);border-left:6px solid #667eea;border-top:3px solid rgba(102, 126, 234, 0.3);'>
                <div style='display:flex;justify-content:space-between;align-items:center;transform-style:preserve-3d;'>
                    <div style='transform:translateZ(15px);'>
                        <h3 style='color:#1a2636;font-size:1.5rem;margin:0 0 0.3em 0;text-shadow:2px 2px 4px rgba(0,0,0,0.2), 0 0 15px rgba(102, 126, 234, 0.3);'>🌟 {name}</h3>
                        <p style='color:#555;margin:0;font-size:0.95rem;text-shadow:1px 1px 2px rgba(255,255,255,0.8);'>Cluster {i} • {cluster_count} students ({cluster_pct:.1f}%)</p>
                    </div>
                    <div style='text-align:right;transform:translateZ(20px);'>
                        <div style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);color:#fff;padding:0.5em 1em;border-radius:12px;font-weight:600;box-shadow:0 6px 12px rgba(102, 126, 234, 0.4), inset 0 1px 0 rgba(255,255,255,0.3), inset 0 -2px 0 rgba(0,0,0,0.2);transform:translateZ(10px);'>
                            {cluster_count}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"📊 View Details: {name}", expanded=False):
                # Controls: show all images toggle and regenerate button
                cols = st.columns((1, 1))
                show_all = st.checkbox("Show all image variations", key=f"show_all_{i}")
                regen_clicked = cols[1].button("Regenerate images", key=f"regen_btn_{i}")

                # If regenerate was clicked, set a session flag for this persona
                if regen_clicked:
                    st.session_state[f"regen_{i}"] = True
                regenerate = st.session_state.pop(f"regen_{i}", False)

                # Use cached results if available, otherwise generate
                if regenerate:
                    with st.spinner("Regenerating images and summary — this may take a few seconds..."):
                        summary_res = openai_summary_and_image(center_row, required_cols, name, n_images=2, regenerate=True, debug=True)
                    if isinstance(summary_res, tuple) and len(summary_res) == 3:
                        summary_dict, img_buf, raw_text = summary_res
                    else:
                        summary_dict, img_buf = summary_res
                        raw_text = None
                else:
                    # Use pre-generated cached results
                    if name in persona_results:
                        summary_res = persona_results[name]
                        if isinstance(summary_res, tuple) and len(summary_res) == 3:
                            summary_dict, img_buf, raw_text = summary_res
                        else:
                            summary_dict, img_buf = summary_res
                            raw_text = None
                    else:
                        # Fallback if not in cache
                        summary_res = openai_summary_and_image(center_row, required_cols, name, n_images=2, regenerate=False, debug=False)
                        if isinstance(summary_res, tuple) and len(summary_res) == 3:
                            summary_dict, img_buf, raw_text = summary_res
                        else:
                            summary_dict, img_buf = summary_res
                            raw_text = None

                # Persona card layout with 3D styling
                st.markdown("""
                <div class="card-3d" style='background:linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);border-radius:20px;padding:1.5em;margin-bottom:1.5em;border:3px solid #667eea;transform-style:preserve-3d;transform:perspective(1200px) rotateX(2deg) translateZ(25px);box-shadow:0 12px 24px rgba(0,0,0,0.2), 0 24px 48px rgba(102, 126, 234, 0.15), inset 0 1px 0 rgba(255,255,255,0.5), inset 0 -3px 0 rgba(0,0,0,0.1);'>
                """, unsafe_allow_html=True)
                if summary_dict:
                    st.markdown(f"""
                    <div style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);padding:1.2em;border-radius:15px;margin-bottom:1em;transform-style:preserve-3d;transform:perspective(1000px) rotateX(1deg) translateZ(15px);box-shadow:0 8px 16px rgba(0,0,0,0.3), 0 16px 32px rgba(102, 126, 234, 0.2), inset 0 1px 0 rgba(255,255,255,0.3), inset 0 -2px 0 rgba(0,0,0,0.2);'>
                        <h2 style='font-size:1.8rem;font-weight:700;color:#ffffff;margin:0;text-shadow:0 4px 8px rgba(0,0,0,0.4), 0 0 20px rgba(255,255,255,0.3);transform:translateZ(10px);'>{summary_dict.get('title', name)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    bullets = summary_dict.get('bullets') or []
                    if bullets:
                        st.markdown("""
                        <div style='background:#e8f4f8;padding:1.2em;border-radius:12px;margin:1em 0;border-left:5px solid #4F8A8B;transform-style:preserve-3d;transform:perspective(800px) rotateX(1deg) translateZ(10px);box-shadow:0 6px 12px rgba(79, 138, 139, 0.2), inset 0 1px 0 rgba(255,255,255,0.4), inset 0 -2px 0 rgba(0,0,0,0.1);'>
                            <h3 style='color:#1a2636;font-size:1.2rem;margin-top:0;text-shadow:1px 1px 2px rgba(0,0,0,0.1), 0 0 10px rgba(79, 138, 139, 0.2);transform:translateZ(5px);'>🔑 Key Traits</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        for b in bullets:
                            st.markdown(f"<div style='padding:0.5em 0.8em;margin:0.3em 0;background:rgba(255,255,255,0.7);border-radius:8px;transform:translateZ(5px);box-shadow:0 2px 4px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.5);'>✓ {b}</div>", unsafe_allow_html=True)
                    
                    food_items = summary_dict.get('food_items') or []
                    if food_items:
                        st.markdown("""
                        <div style='background:#fff3e0;padding:1.2em;border-radius:15px;margin:1em 0;border-left:6px solid #ff9800;transform-style:preserve-3d;transform:perspective(800px) rotateX(1deg) translateZ(12px);box-shadow:0 8px 16px rgba(255, 152, 0, 0.25), inset 0 1px 0 rgba(255,255,255,0.4), inset 0 -2px 0 rgba(0,0,0,0.1);'>
                            <h3 style='color:#1a2636;font-size:1.3rem;margin-top:0;margin-bottom:0.8em;font-weight:700;text-shadow:2px 2px 4px rgba(0,0,0,0.2), 0 0 15px rgba(255, 152, 0, 0.3);transform:translateZ(8px);'>🍽️ Representative Food Items</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        # Display food items in 3D grid
                        num_cols = min(4, len(food_items))
                        food_cols = st.columns(num_cols)
                        for idx, f in enumerate(food_items):
                            with food_cols[idx % num_cols]:
                                st.markdown(f"""
                                <div class="card-3d" style='padding:1.2em;background:linear-gradient(135deg, #fff 0%, #fff9e6 100%);border-radius:15px;margin:0.3em 0;border:3px solid #ffc107;text-align:center;transform-style:preserve-3d;transform:perspective(600px) rotateX(2deg) translateZ(15px);box-shadow:0 6px 12px rgba(255, 193, 7, 0.3), 0 12px 24px rgba(255, 152, 0, 0.2), inset 0 1px 0 rgba(255,255,255,0.5), inset 0 -2px 0 rgba(0,0,0,0.1);transition:all 0.3s ease;'>
                                    <div style='font-size:2.5rem;margin-bottom:0.3em;transform:translateZ(10px);filter:drop-shadow(0 4px 8px rgba(0,0,0,0.2));'>🍴</div>
                                    <div style='color:#333;font-weight:600;font-size:1rem;text-shadow:1px 1px 2px rgba(255,255,255,0.8);transform:translateZ(5px);'>{f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    program_ideas = summary_dict.get('program_ideas') or []
                    if program_ideas:
                        st.markdown("""
                        <div style='background:#f3e5f5;padding:1.2em;border-radius:12px;margin:1em 0;border-left:5px solid #9c27b0;transform-style:preserve-3d;transform:perspective(800px) rotateX(1deg) translateZ(10px);box-shadow:0 6px 12px rgba(156, 39, 176, 0.2), inset 0 1px 0 rgba(255,255,255,0.4), inset 0 -2px 0 rgba(0,0,0,0.1);'>
                            <h3 style='color:#1a2636;font-size:1.2rem;margin-top:0;text-shadow:1px 1px 2px rgba(0,0,0,0.1), 0 0 10px rgba(156, 39, 176, 0.2);transform:translateZ(5px);'>💡 Program Ideas</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        for p in program_ideas:
                            st.markdown(f"<div style='padding:0.5em 0.8em;margin:0.3em 0;background:rgba(255,255,255,0.7);border-radius:8px;transform:translateZ(5px);box-shadow:0 2px 4px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.5);'>💼 {p}</div>", unsafe_allow_html=True)
                    
                    diet_tips = summary_dict.get('diet_tips') or []
                    if diet_tips:
                        st.markdown("""
                        <div style='background:#e8f5e9;padding:1.2em;border-radius:12px;margin:1em 0;border-left:5px solid #4caf50;transform-style:preserve-3d;transform:perspective(800px) rotateX(1deg) translateZ(10px);box-shadow:0 6px 12px rgba(76, 175, 80, 0.2), inset 0 1px 0 rgba(255,255,255,0.4), inset 0 -2px 0 rgba(0,0,0,0.1);'>
                            <h3 style='color:#1a2636;font-size:1.2rem;margin-top:0;text-shadow:1px 1px 2px rgba(0,0,0,0.1), 0 0 10px rgba(76, 175, 80, 0.2);transform:translateZ(5px);'>🥗 Diet Tips</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        for t in diet_tips:
                            st.markdown(f"<div style='padding:0.5em 0.8em;margin:0.3em 0;background:rgba(255,255,255,0.7);border-radius:8px;transform:translateZ(5px);box-shadow:0 2px 4px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.5);'>✓ {t}</div>", unsafe_allow_html=True)
                    
                    advice = summary_dict.get('advice') or []
                    if advice:
                        st.markdown("""
                        <div style='background:#e1f5fe;padding:1.2em;border-radius:12px;margin:1em 0;border-left:5px solid #03a9f4;transform-style:preserve-3d;transform:perspective(800px) rotateX(1deg) translateZ(10px);box-shadow:0 6px 12px rgba(3, 169, 244, 0.2), inset 0 1px 0 rgba(255,255,255,0.4), inset 0 -2px 0 rgba(0,0,0,0.1);'>
                            <h3 style='color:#1a2636;font-size:1.2rem;margin-top:0;text-shadow:1px 1px 2px rgba(0,0,0,0.1), 0 0 10px rgba(3, 169, 244, 0.2);transform:translateZ(5px);'>💬 Communication Advice</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        for a in advice:
                            st.markdown(f"<div style='padding:0.5em 0.8em;margin:0.3em 0;background:rgba(255,255,255,0.7);border-radius:8px;transform:translateZ(5px);box-shadow:0 2px 4px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.5);'>📢 {a}</div>", unsafe_allow_html=True)

                    # JSON download of structured summary
                    try:
                        json_bytes = json.dumps(summary_dict, indent=2).encode('utf-8')
                        st.download_button(label="Download persona JSON", data=json_bytes, file_name=f"persona_{i}_{name}.json", mime="application/json")
                    except Exception:
                        pass
                else:
                    st.markdown("<b>Summary (rule-based):</b>", unsafe_allow_html=True)
                    st.write(
                        f"Cluster {i} ({name}) — eating out: {center_row[0]:.1f}/week, budget: ₹{center_row[1]:.0f}, sweet-tooth: {center_row[2]:.1f}, hobby hours: {center_row[3]:.1f}."
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        # Save models to workspace
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "kmeans.pkl"), "wb") as f:
            pickle.dump(kmeans, f)
        with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(out_dir, "mms.pkl"), "wb") as f:
            pickle.dump(mms, f)
        df_imp.to_csv(os.path.join(out_dir, "wellness_labeled.csv"), index=False)

        # Calculate and display performance metrics
        load_time = time.time() - start_time
        st.success(f"✅ Models and labeled data saved to: {out_dir}")
        
        # Performance summary (collapsible)
        with st.expander("⚡ Performance Summary", expanded=False):
            st.markdown(f"""
            <div style='background:#e8f5e9;padding:1em;border-radius:10px;border-left:4px solid #4caf50;'>
                <h4 style='color:#1a2636;margin-top:0;'>🚀 Optimization Features Enabled:</h4>
                <ul style='color:#333;'>
                    <li>✅ <strong>Caching:</strong> Data processing, clustering, and charts are cached for faster reloads</li>
                    <li>✅ <strong>Parallel Processing:</strong> Persona summaries and images generated in parallel</li>
                    <li>✅ <strong>Lazy Loading:</strong> Images load on-demand for better initial page speed</li>
                    <li>✅ <strong>Optimized Charts:</strong> PCA and radar charts are pre-computed and cached</li>
                    <li>✅ <strong>Multi-threading:</strong> KMeans uses all CPU cores (n_jobs=-1)</li>
                </ul>
                <p style='color:#666;margin-bottom:0;'><strong>Total load time:</strong> {load_time:.2f} seconds</p>
            </div>
            """, unsafe_allow_html=True)

    # Sidebar: single-user predictor
    st.sidebar.header("Single-user persona predictor")
    eating_out = st.sidebar.slider("Eating out per week", 0, 14, 2)
    food_budget = st.sidebar.number_input("Food budget per meal (INR)", min_value=0, max_value=2000, value=150)
    sweet_tooth = st.sidebar.slider("Sweet-tooth level (1-10)", 0, 10, 4)
    hobby_hours = st.sidebar.slider("Weekly hobby hours", 0, 40, 5)

    if st.sidebar.button("Predict persona"):
        user_df = pd.DataFrame([
            {
                "eating_out_per_week": eating_out,
                "food_budget_per_meal_inr": food_budget,
                "sweet_tooth_level": sweet_tooth,
                "weekly_hobby_hours": hobby_hours,
            }
        ])

        model_path = os.path.join(os.getcwd(), "outputs", "kmeans.pkl")
        scaler_path = os.path.join(os.getcwd(), "outputs", "scaler.pkl")
        mms_path = os.path.join(os.getcwd(), "outputs", "mms.pkl")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, "rb") as f:
                kmeans = pickle.load(f)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            with open(mms_path, "rb") as f:
                mms = pickle.load(f)

            user_scaled = scaler.transform(user_df)
            cluster = int(kmeans.predict(user_scaled)[0])
            st.sidebar.success(f"Assigned to Cluster {cluster}")

            centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
            center_vals = centers_orig[cluster]
            persona_name = persona_label_from_center(center_vals, required_cols)
            st.sidebar.markdown(f"**Persona:** {persona_name}")

            summary_dict, img_buf = openai_summary_and_image(center_vals, required_cols, persona_name)
            if summary_dict:
                st.sidebar.markdown("### LLM Summary")
                st.sidebar.markdown(f"**{summary_dict.get('title', persona_name)}**")
                bullets = summary_dict.get('bullets', [])
                if bullets:
                    for b in bullets:
                        st.sidebar.write(f"- {b}")
            else:
                st.sidebar.markdown("### Summary (rule-based)")
                st.sidebar.write(
                    f"Cluster {cluster} ({persona_name}) — typical eating out: {center_vals[0]:.1f}/week, budget: ₹{center_vals[1]:.0f}, sweet-tooth: {center_vals[2]:.1f}, hobby hours: {center_vals[3]:.1f}."
                )

            if img_buf:
                st.sidebar.image(img_buf, caption=f"Example foods for {persona_name}")
            else:
                # try local preview for sidebar too
                fallback_img = create_persona_preview_image(center_vals, required_cols, persona_name)
                if fallback_img:
                    st.sidebar.image(fallback_img, caption=f"Preview for {persona_name} (generated locally)")

        else:
            st.sidebar.error("No trained model found — please upload data or run full clustering first.")


if __name__ == "__main__":
    main()
