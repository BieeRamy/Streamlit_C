import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

# You need to have your trained RandomForest model loaded here:
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(page_title="Fraud Detection Streamlit App", layout="wide")
st.title("Fraud Detection Streamlit App")

@st.cache_data
def load_data():
    df = pd.read_csv('Fraud_Detection.csv', parse_dates=['timestamp'])
    return df

df = load_data()

if st.button("Cluster-Based Profiling"):
    st.header("Cluster-Based Profiling")

    features = ['amount', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score']
    
    # Fill missing values
    X = df[features].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    st.subheader("Step 1: Cluster Summary Table")
    cluster_summary = df.groupby('cluster').agg(
        total_transactions=('transaction_id', 'count'),
        fraud_transactions=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean'),
    ).reset_index()
    cluster_summary['fraud_rate'] = (cluster_summary['fraud_rate'] * 100).round(2)
    st.dataframe(cluster_summary)
    
    st.subheader("Fraud Rate by Cluster (%)")
    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.barplot(data=cluster_summary, x='cluster', y='fraud_rate', palette='viridis', ax=ax1)
    ax1.set_title('Fraud Rate by Cluster (%)')
    ax1.set_ylabel('Fraud Rate (%)')
    ax1.set_xlabel('Cluster')
    st.pyplot(fig1)
    
    st.subheader("Fraud vs Non-Fraud Counts per Cluster")
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.countplot(data=df, x='cluster', hue='is_fraud', palette='Set2', ax=ax2)
    ax2.set_title('Fraud vs Non-Fraud Counts per Cluster')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Transaction Count')
    ax2.legend(title='Is Fraud')
    st.pyplot(fig2)
    
    st.subheader("Cluster Profiles (Normalized Radar Plot)")
    cluster_features = df.groupby('cluster')[features].mean().round(2)
    cluster_norm = (cluster_features - cluster_features.min()) / (cluster_features.max() - cluster_features.min())
    
    labels = features
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig3, ax3 = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    for i in range(len(cluster_norm)):
        values = cluster_norm.iloc[i].tolist()
        values += values[:1]
        ax3.plot(angles, values, label=f'Cluster {i}')
        ax3.fill(angles, values, alpha=0.1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(labels)
    ax3.set_title('Normalized Cluster Profiles')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig3)
    
    st.subheader("Temporal Patterns per Cluster")
    df['date'] = df['timestamp'].dt.date
    cluster_time = df.groupby(['cluster', 'date']).size().unstack(fill_value=0).T
    fig4, ax4 = plt.subplots(figsize=(12,6))
    cluster_time.plot(ax=ax4)
    ax4.set_title("Transaction Volume by Cluster Over Time")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Transaction Count")
    st.pyplot(fig4)
    
    st.subheader("Feature Importance (Random Forest)")
    # NOTE: You must have a trained Random Forest model saved as 'rf_model.joblib'
    # and a list of feature names 'model_features' saved or hardcoded
    
    try:
        model = joblib.load('rf_model.joblib')
        model_features = features  # Or replace with your actual features list
        
        importances = model.feature_importances_
        feat_importance = pd.Series(importances, index=model_features).sort_values(ascending=False)

        fig5, ax5 = plt.subplots(figsize=(8,5))
        sns.barplot(x=feat_importance.values, y=feat_importance.index, ax=ax5)
        ax5.set_title("🔍 Feature Importance (Random Forest)")
        ax5.set_xlabel("Importance")
        st.pyplot(fig5)
    except Exception as e:
        st.warning("Random Forest model or feature list not found. Please place 'rf_model.joblib' in app directory.")
        st.write(str(e))
