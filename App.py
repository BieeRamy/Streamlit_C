import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Fraud Detection Streamlit App", layout="wide")
st.title("🚨 Fraud Detection Streamlit App")

uploaded_file = st.file_uploader("Upload your 'Fraud_Detection.csv' file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date

    # Sidebar selection
    analysis_option = st.sidebar.radio("📊 Select Analysis Type", [
        "Fraud Rate by Hour of Day",
        "Fraud Rate by Daypart",
        "Geolocation Heatmaps",
        "Correlation Heatmap of Temporal Features + Fraud Label",
        "Cluster-Based Profiling"
    ])

    # Fraud Rate by Hour of Day
    if analysis_option == "Fraud Rate by Hour of Day":
        st.subheader("Fraud Rate by Hour of Day")
        if st.button("Show Chart"):
            fig, ax = plt.subplots(figsize=(12,6))
            sns.barplot(data=df, x='hour', y='is_fraud', ci=None, ax=ax)
            ax.set_title('Fraud Rate by Hour of Day')
            ax.set_ylabel('Fraud Rate (Mean)')
            ax.set_xlabel('Hour of Day')
            st.pyplot(fig)

    # Fraud Rate by Daypart
    elif analysis_option == "Fraud Rate by Daypart":
        st.subheader("Fraud Rate by Daypart")
        if st.button("Show Chart"):
            fig, ax = plt.subplots(figsize=(10,5))
            sns.barplot(data=df, x='daypart', y='is_fraud', order=['early_morning', 'morning', 'afternoon', 'evening', 'night'], ax=ax)
            ax.set_title('Fraud Rate by Daypart')
            ax.set_ylabel('Fraud Rate (Mean)')
            ax.set_xlabel('Daypart')
            st.pyplot(fig)

    # Geolocation Heatmaps
    elif analysis_option == "Geolocation Heatmaps":
        st.subheader("Top Locations for Fraudulent Transactions")
        if st.button("Show Chart"):
            city_counts = df[df['is_fraud'] == 1]['location'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(x=city_counts.index, y=city_counts.values, ax=ax)
            ax.set_title("Top Locations for Fraudulent Transactions")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

    # Correlation Heatmap
    elif analysis_option == "Correlation Heatmap of Temporal Features + Fraud Label":
        st.subheader("Correlation Heatmap of Temporal Features and Fraud")
        if st.button("Show Chart"):
            features = ['is_off_hour', 'is_weekend', 'is_night', 'abs_hour_deviation', 'time_since_last', 'hour_entropy', 'temporal_risk_score', 'is_fraud']
            corr_df = df[features].dropna()
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Heatmap of Temporal Features and Fraud')
            st.pyplot(fig)

    # Cluster-Based Profiling
    elif analysis_option == "Cluster-Based Profiling":
        st.subheader("Cluster-Based Fraud Profiling")

        features = ['amount', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score']
        X = df[features].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=4, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)

        cluster_id = st.radio("Select Cluster for Analysis", [0, 1, 2, 3])

        cluster_summary = df.groupby('cluster').agg(
            total_transactions=('transaction_id', 'count'),
            fraud_transactions=('is_fraud', 'sum'),
            fraud_rate=('is_fraud', 'mean')
        ).reset_index()
        cluster_summary['fraud_rate'] = (cluster_summary['fraud_rate'] * 100).round(2)

        st.write("### Fraud Rate by Cluster")
        fig1, ax1 = plt.subplots(figsize=(8,4))
        sns.barplot(data=cluster_summary, x='cluster', y='fraud_rate', palette='viridis', ax=ax1)
        ax1.set_title('Fraud Rate by Cluster (%)')
        ax1.set_ylabel('Fraud Rate (%)')
        ax1.set_xlabel('Cluster')
        st.pyplot(fig1)

        st.write("### Radar Plot: Normalized Cluster Profiles")
        cluster_features = df.groupby('cluster')[features].mean()
        cluster_norm = (cluster_features - cluster_features.min()) / (cluster_features.max() - cluster_features.min())

        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]

        fig2, ax2 = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        for i in range(len(cluster_norm)):
            values = cluster_norm.iloc[i].tolist()
            values += values[:1]
            ax2.plot(angles, values, label=f'Cluster {i}')
            ax2.fill(angles, values, alpha=0.1)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(features)
        ax2.set_title('Normalized Cluster Profiles')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig2)

        st.write("### Transaction Volume by Cluster Over Time")
        cluster_time = df.groupby(['cluster', 'date']).size().unstack(fill_value=0).T
        st.line_chart(cluster_time[[cluster_id]])
else:
    st.warning("Please upload a 'Fraud_Detection.csv' file to begin.")
