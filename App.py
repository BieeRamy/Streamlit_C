import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# App settings
st.set_page_config(page_title="Fraud Detection Streamlit App", layout="wide")
st.title("🚨 Fraud Detection Streamlit App")

# File upload
uploaded_file = st.file_uploader("Upload your 'Fraud_Detection.csv' file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date

    analysis_option = st.sidebar.radio("📊 Select Analysis Type", [
        "Fraud Rate by Hour of Day",
        "Fraud Rate by Daypart",
        "Geolocation Heatmaps",
        "Correlation Heatmap of Temporal Features + Fraud Label",
        "Cluster-Based Profiling",
        "Model Prediction"
    ])

    # 1. Hour of Day
    if analysis_option == "Fraud Rate by Hour of Day":
        st.subheader("Fraud Rate by Hour of Day")
        if 'hour' in df.columns and 'is_fraud' in df.columns:
            fig, ax = plt.subplots(figsize=(12,6))
            sns.barplot(data=df, x='hour', y='is_fraud', ci=None, ax=ax)
            ax.set_title('Fraud Rate by Hour of Day')
            ax.set_ylabel('Fraud Rate (Mean)')
            ax.set_xlabel('Hour of Day')
            st.pyplot(fig)
        else:
            st.warning("Missing 'hour' or 'is_fraud' column.")

    # 2. Daypart
    elif analysis_option == "Fraud Rate by Daypart":
        st.subheader("Fraud Rate by Daypart")
        if 'daypart' in df.columns:
            fig, ax = plt.subplots(figsize=(10,5))
            sns.barplot(data=df, x='daypart', y='is_fraud',
                        order=['early_morning', 'morning', 'afternoon', 'evening', 'night'], ax=ax)
            ax.set_title('Fraud Rate by Daypart')
            st.pyplot(fig)
        else:
            st.warning("Missing 'daypart' column in dataset.")

    # 3. Geolocation Heatmap
    elif analysis_option == "Geolocation Heatmaps":
        st.subheader("Top Locations for Fraudulent Transactions")
        if 'location' in df.columns:
            city_counts = df[df['is_fraud'] == 1]['location'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(x=city_counts.index, y=city_counts.values, ax=ax)
            ax.set_title("Top Locations for Fraudulent Transactions")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        else:
            st.warning("Missing 'location' column in dataset.")

    # 4. Temporal Feature Correlation
    elif analysis_option == "Correlation Heatmap of Temporal Features + Fraud Label":
        st.subheader("Correlation Heatmap of Temporal Features and Fraud")
        features = ['is_off_hour', 'is_weekend', 'is_night', 'abs_hour_deviation', 'time_since_last',
                    'hour_entropy', 'temporal_risk_score', 'is_fraud']
        existing = [col for col in features if col in df.columns]
        if len(existing) >= 2:
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(df[existing].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)
        else:
            st.warning("Temporal features not found in dataset.")

    # 5. Cluster-Based Profiling
    elif analysis_option == "Cluster-Based Profiling":
        st.subheader("Cluster-Based Fraud Profiling")
        cluster_features = ['amount', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score']
        existing = [f for f in cluster_features if f in df.columns]

        if len(existing) < 2:
            st.warning("Not enough clustering features found.")
        else:
            X = df[existing].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)
            kmeans = KMeans(n_clusters=4, random_state=42)
            df['cluster'] = kmeans.fit_predict(X_scaled)

            cluster_id = st.radio("Select Cluster", [0, 1, 2, 3])
            cluster_summary = df.groupby('cluster').agg(
                total=('transaction_id', 'count') if 'transaction_id' in df.columns else ('hour', 'count'),
                frauds=('is_fraud', 'sum'),
                rate=('is_fraud', 'mean')
            ).reset_index()
            cluster_summary['rate'] = (cluster_summary['rate'] * 100).round(2)

            st.write("### Fraud Rate by Cluster")
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(data=cluster_summary, x='cluster', y='rate', palette='viridis', ax=ax)
            ax.set_ylabel('Fraud Rate (%)')
            st.pyplot(fig)

            st.write("### Radar Plot")
            cluster_avg = df.groupby('cluster')[existing].mean()
            cluster_norm = (cluster_avg - cluster_avg.min()) / (cluster_avg.max() - cluster_avg.min())
            angles = np.linspace(0, 2 * np.pi, len(existing), endpoint=False).tolist()
            angles += angles[:1]

            fig2, ax2 = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
            for i in range(len(cluster_norm)):
                values = cluster_norm.iloc[i].tolist()
                values += values[:1]
                ax2.plot(angles, values, label=f'Cluster {i}')
                ax2.fill(angles, values, alpha=0.1)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(existing)
            ax2.set_title('Normalized Cluster Profiles')
            ax2.legend(loc='upper right')
            st.pyplot(fig2)

            st.write("### Transactions Over Time")
            if 'date' in df.columns:
                cluster_time = df.groupby(['cluster', 'date']).size().unstack(fill_value=0).T
                st.line_chart(cluster_time[[cluster_id]])

    # 6. Model Prediction
    elif analysis_option == "Model Prediction":
        st.subheader("📌 Model Prediction Result (Random Forest)")

        if 'is_fraud' not in df.columns:
            st.warning("Missing 'is_fraud' column.")
        else:
            df_model = df.dropna(subset=['is_fraud'])
            X = df_model.drop(columns=['is_fraud', 'transaction_id', 'timestamp', 'date'], errors='ignore')
            y = df_model['is_fraud']

            cat_cols = X.select_dtypes(include='object').columns.tolist()
            num_cols = X.select_dtypes(include='number').columns.tolist()

            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ])

            pipeline = Pipeline([
                ('prep', preprocessor),
                ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            st.metric("Accuracy on Test Set", f"{acc:.4f}")
            st.write("### Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

else:
    st.warning("Please upload a 'Fraud_Detection.csv' file to begin.")
