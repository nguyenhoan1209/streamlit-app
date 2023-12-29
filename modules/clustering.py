import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class Clustering:
    def kmeans_clustering(data):
    # Create a copy of the data
        data_copy = data.copy()

        data_number_colum = data_copy.select_dtypes(include=["int", "float"]).columns
        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_number_colum)

        # Select the number of clusters
        n_clusters = st.slider("Chọn số lượng cụm", 1, 10)

        # Get the data for clustering
        X = data_copy[feature_columns]

        # Standardize the data before clustering
        

        # Perform K-Means clustering
        if not feature_columns:
            st.warning("Chon cot tinh nang")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_scaled)
            

            # Add cluster labels to the data
            X["cluster"] = kmeans.labels_
            X["cluster"] = X["cluster"].astype(str)
            fig = px.scatter(X, x=X.iloc[:, 0], y=X.iloc[:, 1], color="cluster")
            st.markdown("Number of Clusters: {}".format(n_clusters))
            st.plotly_chart(fig)

            silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)  # Calculate silhouette score on scaled data
            st.markdown(f"Silhouette Score: {silhouette_avg:.4f}")
            