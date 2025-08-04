# clustering.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Utility Functions ---

def load_data(file_path="mall_customers.csv"):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def train_kmeans(data, n_clusters=5):
    """Train a KMeans clustering model."""
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(data)
    return model

def calculate_wcss(data, max_k=10):
    """Calculate WCSS for different k values for the Elbow Method."""
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

def calculate_silhouette(data, model):
    """Compute the silhouette score for a given model."""
    labels = model.labels_
    return silhouette_score(data, labels)
