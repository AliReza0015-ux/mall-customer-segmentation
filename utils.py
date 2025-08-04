import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Utility Functions ---

def load_data(path):
    """Load dataset from CSV file."""
    return pd.read_csv(path)

def train_kmeans(data, k):
    """Train KMeans clustering model."""
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(data)
    return model

def calculate_wcss(data):
    """Calculate Within-Cluster Sum of Squares (WCSS) for Elbow Method."""
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

def calculate_silhouette(data, model):
    """Compute Silhouette Score for clustering evaluation."""
    labels = model.labels_
    score = silhouette_score(data, labels)
    return score
