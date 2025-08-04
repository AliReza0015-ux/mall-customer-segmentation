import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
def load_data(path):
    return pd.read_csv(path)

# Train KMeans model with selected number of clusters
def train_kmeans(data, k):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(data)
    return model

# Calculate Within-Cluster Sum of Squares for Elbow Method
def calculate_wcss(data):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# Calculate Silhouette Score
def calculate_silhouette(data, model):
    labels = model.labels_
    score = silhouette_score(data, labels)
    return score
