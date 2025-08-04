# clustering.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data(file_path="data/mall_customers.csv"):
    return pd.read_csv(file_path)

def train_kmeans(data, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(data)
    return model

def calculate_wcss(data, max_k=10):
    wcss = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

def calculate_silhouette(data, model):
    labels = model.labels_
    return silhouette_score(data, labels)
