import streamlit as st
import pandas as pd

# Add for utils
from utils import load_data, train_kmeans, calculate_wcss, calculate_silhouette

# --- Streamlit App Config ---
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🏍️ Mall Customer Segmentation using KMeans")

# --- Load Data ---
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    df = load_data("mall_customers.csv")
    st.info("ℹ️ Using default dataset.")

# --- Show Raw Data ---
st.subheader("📄 Raw Data")
st.dataframe(df.head())

# --- Feature Selection ---
numerical_features = df.select_dtypes(include='number').columns.tolist()
st.write("📊 Available numerical columns:")
st.write(numerical_features)

features = st.multiselect(
    "🔧 Select features for clustering:",
    numerical_features,
    default=numerical_features[:2]
)

if len(features) < 2:
    st.warning("⚠️ Please select at least two features.")
    st.stop()

data_selected = df[features]

# --- Elbow Plot (Optional) ---
if st.checkbox("📈 Show Elbow Method Plot"):
    wcss = calculate_wcss(data_selected)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title("Elbow Method for Optimal k")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

# --- Cluster Model Training ---
k = st.slider("🔹 Number of clusters (k)", min_value=2, max_value=10, value=5)
model = train_kmeans(data_selected, k)
labels = model.labels_

# --- Results Display ---
df["Cluster"] = labels
score = calculate_silhouette(data_selected, model)

st.info(f"📏 Silhouette Score: {score:.2f}")
st.subheader("🎨 Clustered Data")
st.dataframe(df)

# --- 2D Cluster Visualization ---
if len(features) == 2:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=data_selected[features[0]],
        y=data_selected[features[1]],
        hue=labels,
        palette="Set2",
        s=100,
        ax=ax
    )
    ax.set_title("Customer Segments")
    st.pyplot(fig)
else:
    st.warning("📉 Select exactly 2 features to visualize clusters.")
