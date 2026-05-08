import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Resume Clustering Dashboard",
    layout="wide"
)

st.title("Resume Clustering Dashboard")
st.markdown("Semantic clustering visualization using embeddings, PCA/UMAP and HDBSCAN.")


BASE_DIR = Path(__file__).resolve().parents[2]

emb_path = BASE_DIR / "results" / "embeddings" / "resume_embeddings.npy"
cluster_dir = BASE_DIR / "results" / "clusters"
text_dir = BASE_DIR / "data" / "extracted_text"

if not emb_path.exists():
    st.error("Embedding file not found.")
    st.stop()

embeddings = np.load(emb_path)

label_files = list(cluster_dir.glob("*labels*.npy"))

if len(label_files) == 0:
    st.error("No cluster label files found.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select Cluster Result",
    label_files,
    format_func=lambda x: x.name
)

labels = np.load(selected_file)

resume_files = sorted(text_dir.glob("*.txt"))

resume_names = [f.stem for f in resume_files[:len(labels)]]

df = pd.DataFrame({
    "Resume": resume_names,
    "Cluster": labels
})

noise_points = np.sum(labels == -1)
clusters = sorted(set(labels) - {-1})

col1, col2, col3 = st.columns(3)

col1.metric("Total Resumes", len(labels))
col2.metric("Clusters", len(clusters))
col3.metric("Noise Points", int(noise_points))


st.subheader("Cluster Distribution")

cluster_counts = df["Cluster"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 4))

cluster_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Cluster")
ax.set_ylabel("Count")
ax.set_title("Resumes per Cluster")

st.pyplot(fig)


st.subheader("PCA Visualization")

pca = PCA(
    n_components=2,
    random_state=42
)

reduced = pca.fit_transform(embeddings)

plot_df = pd.DataFrame({
    "PC1": reduced[:, 0],
    "PC2": reduced[:, 1],
    "Cluster": labels,
    "Resume": resume_names
})

fig2, ax2 = plt.subplots(figsize=(10, 7))

scatter = ax2.scatter(
    plot_df["PC1"],
    plot_df["PC2"],
    c=plot_df["Cluster"],
    alpha=0.8
)

ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_title("2D PCA Projection")

st.pyplot(fig2)


st.subheader("Cluster Table")

st.dataframe(
    df,
    use_container_width=True
)


st.subheader("Explore Cluster")

selected_cluster = st.selectbox(
    "Choose Cluster",
    sorted(df["Cluster"].unique())
)

cluster_resumes = df[df["Cluster"] == selected_cluster]

st.write(f"Resumes in Cluster {selected_cluster}: {len(cluster_resumes)}")

for _, row in cluster_resumes.iterrows():

    with st.expander(row["Resume"]):

        file_path = text_dir / f"{row['Resume']}.txt"

        if file_path.exists():

            content = file_path.read_text(errors="ignore")

            st.text(content[:3000])

        else:
            st.warning("Resume text not found.")


st.subheader("Nearest Neighbor Search")

selected_resume = st.selectbox(
    "Choose Resume",
    resume_names
)

selected_index = resume_names.index(selected_resume)

similarities = cosine_similarity(
    [embeddings[selected_index]],
    embeddings
)[0]

top_indices = similarities.argsort()[::-1][1:6]

neighbors = []

for idx in top_indices:

    neighbors.append({
        "Resume": resume_names[idx],
        "Cosine Similarity": round(float(similarities[idx]), 4),
        "Cluster": int(labels[idx])
    })

neighbor_df = pd.DataFrame(neighbors)

st.dataframe(
    neighbor_df,
    use_container_width=True
)