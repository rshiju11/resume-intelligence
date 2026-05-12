import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(
    page_title="Resume Clustering Dashboard",
    layout="wide"
)

st.title("Resume Clustering Dashboard")
st.markdown(
    "Semantic clustering visualization using transformer embeddings, dimensionality reduction, and HDBSCAN."
)


BASE_DIR = Path(__file__).resolve().parents[2]

emb_path = BASE_DIR / "results" / "embeddings" / "resume_embeddings.npy"
cluster_dir = BASE_DIR / "results" / "clusters"
text_dir = BASE_DIR / "data" / "extracted_text"


st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Use this panel to explore different clustering outputs.")


if not emb_path.exists():
    st.error("Embedding file not found: results/embeddings/resume_embeddings.npy")
    st.stop()

embeddings = np.load(emb_path)

if not cluster_dir.exists():
    st.error("Cluster folder not found: results/clusters/")
    st.stop()

label_files = sorted(cluster_dir.glob("*labels*.npy"))

if len(label_files) == 0:
    st.error("No cluster label files found in results/clusters/")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select clustering result",
    label_files,
    format_func=lambda x: x.name
)

labels = np.load(selected_file)

resume_files = sorted(text_dir.glob("*.txt"))

if len(resume_files) == 0:
    st.warning("No extracted resume text files found in data/extracted_text/")

resume_names = [f.stem for f in resume_files[:len(labels)]]

if len(resume_names) < len(labels):
    resume_names += [f"Resume_{i}" for i in range(len(resume_names), len(labels))]

df = pd.DataFrame({
    "Resume": resume_names,
    "Cluster": labels
})

noise_points = int(np.sum(labels == -1))
clusters = sorted(set(labels) - {-1})
noise_percent = round((noise_points / len(labels)) * 100, 2)

st.subheader("Project Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Resumes", len(labels))
col2.metric("Clusters Found", len(clusters))
col3.metric("Noise Points", noise_points)
col4.metric("Noise %", f"{noise_percent}%")

st.info(
    "Cluster label -1 represents noise points identified by HDBSCAN. "
    "These resumes were not strongly assigned to any stable cluster."
)

st.subheader("Internal Cluster Evaluation Metrics")

valid_mask = labels != -1
valid_labels = labels[valid_mask]
valid_embeddings = embeddings[valid_mask]

if len(set(valid_labels)) > 1:
    sil = silhouette_score(valid_embeddings, valid_labels)
    dbi = davies_bouldin_score(valid_embeddings, valid_labels)
    ch = calinski_harabasz_score(valid_embeddings, valid_labels)

    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette Score", round(sil, 4))
    m2.metric("Davies-Bouldin Index", round(dbi, 4))
    m3.metric("Calinski-Harabasz Score", round(ch, 4))

    st.caption(
        "Higher Silhouette and Calinski-Harabasz scores are generally better. "
        "Lower Davies-Bouldin Index is generally better."
    )
else:
    st.warning("Not enough valid clusters to calculate evaluation metrics.")


st.subheader("Cluster Distribution")

cluster_counts = df["Cluster"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 4))

cluster_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Cluster Label")
ax.set_ylabel("Number of Resumes")
ax.set_title("Number of Resumes per Cluster")

st.pyplot(fig)
st.subheader("2D PCA Visualization")

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

ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.set_title("Resume Embeddings Projected into 2D using PCA")

st.pyplot(fig2)

st.caption(
    "This plot is only a 2D projection for visualization. "
    "The actual clustering was performed using the selected saved cluster labels."
)

st.subheader("Cluster Table")

cluster_options = ["All"] + sorted(df["Cluster"].unique().tolist())

selected_cluster_filter = st.selectbox(
    "Filter by cluster",
    cluster_options
)

search_text = st.text_input("Search resume name")

filtered_df = df.copy()

if selected_cluster_filter != "All":
    filtered_df = filtered_df[filtered_df["Cluster"] == selected_cluster_filter]

if search_text:
    filtered_df = filtered_df[
        filtered_df["Resume"].str.contains(search_text, case=False, na=False)
    ]

st.dataframe(
    filtered_df,
    use_container_width=True
)

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Cluster Results as CSV",
    data=csv,
    file_name="cluster_results.csv",
    mime="text/csv"
)

st.subheader("Explore Resumes by Cluster")

selected_cluster = st.selectbox(
    "Choose cluster to explore",
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
            st.warning("Resume text file not found.")

st.subheader("Nearest Neighbor Search")

selected_resume = st.selectbox(
    "Choose a resume",
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

st.caption(
    "Nearest neighbors are calculated using cosine similarity between resume embedding vectors."
)

st.subheader("Cluster Interpretation Notes")

st.markdown(
    """
    - **Cluster -1** means HDBSCAN treated those resumes as noise.
    - Larger clusters may represent resumes with more shared skills, experience, or academic background.
    - Smaller clusters may capture more specific resume patterns.
    - Nearest-neighbor similarity helps validate whether semantically similar resumes are grouped closely.
    - PCA visualization is mainly for presentation and interpretation, not the final clustering decision.
    """
)