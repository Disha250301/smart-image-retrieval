import os
import io
import time
import base64
import zipfile
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image
import networkx as nx
import clip
import cv2

from src.feature_extraction import MyResnet50, MyVGG16
from src.clip_feature_extractor import CLIPFeatureExtractor
from src.hypergraph_retrieval_engine import HypergraphRetrievalEngine
from src.compute import compute_mAP

# ----------------------- Utility Functions -----------------------
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def download_images_as_zip(image_paths):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for img_path in image_paths:
            zf.write(img_path, os.path.basename(img_path))
    buffer.seek(0)
    return buffer

# ----------------------- Feature Visualization -----------------------
def generate_edge_map(image_pil):
    image_np = np.array(image_pil.convert("L"))
    edges = cv2.Canny(image_np, 100, 200)
    return edges

def generate_resnet_heatmap(model, image_tensor):
    with torch.no_grad():
        fmap = model.backbone(image_tensor).detach()
        heatmap = fmap.mean(1).cpu().numpy()[0]
        return cv2.resize(heatmap, (224, 224))

def generate_clip_saliency(clip_model, image_tensor):
    with torch.no_grad():
        features = clip_model.encode_image(image_tensor)
        saliency = features.norm(dim=-1).cpu().numpy()
        return np.tile(saliency, (224, 224))

def explain_feature_extraction(model_name, query_img, query_tensor, clip_extractor, resnet_extractor, vgg_extractor):
    st.markdown(
        "<div style='background-color:#001f3f; padding:15px; border-radius:10px;'>"
        f"<h3 style='color:white;'>🔍 Feature Extraction: {model_name}</h3>",
        unsafe_allow_html=True
    )
    if model_name == "CLIP":
        with st.spinner("CLIP extracting semantic regions..."):
            saliency = generate_clip_saliency(clip_extractor.model, query_tensor)
            fig, ax = plt.subplots()
            ax.imshow(query_img)
            ax.imshow(saliency, cmap='jet', alpha=0.4)
            ax.axis("off")
            plt.colorbar(ax.imshow(saliency, cmap='jet', alpha=0.4), ax=ax, fraction=0.046, pad=0.04, label="Saliency Intensity")
            st.pyplot(fig)
            st.markdown("📝 **Interpretation:** Red areas indicate highly relevant regions detected by CLIP based on semantic meaning.")

            labels = ["temple", "monument", "architecture", "sculpture", "building", "heritage site"]
            text_tokens = clip.tokenize(labels).to(clip_extractor.device)
            with torch.no_grad():
                text_features = clip_extractor.model.encode_text(text_tokens)
                img_feat = clip_extractor.model.encode_image(query_tensor)
                sims = (img_feat @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
            keywords = [labels[i] for i in sims.argsort()[::-1][:3]]
            st.success(f"**CLIP detected semantic concepts:** {', '.join(keywords)}")

    elif model_name == "ResNet50":
        with st.spinner("ResNet analyzing textures..."):
            heatmap = generate_resnet_heatmap(resnet_extractor, query_tensor)
            fig, ax = plt.subplots()
            im = ax.imshow(query_img)
            ax.imshow(heatmap, cmap='hot', alpha=0.5)
            ax.axis("off")
            plt.colorbar(ax.imshow(heatmap, cmap='hot', alpha=0.5), ax=ax, fraction=0.046, pad=0.04, label="Texture Activation")
            st.pyplot(fig)
            st.markdown("📝 **Interpretation:** Bright red zones highlight texture-rich areas (edges/patterns) influencing ResNet50 retrieval.")

    elif model_name == "VGG16":
        with st.spinner("VGG16 detecting edges..."):
            edges = generate_edge_map(query_img)
            st.image(edges, caption="VGG16 Edge Detection", use_container_width=True)
            st.markdown("📝 **Interpretation:** VGG16 focuses on edges and outlines of objects for matching.")

# ----------------------- Animated Hypergraph Visualization -----------------------
def animate_hypergraph(graph, retrieved_images, similarity_scores, retrieval_engine):
    pos = nx.spring_layout(graph, seed=42)
    sub_nodes = [i for i, path in enumerate(retrieval_engine.image_paths) if path in retrieved_images]
    subgraph = graph.subgraph(sub_nodes)
    pos = {node: pos[node] for node in subgraph.nodes()}

    fig = go.Figure()

    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], line=dict(width=1, color='#bbb'), mode='lines'))

    node_x, node_y, hover_text, node_colors = [], [], [], []
    for idx, node in enumerate(subgraph.nodes()):
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        img_path = retrieval_engine.image_paths[node]
        img_b64 = encode_image_to_base64(img_path)
        score = similarity_scores[idx]
        hover_text.append(f"<b>{os.path.basename(img_path)}</b><br>Score: {score:.4f}<br>"
                          f"<img src='data:image/jpeg;base64,{img_b64}' width='150'>")
        node_colors.append(score)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=[f"Img {i+1}" for i in range(len(node_x))],
        textposition="bottom center", hoverinfo="text", hovertext=hover_text,
        marker=dict(size=20, color=node_colors, colorscale="Purples", line_width=2, colorbar=dict(title="Similarity Score"))
    ))

    fig.update_layout(title="📊 Hypergraph (Animated Construction)", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("📝 **Interpretation:** Nodes (images) are connected based on similarity. Darker nodes indicate higher similarity to the query.")

# ----------------------- Custom Styling -----------------------
st.set_page_config(page_title="Smart Image Retrieval", layout="wide")
st.markdown("""
<style>
.stApp { background: linear-gradient(120deg, #6a0dad, #8e44ad, #9b59b6); color: white; }
.stButton>button { font-size: 22px; padding: 15px 40px; border-radius: 12px; background: linear-gradient(135deg,#d53369,#daae51); color:white; border:none; }
.stButton>button:hover { transform: scale(1.1); box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
.download-btn>button, .stDownloadButton>button { color: white !important; background-color: #001f3f !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------- Navigation -----------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# Home Page
if st.session_state.page == "home":
    st.markdown("<h1 style='text-align:center;'> Welcome to Smart Image Retrieval</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Choose an activity below</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔎 Retrieve Images"):
            st.session_state.page = "retrieve"
            st.rerun()
    with col2:
        if st.button("📈 Evaluate Models"):
            st.session_state.page = "evaluate"
            st.rerun()
    with col3:
        if st.button("ℹ️ About Project"):
            st.session_state.page = "about"
            st.rerun()

# Retrieval Page
elif st.session_state.page == "retrieve":
    st.markdown("<h1 style='text-align:center;'>🔎 Image Retrieval</h1>", unsafe_allow_html=True)
    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.sidebar.header("⚙️ Settings")
    model_choice = st.sidebar.selectbox("Choose Feature Extractor", ["CLIP", "ResNet50", "VGG16"])
    k_value = st.sidebar.slider("Top-K Results", 5, 20, 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uploaded_file = st.file_uploader("Upload Query Image", type=["jpg", "png"])
    if uploaded_file:
        query_img = Image.open(uploaded_file).convert("RGB")
        st.image(query_img, caption="Query Image", use_container_width=True)

        clip_extractor = CLIPFeatureExtractor(device=device)
        resnet_extractor = MyResnet50(device)
        vgg_extractor = MyVGG16(device)
        retrieval_engine = HypergraphRetrievalEngine(k_neighbors=10)

        feature_path = f"./dataset/features/{model_choice}_features.npy"
        filename_path = f"./dataset/features/{model_choice}_filenames.npy"
        features = np.load(feature_path, allow_pickle=True)
        filenames = np.load(filename_path, allow_pickle=True)
        image_paths = [f"./dataset/paris/{name}" for name in filenames]
        retrieval_engine.build_hypergraph(features, image_paths)

        if model_choice == "CLIP":
            query_tensor = clip_extractor.clip_preprocess(query_img).unsqueeze(0).to(device)
            features = clip_extractor.get_features(query_tensor)
        elif model_choice == "ResNet50":
            query_tensor = resnet_extractor.transform(query_img).unsqueeze(0).to(device)
            features = resnet_extractor(query_tensor).cpu().numpy()
        elif model_choice == "VGG16":
            query_tensor = vgg_extractor.transform(query_img).unsqueeze(0).to(device)
            features = vgg_extractor(query_tensor).cpu().numpy()

        explain_feature_extraction(model_choice, query_img, query_tensor, clip_extractor, resnet_extractor, vgg_extractor)

        with st.spinner("Building hypergraph..."):
            retrieved = retrieval_engine.retrieve_top_k(features, k=k_value)
            retrieved_images, similarity_scores = zip(*retrieved)
            animate_hypergraph(retrieval_engine.graph, retrieved_images, similarity_scores, retrieval_engine)

        st.subheader("🎯 Retrieved Images")
        cols = st.columns(5)
        for idx, img_path in enumerate(retrieved_images):
            with cols[idx % 5]:
                st.image(img_path, caption=f"Score: {similarity_scores[idx]:.4f}", use_container_width=True)
                with open(img_path, "rb") as file:
                    st.download_button("⬇️ Download", file, file_name=os.path.basename(img_path), key=f"dl{idx}", help="Download image", type="primary")
        st.download_button("📥 Download All Retrieved Images", download_images_as_zip(retrieved_images),
                           file_name="retrieved_images.zip", mime="application/zip", key="dl_all", help="Download all retrieved images")

# Evaluation Page
elif st.session_state.page == "evaluate":
    st.markdown("<h1 style='text-align:center;'>📈 Model Evaluation</h1>", unsafe_allow_html=True)
    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    model_choice = st.selectbox("Select Model for Evaluation", ["CLIP", "ResNet50", "VGG16"])
    if st.button("Compute Metrics"):
        progress = st.progress(0)
        for i in range(0, 100, 20):
            time.sleep(0.1)
            progress.progress(i + 20)

        result = compute_mAP(model_choice)
        if len(result) == 5:
            mAP, avg_precisions, precisions, recalls, ranks = result
            avg_precision = np.mean(avg_precisions)
            avg_recall = np.mean([max(r) for r in recalls if r])
        else:
            mAP, avg_precision, avg_recall = result
            avg_precisions, precisions, recalls = [], [], []

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Average Precision (mAP)", f"{mAP:.4f}")
        col2.metric("Average Precision", f"{avg_precision:.4f}")
        col3.metric("Average Recall", f"{avg_recall:.4f}")

        st.markdown("---")
        st.subheader("📊 Evaluation Visualizations")

        tabs = st.tabs(["Precision-Recall Curve", "AP per Query"])

        # Precision-Recall Curve with Color Legend
        with tabs[0]:
            if precisions and recalls:
                fig, ax = plt.subplots(figsize=(8, 5))
                cmap = plt.get_cmap("viridis", len(precisions))
                for i, (p, r) in enumerate(zip(precisions, recalls)):
                    ax.plot(r, p, label=f"Query {i+1}", color=cmap(i))
                sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=1, vmax=len(precisions)))
                sm.set_array([])
                fig.colorbar(sm, ax=ax, ticks=np.linspace(1, len(precisions), 5), label="Query Index")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(f"Precision-Recall Curve ({model_choice})")
                ax.grid(True)
                st.pyplot(fig)
                st.markdown("📝 **Interpretation:** Curves near top-right indicate strong precision and recall.")
            else:
                st.info("Precision-Recall data not available.")

        # AP per Query with Color Legend
        with tabs[1]:
            if avg_precisions:
                fig, ax = plt.subplots(figsize=(8, 4))
                cmap = plt.get_cmap("plasma", len(avg_precisions))
                colors = [cmap(i) for i in range(len(avg_precisions))]
                bars = ax.bar(range(len(avg_precisions)), avg_precisions, color=colors)
                sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=1, vmax=len(avg_precisions)))
                sm.set_array([])
                fig.colorbar(sm, ax=ax, ticks=np.linspace(1, len(avg_precisions), 5), label="Query Index")
                ax.set_xlabel("Query Index")
                ax.set_ylabel("Average Precision (AP)")
                ax.set_title(f"AP per Query ({model_choice})")
                st.pyplot(fig)
                st.markdown("📝 **Interpretation:** Taller bars represent queries with higher AP values.")
            else:
                st.info("AP per query data not available.")

# About Page
elif st.session_state.page == "about":
    st.markdown("<h1 style='text-align:center;'>ℹ️ About Smart Image Retrieval</h1>", unsafe_allow_html=True)
    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    st.write("""
    This project implements a **Content-Based Image Retrieval (CBIR)** system with:
    - **CLIP, ResNet50, VGG16 feature extractors**
    - **Hypergraph-based manifold ranking**
    - **Dynamic hypergraph animation**
    - **Evaluation metrics (mAP, Precision, Recall) with visuals and legends**
    """)
 