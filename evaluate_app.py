import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
from src.compute import compute_mAP

st.set_page_config(layout="wide", page_title="Smart Image Retrieval Evaluation")

st.title("📈 Model Evaluation Dashboard")
st.markdown("Evaluate & compare **Precision**, **Recall**, and **mAP** across extractors.")


def show_badge(value, good_thresh=0.75, average_thresh=0.5):
    if value >= good_thresh:
        return f"🟢 **{value:.4f}**"
    elif value >= average_thresh:
        return f"🟡 **{value:.4f}**"
    else:
        return f"🔴 **{value:.4f}**"


def run_ranking_if_needed(extractor):
    eval_path = f"./dataset/evaluation/original/{extractor}"
    if not os.path.exists(eval_path) or len(os.listdir(eval_path)) == 0:
        st.warning(f"⏳ Ranking for `{extractor}` not found. Running ranking...")
        subprocess.run(
            ["python", "ranking.py", "--feature_extractor", extractor, "--device", "cpu"],
            shell=True,
        )


# --- Main Evaluation ---
extractors = ["CLIP", "Resnet50", "VGG16"]
results = {
    "Extractor": [],
    "mAP": [],
    "Precision": [],
    "Recall": [],
    "mAP Badge": [],
    "Precision Badge": [],
    "Recall Badge": [],
}

with st.spinner("Evaluating models..."):
    for extractor in extractors:
        try:
            run_ranking_if_needed(extractor)
            mAP, _, precisions, recalls, _ = compute_mAP(extractor, crop=False)
            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)

            # Populate results
            results["Extractor"].append(extractor)
            results["mAP"].append(round(mAP, 4))
            results["Precision"].append(round(avg_precision, 4))
            results["Recall"].append(round(avg_recall, 4))

            results["mAP Badge"].append(show_badge(mAP))
            results["Precision Badge"].append(show_badge(avg_precision))
            results["Recall Badge"].append(show_badge(avg_recall))

        except Exception as e:
            st.error(f"Failed for `{extractor}`: {e}")

# --- Results Table ---
st.subheader("Metrics Table with Performance Badges")
df_badges = pd.DataFrame({
    "Extractor": results["Extractor"],
    "mAP": results["mAP Badge"],
    "Precision": results["Precision Badge"],
    "Recall": results["Recall Badge"],
})
st.write(df_badges, unsafe_allow_html=True)

# --- Chart Section ---
st.subheader("Bar Chart Comparison")

fig, ax = plt.subplots(figsize=(10, 5))
x = results["Extractor"]
ax.bar(x, results["mAP"], width=0.25, label="mAP", align="center")
ax.bar(x, results["Precision"], width=0.25, label="Precision", align="edge")
ax.bar(x, results["Recall"], width=0.25, label="Recall", align="edge")
ax.set_ylabel("Score")
ax.set_title("Evaluation Metrics by Extractor")
ax.legend()
st.pyplot(fig)

# --- Download ---
df_csv = pd.DataFrame({
    "Extractor": results["Extractor"],
    "mAP": results["mAP"],
    "Precision": results["Precision"],
    "Recall": results["Recall"],
})
st.download_button("Download CSV", df_csv.to_csv(index=False), "evaluation_metrics.csv", "text/csv")
