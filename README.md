# Smart Image Retrieval Using Image Analytics

This project is an intelligent image retrieval system that returns visually and semantically similar images for a given query image.

## Tech Stack
- Python
- Deep Learning (feature extraction)
- Streamlit (web UI)
- Evaluation metrics: Precision, Recall, mAP

## Features
- Extracts deep features from images and builds an index for fast retrieval
- Ranks images using a hypergraph-based approach to capture visual and semantic similarity
- Streamlit interface to upload a query image and view ranked similar images
- Evaluation pipeline with scripts to compute Precision, Recall and mAP (achieved mAP ≈ 0.978 on a benchmark dataset)

## Project Structure
- `indexing.py` – builds the image index / feature database  
- `ranking.py` – ranks images based on similarity  
- `evaluate.py` – evaluation logic for Precision, Recall, mAP  
- `evaluate_app.py` – evaluation integrated with the Streamlit app  
- `generate_groundtruth.py` – generates ground truth labels for evaluation  
- `evaluation_results.csv` – stored evaluation metrics  
- `monuments.csv` – sample dataset / metadata  
- `practice.py`, `demo.py`, `test.py` – experimental and demo scripts  
- `temp_query.jpg` – example query image

