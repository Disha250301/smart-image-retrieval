# Smart Image Retrieval Using Image Analytics

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

An intelligent image retrieval system that retrieves visually and semantically similar images using deep feature extraction and hypergraph-based ranking. The project supports multiple feature extractors and includes an evaluation pipeline to measure retrieval performance using Precision, Recall, and Mean Average Precision (mAP).

---

## Project Overview

Traditional image search relies on manually assigned tags or metadata, which can be incomplete or inconsistent. This project follows a **Content-Based Image Retrieval (CBIR)** approach, where images are retrieved based on their visual content instead of text descriptions.

The system extracts deep features from a query image, compares them with precomputed features from the dataset, ranks the most similar images using a hypergraph-based retrieval algorithm, and displays the results through a Streamlit web application.

The project also evaluates different feature extraction models to compare their retrieval performance.

---

## Features

- Retrieve visually similar images from a dataset
- Deep feature extraction using pretrained models
- Supports **CLIP**, **ResNet50**, and **VGG16**
- Hypergraph-based similarity ranking
- Interactive Streamlit web interface
- Performance evaluation using Precision, Recall, and mAP
- Automatic ground truth generation for evaluation

---

## Tech Stack

- Python
- PyTorch
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Pillow

---

## Project Workflow

```text
               Query Image
                    │
                    ▼
        Deep Feature Extraction
                    │
                    ▼
         Feature Vector Generation
                    │
                    ▼
     Hypergraph-Based Ranking
                    │
                    ▼
      Retrieve Similar Images
                    │
                    ▼
          Display Results
```

---

## Supported Feature Extractors

| Model | Description |
|-------|-------------|
| CLIP | Generates semantic image embeddings for similarity search |
| ResNet50 | Extracts deep visual features using a residual neural network |
| VGG16 | Extracts high-level visual features using a convolutional neural network |

---

## Project Structure

```text
smart-image-retrieval/
│
├── dataset/
├── src/
├── demo.py
├── index.py
├── ranking.py
├── evaluate.py
├── evaluate_app.py
├── generate_groundtruth.py
├── requirements.txt
├── README.md
└── temp_query.jpg
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/DishaaShankar/smart-image-retrieval.git
```

Navigate to the project directory.

```bash
cd smart-image-retrieval
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Generate image features

```bash
python index.py --feature_extractor CLIP
```

### 2. Perform image ranking

```bash
python ranking.py --feature_extractor CLIP
```

### 3. Launch the Streamlit application

```bash
streamlit run demo.py
```

### 4. Evaluate retrieval performance

```bash
python evaluate.py
```

or

```bash
streamlit run evaluate_app.py
```

---

## Evaluation

The retrieval system is evaluated using standard information retrieval metrics.

| Metric | Value |
|---------|-------|
| Mean Average Precision (mAP) | **0.978** |
| Precision | Evaluated |
| Recall | Evaluated |

The evaluation compares different feature extractors and measures how effectively the system retrieves relevant images from the dataset.

---

## Future Improvements

- Support larger image datasets
- Add FAISS-based indexing for faster retrieval
- Support additional pretrained feature extraction models
- Deploy the application using Docker or a cloud platform
- Add support for custom image datasets

---

## Learning Outcomes

This project helped strengthen my understanding of:

- Content-Based Image Retrieval (CBIR)
- Deep Feature Extraction
- Image Similarity Search
- Hypergraph-Based Ranking
- Information Retrieval Metrics
- Streamlit Application Development
- Model Evaluation and Performance Analysis

---

## Author

**Disha Shankar**

GitHub: https://github.com/DishaaShankar

---

## License

This project is licensed under the MIT License.
