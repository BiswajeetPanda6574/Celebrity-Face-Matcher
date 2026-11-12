# 📸 Multi-Domain Celebrity Face Matcher

A web-based AI application that matches a user's uploaded photo to a dataset of over 15,000 Bollywood and Hollywood celebrity images. It uses MTCNN for robust face detection and a pre-trained ResNet50 model (via Keras VGGFace) to extract facial embeddings and find the closest match using Cosine Similarity.

## 🚀 Live Demo
**https://celebrity-face-matcher-ebityf2uyascoskuzpm6n2.streamlit.app/**

## ✨ Features
* **Accurate Face Detection:** Uses MTCNN to locate and crop faces from uploaded images, even with complex backgrounds.
* **Deep Feature Extraction:** Leverages the ResNet50 architecture (Keras-VGGFace) to map facial features into high-dimensional embeddings.
* **Fast Similarity Matching:** Compares the uploaded face against 15,000+ pre-computed embeddings using Scikit-Learn's Cosine Similarity.
* **Cloud-Ready:** Configured for seamless deployment on Streamlit Community Cloud with all necessary Linux system dependencies.

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Frontend:** Streamlit
* **Computer Vision:** OpenCV, MTCNN, Pillow
* **Deep Learning:** TensorFlow 2.15, Keras, Keras-VGGFace
* **Machine Learning:** Scikit-learn, NumPy

## 📂 Project Structure
```text
Celebrity-Face-Matcher/
│
├── data/
│   ├── features/
│   │   ├── embedding.pkl       # 127MB pre-computed facial embeddings (Git LFS)
│   │   └── filenames.pkl       # Mapped file paths for each embedding
│   └── staging/                # Formatted celebrity image dataset
│
├── notebooks/                  # Jupyter notebooks for data processing & extraction
│   ├── 01_Dataset_Inspection.ipynb
│   ├── 02_Dataset_Processing.ipynb
│   ├── 03_Feature_Extraction.ipynb
│   └── 04_Inference_Testing.ipynb
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── packages.txt                # Linux system dependencies (for Streamlit Cloud)
└── .gitignore
