import os
import sys
import pickle
import numpy as np
from PIL import Image
import cv2
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import tensorflow as tf

# --- DEEP MODULE INJECTION (Required for VGGFace on modern setups) ---
sys.modules['keras.utils.layer_utils'] = tf.keras.utils
sys.modules['keras.utils.data_utils'] = tf.keras.utils
sys.modules['keras.backend'] = tf.keras.backend
sys.modules['keras.engine.topology'] = tf.keras.layers

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Celebrity Face Matcher", page_icon="📸", layout="centered")
st.title("📸 Multi-Domain Celebrity Face Matcher")
st.write("Upload a photo to find out which Bollywood or Hollywood celebrity you look like!")

# --- CACHE RESOURCES ---
@st.cache_resource
def load_models():
    detector = MTCNN()
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    return detector, model

@st.cache_data
def load_data():
    feature_list = np.array(pickle.load(open('data/features/embedding.pkl', 'rb')))
    filenames = pickle.load(open('data/features/filenames.pkl', 'rb'))
    return feature_list, filenames

with st.spinner("Loading AI Models & 15,000+ Embeddings..."):
    detector, model = load_models()
    feature_list, filenames = load_data()

# --- UPLOAD & INFERENCE ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    sample_img_rgb = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(sample_img_rgb, caption="Your Uploaded Image", use_container_width=True)

    if st.button("Find My Celebrity Match"):
        with st.spinner("Detecting face and extracting features..."):
            try:
                # 1. Detect Face
                results = detector.detect_faces(sample_img_rgb)
                
                if len(results) == 0:
                    st.error("No face detected! Please upload a clearer photo with a visible face.")
                else:
                    # 2. Crop Face
                    x, y, width, height = results[0]['box']
                    face = sample_img_rgb[y:y+height, x:x+width]
                    
                    # 3. Preprocess for VGGFace
                    face_image = Image.fromarray(face)
                    face_image = face_image.resize((224, 224))
                    face_array = np.asarray(face_image).astype('float32')
                    expanded_img = np.expand_dims(face_array, axis=0)
                    preprocessed_img = preprocess_input(expanded_img)
                    
                    # 4. Extract Features
                    result = model.predict(preprocessed_img, verbose=0).flatten()
                    
                    # 5. Compute Cosine Similarity
                    similarity = []
                    for i in range(len(feature_list)):
                        similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
                        
                    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
                    matched_image_path = filenames[index_pos]
                    
                    # 6. Clean Path and Extract Celebrity Name
                    clean_path = matched_image_path.replace("\\", "/")
                    if "data/" in clean_path:
                        clean_path = clean_path[clean_path.find("data/"):]
                    
                    # Grab the actual file name (e.g., 'Sunil_Dutt.12.jpg') instead of the folder
                    file_name = clean_path.split("/")[-1]
                    
                    # Split by the period to drop the '.12.jpg' and keep just the name
                    matched_celeb_slug = file_name.split(".")[0]
                    matched_celeb_name = matched_celeb_slug.replace("_", " ").title()
                    
                    # 7. Display Match
                    matched_img = Image.open(clean_path)
                    
                    with col2:
                        st.image(matched_img, caption=f"It seems like {matched_celeb_name}", use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
