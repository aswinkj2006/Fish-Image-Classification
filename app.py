import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import requests
from pathlib import Path
from PIL import Image
import io

MODEL_URL = "https://raw.githubusercontent.com/aswinkj2006/Fish-Image-Classification/main/data/models/best_model.h5"
CLASSES_URL = "https://raw.githubusercontent.com/aswinkj2006/Fish-Image-Classification/main/data/models/class_names.pkl"

model_path = Path("data/models/best_model.h5")
class_names_path = Path("data/models/class_names.pkl")

if not model_path.exists():
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(requests.get(MODEL_URL).content)

if not class_names_path.exists():
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_path, "wb") as f:
        f.write(requests.get(CLASSES_URL).content)

model = tf.keras.models.load_model(model_path)
with open(class_names_path, "rb") as f:
    class_names = pickle.load(f)

st.set_page_config(page_title="Fish Species Classifier", layout="centered")

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🐟 Fish Species Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Upload an image of a fish to identify its species.</p>", unsafe_allow_html=True)

st.markdown("### 📂 Upload Your Image")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

st.markdown("### 📸 Or Try a Sample Image")

SAMPLES_FOLDER = Path("samples")
sample_images = {
    "Head Bream Fish": SAMPLES_FOLDER / "hb.jpg",
    "Horse Mackarel Fish": SAMPLES_FOLDER / "hm.jpg",
    "Sea Sprat Fish": SAMPLES_FOLDER / "ss.jpg",
    "Striped Red Mullet": SAMPLES_FOLDER / "srm.jpg"
}

sample_choice = st.selectbox("Or pick a sample image:", list(sample_images.keys()))

if st.button("Use Sample Image"):
    uploaded_file = open(sample_images[sample_choice], "rb")


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[-4:][::-1]

    st.markdown("## 🏆 Prediction Result")
    st.markdown(f"<h3 style='color: green;'>✅ {class_names[top_indices[0]]} ({predictions[top_indices[0]]*100:.2f}%)</h3>", unsafe_allow_html=True)

    st.markdown("### 🔍 Other Possible Species")
    for idx in top_indices[1:]:
        st.markdown(f"- **{class_names[idx]}** ({predictions[idx]*100:.2f}%)")
