import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from pathlib import Path
from PIL import Image

# ==============================
# Paths
# ==============================
model_json_path = Path("data/models/mobilenetv2_model.json")
model_weights_path = Path("data/models/mobilenetv2_weights.h5")
class_names_path = Path("data/models/class_names.pkl")

if not model_json_path.exists() or not model_weights_path.exists() or not class_names_path.exists():
    raise FileNotFoundError("Model JSON, weights file, or class names file not found in data/models/ folder.")

# ==============================
# Load model from JSON + weights
# ==============================
with open(model_json_path, "r") as json_file:
    loaded_model_json = json_file.read()

model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights(model_weights_path)

# Load class names
with open(class_names_path, "rb") as f:
    class_names = pickle.load(f)

# ==============================
# Streamlit UI
# ==============================
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

# ==============================
# Prediction
# ==============================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)

    predictions = model.predict(img_array)[0]
    st.image(image, caption="Uploaded Image", use_column_width=True)

    top_indices = predictions.argsort()[-4:][::-1]  # top 4

    st.markdown("## 🏆 Prediction Result")
    st.markdown(
        f"<h3 style='color: green;'>✅ {class_names[top_indices[0]]} ({predictions[top_indices[0]]*100:.2f}%)</h3>",
        unsafe_allow_html=True
    )

    st.markdown("### 🔍 Other Possible Species")
    for idx in top_indices[1:]:
        st.markdown(f"- **{class_names[idx]}** ({predictions[idx]*100:.2f}%)")
