# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import pickle
import os

MODEL_PATH = "/data/models/best_model.h5"  # Replace with your trained model path
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES_PATH = r"data\models\class_names.pkl"
with open(CLASS_NAMES_PATH, "rb") as f:
    CLASS_NAMES = pickle.load(f)

def preprocess_image(img):
    img = img.resize((224, 224))  # MobileNetV2 default input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

st.title("🐟 Fish Classifier")
st.write("Upload an image of a fish or choose a sample to predict its type.")

sample_dir = "/samples"
os.makedirs(sample_dir, exist_ok=True)  # Ensure folder exists
sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

option = st.radio("Select Input Method:", ["Upload Image", "Use Sample Image"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Use Sample Image":
    if sample_images:
        sample_choice = st.selectbox("Choose a sample image:", sample_images)
        image_path = os.path.join(sample_dir, sample_choice)
        image = Image.open(image_path).convert("RGB")
    else:
        st.warning("No sample images found. Please add them to the 'samples/' folder.")

# Prediction
if image is not None:
    st.image(image, caption="Selected Image", use_column_width=True)
    if st.button("Predict"):
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img)
        pred_idx = np.argmax(predictions)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = np.max(predictions) * 100

        st.success(f"Prediction: **{pred_class}** ({confidence:.2f}% confidence)")
    else:
        st.info("Click 'Predict' to classify the fish in the image.")