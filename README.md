# 🐟 Fish Species Classification Project

A comprehensive deep learning project for classifying fish species using transfer learning and custom CNN models, with an interactive Streamlit web application for real-time predictions.

## 🎯 Project Overview

This project implements multiclass fish image classification using deep learning techniques. It compares multiple model architectures including custom CNN and pre-trained models (VGG16, ResNet50, MobileNetV2, EfficientNetB0) to find the best performing solution for fish species identification.

## ✨ Features

- **Multiple Model Training**: CNN from scratch + 4 transfer learning models
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score comparison
- **Interactive Web App**: Streamlit-based deployment for real-time predictions
- **Visualization**: Training history plots, confusion matrices, model comparisons
- **Best Model Selection**: Automatic selection and saving of top-performing model
- **Confidence Scoring**: Prediction confidence with visual indicators

## 📁 Dataset Structure

Your dataset should be organized as follows:
```
data/
├── train/
│   ├── animal_fish/
│   ├── animal_fish_bass/
│   ├── fish_sea_food_black_sea_sprat/
│   ├── fish_sea_food_gilt_head_bream/
│   ├── fish_sea_food_hourse_mackarel/
│   ├── fish_sea_food_red_mullet/
│   ├── fish_sea_food_red_sea_bream/
│   ├── fish_sea_food_sea_bass/
│   ├── fish_sea_food_shrimp/
│   ├── fish_sea_food_striped_red_mullet/
│   └── fish_sea_food_trout/
├── valid/
│   └── [same structure as train]
└── test/
    └── [same structure as train]
```

## 🧠 Supported Fish Species

The model can classify the following fish species:
- **Animal Fish** (Generic fish category)
- **Animal Fish Bass** 
- **Black Sea Sprat**
- **Gilt Head Bream**
- **Horse Mackerel** 
- **Red Mullet**
- **Red Sea Bream**
- **Sea Bass**
- **Shrimp**
- **Striped Red Mullet**
- **Trout**

## 📁 Project Structure

```
Fish-Image-Classification/
├── data/                                 # Dataset directory
│   ├── train/                           # Training images
│   ├── valid/                           # Validation images  
│   ├── test/                            # Test images
|   └── models/                          # Generated model files
│       ├── best_model.h5                # Best trained model
│       └── class_names.pkl              # Clean class names
├── Fish Image Classification Aswin K J.ipynb   # Training notebook
├── app.py                     # Streamlit web application
├── requirements.txt                     # Python dependencies cor Streamlit
└── README.md                           # Project documentation
```

## 🏗️ Model Architecture

The project implements and compares 5 different architectures:

1. **Custom CNN** - Built from scratch with 4 convolutional layers
2. **VGG16** - Transfer learning with pre-trained ImageNet weights
3. **ResNet50** - Deep residual network for complex feature learning
4. **MobileNetV2** - Lightweight model optimized for mobile deployment
5. **EfficientNetB0** - Compound scaling method for optimal efficiency

## 📊 Model Evaluation

Each model is evaluated using:
- **Accuracy** (Validation & Test)
- **Precision** (Macro average)
- **Recall** (Macro average) 
- **F1-Score** (Macro average)
- **Confusion Matrix**
- **Training History Plots**

The best performing model is automatically selected and saved for deployment.

## 🌐 Web Application Features

- **Drag & Drop Upload** - Easy image uploading interface
- **Real-time Prediction** - Instant species classification
- **Confidence Scoring** - Visual confidence indicators and gauges
- **Top 3 Predictions** - Shows alternative predictions with confidence
- **Detailed Charts** - Interactive confidence visualization for all classes
- **Responsive Design** - Works on desktop and mobile devices

## 🛠️ Technical Details

- **Framework**: TensorFlow/Keras for deep learning
- **Frontend**: Streamlit for web application
- **Image Processing**: PIL for image preprocessing
- **Visualization**: Plotly for interactive charts
- **Data Augmentation**: Rotation, zoom, flip, shear transforms
- **Transfer Learning**: Fine-tuned pre-trained models
- **Model Persistence**: HDF5 format for model storage
