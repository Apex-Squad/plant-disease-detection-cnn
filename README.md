# 🌿 Plant Disease Detection using CNN

A Deep Learning project for detecting plant diseases from leaf images using a Convolutional Neural Network (CNN).

## 📌 Project Overview

This project uses a CNN model to classify plant leaf images into **38 different classes**, including healthy plants and various diseases.

The model was trained using the **PlantVillage dataset**.

## 🧠 Model

- Architecture: Convolutional Neural Network (CNN)
- Input Size: 64 × 64 × 3
- Number of Classes: 38
- Batch Normalization
- Data Augmentation
- Global Average Pooling
- Dropout
- Adam Optimizer
- Early Stopping
- ReduceLROnPlateau

## 📊 Results

- Validation Accuracy: **96.69%**
- Test Accuracy: **~97%**
- Test Images: **4,345**

## 🚀 Web Demo

The trained model is integrated with a **Gradio** web interface.

Users can upload a plant leaf image and receive:

- Predicted disease
- Prediction confidence

## 📁 Project Files

- `app.py` — Gradio web application
- `best_CNN_model.keras` — trained CNN model
- `Plant_Disease_CNN_Improved.ipynb` — complete training notebook
- `requirements.txt` — required Python packages

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Gradio
- CNN
- PlantVillage Dataset

## 👥 Team

**Apex Squad**

---

Developed as a Deep Learning / Computer Vision project.
