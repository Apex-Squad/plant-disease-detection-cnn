import gradio as gr
import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("best_CNN_model.keras")

# Class names
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


def predict_disease(image_path):

    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Resize
    img_resized = cv2.resize(img, (64, 64))

    # Normalize
    img_normalized = img_resized.astype("float32") / 255.0

    # Add batch dimension
    img_input = np.expand_dims(img_normalized, axis=0)

    # Prediction
    pred_probs = model.predict(img_input, verbose=0)

    # Predicted class
    pred_class = np.argmax(pred_probs[0])

    # Confidence
    confidence = np.max(pred_probs[0]) * 100

    # Class name
    predicted_name = class_names[pred_class]

    return predicted_name, f"{confidence:.2f}%"


# Gradio Interface
demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(
        type="filepath",
        label="Upload Plant Leaf Image"
    ),
    outputs=[
        gr.Textbox(label="Predicted Disease"),
        gr.Textbox(label="Confidence")
    ],
    title="🌿 Plant Disease Detection",
    description="Upload an image of a plant leaf to predict its disease."
)

demo.launch()
