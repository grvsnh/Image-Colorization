import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("colorization_model.h5")

st.title("🎨 Image Colorization App")

st.write("Upload a grayscale image and the model will attempt to colorize it.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    # Resize to model size
    image = cv2.resize(image, (32,32))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=(0,-1))

    # Predict color
    prediction = model.predict(gray)[0]

    st.subheader("Grayscale Input")
    st.image(gray[0], clamp=True)

    st.subheader("Predicted Color")
    st.image(prediction)
