import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from tensorflow.keras.applications.vgg19 import preprocess_input
import time

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('wids_gs_best_mod_tuned.keras')

model = load_model()

# Preprocess image
def preprocess_image(image):
    image = image.resize((224,224))
    image = np.array(image)
    image = np.expand_dims(image, axis = 0)
    image = preprocess_input(image)
    return image

# UI elements
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://seas.umich.edu/sites/default/files/styles/news/public/2023-07/palm-oil-plantations-and-deforestation-in-guatemala-certifying-products-as-sustainable-is-no-panacea.jpg?itok=BpoVcD9S");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 style='color: darkgreen;'>🌴 Palm Oil Plantation Detector 🌴</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color: cornsilk;'> Upload a satellite image to check if it contains a palm oil plantation. </h2>", unsafe_allow_html=True)

# File uploader
# uploaded_file = st.file_uploader("Choose an image:", type = ["jpg", "png", "jpeg"])

st.markdown("<h4 style='color: cornsilk;'> Choose an image: </h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = "Uploaded Image", use_container_width = True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Add loading effect
    with st.spinner("🔍 Analyzing image... Please wait."):
        time.sleep(2)  # Simulates processing delay
        prediction = model.predict(processed_image)

    # Interpret the prediction
    if prediction[0][0] > 0.5:
        st.markdown("<h1 style='color: cornsilk;'> A palm oil plantation is detected! </h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='color: cornsilk;'> No palm oil plantation is detected. </h1>", unsafe_allow_html=True)

    # Probability score rounded to two decimal places
    st.markdown("<h2 style='color: cornsilk;'> Prediction Confidence: </h2>", unsafe_allow_html=True)
    confidence = prediction[0][0] * 100  # Convert to percentage
    st.progress(int(confidence))
    st.markdown(f"<h3 style = 'color:cornsilk;'> Score: {prediction[0][0]:.2f} </h3>", unsafe_allow_html=True)