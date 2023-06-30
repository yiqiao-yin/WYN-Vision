import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Setting page title and header
st.set_page_config(page_title="WYN AI", page_icon=":robot_face:")
st.markdown(
    f"""
        <h1 style='text-align: center;'>W.Y.N. Vision 🤖</h1>
    """,
    unsafe_allow_html=True,
)


# Load model
new_model = tf.keras.models.load_model("models/toy_mnist_model.h5")


# Load image
uploaded_file = st.file_uploader(
    "Upload your file here...", type=["png", "jpeg", "jpg"]
)
if uploaded_file is not None:
    st.image(uploaded_file)

    # Convert to array
    w, h = 28, 28
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.write(f"Dimension of the original image: {image.shape}")
    image = np.resize(image, (w, h))
    st.write(f"Dimension of resized image: {image.shape}")

    # Inference
    pred = new_model.predict(image.reshape((1, w, h)))
    label = np.argmax(pred, axis=1)
    st.write(f"Classification Result: {label}")

else:
    st.warning("Please upload a jpg/png file.")
