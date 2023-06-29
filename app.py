import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Load model
new_model = tf.keras.models.load_model('toy_mnist_model.h5')

# Load image
uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    st.image(uploaded_file)

    # Convert to array
    image = Image.open(uploaded_file)
    image = np.array(image)[:,:,0:3]
    image = np.resize(image, (28, 28))

    # Inference
    st.write(image.shape)
    # pred = new_model.predict(img_array.reshape((1, w, h)))
    # label = np.argmax(pred, axis=1)
    # st.write(f"Classification Result: {label}")

else:
    st.warning("Please upload a jpg/png file.")
