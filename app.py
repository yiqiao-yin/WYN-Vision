import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

# Load model
new_model = tf.keras.models.load_model('toy_mnist_model.h5')

# Load image
uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    st.image(uploaded_file)
    st.write(type(uploaded_file))

    # Inference
    # pred = new_model.predict(uploaded_file)
    # label = np.argmax(pred, axis=1)

else:
    st.warning("Please upload a jpg/png file.")
