import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

# Load model
new_model = tf.keras.models.load_model('toy_mnist_model.h5')

# Load image
# Insert a file uploader that accepts multiple files at a time
uploaded_file = st.file_uploader("Choose a jpg/png file")
if uploaded_file is not None:
    # Success message
    st.success("File uploaded successfully.")

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(uploaded_file)
    st.pyplot(fig)

else:
    st.warning("Please upload a csv file.")
