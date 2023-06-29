import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

# Load model
new_model = tf.keras.models.load_model('toy_mnist_model.h5')

# Load image
img = cv2.imread('/content/sample.png')

# Plot
fig, ax = plt.subplots()
ax.imshow(img, bins=20)
st.pyplot(img)

# Write
st.write("hello world")