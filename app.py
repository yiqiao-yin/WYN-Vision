import cv2
import matplotlib.pyplot as plt
import numpy as np
import openai
from PIL import Image
import streamlit as st
import tensorflow as tf
import urllib.request


# Backend
openai.api_key = st.secrets["OPENAI_API_KEY"]


def text_to_img(prompt: str) -> np.ndarray:
    """ Takes in a prompt and returns an image generated by OpenAI's Image API.
    
    Args:
    prompt : A string, the prompt that will be sent to the OpenAI API to generate the image.
    
    Returns:
    image : A numpy ndarray object, which represents the image generated corresponding to the input prompt.
    """
    # Create an image from prompt using OpenAI's API 
    response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
    
    # Get the url of the generated image and return it as a numpy array object.
    image_url = response["data"][0]["url"]
    image = url_to_image(image_url)

    return image


def url_to_image(url: str) -> np.ndarray:
    """
    Downloads an image from a given URL using urllib, and converts it to a numpy array for use with OpenCV.

    Args:
        url: A string representing URL of the image to be downloaded.

    Returns:
        A numpy array representing the downloaded image.

    """

    # Download the image using urllib.
    with urllib.request.urlopen(url) as url_response:
        img_array = bytearray(url_response.read())

    # Convert the byte array to a numpy array for use with OpenCV.
    img = cv2.imdecode(np.asarray(img_array), cv2.IMREAD_UNCHANGED)

    # Return the image as a numpy array.
    return img

# UI:

# Setting page title and header
st.set_page_config(page_title="WYN AI", page_icon=":robot_face:")
st.markdown(
    f"""
        <h1 style='text-align: center;'>W.Y.N. Vision 👓</h1>
    """,
    unsafe_allow_html=True,
)

# Sidebar
task = st.sidebar.radio(
    "Choose a task:",
    (
        "Image Classification",
        "Image Segmentation",
        "Text-to-Image",
        "Next"
    )
)
st.sidebar.markdown(
    "@ [Yiqiao Yin](https://www.y-yin.io/) | [LinkedIn](https://www.linkedin.com/in/yiqiaoyin/) | [YouTube](https://youtube.com/YiqiaoYin/)"
)

# Load model
new_model = tf.keras.models.load_model("models/toy_mnist_model.h5")
if new_model is not None:
    st.success("Load a neural network model successfully.")


# Choose task
if task == "Image Classification":
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
elif task == "Text-to-Image":
    with st.form(key="my_form", clear_on_submit=True):
        text_prompt = st.text_input('Write what you want to create:', 'a cat in front of a fire place')
        submit_button = st.form_submit_button(label="Create!")
    if submit_button == True:
        img_array = text_to_img(prompt=text_prompt)
        st.write(img_array.shape)
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(img_array)
        st.pyplot(fig)
    else:
        st.warning("Please enter a text and click the 'Create!' button.")
else:
    st.warning("Please select a task from the sidebar on the left.")
