import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openai
import streamlit as st
import tensorflow as tf
from PIL import Image

# Backend
openai.api_key = st.secrets["OPENAI_API_KEY"]


def text_to_img(prompt: str) -> np.ndarray:
    """Takes in a prompt and returns an image generated by OpenAI's Image API.

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
        <h1 style='text-align: center;'>Vision 👓</h1>
    """,
    unsafe_allow_html=True,
)

# Sidebar
task = st.sidebar.radio(
    "Choose a task:",
    (
        "Image Classification",
        "Cancer Segmentation",
        "Denoise Image",
        "Text-to-Image",
        "Next",
    ),
)
st.sidebar.markdown(
    "@ [Yiqiao Yin](https://www.y-yin.io/) | [LinkedIn](https://www.linkedin.com/in/yiqiaoyin/) | [YouTube](https://youtube.com/YiqiaoYin/)"
)


# Choose task
if task == "Image Classification":
    st.markdown(
        """
        To learn more about image classification, please refer to this [notebook](https://github.com/yiqiao-yin/WYNAssociates/blob/main/docs/ref-deeplearning/ex02%20-%20ann%20and%20cnn.ipynb).

        ⚠️⚠️⚠️To interact with the app, you'll need a picture. You can find a sample picture [here](https://github.com/yiqiao-yin/WYN-Vision/tree/main/pics).
    """
    )
    # Load model
    new_model = tf.keras.models.load_model("models/toy_mnist_model.h5")
    if new_model is not None:
        st.success("Load a neural network model successfully.")

    # Load image
    uploaded_file = st.sidebar.file_uploader(
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
elif task == "Cancer Segmentation":
    st.markdown(
        """
        H&E stained images from five prostate cancer Tissue Microarrays 
        (TMAs) and corresponding Gleason annotation masks. In the masks,
        pixel indices correspond to classes as follows: 0=Benign (green),
        1=Gleason_3 (blue), 2=Gleason_4 (yellow), 3=Gleason_5 (red),
        4=unlabelled (white). The original site is from Harvard University.
        Please see [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP) or use the following citation.
        ```
        @data{DVN/OCYCMP_2018,
            author = {Arvaniti, Eirini and Fricker, Kim and Moret, Michael and Rupp, Niels and Hermanns, Thomas and Fankhauser, Christian and Wey, Norbert and Wild, Peter and Rüschoff, Jan Hendrik and Claassen, Manfred},
            publisher = {Harvard Dataverse},
            title = {{Replication Data for: Automated Gleason grading of prostate cancer tissue microarrays via deep learning.}},
            year = {2018},
            version = {V1},
            doi = {10.7910/DVN/OCYCMP},
            url = {https://doi.org/10.7910/DVN/OCYCMP}
        }
        ```

        To learn more about image segmentation, one can use this notebook by Mr. Yin, 
        see [here](https://github.com/yiqiao-yin/WYNAssociates/blob/main/docs/ref-deeplearning/ex09%20-%20image%20segmentation.ipynb).
        For publication, please cite the [source](https://doi.org/10.7910/DVN/OCYCMP).
        Github source, see [here](https://github.com/eiriniar/gleason_CNN). 

        ⚠️⚠️⚠️To interact with the app, you'll need a picture. You can find a sample picture [here](https://github.com/yiqiao-yin/WYN-Vision/tree/main/pics).
    """
    )
    st.success(
        "To demonstrate image segmentation task here, we use a binary mask generator by U-net trained on the above prostate cancer data. The mask highlights the cancerous regions, e.g. Gleason Score greater than 4."
    )
    # Load model
    new_model = tf.keras.models.load_model("models/unet_6_6_allgleason_path1_.h5")
    if new_model is not None:
        st.success("Load a neural network model successfully.")

    # Load image
    uploaded_file = st.sidebar.file_uploader(
        "Upload your file here...", type=["png", "jpeg", "jpg"]
    )
    if uploaded_file is not None:
        st.image(uploaded_file)

        # Convert to array
        w, h = 128, 128
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.write(f"Dimension of the original image: {image.shape}")
        image = cv2.resize(image, (w, h))
        st.write(f"Dimension of resized image: {image.shape}")

        # Inference
        pred = new_model.predict(image.reshape((1, w, h, 3)))
        st.success("AI finished generating the mask!")
        mask = pred[0, :, :, 0]

        # Form
        with st.form("form_to_show_gleason_visualization"):
            st.warning(
                "The transparency level shows highlight of Gleason greater than 4, e.g. likely to be cancerous cells."
            )
            alpha = st.slider("Transparency of mask:", 0, 100, 1)
            submitted = st.form_submit_button("Submit")
            if submitted:
                # Plot image
                fig, ax = plt.subplots()
                ax.axis("off")
                ax.imshow(image)
                ax.imshow(mask, alpha=np.round(float(alpha) / 100, 1), cmap="RdPu")
                st.pyplot(fig)
    else:
        st.warning("Please upload a jpg/png file.")
elif task == "Denoise Image":
    st.markdown(
        """
        **Autoencoders** are based on _Neural Networks (NNs)_ and are known as **Convolutional Neural Networks** (**CNNs** or **convnets**). A **convnet** is a __Deep Learning algorithm__ which takes an input image, assign importance (learnable weight, biases and retains spatial relationships in the data into each one of theirs layers) to various aspects/parts in the image and is able to differentiate/reconstruct the same.

        The general idea behind this kind of code can be visualized here:

        ![General autoencoder](https://www.pyimagesearch.com/wp-content/uploads/2020/02/keras_denoising_autoencoder_overview.png)

        Then, the **autoencoder** compreehends an _encoder_ and a _decoder_. The **encoder** does the _encoding process_, i.e., transforms the image into a _compressed representation_ at the same time that starts the noisy reduction. Then, the _compressed representation_ goes to **decoder** that performs the _decoder process_, restoring the image to its true and recognizable shape. At the end of the process, we remove almost all noise in the image.

        ⚠️⚠️⚠️To interact with the app, you'll need a picture. You can find a sample picture [here](https://github.com/yiqiao-yin/WYN-Vision/tree/main/pics).

    """
    )

    # Load model
    new_model = tf.keras.models.load_model("models/denoising_mnist_ae_model.h5")

    # Load image
    uploaded_file = st.sidebar.file_uploader(
        "Upload your file here...", type=["png", "jpeg", "jpg"]
    )
    if uploaded_file is not None:
        st.image(uploaded_file)

        # Convert to array
        w, h = 28, 28
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.write(f"Dimension of the original image: {image.shape}")
        image = cv2.resize(image, (w, h))
        image = image[:, :, 0]
        st.write(f"Dimension of resized image: {image.shape}")

        # Inference
        pred = new_model.predict(image.reshape((1, w, h, 1)))
        st.success("AI finished generating denoising process!")
        denoised_img = pred[0, :, :, 0]

        # Form
        with st.form("form_to_show_gleason_visualization"):
            st.warning(
                "The transparency level shows highlight of denoised image over the original noisy image."
            )
            alpha = st.slider("Transparency of denoised image:", 0, 100, 1)
            submitted = st.form_submit_button("Submit")
            if submitted:
                # Plot image
                fig, ax = plt.subplots()
                ax.axis("off")
                ax.imshow(image, cmap="Reds")
                ax.imshow(
                    denoised_img, alpha=np.round(float(alpha) / 100, 1), cmap="Reds"
                )
                st.pyplot(fig)
    else:
        st.warning("Please upload a jpg/png file.")
elif task == "Text-to-Image":
    with st.form(key="my_form"):
        text_prompt = st.text_input(
            "Write what you want to create:", "a cat in front of a fire place"
        )
        submit_button = st.form_submit_button(label="Create!")
    if submit_button == True:
        img_array = text_to_img(prompt=text_prompt)
        st.write(img_array.shape)
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(img_array[:, :, ::-1])
        st.pyplot(fig)
    else:
        st.warning("Please enter a text and click the 'Create!' button.")
else:
    st.warning("Please select a task from the sidebar on the left.")
