# W.Y.N. Vision 🤖

This repository contains a simple image classification app called W.Y.N. Vision. It uses a pre-trained TensorFlow model to classify images uploaded by the user.

## Installation

To use this app, you need to have Python 3.x installed on your system. Clone this repository to your local machine and install the required dependencies using the following command:

```
pip install -r requirements.txt
```

## Usage

After installing the dependencies, you can run the app using the following command:

```
streamlit run app.py
```

This will start the app on a local server, and you can access it through your web browser.

## App Functionality

### Upload Image

The app allows you to upload an image file (in JPG or PNG format) by clicking on the "Upload your file here..." button. Once you select an image file, it will be displayed on the app.

### Image Processing

The uploaded image is preprocessed to meet the requirements of the classification model. It is resized to a fixed dimension of 28x28 pixels using bilinear interpolation. The dimensions of the original and resized images are displayed on the app.

### Image Classification

The pre-trained model (toy_mnist_model.h5) is loaded, and the resized image is fed into the model for classification. The app displays the predicted classification result as a single label.

## Note

Please make sure to place the pre-trained model file (toy_mnist_model.h5) in the same directory as the app.py file before running the app.

## License

This project is licensed under the [MIT License](LICENSE).