"""App to detect face masks using a deep learning model."""

import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Load the face mask detection model
model = load_model('model.h5')

# Define class names
class_names = ['No Mask', 'Mask']

# Preprocess image
def preprocess_image(img):
    """
    Preprocess the input image to make it compatible with the model

    Args:
        img (PIL.Image.Image): The Input Image to preprocess
    
    Returns:
        numy.ndarray: The preprocessed image ready for model input
    """
    img = img.resize((128, 128))         # Resize to model input size
    img = img.convert('RGB')             # Ensure RGB format
    img = np.array(img) / 255.0          # Normalize pixel values
    img = np.expand_dims(img, axis=0)    # Add batch dimension
    return img

# Prediction function
def predict(img):
    """
    Return the prediction, if there is mask or not in input image

    Args:
        img (np.ndarray): The Preprocessed image array for model input
    
    Returns:
        Prediction Made by model
    """
    preprocessed = preprocess_image(img)
    result = model.predict(preprocessed)[0]
    prediction = np.argmax(result)
    confidence = result[prediction]
    return class_names[prediction], confidence

# Streamlit UI
st.title("Face Mask Detection App")

option = st.radio("Choose an option", ("Upload Image", "Take Photo"))

# Upload Image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        label, confidence_score = predict(image)
        st.success(f"Prediction: {label} ({confidence_score*100:.2f}% confidence)")

# Take Photo
elif option == "Take Photo":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.image(image, caption="Captured Image", use_column_width=True)
        label, confidence_score = predict(image)
        st.success(f"Prediction: {label} ({confidence_score*100:.2f}% confidence)")
