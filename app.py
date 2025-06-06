import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

# Load the face mask detection model
model = load_model('model.h5')

# Define class names
class_names = ['Mask', 'No Mask']

# Preprocess image
def preprocess_image(image):
    image = image.resize((128, 128))         # Resize to model input size
    image = image.convert('RGB')             # Ensure RGB format
    image = np.array(image) / 255.0          # Normalize pixel values
    image = np.expand_dims(image, axis=0)    # Add batch dimension
    return image

# Prediction function
def predict(image):
    preprocessed = preprocess_image(image)
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
        label, confidence = predict(image)
        st.success(f"Prediction: {label} ({confidence*100:.2f}% confidence)")

# Take Photo
elif option == "Take Photo":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.image(image, caption="Captured Image", use_column_width=True)
        label, confidence = predict(image)
        st.success(f"Prediction: {label} ({confidence*100:.2f}% confidence)")
