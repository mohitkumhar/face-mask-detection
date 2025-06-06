# 😷 Face Mask Detection App

A simple and effective web application built with **Streamlit** and **TensorFlow/Keras** to detect whether a person is wearing a face mask or not in an image. The app supports both image uploads and real-time webcam capture.

---

## 🚀 Features

- 📷 Upload an image or click a photo using your webcam
- 🤖 Real-time face mask detection using a trained CNN model
- 📊 Confidence percentage for prediction
- 🧠 Trained on a balanced dataset with augmentation and dropout for better generalization

---

## 🧠 Model Architecture

The model is a Sequential CNN architecture built with the following layers:

```

Model: "sequential"

---

# Layer (type)                Output Shape              Param \#

Conv2D (32 filters)         (None, 126, 126, 32)      896
MaxPooling2D                (None, 63, 63, 32)        0
Conv2D (64 filters)         (None, 61, 61, 64)        18496
MaxPooling2D                (None, 30, 30, 64)        0
Flatten                     (None, 57600)             0
Dense (128 neurons)         (None, 128)               7,372,928
Dropout                     (None, 128)               0
Dense (64 neurons)          (None, 64)                8,256
Dropout                     (None, 64)                0
Dense (Output - 2 classes)  (None, 2)                 130
=========================================================

Total Trainable Parameters: 7,400,706

````

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/mohitkumhar/face-mask-detection.git
cd face-mask-detection
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
face-mask-detection/
│
├── model.h5                # Trained Keras model
├── app.py                  # Streamlit application code
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📸 App Preview

> ✅ Upload an image or click a photo<br>
> 🎯 Get instant prediction with confidence score<br>
> 🧠 Powered by a deep learning model trained on real face mask datasets<br>
<center><img src="https://github.com/user-attachments/assets/f01b68bf-d80f-4301-b9d6-3f1a85fe0def" alt="App Screenshot" width="600"/></center>



---

## ✍️ Author

**Mohit Kumhar**
[GitHub Profile](https://github.com/mohitkumhar)

---

## 📃 License

This project is open-source and available under the [MIT License](LICENSE).

---

### ✅ Description for GitHub Repo

> A simple face mask detection web app using Streamlit and a custom-trained CNN model. Supports webcam input and image uploads, and shows prediction with confidence. Built with TensorFlow/Keras.


