import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load your model and labels
model = tf.keras.models.load_model('./keras_model.h5')
labels = open('labels.txt').read().splitlines()

# Preprocessing function (same as before)
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit App Title
st.title("Anomaly Detection Inspector")

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Upload product image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)

    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    st.image(image, caption=f'Prediction: {labels[class_idx]} | Confidence: {confidence:.2%}')

# --- Real-Time Camera Section ---

# Define video transformer class for real-time processing
class AnomalyDetector(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")

        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Preprocess and predict
        processed = preprocess_image(pil_img)
        prediction = model.predict(processed)
        class_label = labels[np.argmax(prediction)]

        # Overlay prediction text on frame (BGR format)
        cv2.putText(img, f"{class_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    

        return img

# Add a checkbox to enable live camera anomaly detection
if st.checkbox("Enable Real-Time Camera Anomaly Detection"):
    webrtc_streamer(key="anomaly-detector", video_transformer_factory=AnomalyDetector)
