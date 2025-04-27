import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Model Loading with Error Handling ---
try:
    # Option 1: For HDF5 format
    model = tf.keras.models.load_model('./keras_model.h5')
    
    # Option 2: For SavedModel format (recommended)
    # model = tf.keras.models.load_model('./anomaly_model')  # SavedModel directory
    
except Exception as e:
    st.error(f"""
    **Model Loading Failed**
    Error: {str(e)}
    
    Troubleshooting:
    1. Verify model file exists
    2. Check TensorFlow version compatibility
    3. Convert to SavedModel format if using custom layers
    """)
    st.stop()

# --- Labels with Validation ---
try:
    labels = open('./labels.txt').read().splitlines()
    if not labels:
        st.error("Labels file is empty")
        st.stop()
except FileNotFoundError:
    st.error("labels.txt file not found")
    st.stop()

# --- Preprocessing ---
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- UI ---
st.title("Anomaly Detection Inspector")

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Upload product image", type=["jpg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)
        
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        st.image(image, caption=f'Prediction: {labels[class_idx]} | Confidence: {confidence:.2%}')
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# --- Real-Time Camera Section ---
class AnomalyDetector(VideoTransformerBase):
    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            processed = preprocess_image(pil_img)
            prediction = model.predict(processed)
            class_label = labels[np.argmax(prediction)]

            cv2.putText(img, f"{class_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return img
            
        except Exception as e:
            st.error(f"Camera processing error : {str(e)}")
            return img

if st.checkbox("Enable Real-Time Camera Anomaly Detection "):
    webrtc_streamer(key="anomaly-detector ", video_transformer_factory=AnomalyDetector)
