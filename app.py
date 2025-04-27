import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Enable GPU memory growth to prevent memory allocation errors (if GPU available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        st.warning(f"GPU memory growth could not be set : {e}")

# Streamlit App Title
st.title("Anomaly Detection Inspector")

# Load your model and labels with error handling
try:
    st.info("Loading model....")
    model = tf.keras.models.load_model('./keras_model.h5')
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop the app if model can't be loaded

try:
    labels = open('./labels.txt').read().splitlines()
    st.write(f"Labels loaded: {labels}")
except Exception as e:
    st.error(f"Error loading labels: {e}")
    st.stop()

# Preprocessing function
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    # Ensure float32 dtype for TensorFlow model
    img_array = img_array.astype(np.float32)
    return np.expand_dims(img_array, axis=0)

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Upload product image", type=["jpg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)

        class_idx = np.argmax(prediction)
        confidence = np.max(prediction)

        st.image(image, caption=f'Prediction: {labels[class_idx]} | Confidence: {confidence:.2%}')
    except Exception as e:
        st.error(f"Prediction error: {e}")

# --- Real-Time Camera Section ---

# Define video transformer class for real-time processing
class AnomalyDetector(VideoTransformerBase):
    def transform(self, frame):
        try:
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
        except Exception as e:
            # Log error and return original frame if prediction fails
            print(f"Error in video frame processing: {e}")
            return frame.to_ndarray(format="bgr24")

# Checkbox to enable live camera anomaly detection
if st.checkbox("Enable Real-Time Camera Anomaly Detection"):
    webrtc_streamer(key="anomaly-detector", video_transformer_factory=AnomalyDetector)
