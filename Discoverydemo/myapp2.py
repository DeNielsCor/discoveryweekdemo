import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Image Classifier - Level Up",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

model = load_model()
class_names = open("labels.txt", "r").readlines()

# -----------------------------
# TITLE
# -----------------------------
st.title("🤖 Smart Image Classifier (Webcam Level-Up)")
st.write("Upload, take a picture, or use live webcam AI with prediction overlays!")

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "📤 Upload Images",
    "📸 Take Picture",
    "🎥 Live Webcam"
])

images = []

# -----------------------------
# TAB 1: Upload
# -----------------------------
with tab1:
    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            images.append(Image.open(file).convert("RGB"))

# -----------------------------
# TAB 2: Camera Snapshot
# -----------------------------
with tab2:
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        images.append(Image.open(camera_image).convert("RGB"))

# -----------------------------
# TAB 3: LIVE WEBCAM 🔥
# -----------------------------
with tab3:
    st.info("🎥 Live AI (prediction every 10 seconds)")

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.last_prediction_time = 0
            self.label = "Waiting for prediction..."
            self.confidence = 0

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            current_time = time.time()

            # Predict every 10 seconds
            if current_time - self.last_prediction_time > 10:
                processed = preprocess_frame(img)
                prediction = model.predict(processed, verbose=0)
                index = np.argmax(prediction)

                self.label = class_names[index].strip()
                self.confidence = prediction[0][index]
                self.last_prediction_time = current_time

            # Draw prediction on frame
            text = f"{self.label} ({self.confidence:.2%})"

            # Add background for text
            cv2.rectangle(img, (10, 10), (500, 60), (0, 0, 0), -1)
            cv2.putText(
                img,
                text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            return img

    webrtc_streamer(
        key="webcam_upgrade",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# -----------------------------
# PREDICTION (UPLOAD + CAMERA)
# -----------------------------
if images:
    for idx, image in enumerate(images):
        st.divider()
        st.image(image, caption=f"Image {idx+1}", use_container_width=True)

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image, verbose=0)
        index = np.argmax(prediction)

        class_name = class_names[index].strip()
        confidence = prediction[0][index]

        st.subheader("🔍 Prediction")
        st.success(f"{class_name}")
        st.write(f"Confidence: {confidence:.2%}")

        st.progress(float(confidence))

        if confidence > 0.85:
            st.info("🤖 The model is very confident.")
        elif confidence > 0.6:
            st.warning("🤔 The model is somewhat confident.")
        else:
            st.error("⚠️ Low confidence.")

        st.subheader("📊 All Class Probabilities")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{class_names[i].strip()}: {prob:.2%}")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🧾 Model Info")
st.sidebar.write("**Model Type:** Teachable Machine (Keras)")
st.sidebar.write(f"**Classes:** {len(class_names)}")

with st.sidebar.expander("📋 Show class labels"):
    for label in class_names:
        st.write(label.strip())

st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ using Streamlit")