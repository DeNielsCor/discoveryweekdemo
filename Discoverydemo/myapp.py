import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("keras_model.h5")

# Load labels
class_names = open("labels.txt", "r").readlines()

st.set_page_config(page_title="Image Classifier", page_icon="🤖")

st.title("🤖 AI Image Classifier")
st.write("Upload an image and let the AI predict what it is!")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

def preprocess_image(image):
    image = image.resize((224, 224))  # Teachable Machine default
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence = prediction[0][index]

    st.subheader("🔍 Prediction:")
    st.write(f"**{class_name}**")
    st.write(f"Confidence: {confidence:.2%}")