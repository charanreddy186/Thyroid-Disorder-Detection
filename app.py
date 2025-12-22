import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Thyroid Disorder Detection",
    layout="centered"
)

# ----------------- CUSTOM CSS -----------------
st.markdown("""
    <style>
        .title-text {
            font-size: 40px;
            font-weight: bold;
            color: #1f4e79;
            text-align: center;
        }
        .subtitle-text {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-bottom: 20px;
        }
        .result-normal {
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 10px;
            font-size: 20px;
            color: #2e7d32;
            text-align: center;
            font-weight: bold;
        }
        .result-abnormal {
            background-color: #fdecea;
            padding: 15px;
            border-radius: 10px;
            font-size: 20px;
            color: #c62828;
            text-align: center;
            font-weight: bold;
        }
        .warning-box {
            font-size: 14px;
            color: #555;
            text-align: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- LOGO -----------------
st.image("logo.png", width=150)

# ----------------- TITLE -----------------
st.markdown('<div class="title-text">Thyroid Disorder Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Upload a thyroid ultrasound image to predict Normal or Abnormal condition</div>', unsafe_allow_html=True)

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("thyroid_model.keras")

model = load_model()

# ----------------- FILE UPLOAD -----------------
uploaded_file = st.file_uploader(
    "Upload Ultrasound Image only",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            st.markdown(
                '<div class="result-abnormal">🧠 Abnormal Thyroid Detected</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-normal">✅ Normal Thyroid</div>',
                unsafe_allow_html=True
            )

