import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Thyroid Disorder Detection",
    layout="centered"
)

# ---------------- THEME TOGGLE ----------------
dark_mode = st.toggle("🌙 Dark Mode")
# ---------------- STYLES ----------------
if dark_mode:
    st.markdown("""
    <style>
    .title { color: #e5e7eb; }
    .subtitle { color: #cbd5f5; }
    .card {
        background-color: #111827;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.6);
    }
    .normal {
        background-color: #22c55e;
        color: black;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 600;
        text-align: center;
    }
    .abnormal {
        background-color: #ef4444;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 600;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .title { color: #1e3a8a; }
    .subtitle { color: #475569; }
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .normal {
        background-color: #dcfce7;
        color: #166534;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 600;
        text-align: center;
    }
    .abnormal {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 600;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOGO ----------------
st.markdown("<div class='center'>", unsafe_allow_html=True)
st.image("logo.png", width=140)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<div class='center title'>Thyroid Disorder Detection</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='center subtitle'>Upload a thyroid ultrasound image to predict Normal or Abnormal condition</div>",
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("thyroid_model.keras")

model = load_model()

# ---------------- CARD UI ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Ultrasound Image only",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Ultrasound Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            st.markdown("<div class='abnormal'>🧠 Abnormal Thyroid Detected</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='normal'>✅ Normal Thyroid</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
