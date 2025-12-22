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
dark_mode = st.toggle("🌗 Dark Mode", value=False)

# ---------------- STYLES ----------------
if dark_mode:
    bg = "#0f172a"
    card = "#020617"
    text = "#e5e7eb"
    subtitle = "#94a3b8"
else:
    bg = "#f8fafc"
    card = "#ffffff"
    text = "#1e3a8a"
    subtitle = "#475569"

st.markdown(f"""
<style>
.main {{
    background-color: {bg};
}}
.center {{
    text-align: center;
}}
.title {{
    font-size: 42px;
    font-weight: 700;
    color: {text};
}}
.subtitle {{
    font-size: 18px;
    color: {subtitle};
    margin-bottom: 30px;
}}
.card {{
    background-color: {card};
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}}
.normal {{
    background-color: #16a34a;
    color: white;
    padding: 15px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
}}
.abnormal {{
    background-color: #dc2626;
    color: white;
    padding: 15px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
}}
.footer {{
    margin-top: 40px;
    font-size: 14px;
    color: {subtitle};
    text-align: center;
}}
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
