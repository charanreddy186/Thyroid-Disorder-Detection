import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Thyroid Disorder Detection",
    layout="centered"
)

# ---------------- STYLES ----------------
st.markdown("""
<style>
.center { text-align: center; }

.title {
    color: #1e3a8a;
    font-size: 38px;
    font-weight: 800;
}

.subtitle {
    color: #475569;
    font-size: 18px;
    margin-bottom: 25px;
}

.card {
    background-color: white;
    border-radius: 16px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.1);
}

.normal {
    background-color: #dcfce7;
    color: #166534;
    padding: 14px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
}

.abnormal {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 14px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGO ----------------
st.image("logo.png", width=120)

# ---------------- TITLE ----------------
st.markdown("<div class='center title'>Thyroid Disorder Detection</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='center subtitle'>Upload thyroid ultrasound images to predict Normal or Abnormal condition</div>",
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("thyroid_model.keras")

model = load_model()

# ---------------- FILE UPLOADER ----------------
uploaded_files = st.file_uploader(
    "Upload Ultrasound Image(s) only",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ---------------- DISPLAY IMAGES ----------------
images = []

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        images.append((file.name, image))
        st.image(image, caption=f"Uploaded Image: {file.name}", use_container_width=True)

# ---------------- PREDICT ALL BUTTON ----------------
if uploaded_files and st.button("🔍 Predict All Images"):
    for name, image in images:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0][0]

        if prediction > 0.5:
            st.markdown(
                f"<div class='abnormal'>🧠 {name} → Abnormal Thyroid</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='normal'>✅ {name} → Normal Thyroid</div>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <hr style="margin-top:40px;">
    <div style="text-align:center; color:#6b7280; font-size:14px;">
        ⚠️ <b>Disclaimer:</b> This system is for <b>decision-support only</b> and should not replace
        professional medical diagnosis. Please upload <b>thyroid ultrasound images only</b>.
    </div>
    """,
    unsafe_allow_html=True
)