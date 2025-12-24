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
.center {
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
}

.title {
    color: #1e3a8a;
    font-size: 40px;
    font-weight: 800;
    margin-top: 10px;
}

.subtitle {
    color: #475569;
    font-size: 18px;
    margin-bottom: 30px;
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
    margin-top: 10px;
}

.abnormal {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 14px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGO ----------------
st.markdown("<div class='center'>", unsafe_allow_html=True)
st.image("logo.png", width=120)
st.markdown("</div>", unsafe_allow_html=True)

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

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload Ultrasound Image(s) only",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ---------------- PREDICTION ----------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        image = Image.open(uploaded_file)
        st.image(
            image,
            caption=f"Uploaded Image: {uploaded_file.name}",
            use_container_width=True
        )

        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button(f"🔍 Predict {uploaded_file.name}"):
            prediction = model.predict(img_array, verbose=0)[0][0]

            if prediction > 0.5:
                st.markdown(
                    "<div class='abnormal'>🧠 Abnormal Thyroid Detected</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='normal'>✅ Normal Thyroid</div>",
                    unsafe_allow_html=True
                )

        st.markdown("</div>", unsafe_allow_html=True)