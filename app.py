import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Thyroid Disorder Detection",
    layout="centered"
)

# ---------------- STYLES (LIGHT MODE ONLY) ----------------
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.center {
    text-align: center;
}

.title {
    font-size: 42px;
    font-weight: 700;
    color: #1e3a8a;
    margin-top: 10px;
}

.subtitle {
    font-size: 18px;
    color: #475569;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0px 6px 16px rgba(0,0,0,0.12);
    margin-top: 20px;
}

.normal {
    background-color: #dcfce7;
    color: #166534;
    padding: 16px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
    margin-top: 20px;
}

.abnormal {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 16px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
    margin-top: 20px;
}

.footer {
    margin-top: 40px;
    font-size: 14px;
    color: #64748b;
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

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Ultrasound Image only",
    type=["jpg", "jpeg", "png"]
)

# ---------------- SHOW CARD ONLY AFTER UPLOAD ----------------
if uploaded_file:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Ultrasound Image", use_container_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        prediction = model.predict(img_array)[0][0]

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

# ---------------- FOOTER ----------------
st.markdown(
    "<div class='footer'>⚠️ This system is for decision-support only and should not replace medical diagnosis.</div>",
    unsafe_allow_html=True
)