import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
 
# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Bahraini Currency Recognition",
    page_icon="💵",
    layout="centered",
)
 
# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #1b5e20, #2e7d32, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.6rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 24px rgba(27, 94, 32, 0.35);
        margin: 1rem 0;
    }
    .result-icon { font-size: 3rem; }
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .result-confidence {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 0.2rem;
    }
    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
        border: 2px solid #2e7d32;
    }
    section[data-testid="stSidebar"] { border-right: 1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)
 
st.markdown('<div class="main-title">💵 Bahraini Currency Recognition</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Snap or upload a photo and let the model '
    'identify the note or coin</div>',
    unsafe_allow_html=True)
 
# ----------------------------------------------------------------------
# Model (cached so it only loads once, not on every interaction)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("currency_5model.keras")
 
model = load_model()
 
class_names = ["0.05", "0.100", "0.25", "0.5 BD", "0.50",
               "1BD", "5 BD", "10 BD", "20 BD"]
 
# a simple coin/note icon per class, purely cosmetic
class_icons = {
    "0.05": "🪙", "0.100": "🪙", "0.25": "🪙", "0.50": "🪙",
    "0.5 BD": "💵", "1BD": "💵", "5 BD": "💵", "10 BD": "💵", "20 BD": "💵",
}
 
# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app identifies Bahraini currency (coins and notes) from a "
        "photo using a trained neural network.")
 
    st.header("💱 Denominations")
    for name in class_names:
        st.write(f"{class_icons[name]}  {name}")
 
    st.header("📸 Tips for best results")
    st.write(
        "- Fill most of the frame with the currency\n"
        "- Use a plain, uncluttered background\n"
        "- Good, even lighting helps")
 
# ----------------------------------------------------------------------
# Image input: upload, or open/close the camera
# ----------------------------------------------------------------------
if "camera_open" not in st.session_state:
    st.session_state.camera_open = False
 
col1, col2 = st.columns(2)
with col1:
    uploaded_image = st.file_uploader(
        "📁 Upload a photo", type=["jpg", "jpeg", "png"])
with col2:
    st.write("")  # small vertical spacer to align the button with the uploader
    label = "✖️ Close Camera" if st.session_state.camera_open else "📷 Open Camera"
    if st.button(label, use_container_width=True):
        st.session_state.camera_open = not st.session_state.camera_open
 
camera_image = None
if st.session_state.camera_open:
    camera_image = st.camera_input("Take a live photo")
 
image_file = camera_image if camera_image is not None else uploaded_image
 
# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Your photo", use_container_width=True)
 
    with st.spinner("🔍 Analyzing currency..."):
        model_input = image.resize((64, 64))
        model_input = np.array(model_input) / 255
        model_input = np.expand_dims(model_input, axis=0)
        prediction = model.predict(model_input, verbose=0)
 
    predicted_class = np.argmax(prediction)
    confidence = float(prediction[0][predicted_class])
    name = class_names[predicted_class]
    icon = class_icons[name]
 
    st.markdown(f"""
    <div class="result-card">
        <div class="result-icon">{icon}</div>
        <div class="result-label">{name}</div>
        <div class="result-confidence">{confidence:.0%} confidence</div>
    </div>
    """, unsafe_allow_html=True)
 
    if confidence >= 0.8:
        st.balloons()
    elif confidence < 0.5:
        st.warning(
            "⚠️ Low confidence - the model isn't sure about this one.")
 
    with st.expander("📊 Show full prediction breakdown"):
        probs = {
            f"{class_icons[class_names[i]]} {class_names[i]}": float(prediction[0][i])
            for i in range(len(class_names))}
        st.bar_chart(probs)
 
else:
    st.info("👆 Upload a photo or open the camera to get started.")
