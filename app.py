import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bahraini Currency Recognition",
    page_icon="💵",
    layout="wide"
)


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>

    /* =========================
       MAIN PAGE
       ========================= */

    .stApp {
        background: linear-gradient(
            135deg,
            #f4f8f1 0%,
            #ffffff 50%,
            #eef6ed 100%
        );
    }


    /* =========================
       ALL TEXT
       ========================= */

    .stMarkdown p,
    .stMarkdown li,
    .stText,
    label {
        color: #26352b !important;
    }


    /* =========================
       MAIN TITLE
       ========================= */

    .main-title {
        text-align: center;
        color: #176b3a !important;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #536653 !important;
        font-size: 18px;
        margin-bottom: 30px;
    }


    /* =========================
       HEADINGS
       ========================= */

    h1,
    h2,
    h3,
    h4 {
        color: #176b3a !important;
    }


    /* =========================
       TABS
       ========================= */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 25px;
        color: #176b3a !important;
        font-weight: 700;
        border-radius: 10px 10px 0px 0px;
    }

    .stTabs [data-baseweb="tab"] p {
        color: inherit !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #176b3a !important;
        color: white !important;
    }

    .stTabs [aria-selected="true"] p {
        color: white !important;
    }


    /* =========================
       BUTTONS
       ========================= */

    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(
            90deg,
            #176b3a,
            #2f8f57
        ) !important;

        color: white !important;
        border: none !important;
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 25px;
    }

    .stButton > button p,
    .stFormSubmitButton > button p {
        color: white !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: linear-gradient(
            90deg,
            #12542d,
            #247343
        ) !important;

        color: white !important;
    }


    /* =========================
       INFO CARDS
       ========================= */

    .info-card {
        background-color: #ffffff !important;
        color: #26352b !important;

        padding: 20px;

        border-radius: 15px;

        border-left: 5px solid #c5a34a;

        box-shadow:
            0px 4px 12px rgba(23, 107, 58, 0.08);

        margin-bottom: 20px;
    }

    .info-card h3 {
        color: #176b3a !important;
        margin-bottom: 10px;
    }

    .info-card p {
        color: #26352b !important;
        font-size: 17px;
    }

    .info-card b {
        color: #176b3a !important;
    }


    /* =========================
       RESULT CARD
       ========================= */

    .result-card {
        background: linear-gradient(
            135deg,
            #176b3a,
            #2f8f57
        );

        border-radius: 18px;

        padding: 25px;

        text-align: center;

        color: white;

        box-shadow:
            0px 8px 24px rgba(23, 107, 58, 0.25);

        margin: 15px 0;
    }

    .result-icon {
        font-size: 50px;
    }

    .result-label {
        font-size: 30px;
        font-weight: 700;
        color: white !important;
    }

    .result-confidence {
        font-size: 18px;
        color: white !important;
        opacity: 0.9;
    }


    /* =========================
       METRIC
       ========================= */

    [data-testid="stMetric"] {
        background-color: #ffffff !important;

        border: 2px solid #d4e5d2;

        border-radius: 12px;

        padding: 15px;

        box-shadow:
            0px 3px 10px rgba(23, 107, 58, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: #536653 !important;
    }

    [data-testid="stMetricLabel"] p {
        color: #536653 !important;
    }

    [data-testid="stMetricValue"] {
        color: #176b3a !important;
    }

    [data-testid="stMetricValue"] div {
        color: #176b3a !important;
    }


    /* =========================
       FILE UPLOADER
       ========================= */

    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 2px dashed #9ab88e !important;
        border-radius: 15px !important;
        padding: 10px !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: none !important;
    }

    [data-testid="stFileUploader"] section > div {
        background-color: #ffffff !important;
    }

    [data-testid="stFileUploader"] label {
        color: #26352b !important;
    }

    [data-testid="stFileUploader"] small {
        color: #536653 !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #f4f8f1 !important;
        color: #176b3a !important;
        border: 1px solid #9ab88e !important;
    }

    [data-testid="stFileUploader"] button span {
        color: #176b3a !important;
    }

    [data-testid="stFileUploader"] p {
        color: #26352b !important;
    }


    /* =========================
       CHECKBOX
       ========================= */

    [data-testid="stCheckbox"] label {
        color: #26352b !important;
    }

    [data-testid="stCheckbox"] p {
        color: #26352b !important;
    }


    /* =========================
       TABLE
       ========================= */

    [data-testid="stTable"] {
        background-color: #ffffff !important;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stTable"] table {
        background-color: #ffffff !important;
    }

    [data-testid="stTable"] th {
        background-color: #176b3a !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stTable"] th * {
        color: #ffffff !important;
    }

    [data-testid="stTable"] td {
        background-color: #ffffff !important;
        color: #26352b !important;
    }

    [data-testid="stTable"] td * {
        color: #26352b !important;
    }


    /* =========================
       TIPS
       ========================= */

    .tips-text {
        color: #26352b !important;
        font-size: 17px;
        line-height: 1.9;
    }

    .tips-text p {
        color: #26352b !important;
    }


    /* =========================
       ALERTS
       ========================= */

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "5 fils",
    "25 fils",
    "50 fils",
    "100 fils",
    "BD 0.5",
    "BD 1",
    "BD 5",
    "BD 10",
    "BD 20"
]


# Icons for each currency class
class_icons = {
    "5 fils": "🪙",
    "25 fils": "🪙",
    "50 fils": "🪙",
    "100 fils": "🪙",
    "BD 0.5": "💵",
    "BD 1": "💵",
    "BD 5": "💵",
    "BD 10": "💵",
    "BD 20": "💵"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_currency_model():
    return load_model("currency_5model.keras")


model = load_currency_model()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💵 Bahraini Currency Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Currency Classification System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ℹ️ About")

    st.write(
        "This app identifies Bahraini currency coins and notes "
        "from an image using a trained Deep Learning model."
    )

    st.header("💱 Denominations")

    for name in class_names:
        st.write(f"{class_icons[name]}  {name}")

    st.header("📸 Tips for Best Results")

    st.write(
        "- Fill most of the frame with the currency\n"
        "- Use a plain background\n"
        "- Use good, even lighting\n"
        "- Avoid blurry images"
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(["💵 Recognize Currency",
                            "📊 Model Information",
                            "📖 How to Use"])


# ============================================================
# TAB 1 - RECOGNIZE CURRENCY
# ============================================================

with tab1:

    st.header("💵 Recognize Bahraini Currency")

    st.write(
        "Upload an image or use the camera and let the Deep Learning "
        "model identify the currency."
    )


    # --------------------------------------------------------
    # Image Input
    # --------------------------------------------------------

    if "camera_open" not in st.session_state:
        st.session_state.camera_open = False


    col1, col2 = st.columns(2)


    with col1:

        uploaded_file = st.file_uploader(
            "📁 Upload a currency image",
            type=["jpg", "jpeg", "png"]
        )


    with col2:

        st.write("")

        camera_label = (
            "✖️ Close Camera"
            if st.session_state.camera_open
            else "📷 Open Camera"
        )

        if st.button(
            camera_label,
            use_container_width=True
        ):

            st.session_state.camera_open = (
                not st.session_state.camera_open
            )


    camera_image = None


    if st.session_state.camera_open:

        camera_image = st.camera_input(
            "Take a live photo"
        )


    # Use camera image if available, otherwise uploaded image
    image_file = (
        camera_image
        if camera_image is not None
        else uploaded_file
    )


    # --------------------------------------------------------
    # Predict Button
    # --------------------------------------------------------

    predict_button = st.button(
        "🔍 Predict Currency",
        use_container_width=True
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if predict_button:

        if image_file is None:

            st.warning(
                "Please upload an image or take a photo first."
            )

        else:

            image = Image.open(
                image_file
            ).convert("RGB")


            # Display original image
            col1, col2 = st.columns(2)


            with col1:

                st.subheader("Uploaded Image")

                st.image(
                    image,
                    use_container_width=True
                )


            # ------------------------------------------------
            # Preprocessing
            # ------------------------------------------------

            resized_image = image.resize(
                (64, 64)
            )

            image_array = np.array(
                resized_image
            ) / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            with st.spinner(
                "🔍 Analyzing currency..."
            ):

                try:

                    prediction = model.predict(
                        image_array,
                        verbose=0
                    )

                    predicted_index = int(
                        np.argmax(prediction[0])
                    )

                    predicted_currency = (
                        class_names[predicted_index]
                    )

                    confidence = float(
                        prediction[0][predicted_index]
                        * 100
                    )

                except Exception as e:

                    st.error(
                        f"Prediction Error: {e}"
                    )

                    st.stop()


            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            with col2:

                st.subheader("Prediction Result")

                icon = class_icons[
                    predicted_currency
                ]

                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-icon">
                            {icon}
                        </div>

                        <div class="result-label">
                            {predicted_currency}
                        </div>

                        <div class="result-confidence">
                            {confidence:.2f}% confidence
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.metric(
                    label="Confidence",
                    value=f"{confidence:.2f}%"
                )


            # ------------------------------------------------
            # Confidence Feedback
            # ------------------------------------------------

            if confidence >= 80:

                st.success(
                    "✅ The model is highly confident "
                    "about this prediction."
                )

                st.balloons()


            elif confidence < 50:

                st.warning(
                    "⚠️ The confidence is low. "
                    "Try a clearer image with better lighting."
                )


            # ------------------------------------------------
            # Probability Breakdown
            # ------------------------------------------------

            show_probabilities = st.checkbox(
                "📊 Show probabilities for all classes"
            )


            if show_probabilities:

                probabilities_df = pd.DataFrame(
                    {
                        "Currency": class_names,
                        "Probability (%)": np.round(
                            prediction[0] * 100,
                            2
                        )
                    }
                )


                st.subheader(
                    "All Class Probabilities"
                )


                st.dataframe(
                    probabilities_df,
                    use_container_width=True
                )


                st.bar_chart(
                    probabilities_df.set_index(
                        "Currency"
                    )
                )


# ============================================================
# TAB 2 - MODEL INFORMATION
# ============================================================

with tab2:

    st.header("📊 Model Information")


    st.markdown(
        """
        <div class="info-card">

        <h3>About the Model</h3>

        <p>
        This project uses a Deep Learning classification model
        to recognize Bahraini currency from an image.
        The model predicts one of nine Bahraini currency classes.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Number of classes

    st.metric(
        "Number of Currency Classes",
        len(class_names)
    )


    # Currency classes

    st.subheader("💰 Currency Classes")


    classes_table = pd.DataFrame(
        {
            "Class Number": range(
                1,
                len(class_names) + 1
            ),

            "Currency": class_names
        }
    )


    st.table(
        classes_table
    )


    # Model pipeline

    st.subheader("🔄 Model Steps")


    st.markdown(
        """
        <div class="info-card">

        <p>1. Upload an image or take a photo</p>

        <p>2. Convert the image to RGB</p>

        <p>3. Resize the image to 224 × 224</p>

        <p>4. Normalize pixel values</p>

        <p>5. Add a batch dimension</p>

        <p>6. Send the image to the trained model</p>

        <p>7. Predict the currency class</p>

        <p>8. Display the predicted currency and confidence</p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TAB 3 - HOW TO USE
# ============================================================

with tab3:

    st.header("📖 How to Use")


    st.markdown(
        """
        <div class="info-card">

        <h3>Step 1</h3>

        <p>
        Upload a clear image of a Bahraini currency note or coin,
        or use the camera to take a photo.
        </p>

        </div>


        <div class="info-card">

        <h3>Step 2</h3>

        <p>
        Click the <b>Predict Currency</b> button.
        </p>

        </div>


        <div class="info-card">

        <h3>Step 3</h3>

        <p>
        The model will display the predicted currency
        and its confidence.
        </p>

        </div>


        <div class="info-card">

        <h3>Step 4</h3>

        <p>
        Enable <b>Show probabilities for all classes</b>
        to see the probability for every currency class.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Tips

    st.subheader("💡 Tips for Better Results")


    st.markdown(
        """
        <div class="tips-text">

        • Use a clear image<br>

        • Make sure the currency is visible<br>

        • Use good lighting<br>

        • Avoid excessive blur<br>

        • Use a plain background<br>

        • Try different angles

        </div>
        """,
        unsafe_allow_html=True
    )
