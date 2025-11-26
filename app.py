import streamlit as st
import google.generativeai as genai
from ultralytics import YOLO
from PIL import Image
import numpy as np
from gtts import gTTS
import base64
import io
import time

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pre-Exam Proctoring System",
    layout="wide",
    page_icon="🎓"
)

st.markdown("""
    <h1 style='text-align:center; color:#4CAF50;'>🎓 Pre-Exam Proctoring System</h1>
    <p style='text-align:center; font-size:18px;'>AI-powered detection of restricted items during online exams.</p>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD GEMINI
# ---------------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


# ---------------------------------------------------------
# LOAD YOLO MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

CLASS_NAMES = ['bag', 'book', 'calculator', 'mobile', 'notes', 'smart watch']


# ---------------------------------------------------------
# VOICE ALERT
# ---------------------------------------------------------
def speak(text):
    try:
        tts = gTTS(text=text, lang="en")
        audio = io.BytesIO()
        tts.write_to_fp(audio)
        audio.seek(0)

        b64 = base64.b64encode(audio.read()).decode()
        md = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        st.markdown(md, unsafe_allow_html=True)
    except:
        pass


# ---------------------------------------------------------
# GEMINI VALIDATION
# ---------------------------------------------------------
def validate_with_gemini(pil_img, obj_name):
    prompt = f"""
    Look at this exam proctoring image.
    Confirm ONLY if you see the object: {obj_name}.
    If yes, reply: "WARNING: {obj_name} detected. Remove immediately."
    If not, reply: "Clear".
    """
    try:
        response = gemini_model.generate_content(
            contents=prompt,
            image=pil_img
        )
        return response.text.strip()
    except:
        return "Clear"


# ---------------------------------------------------------
# CAMERA INPUT
# ---------------------------------------------------------
st.subheader("📷 Live Camera Feed")
camera_frame = st.camera_input("Start Camera")

alert_box = st.empty()

if camera_frame:
    # Convert to PIL image
    img = Image.open(camera_frame)
    img_arr = np.array(img)

    # YOLO prediction
    results = model.predict(img_arr, conf=0.40)

    annotated = results[0].plot()
    st.image(annotated, caption="Detected Frame", use_column_width=True)

    detected_classes = [model.names[int(c.cls)] for c in results[0].boxes]

    if detected_classes:
        for obj in detected_classes:
            gemini_result = validate_with_gemini(img, obj)

            if "warning" in gemini_result.lower():
                alert_box.error(gemini_result)
                speak(gemini_result)
            else:
                alert_box.success("Clear Frame")
    else:
        alert_box.info("No prohibited items detected.")
