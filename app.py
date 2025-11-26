import streamlit as st
import google.generativeai as genai
from ultralytics import YOLO
from PIL import Image
import numpy as np
from gtts import gTTS
import base64
import io
import time
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import queue

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Live Exam Proctoring System",
    layout="wide",
    page_icon="🎓"
)

st.markdown("""
    <h1 style='text-align:center; color:#4CAF50;'>🎓 Live Exam Proctoring System</h1>
    <p style='text-align:center; font-size:18px;'>AI-powered real-time detection of restricted items during online exams.</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Detection Settings")

# Confidence threshold slider
confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.40,
    step=0.05,
    help="Higher values = more confident detections but might miss some objects"
)

# Detection delay control
detection_delay = st.sidebar.slider(
    "Detection Delay (seconds)",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="Time between consecutive detections to avoid spam alerts"
)

# Speech enable/disable
enable_speech = st.sidebar.checkbox("Enable Voice Alerts", value=True)

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
# GLOBAL VARIABLES FOR STATE MANAGEMENT
# ---------------------------------------------------------
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0
    
if 'alert_cooldown' not in st.session_state:
    st.session_state.alert_cooldown = False
    
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# ---------------------------------------------------------
# VOICE ALERT WITH COOLDOWN
# ---------------------------------------------------------
def speak(text):
    if not enable_speech:
        return
        
    current_time = time.time()
    # Prevent speech spam
    if current_time - st.session_state.last_detection_time < detection_delay:
        return
        
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
        st.session_state.last_detection_time = current_time
    except Exception as e:
        st.sidebar.warning(f"Speech error: {e}")

# ---------------------------------------------------------
# GEMINI VALIDATION
# ---------------------------------------------------------
def validate_with_gemini(pil_img, obj_name):
    prompt = f"""
    Look at this exam proctoring image.
    Confirm ONLY if you see the object: {obj_name}.
    If yes, reply: "WARNING: {obj_name} detected. Remove immediately."
    If not, reply: "Clear".
    Be very strict and accurate.
    """
    try:
        response = gemini_model.generate_content(
            contents=prompt,
            image=pil_img
        )
        return response.text.strip()
    except Exception as e:
        return f"Clear - Validation error: {e}"

# ---------------------------------------------------------
# VIDEO PROCESSING CLASS
# ---------------------------------------------------------
class VideoProcessor:
    def __init__(self):
        self.confidence_threshold = confidence_threshold
        self.last_processed_time = 0
        self.processing_interval = 1.0  # Process every 1 second
        
    def recv(self, frame):
        current_time = time.time()
        
        # Limit processing rate to avoid overwhelming the system
        if current_time - self.last_processed_time < self.processing_interval:
            return frame
            
        self.last_processed_time = current_time
        
        # Convert frame to PIL Image
        img = frame.to_image()
        img_arr = np.array(img)
        
        # YOLO prediction
        results = model.predict(img_arr, conf=self.confidence_threshold, verbose=False)
        
        # Get detections
        detected_objects = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                detected_objects.append({
                    'name': class_name,
                    'confidence': confidence,
                    'bbox': box.xyxy[0].tolist()
                })
        
        # Update session state with current detections
        st.session_state.current_detections = detected_objects
        st.session_state.last_frame = img_arr
        st.session_state.annotated_frame = results[0].plot()
        
        return frame

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
st.subheader("📹 Live Video Proctoring")

# Initialize video processor
video_processor = VideoProcessor()

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎥 Live Camera Feed")
    
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="proctoring",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("### 📊 Detection Panel")
    
    # Efficiency metrics
    efficiency_metrics = st.empty()
    alert_display = st.empty()
    detection_log = st.empty()

# Display current detections and alerts
if hasattr(st.session_state, 'current_detections') and st.session_state.current_detections:
    detected_objects = st.session_state.current_detections
    
    # Calculate efficiency metrics
    total_detections = len(detected_objects)
    avg_confidence = sum([obj['confidence'] for obj in detected_objects]) / total_detections * 100
    
    # Display metrics
    efficiency_metrics.markdown(f"""
    **Detection Efficiency:**
    - **Objects Detected:** {total_detections}
    - **Average Confidence:** {avg_confidence:.1f}%
    - **Detection Delay:** {detection_delay}s
    """)
    
    # Process alerts
    warning_detected = False
    warning_message = ""
    
    for obj in detected_objects:
        obj_name = obj['name']
        confidence = obj['confidence'] * 100
        
        # Only alert for high confidence detections
        if confidence > (confidence_threshold * 100):
            warning_detected = True
            warning_message = f"🚨 {obj_name.upper()} detected! ({confidence:.1f}% confidence)"
            
            # Voice alert
            if enable_speech and not st.session_state.alert_cooldown:
                speak(f"Warning: {obj_name} detected")
                st.session_state.alert_cooldown = True
                # Reset cooldown after delay
                st.session_state.cooldown_time = time.time()
            break
    
    if warning_detected:
        alert_display.error(warning_message)
    else:
        alert_display.success("✅ No prohibited items detected")
        
    # Display detection log
    log_entries = []
    for obj in detected_objects:
        log_entries.append(f"- {obj['name']} ({obj['confidence']*100:.1f}%)")
    
    detection_log.markdown("**Recent Detections:**\n" + "\n".join(log_entries))
    
else:
    efficiency_metrics.info("⏳ Waiting for camera feed...")
    alert_display.info("🔍 Monitoring for restricted items")
    detection_log.markdown("**Recent Detections:**\n- None")

# Display annotated frame if available
if hasattr(st.session_state, 'annotated_frame'):
    st.markdown("### 📸 Latest Processed Frame")
    st.image(st.session_state.annotated_frame, caption="Live Detection View", use_column_width=True)

# Cooldown management
if hasattr(st.session_state, 'alert_cooldown') and st.session_state.alert_cooldown:
    if time.time() - st.session_state.cooldown_time > detection_delay:
        st.session_state.alert_cooldown = False

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "🔒 Secure Exam Proctoring System | Real-time AI Monitoring"
    "</p>",
    unsafe_allow_html=True
)
