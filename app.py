import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import io
import time
from collections import deque
import os

# Page config
st.set_page_config(page_title="Pre-Exam Proctoring System", layout="wide")
st.title("🛡️ Pre-Exam Proctoring System")

# Configuration
MODEL_PATH = "best.pt"
CLASS_NAMES = ['bag', 'book', 'calculator', 'mobile', 'notes', 'smart watch']
CLASS_MESSAGES = {
    'bag': "A bag is detected. Remove any unauthorized material immediately.",
    'book': "A book is detected. Remove the book from the exam area.",
    'calculator': "A calculator is detected. Remove it if not permitted.",
    'mobile': "A mobile phone is detected. Switch it off and remove it.",
    'notes': "Notes detected. Remove all written material now.",
    'smart watch': "A smart watch detected. Please remove it immediately."
}

# Initialize session state
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {name: 0.0 for name in CLASS_NAMES}
if 'alert_queue' not in st.session_state:
    st.session_state.alert_queue = deque(maxlen=10)

# Load model
@st.cache_resource
def load_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

if model is None:
    st.warning("⚠️ Please ensure 'best.pt' is in the deployment directory.")
    st.stop()

# TTS function
def tts_bytes(text):
    try:
        tts = gTTS(text=text, lang="en")
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# Video Transformer
class YoloTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.names = CLASS_NAMES

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            results = self.model.predict(img, conf=0.35, verbose=False)
        except Exception:
            return frame
            
        annotated = results[0].plot()
        
        # Process detections
        detected = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_idx = int(box.cls)
                if cls_idx < len(self.names):
                    detected.append(self.names[cls_idx])
        
        # Handle alerts
        now = time.time()
        cooldown = st.session_state.get('cooldown', 4.0)
        
        for cls in set(detected):
            if now - st.session_state.last_alert_time.get(cls, 0) > cooldown:
                st.session_state.alert_queue.append({'class': cls, 'time': now})
                st.session_state.last_alert_time[cls] = now
                
                # Visual alert
                try:
                    pil_img = Image.fromarray(annotated[..., ::-1])
                    draw = ImageDraw.Draw(pil_img)
                    w, h = pil_img.size
                    
                    # Draw alert banner
                    banner_height = h // 8
                    overlay = Image.new('RGBA', (w, banner_height), (255, 0, 0, 200))
                    pil_img.paste(overlay, (0, 0), overlay)
                    
                    # Draw text
                    draw = ImageDraw.Draw(pil_img)
                    text = f"REMOVE: {cls.upper()}"
                    font_size = min(40, w // 15)
                    
                    try:
                        font = ImageFont.truetype("Arial", font_size)
                    except:
                        font = ImageFont.load_default()
                    
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_x = (w - text_w) // 2
                    text_y = (banner_height - (bbox[3] - bbox[1])) // 2
                    
                    draw.text((text_x, text_y), text, fill="white", font=font, stroke_width=2, stroke_fill="black")
                    annotated = np.array(pil_img.convert('RGB'))[..., ::-1]
                    
                except Exception:
                    continue
                    
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.2, 0.9, 0.35, 0.05)
cooldown = st.sidebar.slider("Alert Cooldown (seconds)", 2, 10, 4)
st.session_state.cooldown = float(cooldown)

# Main interface
st.markdown("### 🔍 Live Object Detection")

# WebRTC Configuration
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]
}

# WebRTC Streamer
webrtc_ctx = webrtc_streamer(
    key="yolo-proctoring",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"min": 640, "ideal": 1280},
            "height": {"min": 480, "ideal": 720},
            "frameRate": {"ideal": 20, "max": 30}
        },
        "audio": False
    },
    video_transformer_factory=YoloTransformer,
    async_transform=True,
)

# Alert processing
if st.session_state.alert_queue:
    alert = st.session_state.alert_queue.popleft()
    cls = alert['class']
    message = CLASS_MESSAGES.get(cls, f"{cls} detected!")
    
    st.error(f"🚨 **Alert:** {message}")
    
    audio_bytes = tts_bytes(message)
    if audio_bytes:
        st.audio(audio_bytes, format='audio/mp3')

# Instructions
st.markdown("""
---
### 📝 Instructions
1. **Click 'START'** to begin webcam detection
2. **Allow camera permissions** when prompted
3. **Hold up objects** to test detection
4. **Audio alerts** will play when restricted items are detected
5. **Adjust settings** in the sidebar as needed

**Note:** If webcam fails to load, try refreshing the page or check browser permissions.
""")
