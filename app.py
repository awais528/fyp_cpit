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
import queue

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="Pre-Exam Proctoring System (Real-time)", layout="wide")
st.title("🛡️ Pre-Exam Proctoring System — Real-time Detector")

MODEL_PATH = "best.pt"  # make sure best.pt is in repository / deploy root

CLASS_NAMES = ['bag', 'book', 'calculator', 'mobile', 'notes', 'smart watch']
# Message to speak when detected
CLASS_MESSAGES = {
    'bag': "A bag is detected. Remove any unauthorized material immediately.",
    'book': "A book is detected. Remove the book from the exam area.",
    'calculator': "A calculator is detected. Remove it if not permitted.",
    'mobile': "A mobile phone is detected. Switch it off and remove it.",
    'notes': "Notes detected. Remove all written material now.",
    'smart watch': "A smart watch detected. Please remove it immediately."
}

# Initialize session state
if 'alert_queue' not in st.session_state:
    st.session_state.alert_queue = deque(maxlen=10)
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {name: 0.0 for name in CLASS_NAMES}
if 'last_audio' not in st.session_state:
    st.session_state.last_audio = None
if 'last_message' not in st.session_state:
    st.session_state.last_message = ""
if 'last_msg_time' not in st.session_state:
    st.session_state.last_msg_time = 0

# ----------------------
# Load YOLO model (cached)
# ----------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.warning("⚠️ Model not found. Please ensure 'best.pt' is in the root directory.")
    st.stop()

# ----------------------
# Utility: create an audio bytes object from text
# ----------------------
def tts_bytes(text):
    """Return audio bytes (mp3) for given text using gTTS."""
    try:
        tts = gTTS(text=text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# ----------------------
# Video Transformer
# ----------------------
class YoloTransformer(VideoTransformerBase):
    """
    Runs YOLO on each incoming frame (bgr24 ndarray).
    Overlays boxes and big text 'REMOVE: <class>' on the frame when a restricted object is found.
    """
    def __init__(self):
        self.model = model
        self.names = CLASS_NAMES

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert to numpy (BGR)
        img = frame.to_ndarray(format="bgr24")

        # Run prediction
        try:
            results = self.model.predict(img, conf=0.35, imgsz=640, verbose=False)
        except Exception as e:
            return frame

        # Plot annotated image
        annotated = results[0].plot()  # numpy array BGR

        # Collect detected classes in this frame
        detected = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_idx = int(box.cls)
                if cls_idx < len(self.names):
                    detected.append(self.names[cls_idx])

        # If any restricted object found, create alert(s)
        now = time.time()
        current_cooldown = st.session_state.get('current_cooldown', 4.0)
        
        for cls in set(detected):
            # check cooldown
            if now - st.session_state.last_alert_time.get(cls, 0) > current_cooldown:
                # push an alert into queue for main thread
                st.session_state.alert_queue.append({'class': cls, 'time': now})
                # update last alert time
                st.session_state.last_alert_time[cls] = now

                # Draw alert on frame
                try:
                    # Convert BGR to RGB for PIL
                    pil_img = Image.fromarray(annotated[..., ::-1])  # BGR->RGB
                    draw = ImageDraw.Draw(pil_img)
                    
                    # Get image dimensions
                    w, h = pil_img.size
                    rect_h = int(h * 0.12)
                    
                    # Create semi-transparent overlay
                    overlay = Image.new('RGBA', (w, rect_h), (220, 20, 60, 200))
                    pil_img.paste(overlay, (0, 0), overlay)
                    
                    # Draw text
                    draw = ImageDraw.Draw(pil_img)
                    text = f"REMOVE: {cls.upper()}"
                    font_size = max(24, w // 18)
                    
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        try:
                            font = ImageFont.load_default()
                        except:
                            font = None
                    
                    # Calculate text position
                    if font:
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                    else:
                        text_w, text_h = draw.textsize(text)
                    
                    text_x = (w - text_w) // 2
                    text_y = (rect_h - text_h) // 2
                    
                    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
                    
                    # Convert back to BGR for output
                    annotated = np.array(pil_img)[..., ::-1]
                    
                except Exception as e:
                    # Continue without overlay if drawing fails
                    continue

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ----------------------
# Alert Processing Function
# ----------------------
def process_alerts():
    """Process alerts from the queue and update session state"""
    if st.session_state.alert_queue:
        item = st.session_state.alert_queue.popleft()
        cls = item['class']
        msg = CLASS_MESSAGES.get(cls, f"{cls} detected. Remove it immediately.")
        
        # Create audio bytes
        audio_b = tts_bytes(msg)
        
        # Update session state
        st.session_state.last_audio = audio_b
        st.session_state.last_message = msg
        st.session_state.last_msg_time = time.time()

# ----------------------
# Main UI
# ----------------------
st.sidebar.header("Controls")

# Configuration sliders
conf_thresh = st.sidebar.slider("Confidence threshold", 0.2, 0.9, 0.35, 0.05)
cooldown = st.sidebar.slider("Alert cooldown (sec)", 1, 10, 4)

# Update cooldown in session state
st.session_state.current_cooldown = float(cooldown)

st.markdown("### Live detection")
st.write("Start your webcam, allow access, and the model will detect restricted items in real-time. When an item is detected, you will see an instruction on the video and hear a voice alert.")

# Process any pending alerts
process_alerts()

# Display current alert
alert_placeholder = st.empty()

if st.session_state.last_message and (time.time() - st.session_state.last_msg_time < 6):
    alert_placeholder.markdown(f"### ⚠️ {st.session_state.last_message}")
    
    # Play audio if available
    if st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format='audio/mp3')
else:
    alert_placeholder.info("No active alerts.")
    st.session_state.last_message = ""

# WebRTC Streamer
webrtc_ctx = webrtc_streamer(
    key="yolo-stream",
    video_transformer_factory=YoloTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# Notes / tips
st.markdown("""
---
**Notes**
- The browser will ask for camera permission — allow it once.
- If audio does not auto-play, click on the page once to grant user gesture permission.
- Adjust confidence and cooldown in the sidebar.
- Red rectangle with text will appear when restricted items are detected.
""")
