import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import av
import numpy as np
from PIL import Image
from gtts import gTTS
import io
import base64
import time
import threading
from collections import deque

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

# global alert queue used to communicate from transformer -> main thread
alert_queue = deque(maxlen=10)

# cooldown (seconds) per class to avoid repeated alerts
ALERT_COOLDOWN = 4.0
last_alert_time = {name: 0.0 for name in CLASS_NAMES}

# ----------------------
# Load YOLO model (cached)
# ----------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    return YOLO(path)

model = load_model()

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
        # return None on failure
        return None

# ----------------------
# Video Transformer
# ----------------------
class YoloTransformer(VideoTransformerBase):
    """
    Runs YOLO on each incoming frame (bgr24 ndarray).
    Overlays boxes and big text 'REMOVE: <class>' on the frame when a restricted object is found.
    Pushes a small alert event into alert_queue for the main thread to handle voice & UI.
    """
    def __init__(self):
        self.model = model  # shared cached model
        self.names = CLASS_NAMES

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert to numpy (BGR)
        img = frame.to_ndarray(format="bgr24")

        # Run prediction (fast)
        # NOTE: set verbose=False to reduce logs
        results = self.model.predict(img, conf=0.35, imgsz=640, verbose=False)

        # Plot annotated image (Ultralytics returns BGR by default)
        annotated = results[0].plot()  # numpy array BGR

        # Collect detected classes in this frame
        detected = []
        for box in results[0].boxes:
            cls_idx = int(box.cls)
            if cls_idx < len(self.names):
                detected.append(self.names[cls_idx])

        # If any restricted object found, create alert(s)
        now = time.time()
        for cls in set(detected):
            # check cooldown
            if now - last_alert_time.get(cls, 0) > ALERT_COOLDOWN:
                # push an alert into queue for main thread
                alert_queue.append({'class': cls, 'time': now})
                # update last alert time
                last_alert_time[cls] = now

                # draw a big remove text onto annotated frame (so user sees on video)
                # We draw by converting to PIL and then back for convenience
                try:
                    pil = Image.fromarray(annotated[..., ::-1])  # BGR->RGB
                    draw = Image.ImageDraw.Draw(pil)  # type: ignore
                    # Draw a red rectangle and text at top
                    w, h = pil.size
                    # semi-transparent rectangle:
                    rect_h = int(h * 0.12)
                    overlay = Image.new('RGBA', pil.size, (0,0,0,0))
                    ov_draw = Image.ImageDraw.Draw(overlay)  # type: ignore
                    ov_draw.rectangle([(0, 0), (w, rect_h)], fill=(220, 20, 60, 200))
                    pil = Image.alpha_composite(pil.convert('RGBA'), overlay)
                    ov_draw = Image.ImageDraw.Draw(pil)  # type: ignore
                    # large text
                    text = f"REMOVE: {cls.upper()}"
                    # choose font size based on width
                    font_size = max(24, w // 18)
                    try:
                        from PIL import ImageFont
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = None
                    text_w, text_h = ov_draw.textsize(text, font=font)
                    ov_draw.text(((w - text_w) / 2, (rect_h - text_h) / 2), text, fill=(255,255,255,255), font=font)
                    annotated = np.array(pil.convert('RGB'))[..., ::-1]  # back to BGR
                except Exception:
                    # fallback: do nothing special
                    pass

        # Return an av.VideoFrame
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ----------------------
# Controls & UI
# ----------------------
st.sidebar.header("Controls")
conf_thresh = st.sidebar.slider("Confidence threshold", 0.2, 0.9, 0.35, 0.05)
cooldown = st.sidebar.slider("Alert cooldown (sec)", 1, 10, 4)
st.sidebar.write("Click Start to begin real-time detection")
start_button = st.sidebar.button("Start")
stop_button = st.sidebar.button("Stop")

# update globals from UI
ALERT_COOLDOWN = float(cooldown)

st.markdown("### Live detection")
st.write("Start your webcam, allow access, and the model will detect restricted items in real-time. When an item is detected, you will see an instruction on the video and hear a voice alert.")

# placeholder for alerts text
alert_placeholder = st.empty()

# webrtc client settings
client_settings = ClientSettings(
    media_stream_constraints={"video": True, "audio": False},
)

webrtc_ctx = None

# start or stop the webrtc streamer based on button clicks
if start_button:
    # create the streamer (start camera)
    webrtc_ctx = webrtc_streamer(
        key="yolo-stream",
        video_transformer_factory=YoloTransformer,
        client_settings=client_settings,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

if stop_button:
    # stopping: webrtc_streamer has no direct stop call; user may refresh or close stream.
    st.experimental_rerun()

# ----------------------
# Alert polling loop (main thread)
# ----------------------
# We'll poll the alert_queue for new alerts and play TTS when found.
# This non-blocking poll will run while the streamer is active.
def alert_poller():
    # This function runs in a background thread and pushes audio events into a Streamlit placeholder via session state.
    # Note: We only generate the audio bytes here and set a flag in st.session_state.
    while True:
        time.sleep(0.2)
        if not webrtc_ctx or not webrtc_ctx.state.playing:
            continue
        if len(alert_queue) > 0:
            item = alert_queue.popleft()
            cls = item['class']
            msg = CLASS_MESSAGES.get(cls, f"{cls} detected. Remove it immediately.")
            # create audio bytes
            audio_b = tts_bytes(msg)
            if audio_b:
                # set in session state for main thread to render
                st.session_state['last_audio'] = audio_b
                st.session_state['last_message'] = msg
                st.session_state['last_msg_time'] = time.time()
            else:
                st.session_state['last_audio'] = None
                st.session_state['last_message'] = msg
                st.session_state['last_msg_time'] = time.time()

# start poller thread (daemon)
if 'alert_thread_started' not in st.session_state:
    st.session_state['alert_thread_started'] = True
    t = threading.Thread(target=alert_poller, daemon=True)
    t.start()

# Render the latest alert if present
if st.session_state.get('last_message'):
    # show message
    alert_placeholder.markdown(f"### ⚠️ {st.session_state['last_message']}")
    # play audio if available
    last_t = st.session_state.get('last_msg_time', 0)
    # To avoid re-playing same audio repeatedly, ensure it is recent
    if time.time() - last_t < 6:
        audio_bytes = st.session_state.get('last_audio')
        if audio_bytes:
            st.audio(audio_bytes, format='audio/mp3')
else:
    alert_placeholder.info("No active alerts.")

# Notes / tips
st.markdown("""
---
**Notes**
- The browser will ask for camera permission — allow it once.
- If audio does not auto-play, click on the page once to grant user gesture permission.
- Adjust confidence and cooldown in the sidebar.
""")
