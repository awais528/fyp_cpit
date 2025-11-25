import streamlit as st
from ultralytics import YOLO
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import io
import time
from collections import deque
import cv2
import tempfile
import os

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="Pre-Exam Proctoring System (Real-time)", layout="wide")
st.title("🛡️ Pre-Exam Proctoring System — Real-time Detector")

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
if 'last_audio' not in st.session_state:
    st.session_state.last_audio = None
if 'last_message' not in st.session_state:
    st.session_state.last_message = ""
if 'last_msg_time' not in st.session_state:
    st.session_state.last_msg_time = 0
if 'camera_started' not in st.session_state:
    st.session_state.camera_started = False
if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = None

# ----------------------
# Load YOLO model (cached)
# ----------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        model = YOLO(path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.warning("⚠️ Model not found. Please ensure 'best.pt' is in the root directory.")
    st.stop()

# ----------------------
# TTS Function
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
# Process Frame with YOLO
# ----------------------
def process_frame(frame, conf_threshold=0.35):
    """Process a single frame with YOLO and return annotated frame + alerts"""
    # Convert frame to numpy array
    img = np.array(frame)
    
    # Run YOLO prediction
    results = model.predict(img, conf=conf_threshold, imgsz=640, verbose=False)
    
    # Get annotated frame
    annotated = results[0].plot()
    
    # Check for detections
    alerts = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_idx = int(box.cls)
            if cls_idx < len(CLASS_NAMES):
                class_name = CLASS_NAMES[cls_idx]
                confidence = float(box.conf)
                alerts.append(class_name)
    
    return annotated, alerts

# ----------------------
# Draw Alert on Frame
# ----------------------
def draw_alert_on_frame(frame, alert_text):
    """Draw alert text on the frame"""
    try:
        # Convert to PIL Image
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)
        
        # Get dimensions
        width, height = pil_img.size
        rect_height = int(height * 0.1)
        
        # Draw semi-transparent red rectangle
        overlay = Image.new('RGBA', (width, rect_height), (255, 0, 0, 180))
        pil_img.paste(overlay, (0, 0), overlay)
        
        # Draw text
        draw = ImageDraw.Draw(pil_img)
        text = f"ALERT: {alert_text.upper()}"
        
        # Try to use a font
        try:
            font_size = min(40, width // 20)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Calculate text position
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
        else:
            text_width = draw.textlength(text)
        
        text_x = (width - text_width) // 2
        text_y = (rect_height - 30) // 2
        
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        
        return np.array(pil_img)
    except Exception as e:
        st.error(f"Error drawing alert: {e}")
        return frame

# ----------------------
# Main Application
# ----------------------
def main():
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.35, 
        step=0.05,
        help="Adjust how confident the model should be before detecting objects"
    )
    
    alert_cooldown = st.sidebar.slider(
        "Alert Cooldown (seconds)",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
        help="Time between repeated alerts for the same object"
    )
    
    # Camera selection
    st.sidebar.subheader("📷 Camera Settings")
    
    # Try to get available cameras
    try:
        available_cameras = []
        for i in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        if available_cameras:
            camera_index = st.sidebar.selectbox(
                "Select Camera",
                available_cameras,
                format_func=lambda x: f"Camera {x}"
            )
        else:
            st.sidebar.warning("No cameras found. Using default camera.")
            camera_index = 0
    except:
        camera_index = 0
    
    # Start/Stop buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_camera = st.button("🚀 Start Camera", use_container_width=True)
    with col2:
        stop_camera = st.button("🛑 Stop Camera", use_container_width=True)
    
    if start_camera:
        st.session_state.camera_started = True
        st.rerun()
    
    if stop_camera:
        st.session_state.camera_started = False
        st.rerun()
    
    # Main content area
    st.markdown("### 🔍 Live Camera Feed")
    
    if st.session_state.camera_started:
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)
        
        if not cap.isOpened():
            st.error("❌ Could not open camera. Please check your camera connection.")
            st.session_state.camera_started = False
            return
        
        # Create placeholder for video feed
        frame_placeholder = st.empty()
        alert_placeholder = st.empty()
        
        st.info("🎥 Camera is running. Press 'Stop Camera' to end the session.")
        
        # Process frames
        while st.session_state.camera_started:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Failed to grab frame from camera")
                break
            
            # Process frame with YOLO
            processed_frame, alerts = process_frame(frame, conf_threshold)
            
            # Handle alerts
            current_time = time.time()
            for alert in set(alerts):
                if current_time - st.session_state.last_alert_time.get(alert, 0) > alert_cooldown:
                    # Update alert time
                    st.session_state.last_alert_time[alert] = current_time
                    
                    # Get alert message
                    message = CLASS_MESSAGES.get(alert, f"{alert} detected!")
                    
                    # Generate TTS
                    audio_bytes = tts_bytes(message)
                    if audio_bytes:
                        st.session_state.last_audio = audio_bytes
                        st.session_state.last_message = message
                        st.session_state.last_msg_time = current_time
                    
                    # Draw alert on frame
                    processed_frame = draw_alert_on_frame(processed_frame, alert)
            
            # Convert BGR to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            frame_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
            
            # Display current alert if recent
            if st.session_state.last_message and (time.time() - st.session_state.last_msg_time < 6):
                alert_placeholder.warning(f"⚠️ {st.session_state.last_message}")
                
                # Play audio alert
                if st.session_state.last_audio:
                    st.audio(st.session_state.last_audio, format='audio/mp3')
            else:
                alert_placeholder.info("✅ No alerts - All clear!")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.05)
        
        # Release camera when stopped
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        st.info("👆 Click 'Start Camera' to begin proctoring session")
        
        # Show sample detection image if available
        st.markdown("### 📋 Detection Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**What we detect:**")
            st.write("• 📱 Mobile phones")
            st.write("• 📚 Books & Notes")
            st.write("• 🎒 Bags")
            st.write("• ⌚ Smart watches")
            st.write("• 🧮 Calculators")
        
        with col2:
            st.markdown("**How it works:**")
            st.write("• Real-time object detection")
            st.write("• Instant audio alerts")
            st.write("• Visual warnings on screen")
            st.write("• Configurable sensitivity")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📖 Instructions
    1. **Select your camera** from the dropdown
    2. **Adjust confidence threshold** if needed (higher = fewer false alarms)
    3. **Set alert cooldown** to prevent repeated alerts
    4. **Click 'Start Camera'** to begin monitoring
    5. **Allow camera permissions** when prompted by your browser
    6. **Click 'Stop Camera'** when finished
    
    **💡 Tip:** Ensure good lighting and camera positioning for best results.
    """)

if __name__ == "__main__":
    main()
