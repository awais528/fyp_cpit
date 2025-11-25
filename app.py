import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import io
import time
import tempfile
import os

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="Pre-Exam Proctoring System", layout="wide")
st.title("🛡️ Pre-Exam Proctoring System — Real-time Detector")

# Try to import Ultralytics with error handling
try:
    from ultralytics import YOLO
    ULTRAlytics_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Ultralytics import error: {e}")
    ULTRAlytics_AVAILABLE = False

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
if 'model' not in st.session_state:
    st.session_state.model = None

# ----------------------
# Load YOLO model
# ----------------------
@st.cache_resource
def load_model():
    if not ULTRAlytics_AVAILABLE:
        return None
        
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            st.sidebar.success("✅ Model loaded successfully!")
            return model
        else:
            st.sidebar.warning(f"⚠️ Model file '{MODEL_PATH}' not found")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model only if ultralytics is available
if ULTRAlytics_AVAILABLE and st.session_state.model is None:
    with st.spinner("Loading YOLO model..."):
        st.session_state.model = load_model()

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
        st.sidebar.error(f"TTS Error: {e}")
        return None

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
        rect_height = int(height * 0.12)
        
        # Draw semi-transparent red rectangle
        overlay = Image.new('RGBA', (width, rect_height), (255, 0, 0, 180))
        pil_img.paste(overlay, (0, 0), overlay)
        
        # Draw text
        draw = ImageDraw.Draw(pil_img)
        text = f"ALERT: {alert_text.upper()}"
        
        # Use default font
        try:
            font_size = min(36, width // 15)
            font = ImageFont.truetype("Arial", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Calculate text position
        if font:
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(text) * 10
            text_height = 20
        
        text_x = (width - text_width) // 2
        text_y = (rect_height - text_height) // 2
        
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
        
        return np.array(pil_img)
    except Exception as e:
        st.error(f"Error drawing alert: {e}")
        return frame

# ----------------------
# Process Frame with YOLO
# ----------------------
def process_frame(frame, conf_threshold=0.35):
    """Process a single frame with YOLO and return annotated frame + alerts"""
    if st.session_state.model is None:
        return frame, []
    
    try:
        # Run YOLO prediction
        results = st.session_state.model.predict(
            frame, 
            conf=conf_threshold, 
            imgsz=640, 
            verbose=False
        )
        
        # Get annotated frame
        annotated = results[0].plot()
        
        # Check for detections
        alerts = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_idx = int(box.cls)
                if cls_idx < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[cls_idx]
                    alerts.append(class_name)
        
        return annotated, alerts
        
    except Exception as e:
        st.error(f"Error processing frame: {e}")
        return frame, []

# ----------------------
# Camera Functions
# ----------------------
def get_available_cameras():
    """Get list of available cameras"""
    available_cameras = []
    for i in range(3):  # Check first 3 camera indices
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        except:
            pass
    return available_cameras if available_cameras else [0]

# ----------------------
# Main Application
# ----------------------
def main():
    st.sidebar.header("🎛️ Controls")
    
    # Model status
    if not ULTRAlytics_AVAILABLE:
        st.sidebar.error("❌ Ultralytics not available. Check requirements.txt")
        return
    
    if st.session_state.model is None:
        st.sidebar.warning("⚠️ Model not loaded. Check if best.pt exists.")
        st.info("Please ensure 'best.pt' is in your repository root directory.")
        return
    
    # Configuration
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.35, 
        step=0.05
    )
    
    alert_cooldown = st.sidebar.slider(
        "Alert Cooldown (seconds)",
        min_value=2,
        max_value=10,
        value=4,
        step=1
    )
    
    # Camera selection
    st.sidebar.subheader("📷 Camera Settings")
    
    try:
        available_cameras = get_available_cameras()
        camera_index = st.sidebar.selectbox(
            "Select Camera",
            available_cameras,
            format_func=lambda x: f"Camera {x}"
        )
    except:
        camera_index = 0
        st.sidebar.info("Using default camera")
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if not st.session_state.camera_started:
            if st.button("🚀 Start Camera", use_container_width=True):
                st.session_state.camera_started = True
                st.rerun()
    with col2:
        if st.session_state.camera_started:
            if st.button("🛑 Stop Camera", use_container_width=True):
                st.session_state.camera_started = False
                st.rerun()
    
    # Main content
    st.markdown("### 🔍 Live Camera Feed")
    
    if st.session_state.camera_started:
        # Initialize camera
        try:
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 20)
            
            if not cap.isOpened():
                st.error("❌ Could not open camera. Please check your camera connection.")
                st.session_state.camera_started = False
                return
            
            # Create placeholders
            frame_placeholder = st.empty()
            alert_placeholder = st.empty()
            status_placeholder = st.empty()
            
            st.info("🎥 Camera is running. Press 'Stop Camera' to end the session.")
            
            # Process frames
            while st.session_state.camera_started:
                ret, frame = cap.read()
                
                if not ret:
                    status_placeholder.error("❌ Failed to grab frame from camera")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with YOLO
                processed_frame, alerts = process_frame(frame_rgb, conf_threshold)
                
                # Handle alerts
                current_time = time.time()
                alert_triggered = False
                
                for alert in set(alerts):
                    if current_time - st.session_state.last_alert_time.get(alert, 0) > alert_cooldown:
                        # Update alert time
                        st.session_state.last_alert_time[alert] = current_time
                        alert_triggered = True
                        
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
                
                # Display frame
                frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                
                # Display alerts
                if alert_triggered:
                    alert_placeholder.warning(f"⚠️ {st.session_state.last_message}")
                    if st.session_state.last_audio:
                        st.audio(st.session_state.last_audio, format='audio/mp3')
                else:
                    if alerts:
                        alert_placeholder.info(f"🔍 Detected: {', '.join(set(alerts))}")
                    else:
                        alert_placeholder.success("✅ No restricted items detected")
                
                # Small delay
                time.sleep(0.03)
            
            # Release camera
            cap.release()
            
        except Exception as e:
            st.error(f"❌ Camera error: {str(e)}")
            st.session_state.camera_started = False
            
    else:
        st.info("👆 Click 'Start Camera' to begin proctoring session")
        
        # Show detection preview
        st.markdown("### 📋 Detection Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Restricted Items:**")
            for item in CLASS_NAMES:
                st.write(f"• {item.title()}")
        
        with col2:
            st.markdown("**Features:**")
            st.write("• Real-time detection")
            st.write("• Audio alerts")
            st.write("• Visual warnings")
            st.write("• Adjustable sensitivity")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📖 Instructions
    1. Select your camera from dropdown
    2. Adjust confidence threshold if needed
    3. Set alert cooldown to prevent spam
    4. Click 'Start Camera' to begin
    5. Allow camera permissions if prompted
    6. Click 'Stop Camera' when done
    
    **💡 Tip:** Ensure good lighting for best detection accuracy.
    """)

if __name__ == "__main__":
    main()
