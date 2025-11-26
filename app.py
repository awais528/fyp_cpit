import streamlit as st
import google.generativeai as genai
from ultralytics import YOLO
from PIL import Image
import numpy as np
from gtts import gTTS
import base64
import io
import time
import pandas as pd
from datetime import datetime

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
    min_value=2,
    max_value=15,
    value=5,
    step=1,
    help="Time between consecutive detections to avoid spam alerts"
)

# Speech enable/disable
enable_speech = st.sidebar.checkbox("Enable Voice Alerts", value=True)

# Auto-refresh rate
refresh_rate = st.sidebar.slider(
    "Camera Refresh Rate (seconds)",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="How often to capture new frames from camera"
)

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel("gemini-2.0-flash")

@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

gemini_model = load_gemini()
model = load_yolo()
CLASS_NAMES = ['bag', 'book', 'calculator', 'mobile', 'notes', 'smart watch']

# ---------------------------------------------------------
# INITIALIZE SESSION STATE
# ---------------------------------------------------------
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0
    
if 'alert_cooldown' not in st.session_state:
    st.session_state.alert_cooldown = False
    
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
    
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = "ready"
    
if 'last_capture_time' not in st.session_state:
    st.session_state.last_capture_time = 0
    
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_frames_processed': 0,
        'total_detections': 0,
        'alerts_triggered': 0,
        'start_time': datetime.now()
    }

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
        st.session_state.stats['alerts_triggered'] += 1
    except Exception as e:
        st.sidebar.warning(f"Speech error: {e}")

# ---------------------------------------------------------
# GEMINI VALIDATION
# ---------------------------------------------------------
def validate_with_gemini(pil_img, obj_name):
    prompt = f"""
    Examine this exam proctoring image carefully. 
    Confirm ONLY if you clearly see a {obj_name} in the image.
    If {obj_name} is definitely present, reply: "WARNING: {obj_name} detected. Remove immediately."
    If {obj_name} is not clearly visible or not present, reply: "Clear".
    Be very strict and accurate. Only confirm if you're absolutely sure.
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
# PROCESS FRAME FUNCTION
# ---------------------------------------------------------
def process_frame(image):
    """Process a single frame and return results"""
    # Convert to numpy array
    img_arr = np.array(image)
    
    # YOLO prediction
    results = model.predict(img_arr, conf=confidence_threshold, verbose=False)
    
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
                'bbox': box.xyxy[0].tolist() if box.xyxy is not None else []
            })
    
    # Create annotated image
    annotated_img = results[0].plot()
    annotated_pil = Image.fromarray(annotated_img)
    
    return detected_objects, annotated_pil

# ---------------------------------------------------------
# MAIN APP LAYOUT
# ---------------------------------------------------------

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📷 Live Camera Monitoring")
    
    # Auto-refresh logic
    current_time = time.time()
    should_capture = current_time - st.session_state.last_capture_time >= refresh_rate
    
    # Camera input with auto-refresh
    camera_placeholder = st.empty()
    
    if should_capture:
        with camera_placeholder.container():
            camera_frame = st.camera_input(
                "Take a picture for proctoring",
                key=f"camera_{int(current_time)}"
            )
        st.session_state.last_capture_time = current_time
    else:
        # Show last frame with countdown
        time_remaining = refresh_rate - (current_time - st.session_state.last_capture_time)
        camera_placeholder.info(f"🔄 Next capture in {int(time_remaining)} seconds...")
        
        # Show last processed frame if available
        if 'last_processed_frame' in st.session_state:
            st.image(st.session_state.last_processed_frame, 
                    caption="Last Processed Frame", 
                    use_column_width=True)

with col2:
    st.markdown("### 📊 Detection Dashboard")
    
    # Real-time stats
    stats_placeholder = st.empty()
    alert_placeholder = st.empty()
    efficiency_placeholder = st.empty()

# ---------------------------------------------------------
# PROCESSING LOGIC
# ---------------------------------------------------------
if 'camera_frame' in locals() and camera_frame is not None:
    # Update processing state
    st.session_state.processing_state = "processing"
    
    try:
        # Process the frame
        image = Image.open(camera_frame)
        detected_objects, annotated_frame = process_frame(image)
        
        # Update statistics
        st.session_state.stats['total_frames_processed'] += 1
        st.session_state.last_processed_frame = annotated_frame
        
        # Display results in dashboard
        with col2:
            # Stats display
            stats_text = f"""
            **System Statistics:**
            - Frames Processed: {st.session_state.stats['total_frames_processed']}
            - Total Detections: {st.session_state.stats['total_detections']}
            - Alerts Triggered: {st.session_state.stats['alerts_triggered']}
            - Uptime: {(datetime.now() - st.session_state.stats['start_time']).seconds // 60} min
            """
            stats_placeholder.info(stats_text)
            
            # Current detection results
            if detected_objects:
                st.session_state.stats['total_detections'] += len(detected_objects)
                
                # Calculate efficiency metrics
                total_confidence = sum(obj['confidence'] for obj in detected_objects)
                avg_confidence = (total_confidence / len(detected_objects)) * 100
                
                efficiency_text = f"""
                **Current Detection:**
                - Objects Found: {len(detected_objects)}
                - Avg Confidence: {avg_confidence:.1f}%
                - Detection Delay: {detection_delay}s
                """
                efficiency_placeholder.warning(efficiency_text)
                
                # Check for high-confidence detections
                high_confidence_objects = [
                    obj for obj in detected_objects 
                    if obj['confidence'] > confidence_threshold
                ]
                
                if high_confidence_objects:
                    # Get the highest confidence object
                    main_object = max(high_confidence_objects, key=lambda x: x['confidence'])
                    
                    # Gemini validation for the main object
                    gemini_result = validate_with_gemini(image, main_object['name'])
                    
                    if "WARNING" in gemini_result:
                        alert_placeholder.error(f"🚨 {gemini_result}")
                        speak(f"Warning: {main_object['name']} detected")
                    else:
                        alert_placeholder.warning(f"⚠️ Potential {main_object['name']} detected but not confirmed")
                else:
                    alert_placeholder.warning("⚠️ Low confidence detections - monitoring...")
                    
                # Display detection details
                st.markdown("**Detected Objects:**")
                for obj in detected_objects:
                    confidence_percent = obj['confidence'] * 100
                    st.write(f"- {obj['name']}: {confidence_percent:.1f}%")
                    
            else:
                efficiency_placeholder.success("✅ No objects detected")
                alert_placeholder.success("🎉 Clear - No prohibited items")
                
        # Show annotated frame in main area
        with col1:
            st.image(annotated_frame, 
                    caption=f"Processed Frame - {len(detected_objects)} objects detected", 
                    use_column_width=True)
            
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        st.session_state.processing_state = "error"
        
else:
    # Initial state or waiting for input
    with col2:
        stats_placeholder.info("""
        **System Ready**
        - Waiting for camera input
        - Adjust settings in sidebar
        - Camera auto-refreshes periodically
        """)
        
        efficiency_placeholder.info("⏳ Monitoring will begin when camera is active")
        alert_placeholder.info("🔍 System ready for detection")

# ---------------------------------------------------------
# DETECTION HISTORY & LOGS
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Detection History")

# Create a simple log of recent activities
if st.session_state.stats['total_detections'] > 0:
    log_data = {
        "Timestamp": [datetime.now().strftime("%H:%M:%S")],
        "Frames Processed": [st.session_state.stats['total_frames_processed']],
        "Current Detections": [len(detected_objects) if 'detected_objects' in locals() else 0],
        "System Status": ["Active"]
    }
    st.dataframe(log_data, use_container_width=True)
else:
    st.info("No detection history yet. System monitoring in progress...")

# ---------------------------------------------------------
# FOOTER WITH STATUS
# ---------------------------------------------------------
st.markdown("---")
status_color = "#4CAF50" if st.session_state.processing_state == "ready" else "#FF9800"
status_text = "Ready" if st.session_state.processing_state == "ready" else "Monitoring"

st.markdown(
    f"<p style='text-align: center; color: {status_color}; font-weight: bold;'>"
    f"🔒 Status: {status_text} | Auto-refresh: {refresh_rate}s | Confidence: {confidence_threshold*100:.0f}%"
    "</p>",
    unsafe_allow_html=True
)

# Auto-refresh the app to simulate live monitoring
st.rerun()
