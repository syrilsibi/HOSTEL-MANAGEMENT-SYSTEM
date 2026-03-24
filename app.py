import streamlit as st
import cv2
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime
from mtcnn import MTCNN
from keras_facenet import FaceNet

# ---------------- CONFIG ----------------
ENCODINGS_PATH = r"D:\New folder\SKILLPARK\HOSTEL_MANAGEMENT_SYSTEM\encodings.pkl"

st.set_page_config(page_title="Hostel Monitoring", layout="wide", page_icon="🏨")

# ---------------- SESSION STATE INIT ----------------
if "violations" not in st.session_state:
    st.session_state.violations = []
if "detected" not in st.session_state:
    st.session_state.detected = set()
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "restrict_start" not in st.session_state:
    st.session_state.restrict_start = datetime.strptime("22:00", "%H:%M").time()
if "restrict_end" not in st.session_state:
    st.session_state.restrict_end = datetime.strptime("06:00", "%H:%M").time()
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.7
if "camera_source" not in st.session_state:
    st.session_state.camera_source = "0"
if "enable_alerts" not in st.session_state:
    st.session_state.enable_alerts = False
if "alert_email" not in st.session_state:
    st.session_state.alert_email = "admin@hostel.com"

# ---------------- FUNCTIONS ----------------
def is_restricted_time():
    now = datetime.now().time()
    start = st.session_state.restrict_start
    end = st.session_state.restrict_end
    
    if start <= end:
        return start <= now <= end
    else:
        # Crosses midnight (e.g., 22:00 to 06:00)
        return start <= now or now <= end

def send_alert(name, time_str):
    if st.session_state.enable_alerts:
        # Placeholder for actual email/SMS sending logic
        print(f"[MOCK ALERT] Email sent to {st.session_state.alert_email}: Unauthorized movement by {name} at {time_str}")

@st.cache_resource
def load_models():
    return MTCNN(), FaceNet()

@st.cache_data
def load_db():
    if not os.path.exists(ENCODINGS_PATH):
        return None, None
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

detector, embedder = load_models()
known_encodings, known_names = load_db()

if known_encodings is None:
    st.error(f"❌ '{ENCODINGS_PATH}' not found! Please run the training script first.")
    st.stop()

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Dashboard", "Admin Panel"])

# ---------------- ADMIN PANEL ----------------
if menu == "Admin Panel":
    st.title("⚙️ Admin Settings")
    
    if not st.session_state.admin_logged_in:
        st.subheader("Login Required")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == "admin":  # simple hardcoded password
                st.session_state.admin_logged_in = True
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Incorrect password.")
    else:
        st.success("Logged in as Administrator")
        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()
            
        st.markdown("---")
        st.subheader("System Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.restrict_start = st.time_input("Restricted Start Time", st.session_state.restrict_start)
            st.session_state.restrict_end = st.time_input("Restricted End Time", st.session_state.restrict_end)
            st.session_state.threshold = st.slider("Recognition Threshold", 0.1, 1.5, st.session_state.threshold, 0.05, help="Lower value = stricter recognition.")
            
        with col2:
            cam_input = st.text_input("Camera Source (0 for default, or IP Camera URL)", st.session_state.camera_source)
            st.session_state.camera_source = cam_input
            
            st.markdown("### Alerts")
            st.session_state.enable_alerts = st.checkbox("Enable Email/SMS Alerts", st.session_state.enable_alerts)
            st.session_state.alert_email = st.text_input("Alert Email Address", st.session_state.alert_email)
            
        st.info("Settings are applied automatically across the dashboard.")

# ---------------- DASHBOARD ----------------
elif menu == "Dashboard":
    st.title("🏨 AI Hostel Monitoring System")
    
    # Render dashboard metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Registered Students", len(set(known_names)))
        
    with col2:
        status_text = "🔴 Restricted Time" if is_restricted_time() else "🟢 Allowed Time"
        st.metric("System Mode", status_text)
        
    with col3:
        st.metric("Violations Logged", len(st.session_state.violations))
    
    st.markdown("---")
    
    cam_col, log_col = st.columns([2, 1])
    
    with cam_col:
        st.subheader("Live Camera Feed")
        run_camera = st.checkbox("Start Monitoring System", value=is_restricted_time(), help="Automatically turns on if it is currently restricted time.")
        alert_placeholder = st.empty()
        frame_placeholder = st.image([])
        
    with log_col:
        st.subheader("Real-time Violation Logs")
        
        # We need a placeholder for the dataframe so we can update it in the loop
        log_box = st.empty()
        
        def show_logs():
            if st.session_state.violations:
                df = pd.DataFrame(st.session_state.violations)
                # Reverse to show newest first
                df = df.iloc[::-1].reset_index(drop=True)
                log_box.dataframe(df, use_container_width=True)
            else:
                log_box.info("No violations recorded.")
                
        show_logs()
        
        # We place the download button outside so it only renders once per Streamlit cycle
        if st.session_state.violations:
            st.markdown("### Export")
            df_export = pd.DataFrame(st.session_state.violations)
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
            
    # Main Camera Loop
    if run_camera:
        cam_src = st.session_state.camera_source
        if cam_src.isdigit():
            cam_src = int(cam_src)
            
        cap = cv2.VideoCapture(cam_src)
        
        if not cap.isOpened():
            st.error(f"❌ Could not open camera source: {cam_src}")
        else:
            try:
                while run_camera:
                    ret, img = cap.read()
                    if not ret:
                        st.error("Failed to read frame from camera.")
                        break
                        
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = detector.detect_faces(rgb)
                    
                    for face in faces:
                        x, y, w, h = face["box"]
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = x1 + w, y1 + h
                        
                        face_crop = rgb[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue
                            
                        face_crop = cv2.resize(face_crop, (160, 160))
                        face_crop = np.expand_dims(face_crop, axis=0)
                        
                        enc = embedder.embeddings(face_crop)[0]
                        distances = np.linalg.norm(known_encodings - enc, axis=1)
                        idx = np.argmin(distances)
                        score = distances[idx]
                        
                        name = "Unknown"
                        color = (255, 0, 0) # Red for unknown
                        
                        if score < st.session_state.threshold:
                            name = known_names[idx]
                            color = (0, 255, 0) # Green for known
                            
                            # Only log violation if it's restricted time AND the person is KNOWN
                            if is_restricted_time():
                                color = (255, 165, 0) # Orange for warning
                                if name not in st.session_state.detected:
                                    st.session_state.detected.add(name)
                                    time_str = datetime.now().strftime("%H:%M:%S")
                                    
                                    # Log Violation
                                    st.session_state.violations.append({
                                        "Name": name,
                                        "Time": time_str,
                                        "Status": "Unauthorized"
                                    })
                                    
                                    # Send Mock Alert
                                    send_alert(name, time_str)
                                    
                                    # UI Alert
                                    alert_placeholder.warning(f"🚨 ALERT: {name} detected during restricted hours!")
                                    show_logs()
                                    
                        # Draw bounding box and label
                        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(rgb, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                    frame_placeholder.image(rgb, channels="RGB")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                cap.release()