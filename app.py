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

RESTRICT_START = 22  # 10 PM
RESTRICT_END = 6     # 6 AM

# ---------------- UI ----------------
st.set_page_config(page_title="Hostel Monitoring System", layout="wide")
st.title("🏨 AI Hostel Monitoring System")

# ---------------- FUNCTIONS ----------------
def is_restricted_time():
    hour = datetime.now().hour
    return hour >= RESTRICT_START or hour < RESTRICT_END

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
    st.error("❌ encodings.pkl not found! Run train_system.py first.")
    st.stop()

# ---------------- SESSION ----------------
if "violations" not in st.session_state:
    st.session_state.violations = []

if "detected" not in st.session_state:
    st.session_state.detected = set()

# ---------------- SIDEBAR ----------------
THRESHOLD = st.sidebar.slider("Recognition Threshold", 0.1, 1.5, 0.7)

# ---------------- DASHBOARD ----------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Students", len(set(known_names)))

with col2:
    status = "🔴 Restricted Time" if is_restricted_time() else "🟢 Allowed Time"
    st.metric("System Status", status)

# ---------------- CAMERA ----------------
run = st.checkbox("Start Monitoring Camera")
frame_placeholder = st.image([])

# ---------------- LOG ----------------
log_placeholder = st.empty()

def show_logs():
    if st.session_state.violations:
        df = pd.DataFrame(st.session_state.violations)
    else:
        df = pd.DataFrame(columns=["Name", "Time", "Status"])
    log_placeholder.dataframe(df, use_container_width=True)

show_logs()

# ---------------- MAIN LOOP ----------------
if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            faces = detector.detect_faces(rgb)
        except:
            continue

        for face in faces:
            x, y, w, h = face["box"]

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = x1 + w, y1 + h

            face_crop = rgb[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            face_crop = cv2.resize(face_crop, (160, 160))
            face_crop = np.expand_dims(face_crop, axis=0)

            encoding = embedder.embeddings(face_crop)[0]

            distances = np.linalg.norm(known_encodings - encoding, axis=1)
            idx = np.argmin(distances)
            score = distances[idx]

            name = "Unknown"
            color = (255, 0, 0)

            if score < THRESHOLD:
                name = known_names[idx]
                color = (0, 255, 0)

                # 🚨 VIOLATION LOGIC
                if is_restricted_time():
                    if name not in st.session_state.detected:
                        st.session_state.detected.add(name)

                        st.session_state.violations.append({
                            "Name": name,
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Status": "Unauthorized Movement"
                        })

                        st.warning(f"🚨 ALERT: {name} detected during restricted hours!")

                        show_logs()

            cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(rgb, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frame_placeholder.image(rgb)

    cap.release()

# ---------------- DOWNLOAD ----------------
if st.button("Download Violation Report"):
    if st.session_state.violations:
        df = pd.DataFrame(st.session_state.violations)
        filename = f"violations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)
        st.success(f"Saved {filename}")