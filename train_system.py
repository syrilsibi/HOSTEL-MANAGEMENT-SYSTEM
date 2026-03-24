import cv2
import os
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

# ---------------- CONFIG ----------------
DATASET_PATH = r"D:\New folder\SKILLPARK\HOSTEL_MANAGEMENT_SYSTEM\PROJECT\Dataset"
SAVE_PATH = r"D:\New folder\SKILLPARK\HOSTEL_MANAGEMENT_SYSTEM\encodings.pkl"

print("📂 Loading Models...")
detector = MTCNN()
embedder = FaceNet()

known_encodings = []
known_names = []

if not os.path.exists(DATASET_PATH):
    print("❌ Dataset not found!")
    exit()

print("🚀 Encoding Faces...")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
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

                embedding = embedder.embeddings(face_crop)[0]

                known_encodings.append(embedding)
                known_names.append(person)

                print(f"✅ Encoded: {person}")

        except:
            continue

# ---------------- SAVE ----------------
with open(SAVE_PATH, "wb") as f:
    pickle.dump({
        "encodings": np.array(known_encodings),
        "names": known_names
    }, f)

print(f"✅ Saved encodings to {SAVE_PATH}")