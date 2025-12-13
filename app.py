<<<<<<< HEAD

#   CAR DETECTION PROJECT — Combined Training + Streamlit UI
#   - Auto-train if model missing
#   - Train button inside Streamlit
#   - MobileNetV2 + CV Bounding Boxes


import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from PIL import Image
import streamlit as st
from tqdm import tqdm

# ============================================================
# 1. GLOBAL SETTINGS
# ============================================================
IMG_SIZE = (224, 224)
MODEL_NAME = "classifier_mobilenetv2.h5"
DATA_DIR = "data"   # contains /vehicles and /non-vehicles


# ============================================================
# 2. DATA LOADING
# ============================================================
def load_images_from_folder(folder, label):
    imgs, labels = [], []
    for filename in tqdm(os.listdir(folder), desc=f"Loading {folder}"):
        path = os.path.join(folder, filename)
        try:
            img = image.load_img(path, target_size=IMG_SIZE)
            img = image.img_to_array(img)
            imgs.append(img)
            labels.append(label)
        except:
            pass
    return imgs, labels


def load_dataset():
    vehicle_dir = os.path.join(DATA_DIR, "vehicles")
    nonvehicle_dir = os.path.join(DATA_DIR, "non-vehicles")

    X1, y1 = load_images_from_folder(vehicle_dir, 1)
    X0, y0 = load_images_from_folder(nonvehicle_dir, 0)

    X = np.array(X1 + X0)
    y = np.array(y1 + y0)
    X = X / 255.0

    return train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)


# ============================================================
# 3. BUILD MODEL
# ============================================================
def build_model():
    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

=======
import os
import cv2
import numpy as np
import streamlit as st
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from PIL import Image
import joblib
import time

DATA_DIR = "data"  # contains /vehicles and /non-vehicles
MODEL_FILE = "canny_hog_svm.pkl"
SCALER_FILE = "canny_scaler.pkl"


# =============================================================
# 1. CANNY EDGE DETECTOR + HOG FEATURES
# =============================================================
def extract_canny_hog(img):
    img = cv2.resize(img, (128, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Canny edges
    edges = cv2.Canny(gray, 80, 180)

    # Step 2: HOG on the edge map
    features = hog(
        edges,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    return features


# =============================================================
# 2. LOAD TRAINING DATASET
# =============================================================
def load_dataset():
    X, y = [], []

    veh_dir = os.path.join(DATA_DIR, "vehicles")
    nonveh_dir = os.path.join(DATA_DIR, "non-vehicles")

    for f in os.listdir(veh_dir):
        img = cv2.imread(os.path.join(veh_dir, f))
        if img is not None:
            X.append(extract_canny_hog(img))
            y.append(1)

    for f in os.listdir(nonveh_dir):
        img = cv2.imread(os.path.join(nonveh_dir, f))
        if img is not None:
            X.append(extract_canny_hog(img))
            y.append(0)

    return np.array(X), np.array(y)


# =============================================================
# 3. TRAIN MODEL
# =============================================================
def train_model():
    st.write("🔧 Training Canny + HOG + SVM model...")

    X, y = load_dataset()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = LinearSVC()
    svm.fit(X_scaled, y)

    joblib.dump(svm, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    st.success("Training complete!")
    return svm, scaler


# =============================================================
# 4. PREDICT IMAGE
# =============================================================
def predict_vehicle(img, svm, scaler):
    feat = extract_canny_hog(img)
    feat_scaled = scaler.transform([feat])

    pred = svm.predict(feat_scaled)[0]
    conf = svm.decision_function(feat_scaled)[0]

    conf = min(max((conf + 5) * 10, 0), 100)
    label = "Vehicle" if pred == 1 else "Non-Vehicle"

    return label, conf


# =============================================================
# 5. STREAMLIT UI
# =============================================================
def run_app():
    st.title("🚗 Vehicle Detection using Canny Edge + HOG + SVM")

    if not os.path.exists(MODEL_FILE):
        svm, scaler = train_model()
    else:
        svm = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)

    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

    if file:
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        st.image(img_np, caption="Uploaded Image", use_column_width=True)

        start = time.time()
        label, conf = predict_vehicle(img_bgr, svm, scaler)
        runtime = time.time() - start

        st.subheader(f"Prediction: **{label}**")
        st.write(f"Confidence: {conf:.2f}%")
        st.success(f"Runtime: {runtime:.4f} seconds")

        # Show Canny edge map
        edges = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 80, 180)
        st.image(edges, caption="Canny Edge Map", use_column_width=True)


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    run_app()
>>>>>>> ff5aba06 (Initial commit: Canny + HOG + SVM vehicle detection)
