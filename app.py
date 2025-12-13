
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

