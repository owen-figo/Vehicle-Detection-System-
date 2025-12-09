
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
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(2, activation="softmax")(x)

    model = models.Model(base.input, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ============================================================
# 4. TRAINING FUNCTION
# ============================================================
def train_model():
    st.write("### Training model... This may take a few minutes.")
    X_train, X_test, y_train, y_test = load_dataset()

    model = build_model()
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32
    )

    model.save(MODEL_NAME)
    st.success("Model successfully trained and saved!")

    loss, acc = model.evaluate(X_test, y_test)
    st.write(f"### Test Accuracy: **{acc:.4f}**")

    return model


# ============================================================
# 5. PREPROCESS FOR PREDICTION
# ============================================================
def preprocess_for_model(img_bgr):
    img_rgb = img_bgr[..., ::-1]
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_resized = img_resized.astype("float32") / 255.0
    return np.expand_dims(img_resized, axis=0)


# ============================================================
# 6. CV DETECTOR 
# ============================================================
def detect_vehicle_bboxes(cv_img):
    img = cv_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 900:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append([x, y, w, h])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    return bboxes, img


# ============================================================
# 7. STREAMLIT APP
# ============================================================
def run_app():
    st.title("🚗 Vehicle Detection — Training + Prediction in One App")

    # ------------------------------------------
    # Auto-train if model missing
    # ------------------------------------------
    if not os.path.exists(MODEL_NAME):
        st.warning("⚠ Model not found — training automatically...")
        model = train_model()
    else:
        @st.cache_resource
        def load_model():
            return tf.keras.models.load_model(MODEL_NAME)

        model = load_model()

    # ------------------------------------------
    # Manual training button
    # ------------------------------------------
    if st.button("🔧 Retrain Model"):
        model = train_model()

    # ------------------------------------------
    # Upload Section
    # ------------------------------------------
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        st.image(img_np, caption="Uploaded Image", use_column_width=True)

        # Prediction
        input_tensor = preprocess_for_model(img_bgr)
        preds = model.predict(input_tensor)
        pred_class = np.argmax(preds)
        confidence = float(np.max(preds)) * 100

        st.subheader(f"Prediction: **{'Vehicle' if pred_class == 1 else 'Non-Vehicle'}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        # Bounding box detection
        if pred_class == 1:
            bboxes, annotated = detect_vehicle_bboxes(img_bgr)
            ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, caption=f"Detected {len(bboxes)} region(s)")
        else:
            st.info("No vehicle detected.")


# ============================================================
# 8. STREAMLIT ENTRY POINT  to run streamlit run app.py
# ============================================================
if __name__ == "__main__":
    run_app()
