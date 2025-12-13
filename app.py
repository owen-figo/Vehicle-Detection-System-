
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

