import os
import time
import math
import joblib
import numpy as np
import cv2
import streamlit as st

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# =============================================================
# CONFIG
# =============================================================
DATA_DIR = "data"
POS_DIR = os.path.join(DATA_DIR, "vehicles")
NEG_DIR = os.path.join(DATA_DIR, "non-vehicles")

MODEL_FILE = "canny_hog_svm.pkl"
SCALER_FILE = "canny_scaler.pkl"

WIN_W, WIN_H = 128, 64

# default canny
CANNY_T1_DEFAULT = 80
CANNY_T2_DEFAULT = 180

# pyramid + sliding window
PYR_SCALE_DEFAULT = 1.25
STEP_SIZE_DEFAULT = 16


# =============================================================
# UTILS
# =============================================================
def is_image_file(name: str) -> bool:
    n = name.lower()
    return n.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def safe_listdir(path: str):
    if not os.path.isdir(path):
        return []
    return [f for f in os.listdir(path) if is_image_file(f)]


def sigmoid_conf(score: float) -> float:
    # not a calibrated probability; only for visualization
    return float(100.0 / (1.0 + np.exp(-score)))


def clamp_bbox(bbox, W, H):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W - 1, int(x2)))
    y2 = max(0, min(H - 1, int(y2)))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return (x1, y1, x2, y2)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def norm_to_u8(img_float: np.ndarray) -> np.ndarray:
    x = img_float.astype(np.float32)
    x = x - float(np.min(x))
    denom = float(np.max(x))
    if denom < 1e-9:
        return np.zeros_like(x, dtype=np.uint8)
    x = x / denom
    return (x * 255.0).astype(np.uint8)


# =============================================================
# FEATURE EXTRACTION
# =============================================================
def extract_canny_hog_from_patch(patch_bgr: np.ndarray, canny_t1: int, canny_t2: int) -> np.ndarray:
    patch_bgr = cv2.resize(patch_bgr, (WIN_W, WIN_H))
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_t1, canny_t2)

    feat = hog(
        edges,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )
    return feat.astype(np.float32)


def extract_canny_hog_debug(patch_bgr: np.ndarray, canny_t1: int, canny_t2: int, visualize_hog: bool = True):
    """
    Debug version to SHOW every stage on a patch:
      patch -> resize(128x64) -> grayscale -> canny -> HOG(feature vector + optional visualization)
    """
    patch_resized = cv2.resize(patch_bgr, (WIN_W, WIN_H))
    gray = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_t1, canny_t2)

    if visualize_hog:
        feat, hog_img = hog(
            edges,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True,
            feature_vector=True
        )
    else:
        feat = hog(
            edges,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True
        )
        hog_img = None

    return feat.astype(np.float32), patch_resized, gray, edges, hog_img


def score_patch(patch_bgr: np.ndarray, svm: LinearSVC, scaler: StandardScaler, canny_t1: int, canny_t2: int):
    feat = extract_canny_hog_from_patch(patch_bgr, canny_t1, canny_t2)
    feat_scaled = scaler.transform([feat])
    score = float(svm.decision_function(feat_scaled)[0])
    pred = int(svm.predict(feat_scaled)[0])  # 1 if score >= 0, else 0 (for LinearSVC)
    conf = sigmoid_conf(score)
    return pred, score, conf


# =============================================================
# DATASET LOADING
# =============================================================
def load_base_dataset(canny_t1: int, canny_t2: int):
    X, y = [], []

    pos_files = safe_listdir(POS_DIR)
    neg_files = safe_listdir(NEG_DIR)

    if len(pos_files) == 0 or len(neg_files) == 0:
        raise FileNotFoundError(
            "Dataset not found or empty.\n"
            f"Expected images in:\n- {POS_DIR}\n- {NEG_DIR}"
        )

    for f in pos_files:
        img = cv2.imread(os.path.join(POS_DIR, f))
        if img is None:
            continue
        X.append(extract_canny_hog_from_patch(img, canny_t1, canny_t2))
        y.append(1)

    for f in neg_files:
        img = cv2.imread(os.path.join(NEG_DIR, f))
        if img is None:
            continue
        X.append(extract_canny_hog_from_patch(img, canny_t1, canny_t2))
        y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    if len(X) == 0:
        raise ValueError("No valid images could be read from dataset folders.")

    return X, y


# =============================================================
# SLIDING WINDOW + PYRAMID
# =============================================================
def sliding_window(image_bgr: np.ndarray, step: int, window_size: tuple[int, int], y_start: int = 0):
    wW, wH = window_size
    H, W = image_bgr.shape[:2]
    y_start = max(0, min(H - 1, y_start))
    for y in range(y_start, H - wH + 1, step):
        for x in range(0, W - wW + 1, step):
            yield x, y, image_bgr[y:y + wH, x:x + wW]


def scan_image_collect_scores(
    img_bgr: np.ndarray,
    svm: LinearSVC,
    scaler: StandardScaler,
    canny_t1: int,
    canny_t2: int,
    step: int,
    pyr_scale: float,
    y_start_ratio: float,
    max_levels: int,
    collect_level_stats: bool = False,
):
    """
    Returns:
      windows: list of dicts with bbox and score/pred/conf
      all_scores: list of float (scores for distribution)
      (optional) level_stats: list of per-pyramid-level stats for visualization
    """
    H0, W0 = img_bgr.shape[:2]
    windows = []
    all_scores = []
    level_stats = []

    # if too small, just evaluate whole image as one "window"
    if W0 < WIN_W or H0 < WIN_H:
        pred, score, conf = score_patch(img_bgr, svm, scaler, canny_t1, canny_t2)
        windows.append({
            "bbox": (0, 0, W0 - 1, H0 - 1),
            "pred": pred,
            "score": score,
            "conf": conf
        })
        all_scores.append(score)
        if collect_level_stats:
            level_stats.append({
                "level": 0,
                "scale": 1.0,
                "resized_W": W0,
                "resized_H": H0,
                "y_start": 0,
                "windows": 1,
                "max_score": score,
                "p90_score": score
            })
            return windows, all_scores, level_stats
        return windows, all_scores

    scale = 1.0
    levels = 0
    while True:
        if levels >= max_levels:
            break

        Ws, Hs = int(W0 * scale), int(H0 * scale)
        if Ws < WIN_W or Hs < WIN_H:
            break

        resized = cv2.resize(img_bgr, (Ws, Hs), interpolation=cv2.INTER_AREA)
        y_start = int((resized.shape[0]) * y_start_ratio)

        level_scores = []
        level_win_count = 0

        for x, y, win in sliding_window(resized, step, (WIN_W, WIN_H), y_start=y_start):
            pred, score, conf = score_patch(win, svm, scaler, canny_t1, canny_t2)
            all_scores.append(score)
            level_scores.append(score)
            level_win_count += 1

            # map bbox back to original
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + WIN_W) / scale)
            y2 = int((y + WIN_H) / scale)

            x1 = max(0, min(W0 - 1, x1))
            y1 = max(0, min(H0 - 1, y1))
            x2 = max(0, min(W0 - 1, x2))
            y2 = max(0, min(H0 - 1, y2))

            windows.append({
                "bbox": (x1, y1, x2, y2),
                "pred": pred,
                "score": score,
                "conf": conf,
                "level": levels,
                "scale": scale
            })

        if collect_level_stats:
            if level_win_count > 0:
                p90 = float(np.percentile(np.array(level_scores, dtype=np.float32), 90))
                mx = float(np.max(level_scores))
            else:
                p90 = float("nan")
                mx = float("nan")
            level_stats.append({
                "level": levels,
                "scale": float(scale),
                "resized_W": int(Ws),
                "resized_H": int(Hs),
                "y_start": int(y_start),
                "windows": int(level_win_count),
                "max_score": mx,
                "p90_score": p90
            })

        scale /= pyr_scale
        levels += 1

    if collect_level_stats:
        return windows, all_scores, level_stats
    return windows, all_scores


# =============================================================
# 1-BOX OUTPUT (merge / union)
# =============================================================
def merge_topk_boxes(windows, pos_score_thresh: float, top_k: int):
    """
    Collect all windows with score >= pos_score_thresh (strong positives),
    then merge ONLY the top_k by score into one union box.
    Returns: bbox or None, merged_score (max score), merged_conf, used_count
    """
    positives = [w for w in windows if w["score"] >= pos_score_thresh]
    if len(positives) == 0:
        return None, None, None, 0

    positives.sort(key=lambda w: w["score"])
    selected = positives[-top_k:] if len(positives) > top_k else positives

    xs1 = [w["bbox"][0] for w in selected]
    ys1 = [w["bbox"][1] for w in selected]
    xs2 = [w["bbox"][2] for w in selected]
    ys2 = [w["bbox"][3] for w in selected]

    bbox = (min(xs1), min(ys1), max(xs2), max(ys2))
    best_score = float(max(w["score"] for w in selected))
    best_conf = sigmoid_conf(best_score)
    return bbox, best_score, best_conf, len(selected)


def bbox_to_yolo_line(bbox, img_w: int, img_h: int, class_id: int = 0):
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    # normalize to [0,1]
    xc_n = xc / float(img_w)
    yc_n = yc / float(img_h)
    bw_n = bw / float(img_w)
    bh_n = bh / float(img_h)

    return f"{class_id} {xc_n:.6f} {yc_n:.6f} {bw_n:.6f} {bh_n:.6f}"


# =============================================================
# HEATMAP VISUALIZATION
# =============================================================
def build_heatmap_overlay(
    img_bgr: np.ndarray,
    windows,
    step: int,
    blur_ksize: int = 31,
):
    """
    Builds a max-score heatmap (per-window bbox area) and overlays it.
    Simpler & robust: paint each window area with max(score) then normalize.
    """
    H, W = img_bgr.shape[:2]
    heat = np.full((H, W), -np.inf, dtype=np.float32)

    # paint max score into each bbox area
    for w in windows:
        x1, y1, x2, y2 = w["bbox"]
        score = float(w["score"])
        if x2 <= x1 or y2 <= y1:
            continue
        heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], score)

    # replace -inf with min score
    finite = np.isfinite(heat)
    if not np.any(finite):
        return img_bgr, None

    minv = float(np.min(heat[finite]))
    heat[~finite] = minv

    # normalize 0..255 for colormap
    heat_norm = heat - float(np.min(heat))
    denom = float(np.max(heat_norm)) if float(np.max(heat_norm)) > 1e-9 else 1.0
    heat_norm = heat_norm / denom
    heat_u8 = (heat_norm * 255.0).astype(np.uint8)

    if blur_ksize and blur_ksize > 1:
        # ensure odd
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        heat_u8 = cv2.GaussianBlur(heat_u8, (blur_ksize, blur_ksize), 0)

    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.65, heat_color, 0.35, 0)
    return overlay, heat_u8


# =============================================================
# HARD NEGATIVE MINING
# =============================================================
def train_linear_svm(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    svm = LinearSVC()
    svm.fit(Xs, y)
    return svm, scaler


def mine_hard_negatives(
    svm: LinearSVC,
    scaler: StandardScaler,
    canny_t1: int,
    canny_t2: int,
    hard_thresh: float,
    step: int,
    pyr_scale: float,
    y_start_ratio: float,
    max_levels: int,
    max_hard_per_image: int,
    max_total_hard: int,
):
    """
    Scan NEGATIVE images, collect windows that score HIGH (false positives),
    add them as extra negative training samples.
    """
    neg_files = safe_listdir(NEG_DIR)
    hard_feats = []
    hard_count = 0

    for f in neg_files:
        if hard_count >= max_total_hard:
            break

        img = cv2.imread(os.path.join(NEG_DIR, f))
        if img is None:
            continue

        windows, _ = scan_image_collect_scores(
            img_bgr=img,
            svm=svm,
            scaler=scaler,
            canny_t1=canny_t1,
            canny_t2=canny_t2,
            step=step,
            pyr_scale=pyr_scale,
            y_start_ratio=y_start_ratio,
            max_levels=max_levels,
        )

        # take the strongest false positives
        candidates = [w for w in windows if w["score"] >= hard_thresh]
        if len(candidates) == 0:
            continue

        candidates.sort(key=lambda w: w["score"], reverse=True)
        candidates = candidates[:max_hard_per_image]

        for w in candidates:
            x1, y1, x2, y2 = w["bbox"]
            if x2 <= x1 or y2 <= y1:
                continue
            patch = img[y1:y2, x1:x2]
            feat = extract_canny_hog_from_patch(patch, canny_t1, canny_t2)
            hard_feats.append(feat)
            hard_count += 1
            if hard_count >= max_total_hard:
                break

    if len(hard_feats) == 0:
        return np.empty((0, 1), dtype=np.float32)

    return np.array(hard_feats, dtype=np.float32)


def train_with_hard_negative_mining(
    canny_t1: int,
    canny_t2: int,
    iterations: int,
    hard_thresh: float,
    step: int,
    pyr_scale: float,
    y_start_ratio: float,
    max_levels: int,
    max_hard_per_image: int,
    max_total_hard: int,
):
    """
    Iterative training:
      1) Train on base dataset
      2) Mine hard negatives using current model
      3) Retrain by adding hard negatives (label 0)
    """
    X_base, y_base = load_base_dataset(canny_t1, canny_t2)

    # initial train
    svm, scaler = train_linear_svm(X_base, y_base)

    mined_total = 0
    for it in range(iterations):
        hard_X = mine_hard_negatives(
            svm=svm,
            scaler=scaler,
            canny_t1=canny_t1,
            canny_t2=canny_t2,
            hard_thresh=hard_thresh,
            step=step,
            pyr_scale=pyr_scale,
            y_start_ratio=y_start_ratio,
            max_levels=max_levels,
            max_hard_per_image=max_hard_per_image,
            max_total_hard=max_total_hard,
        )

        if hard_X.shape[0] == 0:
            st.info(f"Hard mining iteration {it+1}: no hard negatives found at threshold {hard_thresh}.")
            break

        mined_total += hard_X.shape[0]
        y_hard = np.zeros((hard_X.shape[0],), dtype=np.int32)

        X_new = np.vstack([X_base, hard_X])
        y_new = np.concatenate([y_base, y_hard])

        svm, scaler = train_linear_svm(X_new, y_new)

        # update base to keep accumulating mined samples across iterations
        X_base, y_base = X_new, y_new

        st.write(f"✅ Iter {it+1}/{iterations}: mined {hard_X.shape[0]} hard negatives (total {mined_total}).")

    joblib.dump(svm, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return svm, scaler, mined_total


def load_or_train_simple(canny_t1: int, canny_t2: int):
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)
    X, y = load_base_dataset(canny_t1, canny_t2)
    svm, scaler = train_linear_svm(X, y)
    joblib.dump(svm, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return svm, scaler


# =============================================================
# STREAMLIT APP
# =============================================================
def run_app():
    st.title("🚗 Vehicle Detection (1 Box) — Canny + HOG + SVM")
    st.caption("Now shows every methodology stage: resize, grayscale, Canny, patch HOG, evaluation, bbox merge, heatmap, export.")

    # Sidebar controls
    st.sidebar.header("Detection settings")
    canny_t1 = st.sidebar.slider("Canny threshold1", 0, 255, CANNY_T1_DEFAULT, 1)
    canny_t2 = st.sidebar.slider("Canny threshold2", 0, 255, CANNY_T2_DEFAULT, 1)

    step = st.sidebar.slider("Sliding step (pixels)", 4, 64, STEP_SIZE_DEFAULT, 1)
    pyr_scale = st.sidebar.slider("Pyramid scale", 1.05, 2.0, float(PYR_SCALE_DEFAULT), 0.01)
    max_levels = st.sidebar.slider("Max pyramid levels", 1, 12, 6, 1)

    y_start_ratio = st.sidebar.slider("Scan from (vertical ratio)", 0.0, 0.8, 0.33, 0.01)
    st.sidebar.caption("0.33 means ignore top 33% (sky/trees) → fewer false positives.")

    pos_score_thresh = st.sidebar.slider("POS_SCORE_THRESH (vehicle filter)", -2.0, 8.0, 2.5, 0.1)
    top_k = st.sidebar.slider("TOP_K windows to merge", 1, 200, 15, 1)

    st.sidebar.header("Visualization toggles")
    show_method_steps = st.sidebar.checkbox("Show methodology steps", True)
    show_level_stats = st.sidebar.checkbox("Show pyramid-level evaluation table", True)
    show_top_windows = st.sidebar.checkbox("Show top positive windows overlay (optional)", False)
    top_windows_n = st.sidebar.slider("Top windows to draw", 1, 100, 20, 1) if show_top_windows else 20

    show_heatmap = st.sidebar.checkbox("Show heatmap overlay", True)
    heat_blur = st.sidebar.slider("Heatmap blur ksize", 0, 101, 31, 2)
    show_hist = st.sidebar.checkbox("Show score distribution (hist)", True)

    st.sidebar.header("Training (Hard Negative Mining)")
    iters = st.sidebar.slider("Mining iterations", 0, 5, 2, 1)
    hard_thresh = st.sidebar.slider("Hard negative threshold", -2.0, 8.0, 1.5, 0.1)
    max_hard_per_image = st.sidebar.slider("Max hard neg per negative image", 1, 50, 10, 1)
    max_total_hard = st.sidebar.slider("Max total hard negatives", 50, 5000, 800, 50)

    colA, colB = st.columns(2)
    with colA:
        if st.button("🧠 Train (simple)"):
            with st.spinner("Training on base dataset..."):
                try:
                    svm, scaler = load_or_train_simple(canny_t1, canny_t2)
                    st.success("Trained & saved model.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    with colB:
        if st.button("🔥 Train + Hard Negative Mining"):
            with st.spinner("Training with hard negative mining..."):
                try:
                    svm, scaler, mined_total = train_with_hard_negative_mining(
                        canny_t1=canny_t1,
                        canny_t2=canny_t2,
                        iterations=iters,
                        hard_thresh=hard_thresh,
                        step=step,
                        pyr_scale=pyr_scale,
                        y_start_ratio=y_start_ratio,
                        max_levels=max_levels,
                        max_hard_per_image=max_hard_per_image,
                        max_total_hard=max_total_hard,
                    )
                    st.success(f"Done. Mined total hard negatives: {mined_total}. Model saved.")
                except Exception as e:
                    st.error(f"Hard negative mining failed: {e}")

    # Load model (or train quickly if missing)
    try:
        svm, scaler = load_or_train_simple(canny_t1, canny_t2)
    except Exception as e:
        st.error(f"Model load/train failed: {e}")
        return

    st.divider()
    st.subheader("Upload image (OpenCV decode)")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if not uploaded:
        st.info("Upload an image to run detection.")
        return

    # Robust decode (avoids PIL.UnidentifiedImageError)
    file_bytes = np.asarray(bytearray(uploaded.getvalue()), dtype=np.uint8)
    img_bgr_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr_raw is None:
        st.error("Failed to decode image. Please upload a valid raster image file.")
        return

    # ---------------------------------------------------------
    # METHODOLOGY STEP: Speed cap resize (this affects scanning speed)
    # ---------------------------------------------------------
    max_w = 1100
    H_raw, W_raw = img_bgr_raw.shape[:2]
    img_bgr = img_bgr_raw.copy()
    resized_for_speed = False
    if W_raw > max_w:
        new_h = int(H_raw * (max_w / W_raw))
        img_bgr = cv2.resize(img_bgr, (max_w, new_h), interpolation=cv2.INTER_AREA)
        resized_for_speed = True

    H0, W0 = img_bgr.shape[:2]

    # Show input
    st.image(bgr_to_rgb(img_bgr), caption="Uploaded Image (used for scanning)", use_column_width=True)

    # Draw scan start line preview
    scan_preview = img_bgr.copy()
    y_start_vis = int(H0 * y_start_ratio)
    cv2.line(scan_preview, (0, y_start_vis), (W0 - 1, y_start_vis), (0, 255, 255), 2)
    cv2.putText(
        scan_preview,
        f"Scan starts at y={y_start_vis} (y_start_ratio={y_start_ratio:.2f})",
        (10, max(20, y_start_vis - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    if show_method_steps:
        with st.expander("✅ Methodology Steps (Resize → Grayscale → Canny → HOG on Patch)", expanded=True):
            st.markdown("### 1) Resize (speed cap for scanning)")
            st.write(f"Original: {W_raw}x{H_raw}  →  Used for scanning: {W0}x{H0}")
            if resized_for_speed:
                st.info("Speed cap applied: image width limited to 1100 pixels for faster scanning.")
            st.image(bgr_to_rgb(img_bgr), caption="Image after speed cap (this is what the scanner uses)", use_column_width=True)

            st.markdown("### 2) Grayscale (global view)")
            gray_global = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            st.image(gray_global, caption="Global Grayscale", use_column_width=True)

            st.markdown("### 3) Canny Edge Detection (global view)")
            edges_global = cv2.Canny(gray_global, canny_t1, canny_t2)
            st.image(edges_global, caption=f"Global Canny Edges (t1={canny_t1}, t2={canny_t2})", use_column_width=True)

            st.markdown("### 4) Scanning Region (evaluation starts below this line)")
            st.image(bgr_to_rgb(scan_preview), caption="Scan start line (yellow)", use_column_width=True)

    # ---------------------------------------------------------
    # EVALUATION: scan + collect scores
    # ---------------------------------------------------------
    st.subheader("Detection / Evaluation (Sliding Window + Pyramid)")
    start = time.time()

    if show_level_stats:
        windows, all_scores, level_stats = scan_image_collect_scores(
            img_bgr=img_bgr,
            svm=svm,
            scaler=scaler,
            canny_t1=canny_t1,
            canny_t2=canny_t2,
            step=step,
            pyr_scale=pyr_scale,
            y_start_ratio=y_start_ratio,
            max_levels=max_levels,
            collect_level_stats=True,
        )
    else:
        windows, all_scores = scan_image_collect_scores(
            img_bgr=img_bgr,
            svm=svm,
            scaler=scaler,
            canny_t1=canny_t1,
            canny_t2=canny_t2,
            step=step,
            pyr_scale=pyr_scale,
            y_start_ratio=y_start_ratio,
            max_levels=max_levels,
            collect_level_stats=False,
        )
        level_stats = None

    bbox, best_score, best_conf, used_count = merge_topk_boxes(windows, pos_score_thresh, top_k)
    runtime = time.time() - start

    # Score distribution (exact)
    if len(all_scores) == 0:
        st.error("No windows were evaluated (image may be too small).")
        return

    scores_np = np.array(all_scores, dtype=np.float32)
    st.write(f"Windows evaluated: **{len(all_scores)}**")
    st.write(f"Runtime (scan + scoring + merge): **{runtime:.4f} s**")
    st.write(
        f"Score stats: min **{float(scores_np.min()):.3f}**, "
        f"p50 **{float(np.percentile(scores_np, 50)):.3f}**, "
        f"p90 **{float(np.percentile(scores_np, 90)):.3f}**, "
        f"p99 **{float(np.percentile(scores_np, 99)):.3f}**, "
        f"max **{float(scores_np.max()):.3f}**"
    )

    if show_level_stats and level_stats is not None:
        st.markdown("#### Pyramid-level evaluation summary")
        st.dataframe(level_stats, use_container_width=True)

    if show_hist:
        fig = plt.figure()
        plt.hist(scores_np, bins=60)
        plt.title("Decision Function Score Distribution")
        plt.xlabel("SVM score (decision_function)")
        plt.ylabel("count")
        st.pyplot(fig, clear_figure=True)

    # ---------------------------------------------------------
    # Bounding Box + (optional) top windows visualization
    # ---------------------------------------------------------
    st.subheader("Bounding Box (Top-K Merge Output)")
    vis = img_bgr.copy()

    if show_top_windows:
        positives = [w for w in windows if w["score"] >= pos_score_thresh]
        positives.sort(key=lambda w: w["score"], reverse=True)
        positives = positives[:top_windows_n]

        for w in positives:
            x1, y1, x2, y2 = clamp_bbox(w["bbox"], W0, H0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.putText(
            vis,
            f"Top {len(positives)} positive windows (red) drawn",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    if bbox is None:
        st.warning("No vehicle box found (try lowering POS_SCORE_THRESH or TOP_K / increase scanning region).")
        st.image(bgr_to_rgb(vis), caption="No merged box (optional windows may still be drawn)", use_column_width=True)
    else:
        x1, y1, x2, y2 = clamp_bbox(bbox, W0, H0)
        label = "Vehicle"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{label} | score={best_score:.2f} | conf={best_conf:.1f}% | merged={used_count}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        st.image(bgr_to_rgb(vis), caption="Result (1 merged box)", use_column_width=True)

        # YOLO-style output (single line)
        yolo_line = bbox_to_yolo_line((x1, y1, x2, y2), img_w=W0, img_h=H0, class_id=0)
        st.subheader("Export: YOLO-style output (single bbox)")
        st.code(yolo_line, language="text")
        st.caption("Format: class_id x_center y_center width height (all normalized 0..1)")

        # Download exports
        st.subheader("Export Files")
        st.download_button(
            "⬇️ Download YOLO .txt",
            data=(yolo_line + "\n").encode("utf-8"),
            file_name="prediction.txt",
            mime="text/plain"
        )

        ok_img, buf_img = cv2.imencode(".png", vis)
        if ok_img:
            st.download_button(
                "⬇️ Download Result Image (.png)",
                data=buf_img.tobytes(),
                file_name="result.png",
                mime="image/png"
            )

    # ---------------------------------------------------------
    # Heatmap overlay
    # ---------------------------------------------------------
    if show_heatmap:
        st.subheader("Heatmap (score map overlay)")
        overlay, heat_u8 = build_heatmap_overlay(
            img_bgr=img_bgr,
            windows=windows,
            step=step,
            blur_ksize=int(heat_blur) if heat_blur > 0 else 0,
        )
        st.image(bgr_to_rgb(overlay), caption="Heatmap overlay (red=high score)", use_column_width=True)

        ok_heat, buf_heat = cv2.imencode(".png", overlay)
        if ok_heat:
            st.download_button(
                "⬇️ Download Heatmap Overlay (.png)",
                data=buf_heat.tobytes(),
                file_name="heatmap_overlay.png",
                mime="image/png"
            )

    # ---------------------------------------------------------
    # Global Canny edge visualization (already in your code)
    # ---------------------------------------------------------
    st.subheader("Canny edges (global)")
    edges = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), canny_t1, canny_t2)
    st.image(edges, caption="Canny Edge Map (global)", use_column_width=True)

    ok_edges, buf_edges = cv2.imencode(".png", edges)
    if ok_edges:
        st.download_button(
            "⬇️ Download Global Canny Edges (.png)",
            data=buf_edges.tobytes(),
            file_name="canny_edges.png",
            mime="image/png"
        )

    # ---------------------------------------------------------
    # Patch-level demonstration (HOG is computed on 128x64 patches)
    # ---------------------------------------------------------
    st.subheader("Patch-level Feature Extraction Demo (128×64 patch)")
    if len(windows) > 0:
        # pick best-scoring window for demo (even if bbox None)
        best_win = max(windows, key=lambda w: w["score"])
        bx1, by1, bx2, by2 = clamp_bbox(best_win["bbox"], W0, H0)
        patch = img_bgr[by1:by2, bx1:bx2].copy()

        feat_vec, patch_resized, patch_gray, patch_edges, hog_img = extract_canny_hog_debug(
            patch_bgr=patch,
            canny_t1=canny_t1,
            canny_t2=canny_t2,
            visualize_hog=True
        )

        c1, c2 = st.columns(2)
        with c1:
            st.image(bgr_to_rgb(patch), caption=f"Best window patch (bbox={bx1},{by1},{bx2},{by2})", use_column_width=True)
            st.image(bgr_to_rgb(patch_resized), caption="Patch resized to 128×64", use_column_width=True)
        with c2:
            st.image(patch_gray, caption="Patch grayscale", use_column_width=True)
            st.image(patch_edges, caption="Patch Canny edges (input to HOG)", use_column_width=True)

        st.write(f"HOG feature vector length: **{len(feat_vec)}**")
        st.write(f"Best-window score: **{best_win['score']:.3f}** | conf(sigmoid): **{best_win['conf']:.1f}%** | pred: **{best_win['pred']}**")

        if hog_img is not None:
            hog_u8 = norm_to_u8(hog_img)
            st.image(hog_u8, caption="HOG visualization (for illustration of gradient structure)", use_column_width=True)

    # Helpful hint
    st.info(
        "If everything becomes 'vehicle': increase POS_SCORE_THRESH (e.g., 2.5→4.0), "
        "increase y_start_ratio (ignore sky), or retrain with hard negatives."
    )


if __name__ == "__main__":
    run_app()
