# realtime_recognition.py
# Run:  py .\realtime_recognition.py
# Quit: click the OpenCV window then press Q or ESC. If stuck: Ctrl+C in terminal.

import cv2
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
import time
from collections import deque

from detect_and_crop import detect_and_crop

# ---------- PATHS / CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Traffic Sign Recognition/
MODEL_PATH = PROJECT_ROOT / "models" / "sign_model.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "labels.json"

IMG_SIZE = 32
CAMERA_INDEX = 0

CONF_THRESHOLD = 60.0  # percent
SHOW_FPS = True

# A) crop padding factor (bigger crop -> less cutting sign)
PAD_FACTOR = 0.15

# B) temporal smoothing window (averaging predictions)
SMOOTHING_FRAMES = 8
# ------------------------------------

# GTSRB class id -> human readable name (official 43 classes)
CLASS_NAME = {
    0: "Speed limit 20",
    1: "Speed limit 30",
    2: "Speed limit 50",
    3: "Speed limit 60",
    4: "Speed limit 70",
    5: "Speed limit 80",
    6: "End speed limit 80",
    7: "Speed limit 100",
    8: "Speed limit 120",
    9: "No passing",
    10: "No passing > 3.5t",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "STOP",
    15: "No vehicles",
    16: "Vehicles > 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed & passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing > 3.5t",
}


def safe_load_labels(labels_path: Path):
    """
    Supports BOTH possible label files:
      1) {"class_names": ["0","1",...]}  (folder loader)
      2) {"class_ids": [0,1,...]}       (older loader)
    Returns a list of class IDs in the order used by training.
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "class_names" in data:
        return [int(x) for x in data["class_names"]]
    if "class_ids" in data:
        return [int(x) for x in data["class_ids"]]

    raise KeyError(
        f"labels.json must contain 'class_names' or 'class_ids'. Found keys: {list(data.keys())}"
    )


def center_crop(frame: np.ndarray):
    """Fallback ROI: center crop. Returns (crop, bbox)."""
    h, w = frame.shape[:2]
    size = min(h, w)
    half = size // 4
    cx, cy = w // 2, h // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    crop = frame[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)


def apply_padding_and_reclip(frame, bbox, pad_factor=0.15):
    """A) Expand bbox by padding factor and keep inside frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad = int(pad_factor * max(bw, bh))

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = frame[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)


def preprocess_bgr_to_model_input(bgr_img: np.ndarray) -> np.ndarray:
    """
    C) Webcam-friendly preprocessing:
    - resize to 32x32
    - BGR->RGB
    - histogram equalization on luminance (improves contrast)
    - normalize to 0..1
    - add batch dim
    """
    img = cv2.resize(bgr_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Contrast normalization (Y channel equalize)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    y = cv2.equalizeHist(y)
    img = cv2.merge([y, cr, cb])
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)  # (1, 32, 32, 3)


def main():
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Make sure training finished and created models/sign_model.keras")
        return

    if not LABELS_PATH.exists():
        print(f"❌ Labels file not found: {LABELS_PATH}")
        print("Make sure training created models/labels.json")
        return

    print(f"✅ Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"✅ Loading labels: {LABELS_PATH}")
    class_ids_in_order = safe_load_labels(LABELS_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Camera not found / cannot open camera.")
        return

    print("✅ Camera started.")
    print("✅ Improvements enabled: A) padding crop, B) smoothing, C) contrast normalization")
    print("✅ Quit: click the OpenCV window then press Q or ESC.")

    # FPS
    prev_time = time.time()
    fps = 0.0

    # B) smoothing buffer
    prob_history = deque(maxlen=SMOOTHING_FRAMES)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect sign ROI
        result = detect_and_crop(frame)
        if result is None:
            crop, bbox = center_crop(frame)
            found = False
        else:
            crop, bbox = result
            found = True

        # A) Apply padding to bbox and recrop (helps border/edges)
        crop, bbox = apply_padding_and_reclip(frame, bbox, PAD_FACTOR)
        x1, y1, x2, y2 = bbox

        # Predict
        inp = preprocess_bgr_to_model_input(crop)
        probs = model.predict(inp, verbose=0)[0]

        # B) Smooth predictions over last N frames
        prob_history.append(probs)
        avg_probs = np.mean(prob_history, axis=0)

        pred_idx = int(np.argmax(avg_probs))
        conf = float(avg_probs[pred_idx]) * 100.0

        class_id = class_ids_in_order[pred_idx] if pred_idx < len(class_ids_in_order) else pred_idx
        label = CLASS_NAME.get(int(class_id), f"Class {class_id}")

        # UNKNOWN threshold
        label_to_show = label if conf >= CONF_THRESHOLD else "UNKNOWN"

        # FPS update
        if SHOW_FPS:
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Draw bbox + text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        status = "SIGN" if found else "FALLBACK"
        cv2.putText(
            frame,
            f"{label_to_show} ({conf:.1f}%) [{status}]",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Threshold: {CONF_THRESHOLD:.0f}% | Smooth: {SMOOTHING_FRAMES} frames",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if SHOW_FPS:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Traffic Sign Recognition", frame)

        # Keys work only if OpenCV window is focused (click it)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
