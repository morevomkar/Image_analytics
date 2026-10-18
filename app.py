"""
Big Data Image Analytics - Spoon / Knife Detector
Streamlit + OpenCV (DNN module, YOLOv4-tiny trained on COCO)

Run:  streamlit run app.py
"""
import os
import urllib.request

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MODEL_DIR = "model"
MODEL_FILES = {
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
}
TARGETS = ["spoon", "knife"]
COLORS = {"spoon": (0, 200, 0), "knife": (0, 0, 255)}  # BGR

st.set_page_config(page_title="Spoon & Knife Detector", page_icon="🥄", layout="wide")
st.write("OpenCV version:", cv2.__version__)
st.write("Has dnn:", hasattr(cv2, "dnn") and hasattr(cv2.dnn, "readNetFromDarknet"))

# ----------------------------------------------------------------------
# Model loading (downloaded once, cached)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading detection model (first run downloads ~25 MB)...")

def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fname, url in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)

    net = cv2.dnn.readNetFromDarknet(
        os.path.join(MODEL_DIR, "yolov4-tiny.cfg"),
        os.path.join(MODEL_DIR, "yolov4-tiny.weights"),
    )
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255.0, swapRB=True)

    with open(os.path.join(MODEL_DIR, "coco.names")) as f:
        classes = [c.strip() for c in f.read().strip().split("\n")]
    return model, classes


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------
def detect(image_bgr, model, classes, conf_thr, nms_thr):
    """Return annotated image and {'spoon': count, 'knife': count}."""
    class_ids, scores, boxes = model.detect(image_bgr, conf_thr, nms_thr)
    counts = {t: 0 for t in TARGETS}
    annotated = image_bgr.copy()

    if len(class_ids) > 0:
        for cid, score, box in zip(np.array(class_ids).flatten(),
                                   np.array(scores).flatten(), boxes):
            label = classes[int(cid)]
            if label not in TARGETS:
                continue
            counts[label] += 1
            x, y, w, h = [int(v) for v in box]
            color = COLORS[label]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, f"{label} {score:.2f}", (x, max(y - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated, counts


def read_upload(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🥄🔪 Image Analytics: Spoon & Knife Detection")
st.caption("Upload one or many images. The app flags each class as present (1) or absent (0).")

with st.sidebar:
    st.header("Settings")
    conf_thr = st.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05,
                         help="Lower it if small cutlery is being missed.")
    nms_thr = st.slider("NMS threshold", 0.1, 0.9, 0.40, 0.05)
    st.subheader("Expected pattern")
    exp_spoon = st.selectbox("spoon =", [1, 0], index=0)
    exp_knife = st.selectbox("knife =", [1, 0], index=1)

model, classes = load_model()

files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "bmp", "webp"],
                         accept_multiple_files=True)

if files:
    rows = []
    progress = st.progress(0.0, text="Analysing images...")

    for i, f in enumerate(files, start=1):
        img = read_upload(f)
        if img is None:
            st.warning(f"Could not read {f.name}")
            continue

        annotated, counts = detect(img, model, classes, conf_thr, nms_thr)
        spoon_flag = int(counts["spoon"] > 0)
        knife_flag = int(counts["knife"] > 0)
        match = (spoon_flag == exp_spoon) and (knife_flag == exp_knife)

        rows.append({
            "image": f.name,
            "spoon": spoon_flag,
            "knife": knife_flag,
            "spoon_count": counts["spoon"],
            "knife_count": counts["knife"],
            "matches_expected": match,
        })

        with st.expander(f"{f.name}  →  spoon = {spoon_flag}, knife = {knife_flag}  "
                         f"{'✅' if match else '❌'}", expanded=len(files) == 1):
            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original",
                     use_container_width=True)
            c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detections",
                     use_container_width=True)
            m1, m2 = st.columns(2)
            m1.metric("spoon", spoon_flag)
            m2.metric("knife", knife_flag)

        progress.progress(i / len(files), text=f"Analysing images... {i}/{len(files)}")

    progress.empty()

    # ------------------------------------------------------------------
    # Batch analytics
    # ------------------------------------------------------------------
    if rows:
        df = pd.DataFrame(rows)
        st.subheader("Batch results")
        st.dataframe(df, use_container_width=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Images processed", len(df))
        k2.metric("With spoon", int(df["spoon"].sum()))
        k3.metric("With knife", int(df["knife"].sum()))
        k4.metric(f"Match (spoon={exp_spoon}, knife={exp_knife})",
                  int(df["matches_expected"].sum()))

        st.bar_chart(df[["spoon", "knife"]].sum())

        st.download_button("Download results (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name="spoon_knife_results.csv",
                           mime="text/csv")
else:
    st.info("Upload images to start.")
