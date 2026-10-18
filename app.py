"""
Big Data Image Analytics - Spoon / Knife Detector
Result code:  spoon = 0,  knife = 1

Stack: Streamlit + OpenCV (DNN, YOLOv4-tiny pretrained on COCO) + optional PySpark summary
Run:   streamlit run app.py
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

# Result code for each class  (as requested: spoon = 0, knife = 1)
CODE = {"spoon": 0, "knife": 1}
NO_DETECTION = -1
COLORS = {"spoon": (0, 200, 0), "knife": (0, 0, 255)}  # BGR

st.set_page_config(page_title="Spoon & Knife Detector", page_icon="🥄", layout="wide")


# ----------------------------------------------------------------------
# Environment check (OpenCV 5 removed the Darknet loader)
# ----------------------------------------------------------------------
if not hasattr(cv2.dnn, "readNetFromDarknet"):
    st.error(
        f"OpenCV {cv2.__version__} has no Darknet loader. "
        "Use OpenCV 4: put `opencv-python-headless>=4.8,<5` in requirements.txt "
        "(on Colab: `pip install 'opencv-python-headless<5'` and restart the runtime)."
    )
    st.stop()


# ----------------------------------------------------------------------
# Model
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


def detect(image_bgr, model, classes, conf_thr, nms_thr):
    """Return annotated image and a list of (label, score) for spoon/knife only."""
    class_ids, scores, boxes = model.detect(image_bgr, conf_thr, nms_thr)
    found = []
    annotated = image_bgr.copy()

    if len(class_ids) > 0:
        for cid, score, box in zip(np.array(class_ids).flatten(),
                                   np.array(scores).flatten(), boxes):
            label = classes[int(cid)]
            if label not in CODE:
                continue
            found.append((label, float(score)))
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), COLORS[label], 2)
            cv2.putText(annotated, f"{label}={CODE[label]} ({score:.2f})",
                        (x, max(y - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        COLORS[label], 2)
    return annotated, found


def decide(found):
    """Pick the single most confident detection -> (label, code, score)."""
    if not found:
        return "none", NO_DETECTION, 0.0
    label, score = max(found, key=lambda t: t[1])
    return label, CODE[label], score


def read_upload(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ----------------------------------------------------------------------
# Optional PySpark summary (Big Data part)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting Spark...")
def get_spark():
    from pyspark.sql import SparkSession
    return (SparkSession.builder.appName("SpoonKnifeImageAnalytics")
            .master("local[*]").getOrCreate())


def summarize(df, use_spark):
    if use_spark:
        try:
            spark = get_spark()
            sdf = spark.createDataFrame(df[["image", "label", "result"]])
            out = sdf.groupBy("label", "result").count().orderBy("label").toPandas()
            return out, "PySpark"
        except Exception as e:  # Java / pyspark missing
            st.warning(f"Spark unavailable ({type(e).__name__}); using pandas instead.")
    out = (df.groupby(["label", "result"]).size()
           .reset_index(name="count").sort_values("label"))
    return out, "pandas"


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🥄🔪 Image Analytics: Spoon & Knife Detection")
st.caption("Result code:  **spoon = 0**,  **knife = 1**")

with st.sidebar:
    st.header("Settings")
    conf_thr = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                         help="Lower it if the object is not being detected.")
    nms_thr = st.slider("NMS threshold", 0.1, 0.9, 0.40, 0.05)
    use_spark = st.checkbox("Use PySpark for batch summary", value=False,
                            help="Needs Java installed. Falls back to pandas if missing.")

model, classes = load_model()

files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "bmp", "webp"],
                         accept_multiple_files=True)

if not files:
    st.info("Upload one or more images to start.")
    st.stop()

rows = []
progress = st.progress(0.0, text="Analysing images...")

for i, f in enumerate(files, start=1):
    img = read_upload(f)
    if img is None:
        st.warning(f"Could not read {f.name}")
        continue

    annotated, found = detect(img, model, classes, conf_thr, nms_thr)
    label, code, score = decide(found)
    rows.append({"image": f.name, "label": label, "result": code,
                 "confidence": round(score, 3)})

    with st.expander(f"{f.name}  →  result = {code}  ({label})", expanded=len(files) == 1):
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original",
                 use_container_width=True)
        c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detections",
                 use_container_width=True)

        if code == 0:
            st.success(f"Spoon detected → **Result = 0**  (confidence {score:.2f})")
        elif code == 1:
            st.error(f"Knife detected → **Result = 1**  (confidence {score:.2f})")
        else:
            st.info("No spoon or knife detected. Try lowering the confidence threshold.")

    progress.progress(i / len(files), text=f"Analysing images... {i}/{len(files)}")

progress.empty()

# ----------------------------------------------------------------------
# Batch analytics
# ----------------------------------------------------------------------
if rows:
    df = pd.DataFrame(rows)
    st.subheader("Batch results")
    st.dataframe(df, use_container_width=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Images processed", len(df))
    k2.metric("Spoon (0)", int((df["result"] == 0).sum()))
    k3.metric("Knife (1)", int((df["result"] == 1).sum()))
    k4.metric("Not detected", int((df["result"] == NO_DETECTION).sum()))

    summary, engine = summarize(df, use_spark)
    st.caption(f"Summary computed with {engine}")
    st.dataframe(summary, use_container_width=True)
    st.bar_chart(summary.set_index("label")["count"])

    st.download_button("Download results (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name="spoon_knife_results.csv", mime="text/csv")
