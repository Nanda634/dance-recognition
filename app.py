import os
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.dataset import CLASSES, discover_classes, extract_dataset
from src.model import train_from_dataframe
from src.inference import predict_image, predict_video
from learning_guide import (DANCE_FORMS, extract as guide_extract, compare as guide_compare,
                            save_reference, list_references, load_reference, analyze_video)

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "dance_model.joblib"
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="Nritya AI | Dance Recognition", page_icon="💃", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root{--ink:#2b201b;--muted:#78675d;--cream:#fbf7f1;--copper:#a95f3d;--line:#eadfd4}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:linear-gradient(180deg,#fbf7f1,#f7f0e8);color:var(--ink)}
.block-container{max-width:1380px;padding-top:1.35rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:#251b17;border-right:1px solid #4a352a}
[data-testid="stSidebar"] *{color:#f8eee5!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:.55rem .65rem;border-radius:10px}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#49362b}
.hero{padding:2rem 2.2rem;border-radius:23px;color:#fff8f0;background:radial-gradient(circle at 90% 10%,rgba(213,164,91,.3),transparent 27%),linear-gradient(115deg,#251b17,#67432f 68%,#94583a);box-shadow:0 15px 38px #39231820;margin-bottom:1.6rem}
.eyebrow{text-transform:uppercase;letter-spacing:.18em;color:#e9c99d;font-size:.72rem;font-weight:700}
.hero h1{font-family:'Playfair Display',serif;font-size:clamp(2rem,4vw,3.1rem);line-height:1.1;color:#fff8f0;margin:.45rem 0 .6rem}
.hero p{color:#f2dfcf;margin:0;max-width:760px}
.kicker{color:#a45f3d;text-transform:uppercase;letter-spacing:.16em;font-weight:700;font-size:.72rem;margin-bottom:.3rem}
.metric{background:#fffdfa;border:1px solid #eadfd4;border-radius:15px;padding:1rem 1.1rem;min-height:90px;box-shadow:0 5px 18px #432b1c08}
.metric small{color:#806d61;font-weight:600}.metric strong{display:block;color:#2d211c;font-size:1.6rem;margin-top:.15rem}
.panel{background:#fffdfa;border:1px solid #eadfd4;border-radius:17px;padding:1.1rem 1.2rem;box-shadow:0 5px 18px #432b1c08}
div.stButton>button[kind="primary"]{background:linear-gradient(110deg,#985333,#bd784b);border:0;color:white;border-radius:10px;font-weight:700;padding:.6rem 1rem}
div.stButton>button{border-radius:10px}
[data-testid="stFileUploader"]{background:#fffdfa;border:1px dashed #c9a88e;border-radius:13px;padding:.5rem}
hr{border-color:#eadfd4} footer,#MainMenu{visibility:hidden}
</style>
""", unsafe_allow_html=True)


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def title(kicker, heading, subtitle):
    st.markdown(
        f'<div class="kicker">{kicker}</div>'
        f'<h2 style="font-family:Playfair Display,serif;margin-top:0">{heading}</h2>'
        f'<p style="color:#78675d;margin-top:-.4rem">{subtitle}</p>',
        unsafe_allow_html=True)


# Fresh class discovery on every app run
DISCOVERED_CLASSES = discover_classes(RAW) or CLASSES

# Session state defaults
for key, default in {"guide_dance": DANCE_FORMS[0], "guide_ref_select": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- Sidebar (defined FIRST — all pages can use these) ----------------
with st.sidebar:
    st.markdown("## 💃 NRITYA AI")
    st.caption("INDIAN CLASSICAL DANCE INTELLIGENCE")
    st.divider()
    page = st.radio("WORKSPACE", ["Overview", "Dataset Studio", "Model Lab",
                                  "Recognition Studio", "Dance Learning Guide"])
    st.divider()

    # Settings used by Recognition Studio (must exist before the video loop runs)
    confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.35, 0.05)
    FRAME_SKIP = st.slider("Analyze every Nth frame", 1, 30, 5,
                           help="Higher = faster, but may miss fast movements")

    st.divider()
    st.markdown("**MODEL STATUS**")
    st.markdown(
        f'<div style="font-weight:600;padding:.35rem 0">'
        f'{"Trained model available" if MODEL_PATH.exists() else "Awaiting training"}</div>',
        unsafe_allow_html=True)
    st.caption("Local academic prototype")

st.markdown(
    """<div class="hero"><div class="eyebrow">Artificial Intelligence · Computer Vision · Heritage</div>
    <h1>Indian Classical Dance<br>Recognition System</h1>
    <p>Analyze body posture with human pose estimation and machine learning — from labeled dataset to image, video and webcam analysis.</p></div>""",
    unsafe_allow_html=True)

# ============================ OVERVIEW ============================
if page == "Overview":
    title("Workspace / Overview", "Your dance intelligence workspace",
          "A guided workflow to prepare data, train a model and evaluate dance postures.")
    metrics = [("Dance categories", f"{len(DISCOVERED_CLASSES):02d}"),
               ("Pose landmarks", "33"),
               ("Model status", "Ready" if MODEL_PATH.exists() else "Pending")]
    cols = st.columns(3)
    for col, (label, value) in zip(cols, metrics):
        col.markdown(f'<div class="metric"><small>{label}</small><strong>{value}</strong></div>',
                     unsafe_allow_html=True)
    st.write("")
    with st.container():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("How the system works")
        st.markdown(
            "**01 · Input** — image, video or webcam snapshot\n\n↓\n\n"
            "**02 · Pose estimation** — MediaPipe detects 33 body landmarks\n\n↓\n\n"
            "**03 · Feature extraction** — normalized body coordinates\n\n↓\n\n"
            "**04 · Classification** — Random Forest predicts a dance category\n\n↓\n\n"
            "**05 · Output** — annotated skeleton, predicted class and score")
        st.markdown('</div>', unsafe_allow_html=True)
    st.info("Start in Dataset Studio. Add labeled images, train and evaluate the model, "
            "then test it in Recognition Studio.")

# ============================ DATASET STUDIO ============================
elif page == "Dataset Studio":
    title("Workspace / Data", "Dataset studio",
          "Upload correctly labeled dance images. Prefer one dancer and a clearly visible full-body pose.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    category = st.selectbox("Select dance category", DISCOVERED_CLASSES)
    uploads = st.file_uploader("Upload JPG, PNG or WEBP images",
                               type=["jpg", "jpeg", "png", "webp"],
                               accept_multiple_files=True)
    if st.button("Save images to dataset", type="primary", disabled=not uploads):
        folder = RAW / category
        folder.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(uploads):
            name = Path(f.name).name.replace(" ", "_")
            dest = folder / name
            if dest.exists():
                dest = folder / f"{dest.stem}_{i}{dest.suffix}"
            dest.write_bytes(f.getbuffer())
        st.success(f"Images saved successfully to data/raw/{category}/")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    counts = {c: len(list((RAW / c).glob("*"))) for c in DISCOVERED_CLASSES if (RAW / c).exists()}
    if counts:
        st.subheader("Images per category")
        st.bar_chart(pd.DataFrame.from_dict(counts, orient="index", columns=["Images"]))
        smallest = min(counts, key=counts.get)
        if max(counts.values()) >= 2 * max(min(counts.values()), 1):
            st.warning(f"Class imbalance detected — '{smallest}' has the fewest images "
                       f"({counts[smallest]}). Consider adding more images for it before training.")
    st.caption("Only use media you own or are permitted to use. "
               "Avoid near-duplicate images across training and testing.")

# ============================ MODEL LAB ============================
elif page == "Model Lab":
    title("Workspace / Training", "Model lab",
          "Extract MediaPipe landmarks, train the classifier and inspect held-out test performance.")

    counts = [len(list((RAW / c).glob("*"))) for c in DISCOVERED_CLASSES if (RAW / c).exists()]
    if len(counts) < 2 or any(n == 0 for n in counts):
        st.warning("At least 2 categories with images are required before training. "
                   "Add images under data/raw/ first.")
    elif st.button("Extract landmarks & train model", type="primary"):
        progress = st.progress(0, text="Scanning labeled images...")
        try:
            df, skipped = extract_dataset(
                RAW, progress=lambda x: progress.progress(x, text=f"Extracting pose features · {x:.0%}"))
            if df.empty:
                st.error("No valid pose features found. Add full-body images to at least two dance categories.")
            else:
                (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
                df.to_csv(ROOT / "data" / "processed" / "pose_features.csv", index=False)
                result = train_from_dataframe(df, ROOT / "models", OUTPUTS)
                st.success("Model training completed.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Held-out accuracy", f"{result['accuracy']:.1%}")
                m2.metric("Usable samples", len(df))
                m3.metric("Skipped images", len(skipped))
                st.subheader("Classification report")
                st.dataframe(pd.DataFrame(result["report"]).T.round(3), use_container_width=True)
                st.subheader("Confusion matrix")
                st.dataframe(pd.DataFrame(result["confusion_matrix"],
                                          index=result["classes"], columns=result["classes"]),
                             use_container_width=True)
                if skipped:
                    with st.expander(f"Skipped images ({len(skipped)})"):
                        st.dataframe(pd.DataFrame(skipped, columns=["File", "Reason"]),
                                     use_container_width=True)
        except Exception as e:
            st.error(f"Training failed: {e}")
    if MODEL_PATH.exists():
        st.success("A trained model is ready.")
    else:
        st.warning("No trained model yet. Add data and train before recognition.")

# ============================ RECOGNITION STUDIO ============================
elif page == "Recognition Studio":
    title("Workspace / Recognition", "Recognition studio",
          "Test a still image, a video or a webcam snapshot.")
    if not MODEL_PATH.exists():
        st.warning("Train a model in Model Lab first.")
        st.stop()
    model = joblib.load(MODEL_PATH)["model"]
    mode = st.radio("Input type", ["Image", "Video", "Webcam snapshot"], horizontal=True)

    # ---------- Image ----------
    if mode == "Image":
        upload = st.file_uploader("Upload a dance image",
                                  type=["jpg", "jpeg", "png", "webp"], key="image_input")
        if upload:
            pil = Image.open(upload).convert("RGB")
            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            with st.spinner("Detecting pose and classifying..."):
                label, score, annotated, probs = predict_image(frame, model, draw=True)
            x, y = st.columns(2)
            x.image(pil, caption="Original image", use_container_width=True)
            y.image(rgb(annotated), caption="Pose landmarks", use_container_width=True)
            if label:
                st.success(f"Predicted dance form: {label}")
                st.metric("Classifier score · not calibrated confidence", f"{score:.1%}")
            else:
                st.error("No pose detected. Use a well-lit image.")

    # ---------- Video ----------
    elif mode == "Video":
        upload = st.file_uploader("Upload MP4, MOV or AVI",
                                  type=["mp4", "mov", "avi", "mkv"], key="video_input")
        if upload:
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=Path(upload.name).suffix or ".mp4") as tmp:
                tmp.write(upload.getbuffer())
                video_path = tmp.name

            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Could not open the video file. Unsupported codec or corrupted file.")
                else:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    st.info(f"Video: {width}x{height} @ {fps:.1f} fps — "
                            f"{total} frames (~{total / fps:.1f}s)")

                    frame_idx = 0
                    results_log = []
                    progress = st.progress(0.0, text="Processing frames...")
                    frame_slot = st.empty()   # reused slot → no memory blowup

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Run pose classification every N frames to save compute
                        if frame_idx % FRAME_SKIP == 0:
                            label, score, annotated, _ = predict_image(frame, model, draw=True)
                            if label:
                                results_log.append({
                                    "frame": frame_idx,
                                    "time_s": round(frame_idx / fps, 2),
                                    "dance_form": label,
                                    "score": round(score, 3),
                                })
                        else:
                            annotated = frame

                        frame_slot.image(rgb(annotated), use_container_width=True)

                        frame_idx += 1
                        progress.progress(min(frame_idx / max(total, 1), 1.0),
                                          text=f"Frame {frame_idx}/{total}")

                    cap.release()
                    progress.progress(1.0, text="Done ✅")

                    if results_log:
                        st.subheader("Dance predictions over time")
                        st.dataframe(pd.DataFrame(results_log), use_container_width=True)
                        csv = pd.DataFrame(results_log).to_csv(index=False).encode()
                        st.download_button("Download predictions CSV", csv,
                                           "dance_predictions.csv", "text/csv")
                    else:
                        st.warning("No pose detected in any analyzed frame. "
                                   "Check lighting / framing, or lower the frame skip.")
            except Exception as e:
                st.exception(f"Error while processing video: {e}")
            finally:
                if os.path.exists(video_path):
                    os.unlink(video_path)  # clean up temp file

    # ---------- Webcam snapshot ----------
    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("Capture a snapshot from your webcam and classify the pose instantly.")
        img = st.camera_input("Take a snapshot")
        if img:
            pil = Image.open(img).convert("RGB")
            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            with st.spinner("Detecting pose and classifying..."):
                label, score, annotated, _ = predict_image(frame, model, draw=True)
            x, y = st.columns(2)
            x.image(pil, caption="Snapshot", use_container_width=True)
            y.image(rgb(annotated), caption="Pose landmarks", use_container_width=True)
            if label:
                st.success(f"Predicted dance form: {label}")
                st.metric("Classifier score", f"{score:.1%}")
            else:
                st.error("No pose detected. Ensure your full body is visible and well lit.")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================ LEARNING GUIDE ============================
else:
    title("Workspace / Learning", "Dance learning guide",
          "Compare your practice against a reference pose and get visual feedback.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.session_state.guide_dance = st.selectbox("Dance form", DANCE_FORMS,
                                                index=DANCE_FORMS.index(st.session_state.guide_dance)
                                                if st.session_state.guide_dance in DANCE_FORMS else 0)
    st.markdown("**Step 1 · Save a reference pose**")
    ref_img = st.camera_input("Capture reference pose", key="ref_capture")
    if ref_img and st.button("Save reference", type="primary"):
        pil = Image.open(ref_img).convert("RGB")
        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        ok, msg = save_reference(frame, st.session_state.guide_dance)
        (st.success if ok else st.error)(msg)
        st.rerun()

    st.markdown("**Step 2 · Compare your practice**")
    refs = list_references(st.session_state.guide_dance)
    if refs:
        st.session_state.guide_ref_select = st.selectbox("Reference pose", refs,
                                                         index=0 if st.session_state.guide_ref_select is None else None)
        practice_img = st.camera_input("Capture your practice pose", key="practice_capture")
        if practice_img and st.button("Compare poses"):
            practice = cv2.cvtColor(np.array(Image.open(practice_img).convert("RGB")), cv2.COLOR_RGB2BGR)
            reference = load_reference(st.session_state.guide_dance, st.session_state.guide_ref_select)
            result = guide_compare(reference, practice)
            if result.get("match") is not None:
                st.metric("Pose similarity", f"{result['match']:.1%}")
            if result.get("annotated") is not None:
                st.image(rgb(result["annotated"]), caption="Practice overlay", use_container_width=True)
            for tip in result.get("feedback", []):
                st.info(tip)
    else:
        st.info("No reference saved for this dance form yet. Capture one above first.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.divider()
st.caption("Built with Streamlit, OpenCV & MediaPipe — Ox Alpha demo app.")



