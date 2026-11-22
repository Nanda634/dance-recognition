from pathlib import Path
import tempfile
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from src.dataset import CLASSES, extract_dataset
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
:root{--ink:#2b201b;--muted:#78675d;--cream:#fbf7f1;--paper:#fffdfa;--copper:#a95f3d;--line:#eadfd4}
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
    st.markdown(f'<div class="kicker">{kicker}</div><h2 style="font-family:Playfair Display,serif;margin-top:0">{heading}</h2><p style="color:#78675d;margin-top:-.4rem">{subtitle}</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 💃 NRITYA AI")
    st.caption("INDIAN CLASSICAL DANCE INTELLIGENCE")
    st.divider()
    page = st.radio("WORKSPACE", ["Overview","Dataset Studio","Model Lab","Recognition Studio","Dance Learning Guide"])
    st.divider()
    st.markdown("**MODEL STATUS**")
    st.markdown(f'<div style="color:#000000;font-weight:600;padding:.35rem 0">{"Trained model available" if MODEL_PATH.exists() else "Awaiting training"}</div>', unsafe_allow_html=True)
    st.caption("Local academic prototype")

st.markdown("""<div class="hero"><div class="eyebrow">Artificial Intelligence · Computer Vision · Heritage</div><h1>Indian Classical Dance<br>Recognition System</h1><p>Analyze body posture with human pose estimation and machine learning — from labeled dataset to image, video and webcam analysis.</p></div>""", unsafe_allow_html=True)

if page == "Overview":
    title("Workspace / Overview","Your dance intelligence workspace","A guided workflow to prepare data, train a model and evaluate dance postures.")
    metrics = [("Dance categories","06"),("Pose landmarks","33"),("Model status","Ready" if MODEL_PATH.exists() else "Pending")]
    cols = st.columns(3)
    for col,(label,value) in zip(cols,metrics):
        col.markdown(f'<div class="metric"><small>{label}</small><strong>{value}</strong></div>',unsafe_allow_html=True)
    st.write("")
    with st.container():
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.subheader("How the system works")
        st.markdown("**01 · Input** — image, video or webcam snapshot\n\n↓\n\n**02 · Pose estimation** — MediaPipe detects 33 body landmarks\n\n↓\n\n**03 · Feature extraction** — normalized body coordinates\n\n↓\n\n**04 · Classification** — Random Forest predicts a dance category\n\n↓\n\n**05 · Output** — annotated skeleton, predicted class and score")
        st.markdown('</div>',unsafe_allow_html=True)
    st.info("Start in Dataset Studio. Add labeled images, train and evaluate the model, then test it in Recognition Studio.")
elif page == "Dataset Studio":
    title("Workspace / Data","Dataset studio","Upload correctly labeled dance images. Prefer one dancer and a clearly visible full-body pose.")
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    category = st.selectbox("Select dance category",CLASSES)
    uploads = st.file_uploader("Upload JPG, PNG or WEBP images",type=["jpg","jpeg","png","webp"],accept_multiple_files=True)
    if st.button("Save images to dataset",type="primary",disabled=not uploads):
        folder = RAW/category
        folder.mkdir(parents=True,exist_ok=True)
        saved=0
        for i,f in enumerate(uploads):
            name=Path(f.name).name.replace(" ","_")
            dest=folder/name
            if dest.exists(): dest=folder/f"{dest.stem}_{i}{dest.suffix}"
            dest.write_bytes(f.getbuffer()); saved+=1
        st.success(f"Images saved successfully to data/raw/{category}/")
        st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)
    st.caption("Only use media you own or are permitted to use. Avoid near-duplicate images across training and testing.")

elif page == "Model Lab":
    title("Workspace / Training","Model lab","Extract MediaPipe landmarks, train the classifier and inspect held-out test performance.")
    if st.button("Extract landmarks & train model",type="primary"):
        progress=st.progress(0,text="Scanning labeled images...")
        try:
            df,skipped=extract_dataset(RAW,progress=lambda x:progress.progress(x,text=f"Extracting pose features · {x:.0%}"))
            if df.empty:
                st.error("No usable poses found. Add full-body images to at least two dance categories.")
            else:
                (ROOT/"data"/"processed").mkdir(parents=True,exist_ok=True)
                df.to_csv(ROOT/"data"/"processed"/"pose_features.csv",index=False)
                result=train_from_dataframe(df,ROOT/"models",OUTPUTS)
                st.success("Model training completed.")
                m1,m2,m3=st.columns(3)
                m1.metric("Held-out accuracy",f"{result['accuracy']:.1%}")
                m2.metric("Usable samples",len(df))
                m3.metric("Skipped images",len(skipped))
                st.subheader("Classification report")
                st.dataframe(pd.DataFrame(result["report"]).T.round(3),use_container_width=True)
                st.subheader("Confusion matrix")
                st.dataframe(pd.DataFrame(result["confusion_matrix"],index=result["classes"],columns=result["classes"]),use_container_width=True)
                if skipped:
                    with st.expander(f"Skipped images ({len(skipped)})"):
                        st.dataframe(pd.DataFrame(skipped,columns=["File","Reason"]),use_container_width=True)
        except Exception as e:
            st.error(f"Training failed: {e}")
    if MODEL_PATH.exists(): st.success("A trained model is ready.")
    else: st.warning("No trained model yet. Add data and train before recognition.")

elif page == "Recognition Studio":
    title("Workspace / Recognition","Recognition studio","Test a still image, a video or a webcam snapshot.")
    if not MODEL_PATH.exists():
        st.warning("Train a model in Model Lab first."); st.stop()
    model=joblib.load(MODEL_PATH)["model"]
    mode=st.radio("Input type",["Image","Video","Webcam snapshot"],horizontal=True)
    if mode=="Image":
        upload=st.file_uploader("Upload a dance image",type=["jpg","jpeg","png","webp"],key="image_input")
        if upload:
            pil=Image.open(upload).convert("RGB")
            frame=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
            with st.spinner("Detecting pose and classifying..."):
                label,score,annotated=predict_image(frame,model,draw=True)
            x,y=st.columns(2)
            x.image(pil,caption="Original image",use_container_width=True)
            y.image(rgb(annotated),caption="Pose landmarks",use_container_width=True)
            if label:
                st.success(f"Predicted dance form: {label}")
                st.metric("Classifier score · not calibrated confidence",f"{score:.1%}")
            else: st.error("No pose detected. Use a well-lit, full-body image.")
    elif mode=="Video":
        upload=st.file_uploader("Upload MP4, MOV or AVI",type=["mp4","mov","avi","mkv"],key="video_input")
        if upload:
            with tempfile.NamedTemporaryFile(delete=False,suffix=Path(upload.name).suffix or ".mp4") as tmp:
                tmp.write(upload.getbuffer()); video_path=tmp.name
            try:
                with st.spinner("Sampling video frames..."):
                    winner,preds,preview=predict_video(video_path,model,max_samples=60)
                if winner:
                    st.success(f"Video result (majority vote): {winner}")
                    st.metric("Frames with detected poses",len(preds))
                    st.bar_chart(pd.Series([p[0] for p in preds]).value_counts())
                    if preview is not None: st.image(rgb(preview),caption="Example annotated frame",use_container_width=True)
                else: st.error("No human poses found in sampled frames.")
            except Exception as e: st.error(f"Video processing failed: {e}")
            finally: Path(video_path).unlink(missing_ok=True)
    else:
        st.info("Allow camera permission. Keep the dancer's full body in view.")
        snap=st.camera_input("Capture webcam frame")
        if snap:
            pil=Image.open(snap).convert("RGB")
            frame=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
            label,score,annotated=predict_image(frame,model,draw=True)
            st.image(rgb(annotated),caption="Detected pose",use_container_width=True)
            if label:
                st.success(f"Predicted dance form: {label}")
                st.metric("Classifier score · not calibrated confidence",f"{score:.1%}")
            else: st.error("No pose detected. Try a full-body image.")
elif page == "Dance Learning Guide":
    title("Workspace / Learning", "Dance learning guide", "Compare a learner with a saved reference posture. Feedback measures joint-angle differences, not authoritative dance technique.")
    st.warning("Reference poses must be checked by a qualified instructor before treating feedback as correct technique. This prototype does not assess mudras, facial expression, rhythm, or choreography.")
    ref_root=ROOT / "data" / "references"
    dance=st.selectbox("Choose dance form",DANCE_FORMS,key="guide_dance")
    tab1,tab2=st.tabs(["Reference library", "Practice & feedback"])
    with tab1:
        st.subheader("Add a reference posture")
        st.caption("Use a clear front-facing full-body photo of a correct pose. Name the specific step/posture; each dance form can have multiple reference poses.")
        pose_name=st.text_input("Posture name (e.g. Araimandi)",key="guide_pose_name")
        notes=st.text_area("Instructor notes / instructions (optional)",key="guide_notes")
        ref_upload=st.file_uploader("Upload reference image",type=["jpg","jpeg","png","webp"],key="guide_reference_upload")
        if ref_upload is not None:
            ref_pil=Image.open(ref_upload).convert("RGB")
            st.image(ref_pil,caption="Proposed reference",width=340)
        if st.button("Save reference pose",type="primary",disabled=not (pose_name.strip() and ref_upload is not None)):
            try:
                ref_frame=cv2.cvtColor(np.asarray(ref_pil),cv2.COLOR_RGB2BGR)
                saved,annotated=save_reference(ref_root,dance,pose_name,ref_frame,notes)
                st.success(f"Reference saved: {saved.name}")
                st.image(rgb(annotated),caption="Reference skeleton",width=340)
            except Exception as e: st.error(str(e))
        existing=list_references(ref_root,dance)
        st.caption(f"Saved reference postures for {dance}: {len(existing)}")
        if existing: st.write(", ".join(p.stem.replace("_"," ") for p in existing))
    with tab2:
        references=list_references(ref_root,dance)
        if not references:
            st.info("First save at least one reference posture for this dance form in the Reference library tab.")
        else:
            selected=st.selectbox("Correct reference posture",references,format_func=lambda p:p.stem.replace("_"," "),key="guide_ref_select")
            reference=load_reference(selected)
            st.write("**Reference instructions:**",reference.get("notes") or "No instructor notes supplied")
            ref_image=selected.with_suffix(".jpg")
            if ref_image.exists(): st.image(str(ref_image),caption="Target reference posture",width=280)
            tolerance=st.slider("Allowed joint-angle difference (degrees)",5,45,15,5)
            mode=st.radio("Practice input",["Image","Video","Webcam snapshot"],horizontal=True,key="guide_mode")
            uploaded=None
            if mode=="Image":
                uploaded=st.file_uploader("Upload learner photo",type=["jpg","jpeg","png","webp"],key="guide_student_image")
            elif mode=="Webcam snapshot":
                st.info("This mode takes a snapshot. Continuous live coaching requires a streaming webcam implementation.")
                uploaded=st.camera_input("Capture learner posture",key="guide_student_camera")
            else:
                uploaded=st.file_uploader("Upload practice video",type=["mp4","mov","avi","mkv"],key="guide_student_video")
            if uploaded is not None and st.button("Analyze posture",type="primary",key="guide_analyze"):
                try:
                    if mode=="Video":
                        with tempfile.NamedTemporaryFile(delete=False,suffix=Path(uploaded.name).suffix or ".mp4") as tmp:
                            tmp.write(uploaded.getbuffer()); video_path=tmp.name
                        try:
                            with st.spinner("Comparing sampled frames with the selected static reference..."):
                                samples,preview=analyze_video(video_path,reference["landmarks"],tolerance)
                        finally: Path(video_path).unlink(missing_ok=True)
                        if not samples: st.error("No suitable body poses detected in sampled frames.")
                        else:
                            st.metric("Average joint-angle match (sampled frames)",f"{np.mean([s['Match %'] for s in samples]):.1f}%")
                            st.line_chart(pd.DataFrame([{'Frame':s['Frame'],'Match %':s['Match %']} for s in samples]).set_index("Frame"))
                            if preview is not None: st.image(rgb(preview),caption="Example analyzed frame",width=420)
                            worst=min(samples,key=lambda x:x["Match %"])
                            st.write(f"**Feedback for sampled frame {worst['Frame']}**")
                            st.dataframe(pd.DataFrame(worst["Result"]["rows"]),hide_index=True,use_container_width=True)
                            st.caption("Video comparison assumes the selected reference pose is held; it does not evaluate sequences or timing.")
                    else:
                        pil=Image.open(uploaded).convert("RGB")
                        frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
                        student,annotated=guide_extract(frame)
                        result=guide_compare(student,reference["landmarks"],tolerance)
                        c1,c2=st.columns(2)
                        c1.image(pil,caption="Learner",use_container_width=True)
                        c2.image(rgb(annotated),caption="Detected body landmarks",use_container_width=True)
                        if result["match_percent"] is None: st.warning("Not enough visible joints to compare reliably.")
                        else: st.metric("Joint-angle match to selected reference",f"{result['match_percent']}%",help="Percentage of measurable joint angles within the chosen tolerance; not a dance proficiency score.")
                        st.dataframe(pd.DataFrame(result["rows"]),hide_index=True,use_container_width=True)
                        for row in result["rows"]:
                            if row["Difference"] is not None and row["Difference"]>tolerance:
                                st.write(f"• **{row['Joint']}**: {row['Feedback'].lower()} (difference {row['Difference']}°)")
                        st.caption("Left/right labels refer to the dancer, not the viewer. Match camera viewpoint and avoid mirrored images for best comparison.")
                except Exception as e: st.error(f"Could not analyze input: {e}")

st.divider()
st.markdown('<p style="text-align:center;color:#8b796e;font-size:.8rem">NRITYA AI · Academic prototype · Recognition quality depends on dataset quality</p>',unsafe_allow_html=True)