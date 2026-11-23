from pathlib import Path
import cv2
import joblib
import numpy as np
from src.pose import get_pose

def load_model(path):
    bundle = joblib.load(path)
    return bundle["model"]

def predict_image(image_bgr, model, draw=True):
    """Returns (label, score, annotated, probs). probs is None if no pose found."""
    features, annotated = get_pose(image_bgr, draw=draw, complexity=0)
    if features is None:
        return None, None, annotated, None
    probs = model.predict_proba([features])[0]
    idx = int(np.argmax(probs))
    return str(model.classes_[idx]), float(probs[idx]), annotated, probs

def predict_video_frame(image_bgr, model, draw=True):
    """Single-frame prediction for video sampling with a progress bar.
    Same signature/returns as predict_image."""
    return predict_image(image_bgr, model, draw=draw)

def predict_video(video_path, model, max_samples=80):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open this video file.")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // max_samples) if frame_count > 0 else 10
    predictions, preview = [], None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            label, score, annotated, _ = predict_video_frame(frame, model, draw=True)
            if label:
                predictions.append((label, score))
                preview = annotated
        idx += 1
    cap.release()
    if not predictions:
        return None, [], preview
    labels, counts = np.unique([p[0] for p in predictions], return_counts=True)
    winner = str(labels[int(np.argmax(counts))])
    return winner, predictions, preview
