import cv2
import mediapipe as mp
import numpy as np
import urllib.request
from pathlib import Path

# MediaPipe 0.10.32 uses the Tasks API (mp.solutions is not available).
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
TASK_MODEL = MODEL_DIR / "pose_landmarker_lite.task"
TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

def ensure_task_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not TASK_MODEL.exists():
        try:
            urllib.request.urlretrieve(TASK_MODEL_URL, TASK_MODEL)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the MediaPipe pose model. Check internet access "
                f"and retry. Details: {exc}"
            ) from exc
    return TASK_MODEL

def get_pose(image_bgr, draw=False, complexity=1):
    """Return (normalized 33x4 feature vector, annotated BGR image)."""
    if image_bgr is None or image_bgr.size == 0:
        return None, image_bgr

    model_path = ensure_task_model()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    annotated = image_bgr.copy()
    if not result.pose_landmarks:
        return None, annotated

    landmarks = result.pose_landmarks[0]
    arr = np.array([[p.x, p.y, p.z, p.visibility] for p in landmarks], dtype=np.float32)

    hip_center = (arr[23, :2] + arr[24, :2]) / 2.0
    shoulder_center = (arr[11, :2] + arr[12, :2]) / 2.0
    scale = float(np.linalg.norm(shoulder_center - hip_center))
    if scale > 1e-5:
        arr[:, :2] = (arr[:, :2] - hip_center) / scale
        arr[:, 2] = arr[:, 2] / scale

    if draw:
        h, w = annotated.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        links = [
            (11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
            (23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),
            (28,30),(30,32),(15,17),(15,19),(15,21),(16,18),(16,20),(16,22)
        ]
        for a,b in links:
            cv2.line(annotated, pts[a], pts[b], (40,190,90), 2)
        for x,y in pts:
            cv2.circle(annotated, (x,y), 3, (20,80,230), -1)

    return arr.reshape(-1), annotated
