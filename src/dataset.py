from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src.pose import get_pose

CLASSES = [
    "Bharatanatyam", "Kuchipudi", "Kathak",
    "Odissi", "Mohiniyattam", "Kathakali"
]
VALID = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def extract_dataset(data_dir, progress=None):
    rows, skipped = [], []
    base = Path(data_dir)
    paths = []
    for cls in CLASSES:
        folder = base / cls
        if folder.exists():
            paths.extend((p, cls) for p in folder.rglob("*") if p.suffix.lower() in VALID)
    total = max(1, len(paths))
    for i, (path, label) in enumerate(paths, 1):
        image = cv2.imread(str(path))
        if image is None:
            skipped.append((str(path), "Unreadable image"))
        else:
            feat, _ = get_pose(image, draw=False, complexity=0)
            if feat is None:
                skipped.append((str(path), "No pose detected"))
            else:
                rows.append({"label": label, "source": str(path), **{f"f{j}": v for j, v in enumerate(feat)}})
        if progress:
            progress(i / total)
    return pd.DataFrame(rows), skipped
