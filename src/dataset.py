from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src.pose import get_pose

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

def discover_classes(data_dir=RAW_DIR):
    """Auto-discover class names from folders in data/raw/."""
    base = Path(data_dir)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))

# Discovered automatically from data/raw/ folders at import time.
# Falls back to the six classical forms if no folders exist yet.
CLASSES = discover_classes() or [
    "Bharatanatyam", "Kuchipudi", "Kathak",
    "Odissi", "Mohiniyattam", "Kathakali", "Manipuri", "Sattriya",
]
VALID = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def extract_dataset(data_dir, progress=None):
    rows, skipped = [], []
    base = Path(data_dir)
    classes = discover_classes(base)  # fresh scan at training time
    paths = []
    for cls in classes:
        folder = base / cls
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
