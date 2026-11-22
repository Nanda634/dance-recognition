from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_from_dataframe(df, model_dir, output_dir):
    if df.empty or "label" not in df:
        raise ValueError("No extracted training samples. Add images first.")
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("At least two dance classes with usable poses are required.")
    if counts.min() < 2:
        raise ValueError("Each class needs at least two usable examples for a stratified test split.")
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=350, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1, min_samples_leaf=1
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, pred))
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, pred, labels=clf.classes_)
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_cols": feature_cols}, model_dir / "dance_model.joblib")
    result = {
        "accuracy": accuracy,
        "classes": clf.classes_.tolist(),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "report": report,
        "confusion_matrix": matrix.tolist()
    }
    (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(matrix, index=clf.classes_, columns=clf.classes_).to_csv(output_dir / "confusion_matrix.csv")
    return result
