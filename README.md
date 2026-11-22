# Indian Classical Dance Recognition — Complete Prototype

A local Streamlit application for posture-based recognition of six Indian classical dance forms using MediaPipe Pose landmarks and a Random Forest classifier.

## Important truth about the model
This project includes the complete application and training pipeline, but **does not contain a trained six-class model or a licensed dance dataset**. A trained model cannot be honestly supplied without training examples. The app therefore includes a guided data collection page and will clearly explain what to do until a model is trained. Do not present placeholder or untrained predictions as real recognition.

## Quick start (Windows / VS Code)

1. Extract this ZIP.
2. Open the `Indian-Classical-Dance-Recognition-Complete` folder in VS Code.
3. Open Terminal (PowerShell) and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_setup.py
streamlit run app.py
```

If `py -3.11` is unavailable, try `python -m venv .venv`. MediaPipe wheels are most reliable on supported 64-bit Python versions; if installation fails, use Python 3.11 64-bit.

## App workflow

1. Open **Dataset Builder**.
2. For each dance category, upload labeled images. Use clear full-body photos with one dancer, varied poses and backgrounds.
3. Use the same person/session only in one split where possible; do not place near-duplicate frames in both train and test.
4. Open **Train Model** and train the Random Forest baseline.
5. Check the held-out report and confusion matrix. If a class has too few examples, collect more.
6. Open **Recognize** to test images, videos or live webcam.

## Six classes

- Bharatanatyam
- Kuchipudi
- Kathak
- Odissi
- Mohiniyattam
- Kathakali

## Data collection guidance

For a meaningful first demo, aim for at least 30–50 usable, varied images per class (more is better). Include full-body images, different performers, camera distances and backgrounds. The minimum accepted by the trainer is lower for smoke testing only; very small datasets are not reliable.

Only use images/videos you own or are permitted to use. Keep source attribution and license records. The model learns visual pose patterns and may confuse dance forms with similar stances. A single static image cannot reliably identify a dance style in every case.

## Video and webcam
Video predictions are sampled frame-by-frame and combined by majority vote. Live webcam prediction runs on the local computer. Browser camera permissions are required.

## Troubleshooting

- **MediaPipe import error:** use 64-bit Python 3.11 and reinstall `pip install -r requirements.txt`.
- **No pose detected:** use a well-lit, full-body image; avoid cropped feet/head and multiple people.
- **No model found:** add labeled examples and train from the app.
- **Low accuracy:** collect more balanced, varied data; verify labels; avoid near-duplicate leakage; inspect per-class precision/recall.
- **Webcam unavailable:** close other apps using the camera and allow browser permission.

## Project layout

```text
app.py                  Streamlit interface
check_setup.py          Dependency / MediaPipe smoke test
src/pose.py             Pose landmarks and normalization
src/dataset.py          Image scanning and feature extraction
src/model.py            Training, evaluation and model saving
src/inference.py        Image/video prediction helpers
data/raw/<class>/       Labeled source images
data/processed/         Extracted feature table
models/                 Trained model artifacts
outputs/                Evaluation report and confusion matrix
docs/                   Project notes
```

## Research limitations
This is a prototype, not a validated production classifier. The model's `predict_proba` values are classifier scores, not calibrated probabilities. Report measured metrics only. The system currently uses one detected person's body pose; hand mudras, facial expression, rhythm and temporal choreography are not modeled.
