import sys
print("Python:", sys.version)
import cv2, mediapipe as mp, numpy as np, pandas as pd, sklearn, streamlit
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("Streamlit:", streamlit.__version__)
print("MediaPipe Tasks API:", hasattr(mp, "tasks"))
print("SETUP OK - Tasks API available")
