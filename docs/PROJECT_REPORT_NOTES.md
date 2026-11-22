# Report notes

## Objective
Develop a posture-based prototype that classifies six Indian classical dance categories using human pose estimation and supervised machine learning.

## Method
MediaPipe Pose detects 33 landmarks. Landmark coordinates are translated and scale-normalized relative to the hip midpoint and shoulder-to-hip distance. A Random Forest classifier learns the resulting feature vectors.

## Evaluation
Use the held-out test split created by the training script. Report accuracy, precision, recall, F1-score and confusion matrix exactly as generated. Small datasets may produce unstable estimates.

## Limitations
- Image dataset size and class balance affect performance.
- Similar postures across dance forms may be confused.
- Static landmarks do not encode choreography or rhythm.
- Hand mudras and facial expressions are not included.
- Predictions are not a substitute for expert assessment.
