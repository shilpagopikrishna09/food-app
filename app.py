import sys
import os

# Redirect error messages to a log file instead of crashing
sys.stderr = open(os.devnull, 'w')

# Now import torch
import torch

import torch
import cv2
import numpy as np
import sys
if sys.stderr is None:
    sys.stderr = sys.stdout  # Redirect errors to normal output if stderr is missing


# Load YOLOv5 Food Model (Update path if needed)
model = torch.hub.load("ultralytics/yolov5", "custom", path="C:/Users/shilp/Documents/AI_project/Food_App/runs/train/exp/weights/yolov5s.pt", force_reload=True)
 # Ensures that the model is freshly downloaded, if necessary

# Start the camera stream (0 for the default camera)
cap = cv2.VideoCapture(0)

while True:	
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLOv5 Inference
    results = model(frame_rgb)

    # Get detected objects
    detections = results.pandas().xyxy[0]

    # Draw bounding boxes on frame
    for _, row in detections.iterrows():
        x1, y1, x2, y2, confidence, class_id, label = (
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
            row["confidence"],
            int(row["class"]),
            row["name"],  # Food name
        )

        if confidence > 0.3:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame with detections
    cv2.imshow("Food Detection - Live", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
