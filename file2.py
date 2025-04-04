import torch
import cv2  # ✅ Use cv2 instead of cv3
import os
import mlflow
import mlflow.pytorch
import numpy as np
from ultralytics import YOLO
import dagshub
dagshub.init(repo_owner='girishdongrekar123', repo_name='signflow', mlflow=True)

# Initialize MLflow experiment
mlflow.set_experiment("YOLOv8 Number Plate Detection 1")
mlflow.pytorch.autolog()
# Load YOLO model
model = YOLO('best10.pt')

# Create a directory for saving detected frames
output_folder = "detected_frames"
os.makedirs(output_folder, exist_ok=True)

def preprocess(frame):
    """Preprocess the frame for YOLO model"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return frame

def predict(frame):
    """Perform object detection"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    return results

# Open webcam
cap = cv2.VideoCapture(0)  # ✅ Use cv2 instead of cv3
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Start MLflow Run
with mlflow.start_run():
    mlflow.log_param("model_name", "YOLOv8")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = predict(frame)

        detected = False  # Flag to check if there is a detection
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                detected = True  # Detection occurred
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                conf = box.conf[0].item()
                cls = int(box.cls[0])  
                label = f'{model.names[cls]} {conf:.2f}'

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

                # Log detection details
                mlflow.log_metric(f"confidence_cls_{cls}", conf)

        # Save and log inference frame only if detection occurred
        if detected:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            mlflow.log_artifact(frame_path)
            frame_count += 1  # Increment only when a frame is saved

        cv2.imshow('YOLOv8 Detection with MLflow', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Log model
    mlflow.pytorch.log_model(model, "YOLOv8_model")

cap.release()
cv2.destroyAllWindows()
