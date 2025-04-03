import torch
import cv2
from ultralytics import YOLO


model = YOLO('best10.pt')  
def preprocess(frame):
  
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return frame

def predict(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb) 
    return results

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = predict(frame)

    for result in results:
        boxes = result.boxes  
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            conf = box.conf[0]  
            cls = int(box.cls[0])  
            label = f'{model.names[cls]} {conf:.2f}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow('YOLOv10 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
