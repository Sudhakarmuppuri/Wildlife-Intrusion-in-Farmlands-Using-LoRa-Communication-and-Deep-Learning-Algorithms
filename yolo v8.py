import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model (using the small 'n' version for speed)
model = YOLO("yolov8n.pt")  # Make sure the model file is in the same directory

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no frame is captured

    # Run YOLOv8 inference
    results = model(frame)

    # Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = model.names[cls]  # Get class label

            if conf > 0.5:  # Confidence threshold
                color = (0, 255, 0)  # Green color for bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                print(f"Detected: {label} with confidence {conf:.2f}")

    # Show the live detection feed
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
