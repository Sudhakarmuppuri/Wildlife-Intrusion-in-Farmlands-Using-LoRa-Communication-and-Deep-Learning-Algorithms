# train_yolo.py

from ultralytics import YOLO

# Load a YOLOv8n model
model = YOLO('yolov8n.pt')  # Nano version of YOLOv8

# Train the model
model.train(
    data=' ', #replace with path address of Yaml File
    epochs=100,
    imgsz=640
)

print("Training Completed!")
