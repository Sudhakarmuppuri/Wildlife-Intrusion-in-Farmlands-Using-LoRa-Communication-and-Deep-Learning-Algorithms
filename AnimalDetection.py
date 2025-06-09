import cv2
import torch
from ultralytics import YOLO
import serial
import warnings

import smtplib 
import os
import time
import numpy as np 
import requests  # To send the photo to Telegram 
from email.mime.text import MIMEText 
from email.mime.multipart import MIMEMultipart 
from email.mime.image import MIMEImage 
from datetime import datetime

warnings.filterwarnings("ignore")

ser = serial.Serial('COM5', baudrate=9600)
 # Adjust COM port and baudrate as needed
print("Serial connection opened successfully!")

# Load the YOLOv8 model
model = YOLO("C:/Users/kmith/Desktop/MITHUN/Major Project/MP/runs/detect/train6/weights/best.pt")

# Email credentials 
sender_email = "xyz@gmail.com" 
receiver_emails = ["xyz@gmail.com"," "] 
sender_password = "xyz" 
smtp_server = "smtp.gmail.com" 
smtp_port = 587

def capture_photo(frame): 
    print("Capturing image from the current frame...") 
 
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    filename = f"photo_{timestamp}.jpg" 
    cv2.imwrite(filename, frame)  # Save the current frame as the image 
    print(f"Photo saved as {filename}") 
     
    return filename 

# Function to send email with the person's name in the subject and body 
def send_mail(person_name, attachment=None): 
    print(f"Sending mail...") 
    time.sleep(2) 
 
    msg = MIMEMultipart() 
    msg['From'] = sender_email 
    msg['To'] = ", ".join(receiver_emails) 
    msg['Subject'] = f"{person_name} detected" 
 
    body_text = f"{person_name} detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\nPlease find the attached photo." 
    msg.attach(MIMEText(body_text, 'plain')) 
 
    if attachment: 
        with open(attachment, 'rb') as fp: 
            img_data = MIMEImage(fp.read()) 
            msg.attach(img_data) 
 
    with smtplib.SMTP(smtp_server, smtp_port) as server: 
        server.starttls() 
        server.login(sender_email, sender_password) 
        server.sendmail(sender_email, receiver_emails, msg.as_string()) 
 
    print("Mail sent") 
    time.sleep(2) 

# Open webcam (0 for default camera)
cap = cv2.VideoCapture(0)

# List of animal classes to detect
#animal_classes = ["dog", "cat", "cow", "horse", "sheep", "elephant", "bear", "zebra", "giraffe", "bird"]
animal_classes = ["Bear", "Elephant", "lion", "rhino"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)
    results = model(frame, verbose=False)  # Suppressing speed metrics output


    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])  # Class ID
            label = model.names[cls_id]  # Class name
            confidence = float(box.conf[0])  # Confidence score
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check if the detected object is an animal and confidence is > 0.7
            if label in animal_classes and confidence > 0.7:
                
                print(f"Detected: {label} with confidence {confidence:.2f}")
                values_string = f"${label}#\n"
                ser.write(bytes(values_string, 'utf-8'))
                print(f"data sent : {values_string}")
                photo_filename = capture_photo(frame)  # Capture photo
                send_mail(f"{label} detected", photo_filename)
                time.sleep(10)
                

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Animal Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("C:/Users/kmith/Desktop/MITHUN/Major Project/MP/runs/detect/train6/weights/best.pt") # replace with your model path

# Evaluate the model on a dataset (e.g., COCO-format or YOLO-format dataset)
metrics = model.val(data="C:/Users/kmith/Desktop/MITHUN/Major Project/MP/Dataset/animals.v1i.yolov8/data.yaml", split='test')  # or split='test'

# Print evaluation metrics
print("Evaluation Metrics:")
print(f"Precision: {metrics.box.mAP50:.4f}")
print(f"Recall: {metrics.box.recall:.4f}")
print(f"mAP@0.5: {metrics.box.mAP50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.mAP50_95:.4f}")
print(f"F1 Score: {(2 * metrics.box.mAP50 * metrics.box.recall) / (metrics.box.mAP50 + metrics.box.recall + 1e-6):.4f}")
