import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Load YOLO model (replace with the correct path to your model)
model = YOLO("path/to/your/yolov8n.pt")

def detect_faces(image):
    # Convert the image to a format suitable for YOLO
    img_np = np.array(image)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    # Perform face detection
    results = model(img_rgb, imgsz=640)
    
    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"Confidence: {confidence:.2f}"
            cv2.putText(img_np, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_np

def app():
    st.title("Face Detection with YOLO")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Load the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Perform face detection
        detected_image = detect_faces(image)
        detected_image_pil = Image.fromarray(detected_image)
        
        # Display the detected image
        st.image(detected_image_pil, caption="Detected Faces", use_column_width=True)

if __name__ == "__main__":
    app()
