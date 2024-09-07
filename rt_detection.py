import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("path/to/your/yolov8n.pt")  # Replace with your model path

def process_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, imgsz=640)

    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"Confidence: {confidence:.2f}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def app():
    st.title("Real-Time Face Detection with YOLO")

    # Create a placeholder for the webcam feed
    stframe = st.empty()

    # Open webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        frame = process_frame(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Allow stopping the stream
        if st.button('Stop Stream'):
            break

    cap.release()
    st.success("Stream stopped")

if __name__ == "__main__":
    app()
