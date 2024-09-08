import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import io

# Load the YOLO model
model = YOLO("best_50.pt")  # Ensure model path is correct

# Function to perform face detection
def face_detection(frame, conf_threshold=0.25):
    img_bgr = frame
    results = model(img_bgr, imgsz=640, conf=conf_threshold)

    # Draw bounding boxes and confidence scores
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
            label = f'Confidence: {confidence:.2f}'
            cv2.putText(img_bgr, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_bgr

# Function to convert image to streamlit compatible format
def image_to_bytes(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()

# The app function
def app():
    st.title("Live Webcam Face Detection")

    # Add a slider to adjust the confidence threshold
    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01,key="slider_confidence")

    # Capture the video stream from the webcam
    video_file = st.camera_input("Capture a video")

    if video_file:
        # Convert video file to a byte stream
        video_bytes = video_file.read()
        video_buffer = io.BytesIO(video_bytes)
        
        # Open the video stream using OpenCV
        cap = cv2.VideoCapture(video_buffer)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            detected_frame = face_detection(frame, conf_threshold=conf_threshold)
            detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)

            # Display the result
            st.image(image_to_bytes(detected_frame_rgb), caption='Detected Faces', use_column_width=True)
        
        cap.release()

if __name__ == "__main__":
    app()
