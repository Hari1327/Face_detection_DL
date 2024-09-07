import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import requests
from io import BytesIO
from PIL import Image

# Load YOLO model
model = YOLO("best.pt")  # Replace with your model path

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.webcam_url = "http://localhost:5000/video_feed"  # Replace with your webcam API URL

    def recv(self, frame):
        # Fetch the video frame from the webcam API
        response = requests.get(self.webcam_url, stream=True)
        image_bytes = BytesIO(response.content)
        img = Image.open(image_bytes)
        img = np.array(img)

        # Convert the frame to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform face detection
        results = self.model(img_rgb, imgsz=640)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f"Confidence: {confidence:.2f}"
                cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

def app():
    st.title("Real-Time Face Detection with YOLO")

    # Create a Streamlit WebRTC video streamer
    webrtc_streamer(key="video", video_processor_factory=VideoProcessor)

if __name__ == "__main__":
    app()
