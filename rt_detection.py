import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# Load YOLOv8 model for face detection
model = YOLO("best.pt")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        try:
            # Convert frame to numpy array (OpenCV format)
            img = frame.to_ndarray(format="bgr24")

            # Perform face detection with YOLO
            results = self.model(img, imgsz=640)

            # Loop through detections and draw bounding boxes
            for result in results:
                for box in result.boxes:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()

                    # Draw rectangle around detected face
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Add confidence label
                    label = f"Confidence: {confidence:.2f}"
                    cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Return processed frame
            return img
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return frame

# Streamlit App
def app():
    st.title("Real-Time Face Detection using YOLOv8")
    
    # Create a real-time webcam feed with YOLO face detection
    webrtc_streamer(key="face-detection", video_transformer_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False})

if __name__ == "__main__":
    app()
