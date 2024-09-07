import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")  # Adjust the path to your model if necessary

class FaceDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Perform face detection using YOLO
        results = model(img, imgsz=640)
        
        # Loop through detections and draw bounding boxes
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].int().tolist()
                confidence = box.conf[0].item()

                # Draw rectangle around face
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                label = f"Confidence: {confidence:.2f}"
                cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img

def app():
    st.title("Webcam Real-Time Face Detection")

    # Start webrtc streamer
    webrtc_streamer(key="face-detection", video_transformer_factory=FaceDetectionTransformer)

if __name__ == "__main__":
    app()
