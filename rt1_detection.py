import streamlit as st
import numpy as np
import cv2
from PIL import Image
import base64
import io
from ultralytics import YOLO
import json
from flask import Flask, request

# Initialize YOLO model
model = YOLO("best_50.pt")

# Flask for handling the backend requests
app = Flask(__name__)

@app.route('/video-frame', methods=['POST'])
def video_frame():
    data = request.json
    image_data = data['image']

    # Decode the image from base64
    image_data = image_data.split(",")[1]
    decoded_data = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(decoded_data))

    # Convert image to BGR format (for OpenCV)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Perform face detection with YOLO
    results = model(image_bgr)

    # Prepare results to send back (bounding boxes and confidences)
    detections = []
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            detections.append({'box': [x_min, y_min, x_max, y_max], 'confidence': confidence})

    return json.dumps(detections)


# Streamlit interface
def streamlit_app():
    st.title("Real-time Video Face Detection")

    # Display HTML component for capturing video
    html_string = """
        <video id="webcam" autoplay playsinline></video>
        <canvas id="canvas" style="display:none;"></canvas>

        <script>
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');

            navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
                video.srcObject = stream;
            });

            function captureFrame() {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataURL = canvas.toDataURL('image/jpeg');
                fetch('/video-frame', {
                    method: 'POST',
                    body: JSON.stringify({ image: dataURL }),
                    headers: {
                        'Content-Type': 'application/json'
                    }
                }).then(response => response.json())
                  .then(data => {
                      console.log(data);
                  });
            }

            setInterval(captureFrame, 100);
        </script>
    """
    st.components.v1.html(html_string, height=400)

if __name__ == "__main__":
    streamlit_app()
