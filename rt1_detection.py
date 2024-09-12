import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import base64

# Load the YOLO model
model = YOLO("best_50.pt")

# Streamlit app for real-time video detection
def app():
    st.title("Real-Time Face Detection with YOLO")

    # HTML5 video capture component
    video_html = """
        <video id="webcam" autoplay playsinline></video>
        <canvas id="canvas" style="display:none;"></canvas>

        <script>
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');

            // Access the user's webcam
            navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
                video.srcObject = stream;
            });

            // Capture frames and send them to Streamlit for processing
            function captureFrame() {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Convert the frame to base64
                const frameData = canvas.toDataURL('image/jpeg');
                
                // Send the frame to Streamlit via WebSocket
                Streamlit.setComponentValue(frameData);
            }

            // Capture frames every 100ms
            setInterval(captureFrame, 100);
        </script>
    """
    
    # Render the HTML video capture
    st.components.v1.html(video_html, height=400)

    # Process the captured frames
    frame_data = st.experimental_get_query_params()
    
    if frame_data:
        # Decode base64 image
        frame_base64 = frame_data["image"][0].split(",")[1]
        frame_bytes = base64.b64decode(frame_base64)
        np_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame_img = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

        # Perform face detection using YOLO
        results = model(frame_img)

        # Draw bounding boxes on the detected faces
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0].item()
                cv2.rectangle(frame_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame_img, f'{confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the detected frame
        st.image(frame_img, channels="BGR")

# Run the app
if __name__ == "__main__":
    app()
