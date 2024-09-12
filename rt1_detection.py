import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64

# Load the YOLO model
from ultralytics import YOLO
model = YOLO("best_50.pt")

# Function to decode base64 to OpenCV image
def base64_to_cv2_image(base64_str):
    img_bytes = base64.b64decode(base64_str.split(",")[1])
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

# Streamlit app interface
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
                
                // Send the frame to Streamlit
                window.parent.postMessage({ type: 'FRAME', data: frameData }, '*');
            }

            // Capture frames every 100ms
            setInterval(captureFrame, 100);
        </script>
    """
    
    # Render the HTML video capture
    st.components.v1.html(video_html, width=1280, height=720)

    # Placeholder for the image
    frame_placeholder = st.empty()

    # Display the captured frames
    def update_frame(base64_frame):
        # Convert base64 image to OpenCV image
        frame_img = base64_to_cv2_image(base64_frame)

        # Perform face detection using YOLO
        results = model(frame_img)

        # Draw bounding boxes on the detected faces
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0].item()
                cv2.rectangle(frame_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame_img, f'{confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert image to RGB for displaying
        frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Update the Streamlit placeholder with the detected frame
        frame_placeholder.image(frame_pil, caption='Detected Faces', use_column_width=True)

    # Process frames from JavaScript
    if st.experimental_get_query_params():
        frame_data = st.experimental_get_query_params().get('data')
        if frame_data:
            update_frame(frame_data[0])

# Run the app
if __name__ == "__main__":
    app()
