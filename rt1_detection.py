import streamlit as st
import numpy as np
import cv2
from PIL import Image
import base64
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best_50.pt")

# Function to decode base64 to OpenCV image
def base64_to_cv2_image(base64_str):
    try:
        img_bytes = base64.b64decode(base64_str.split(",")[1])
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        st.error(f"Error decoding base64 image: {e}")
        return None

# Function to test if the model is running
def test_model():
    try:
        # Create a dummy image (black square)
        dummy_img = np.zeros((640, 480, 3), dtype=np.uint8)
        results = model(dummy_img)
        return True, "Model is running"
    except Exception as e:
        return False, f"Model is not running: {e}"

# Streamlit app interface
def app():
    st.title("Real-Time Face Detection with YOLO")

    # Add a slider for confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%.2f"
    )

    # Add a button to start/stop the webcam
    start_camera = st.button("Start Camera")
    stop_camera = st.button("Stop Camera")
    
    # Initialize camera status
    if 'camera_running' not in st.session_state:
        st.session_state['camera_running'] = False
    
    if start_camera:
        st.session_state['camera_running'] = True
    if stop_camera:
        st.session_state['camera_running'] = False

    # Button to check if the model is running
    if st.button("Check Model Status"):
        status, message = test_model()
        st.success(message) if status else st.error(message)

    if st.session_state['camera_running']:
        # HTML5 video capture component
        video_html = """
            <video id="webcam" autoplay playsinline></video>
            <canvas id="canvas"></canvas>
            <script>
                const video = document.getElementById('webcam');
                const canvas = document.getElementById('canvas');
                const context = canvas.getContext('2d');

                async function startCamera() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                        video.srcObject = stream;
                    } catch (error) {
                        console.error('Error accessing the webcam: ', error);
                        window.parent.postMessage({ type: 'ERROR', data: 'Error accessing the webcam' }, '*');
                    }
                }

                function captureFrame() {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const frameData = canvas.toDataURL('image/jpeg');
                    window.parent.postMessage({ type: 'FRAME', data: frameData }, '*');
                }

                startCamera().then(() => {
                    setInterval(captureFrame, 100);
                });
            </script>
        """
        
        # Render the HTML video capture
        st.components.v1.html(video_html, width=640)
        # Placeholder for the image
        frame_placeholder = st.empty()
        detection_placeholder = st.empty()

        # Function to process and update the frame
        def update_frame(base64_frame):
            frame_img = base64_to_cv2_image(base64_frame)
            if frame_img is None:
                st.write("Error processing frame.")
                return

            # Perform face detection
            try:
                results = model(frame_img)
                detections = results.pandas().xyxy[0]
                print(detections)
                if detections.empty:
                    detection_placeholder.write("No faces detected")
                else:
                    detection_text = ""
                    for _, row in detections.iterrows():
                        confidence = row['confidence']
                        if confidence >= confidence_threshold:
                            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                            cv2.rectangle(frame_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            cv2.putText(frame_img, f'{confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            detection_text += f"Face detected: Confidence {confidence:.2f} at ({x_min}, {y_min}) to ({x_max}, {y_max})\n"

                    # Show detection results
                    detection_placeholder.text(detection_text)
                    
            except Exception as e:
                st.error(f"Error during model inference: {e}")
                return

            frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_placeholder.image(frame_pil, caption='Detected Faces', use_column_width=True)

        # Handle incoming messages from JavaScript
        def handle_message(message):
            if message.get('type') == 'FRAME':
                base64_frame = message.get('data')
                if base64_frame:
                    update_frame(base64_frame)

        # JavaScript to handle messages
        st.components.v1.html("""
            <script>
                window.addEventListener('message', function(event) {
                    const message = event.data;
                    if (message.type === 'FRAME') {
                        window.parent.postMessage(message, '*');
                    }
                });
            </script>
        """, height=0)

if __name__ == "__main__":
    app()
