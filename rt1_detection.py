import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best_50.pt")

# Function to decode binary to OpenCV image
def binary_to_cv2_image(binary_data):
    try:
        img_array = np.frombuffer(binary_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        st.write("Received Image Size:", img.shape)
        return img
    except Exception as e:
        st.error(f"Error decoding binary image: {e}")
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
    # Initialize session state variables if they don't exist
    if 'camera_running' not in st.session_state:
        st.session_state['camera_running'] = False
    if 'stream_status' not in st.session_state:
        st.session_state['stream_status'] = "No stream available"

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

    if start_camera:
        st.session_state['camera_running'] = True
        st.session_state['stream_status'] = "Streaming..."
    if stop_camera:
        st.session_state['camera_running'] = False
        st.session_state['stream_status'] = "No stream available"

    # Button to check if the model is running
    if st.button("Check Model Status"):
        status, message = test_model()
        st.success(message) if status else st.error(message)

    st.write(f"**Stream Status:** {st.session_state['stream_status']}")

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
                        video.play();
                        window.parent.postMessage({ type: 'STREAM_STATUS', data: 'streaming' }, '*');
                    } catch (error) {
                        console.error('Error accessing the webcam: ', error);
                        window.parent.postMessage({ type: 'STREAM_STATUS', data: 'error' }, '*');
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
                    setInterval(captureFrame, 100);  // Adjust frame capture interval as needed
                });
            </script>
        """
        
        # Render the HTML video capture
        st.components.v1.html(video_html, height=400)
        # Placeholder for image and detections
        frame_placeholder = st.empty()

        # Function to process and update the frame
        def update_frame(base64_frame):
            frame_img = base64_to_cv2_image(base64_frame)
            if frame_img is None:
                st.write("No frames")
                return
            
            # Resize the image to 1280x720
            img_resized = cv2.resize(frame_img, (1280, 720))
            st.write("Resized Image Size:", img_resized.shape)

            # Perform face detection with the resized image
            results = model(img_resized, imgsz=1280, conf=confidence_threshold)
            
            # Print results for debugging
            st.write("Detection Results:", results.pandas().xyxy[0])
            
            # Draw bounding boxes and labels on detected faces
            if not results.pandas().xyxy[0].empty:
                for _, row in results.pandas().xyxy[0].iterrows():
                    x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    confidence = row['confidence']
                    cv2.rectangle(img_resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img_resized, f'{confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                st.write("No faces detected")

            frame_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_placeholder.image(frame_pil, caption='Detected Faces', use_column_width=True)

        # Handle incoming messages from JavaScript
        def handle_message(message):
            if message.get('type') == 'FRAME':
                base64_frame = message.get('data')
                if base64_frame:
                    update_frame(base64_frame)
            elif message.get('type') == 'STREAM_STATUS':
                status = message.get('data')
                st.session_state['stream_status'] = "Streaming..." if status == 'streaming' else "No stream available"

        # JavaScript to handle messages
        st.components.v1.html("""
            <script>
                window.addEventListener('message', function(event) {
                    const message = event.data;
                    if (message.type === 'FRAME' || message.type === 'STREAM_STATUS') {
                        window.parent.postMessage(message, '*');
                    }
                });
            </script>
        """, height=0)

if __name__ == "__main__":
    app()
