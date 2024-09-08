import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import io
from io import BytesIO

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
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()

# The app function
def app():
    st.title("Live Webcam Face Detection")

    # Add a slider to adjust the confidence threshold
    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01,key="slider_confidence")

    # Streamlit's HTML and JavaScript to capture live video
    st.markdown("""
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
        <script>
        let video;
        let canvas;
        let context;
        let stream;
        let imageData;

        async function setup() {
            video = document.createElement('video');
            video.width = 640;
            video.height = 480;
            document.body.appendChild(video);
            canvas = document.createElement('canvas');
            canvas.width = video.width;
            canvas.height = video.height;
            context = canvas.getContext('2d');
            document.body.appendChild(canvas);

            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.play();

            const interval = setInterval(() => {
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                imageData = canvas.toDataURL('image/png');
                Streamlit.setComponentValue(imageData);
            }, 100);
        }

        setup();
        </script>
    """, unsafe_allow_html=True)

    # Capture the image data
    image_data = st.experimental_get_query_params().get('imageData')

    if image_data:
        # Decode the image data
        image_data = image_data[0].split(',')[1]
        image_bytes = BytesIO(base64.b64decode(image_data))
        image = Image.open(image_bytes)

        # Convert to NumPy array and process
        frame = np.array(image)
        detected_frame = face_detection(frame, conf_threshold=conf_threshold)

        # Display the result
        st.image(image_to_bytes(detected_frame), caption='Detected Faces', use_column_width=True)

if __name__ == "__main__":
    app()
