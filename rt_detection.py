import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the YOLO model
model = YOLO("best_50.pt")  # Ensure model path is correct

# Function to perform face detection
def face_detection(uploaded_image, conf_threshold=0.25):
    img_array = np.array(uploaded_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Perform face detection
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

# Streamlit app to capture webcam input and perform detection
def app():
    st.title("Webcam Face Detection App")

    # Add a slider to adjust the confidence threshold
    confi_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01,key="confidence_slider_live")

    # Capture an image from the webcam
    webcam_image = st.camera_input("Take a picture")

    if webcam_image:
        # Convert the webcam image to PIL Image
        image = Image.open(webcam_image)

        # Show the original captured image
        st.image(image, caption='Captured Image', use_column_width=True)

        # Perform face detection
        detected_image = face_detection(image, confi_threshold=conf_threshold)

        # Convert BGR image back to RGB for displaying
        detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

        # Show the detected faces image
        st.image(detected_image_rgb, caption='Detected Faces', use_column_width=True)

# To run the app
if __name__ == "__main__":
    app()
