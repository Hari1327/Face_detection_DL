import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load the YOLO model
model = YOLO("best(2).pt")  # Ensure model path is correct

# Function to perform face detection
def face_detection(uploaded_image, conf_threshold=0.25):
    img_array = np.array(uploaded_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Resize the image to higher resolution
    img_resized = cv2.resize(img_bgr, (1280, 1280))

    # Perform face detection with a lower confidence threshold
    results = model(img_resized, imgsz=1280, conf=conf_threshold)

    # Draw bounding boxes and confidence scores
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            cv2.rectangle(img_resized, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
            label = f'Confidence: {confidence:.2f}'
            cv2.putText(img_resized, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_resized

# The app function to be called in main.py
def app():
    st.title("Upload the Image and Detect Faces")
    
    # Add a slider to adjust confidence threshold
    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

    file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if file:
        image = Image.open(file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Detect Faces"):
            detected_image = face_detection(image, conf_threshold=conf_threshold)
            detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(detected_image_rgb, caption='Detected Faces', use_column_width=True)

# To run the app
if __name__ == "__main__":
    app()
