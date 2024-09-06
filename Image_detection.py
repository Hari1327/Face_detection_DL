import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the YOLO model
model = YOLO("best.pt")  # Make sure the path is correct to your model

# Function to perform face detection
def face_detection(uploaded_image):
    # Convert the uploaded file to an OpenCV image
    img_array = np.array(uploaded_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize the image to 640x640 for YOLO input
    img_resized = cv2.resize(img_bgr, (640, 640))

    # Perform face detection
    results = model(img_resized, imgsz=640)

    # Loop through results and draw bounding boxes and confidence scores
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates and confidence score
            x_min, y_min, x_max, y_max = box.xyxy[0].int().tolist()
            confidence = box.conf[0].item()

            # Draw a rectangle for the bounding box
            cv2.rectangle(img_resized, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

            # Add confidence label
            label = f'Confidence: {confidence:.2f}'
            cv2.putText(img_resized, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_resized

# Streamlit app layout
st.title("Face Detection with YOLOv8")
file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if file:
    # Display uploaded image
    st.image(file, caption='Uploaded Image', use_column_width=True)
    
    # Convert uploaded file to PIL Image
    image = Image.open(file)
    
    # Button to trigger detection
    if st.button("Detect Faces"):
        # Perform face detection
        detected_image = face_detection(image)
        
        # Convert OpenCV BGR image back to RGB for displaying in Streamlit
        detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
        
        # Display detected image with bounding boxes and confidence scores
        st.image(detected_image_rgb, caption='Detected Faces', use_column_width=True)
