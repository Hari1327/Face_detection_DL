import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load the YOLO model
model = YOLO("best.pt")  # Replace with your custom model path

# Function to perform face detection on an image
def face_detection_image(image_file):
    image = Image.open(image_file)
    image_np = np.array(image)  # Convert PIL image to numpy array for OpenCV

    # Resize the image to the model input size
    image_resized = cv2.resize(image_np, (640, 640))

    # Perform face detection
    results = model(image_resized, imgsz=640)

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].int().tolist()
            confidence = box.conf[0].item()

            # Draw a rectangle around the face
            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

            # Prepare the text to display
            label = f'Confidence: {confidence:.2f}'
            cv2.putText(image_np, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_np

# Streamlit app for image face detection
def app_image():
    st.title("Image Face Detection")

    # Image uploader
    image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if image_file:
        # Display the uploaded image in the first column
        col1, col2 = st.columns(2)

        with col1:
            st.image(image_file, caption='Uploaded Image', use_column_width=True)

        # Perform face detection when button is clicked
        if st.button("Detect Faces"):
            detected_image = face_detection_image(image_file)

            # Display the detected faces image in the second column
            with col2:
                st.image(detected_image, caption='Detected Faces', use_column_width=True)

# Run the image detection app
if __name__ == "__main__":
    app_image()
