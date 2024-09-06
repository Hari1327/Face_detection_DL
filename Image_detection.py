import streamlit as st
from ultralytics import YOLO
import PIL
import numpy as np
import os

# Fix for duplicate library errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def text_detection(file):
    # Load your YOLOv8 model
    model = YOLO("best.pt")
    
    # Load the uploaded image
    uploaded_image = PIL.Image.open(file)
    
    # Perform inference
    res = model.predict(uploaded_image, conf=0.5)
    
    # Extract bounding boxes
    box = res[0].boxes.xyxy.tolist()
    
    # Plot the results
    res_plotted = res[0].plot()[:, :, ::-1]  # Convert BGR to RGB for Streamlit display
    
    # Display the result in Streamlit
    st.image(res_plotted, caption='Detected Image', use_column_width=True)
    
    # Display the number of detections
    st.write(f"Number of detections: {len(box)}")

def app():
    st.title("Upload an Image and Detect Faces")
    
    # Upload image file
    file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if file is not None:
        # Display the uploaded image before detection
        st.image(file, caption='Uploaded Image', use_column_width=True)
    
        # Add a detect button to trigger the face detection
        if st.button("Detect"):
            # Call the face detection function and display the results
            text_detection(file)

if __name__ == "__main__":
    app()
