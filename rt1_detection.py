import streamlit as st
import cloudinary
import cloudinary.uploader
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
import base64
from io import BytesIO

# Set up Cloudinary config
cloudinary.config(
    cloud_name = 'dckc1r7ub', 
    api_key = '268289947187182', 
    api_secret = 'ITtRnnt18P-7oWVt5KoV8Hx2Hp4'
)

# Load YOLO model
model = YOLO("best_50.pt")

# Function to perform face detection
def face_detection(image_array):
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    results = model(img_bgr, imgsz=640, conf=0.25)
    
    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            label = f'{confidence:.2f}'
            cv2.putText(img_bgr, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_bgr

# Function to upload image to Cloudinary
def upload_to_cloudinary(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    response = cloudinary.uploader.upload(img_str)
    return response['secure_url']

# Main Streamlit app function
def app():
    st.title("Real-Time Face Detection with Cloudinary")

    # Add a slider to adjust the refresh interval (to simulate real-time detection)
    refresh_interval = st.slider("Refresh Interval (seconds)", min_value=1, max_value=5, value=2)

    webcam_enabled = st.checkbox("Enable Webcam", value=True)

    if webcam_enabled:
        # Continuously capture images from webcam and process them
        while webcam_enabled:
            webcam_image = st.camera_input("Take a picture", key="webcam_input")
            
            if webcam_image:
                # Convert the image from webcam to PIL
                image = Image.open(webcam_image)
                
                # Show the captured image
                st.image(image, caption='Captured Image', use_column_width=True)
                
                # Perform face detection on the image
                image_np = np.array(image)
                detected_image = face_detection(image_np)

                # Convert to PIL for Cloudinary upload
                detected_pil_image = Image.fromarray(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
                
                # Upload to Cloudinary
                cloudinary_url = upload_to_cloudinary(detected_pil_image)
                
                # Show detected faces
                st.image(cloudinary_url, caption='Detected Faces from Cloudinary')

                # Wait for the defined refresh interval before capturing the next frame
                time.sleep(refresh_interval)
    else:
        st.write("Webcam is disabled. Enable it to start detection.")

# Run the app
if __name__ == "__main__":
    app()
