import cloudinary
import cloudinary.uploader
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

cloudinary.config(
    cloud_name = 'dckc1r7ub', 
    api_key = '268289947187182', 
    api_secret = 'ITtRnnt18P-7oWVt5KoV8Hx2Hp4'
)

model = YOLO("best_50.pt")

def face_detection(uploaded_image):
    img_array = np.array(uploaded_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results = model(img_bgr, imgsz=640, conf=0.25)
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    return img_bgr

# Upload image to Cloudinary
def upload_to_cloudinary(image):
    response = cloudinary.uploader.upload(image)
    return response['secure_url']

def app():
    st.title("Webcam Face Detection App")

    webcam_image = st.camera_input("Take a picture", key="webcam_input")

    if webcam_image:
        image = Image.open(webcam_image)
        st.image(image, caption='Captured Image', use_column_width=True)

        detected_image = face_detection(image)

        # Save locally and upload to Cloudinary
        cv2.imwrite('detected_image.jpg', detected_image)
        cloudinary_url = upload_to_cloudinary('detected_image.jpg')
        st.image(cloudinary_url, caption='Detected Faces')

if __name__ == "__main__":
    app()
