import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import base64

# Load the YOLO model
model = YOLO("best_50.pt")

# # Function to decode base64 to OpenCV image
# def base64_to_cv2_image(base64_str):
#     try:
#         img_bytes = base64.b64decode(base64_str.split(",")[1])
#         img_array = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#         return img
#     except Exception as e:
#         st.error(f"Error decoding base64 image: {e}")
#         return None

# Function to test if the model is running
# def test_model():
#     try:
#         # Create a dummy image (black square)
#         dummy_img = np.zeros((640, 480, 3), dtype=np.uint8)
#         results = model(dummy_img)
#         return True, "Model is running"
#     except Exception as e:
#         return False, f"Model is not running: {e}"

# Streamlit app interface
def app():
     """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    # """
    # source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(0)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

if __name__ == "__main__":
    app()
