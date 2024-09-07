import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

# Load the YOLO model
model = YOLO("best.pt")  # Make sure the path to your model is correct

# Function to perform face detection on video frames
def face_detection_video(video_file):
    # Create a VideoCapture object to read from the video file
    cap = cv2.VideoCapture(video_file)

    # Create a temporary file to save the output video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to save the video with detections
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_file.name, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize the frame for YOLO input
        frame_resized = cv2.resize(frame, (640, 640))

        # Perform face detection
        results = model(frame_resized, imgsz=640)

        # Draw bounding boxes and confidence scores
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].int().tolist()
                confidence = box.conf[0].item()

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

                # Prepare the text to display
                label = f'Confidence: {confidence:.2f}'
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame with detections to the output video
        out.write(frame)

    # Release everything when done
    cap.release()
    out.release()

    return temp_file.name

# Streamlit app to upload and detect faces in video
def app():
    st.title("Upload a video and Detect Faces")

    # Video uploader
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if video_file:
        # Display the uploaded video
        st.video(video_file)

        if st.button("Detect Faces"):
            # Save the uploaded video to a temporary location
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(video_file.read())

            # Perform face detection
            output_video_path = face_detection_video(temp_video.name)

            # Display the output video with detections
            st.video(output_video_path)

# Run the app
if __name__ == "__main__":
    app()
