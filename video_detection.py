import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

def process_video(video_path, model_path='best.pt'):
    # Load the YOLO model
    model = YOLO(model_path)

    # Capture the uploaded video
    video_capture = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a temporary file to save the output video
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_video.name

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame in the video
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Perform face detection using the YOLO model
        results = model(frame, imgsz=640)

        # Draw bounding boxes and confidence scores
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].int().tolist()
                confidence = box.conf[0].item()

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
                
                # Label with confidence score
                label = f'Confidence: {confidence:.2f}'
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)
    
    # Release everything if job is finished
    video_capture.release()
    out.release()

    return output_path

def app():
    st.title("Video Face Detection App")

    # Video upload
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video.write(uploaded_video.read())
        uploaded_video_path = temp_video.name

        # Display the original uploaded video
        st.video(uploaded_video_path, format='video/mp4')

        # Detect faces and save the processed video
        st.write("Processing video...")
        processed_video_path = process_video(uploaded_video_path)

        # Display the processed video
        st.write("Processed Video:")
        st.video(processed_video_path, format='video/mp4')

if __name__ == "__main__":
    app()
