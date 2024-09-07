import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

def process_frame(frame, model):
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
    
    return frame

def app():
    st.title("Real-Time Face Detection App")

    # Load YOLO model
    model = YOLO('yolov8n.pt')  # Path to your YOLO model

    # Start the webcam
    st.write("Starting webcam...")
    cap = cv2.VideoCapture(0)  # 0 to use default webcam
    
    stframe = st.empty()  # Placeholder to display video frames

    # Process video frames in real-time
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break
        
        # Process the frame to detect faces
        processed_frame = process_frame(frame, model)

        # Convert color (BGR to RGB)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Display the processed frame in Streamlit
        stframe.image(processed_frame, channels="RGB", use_column_width=True)

        # Exit if the user presses "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app()
