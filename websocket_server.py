import streamlit as st
import asyncio
import websockets
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io

# Load YOLOv8 model
model = YOLO('path/to/your/yolov8_model.pt')  # Replace with your YOLOv8 model path

# WebSocket URI
WS_URI = "ws://localhost:8765"

async def receive_frame():
    async with websockets.connect(WS_URI) as websocket:
        while True:
            frame_bytes = await websocket.recv()
            yield frame_bytes

def detect_faces(frame_bytes):
    # Convert bytes to image
    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Perform face detection
    results = model(img)
    
    # Draw bounding boxes on the image
    img_with_boxes = results.render()[0]
    
    # Convert image with bounding boxes to bytes
    _, img_encoded = cv2.imencode('.jpg', img_with_boxes)
    img_bytes = img_encoded.tobytes()
    
    return img_bytes

def app():
    st.title("Real-Time Face Detection with YOLOv8")
    
    stframe = st.empty()
    
    # Run the asynchronous WebSocket receiver in a Streamlit-compatible way
    async def run_websocket():
        async for frame_bytes in receive_frame():
            img_bytes = detect_faces(frame_bytes)
            
            # Convert bytes to image
            img = Image.open(io.BytesIO(img_bytes))
            
            # Display the image
            stframe.image(img, channels="RGB")
    
    # Use Streamlit's ability to run asynchronous code
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_websocket())
