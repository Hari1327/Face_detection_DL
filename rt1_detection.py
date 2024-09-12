import streamlit as st
import asyncio
import websockets
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import base64

# Load YOLOv8 model
model = YOLO('best_50.pt')  # Replace with your YOLOv8 model path

def detect_faces(image):
    # Perform face detection
    results = model(image)
    
    # Draw bounding boxes on the image
    img_with_boxes = results.render()[0]
    
    return img_with_boxes

async def receive_frame(websocket_uri):
    async with websockets.connect(websocket_uri) as websocket:
        while True:
            encoded_frame = await websocket.recv()
            frame_bytes = base64.b64decode(encoded_frame)
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            yield frame

def app():
    st.title("Real-Time Face Detection with YOLOv8")

    websocket_uri = "ws://localhost:8765"  # WebSocket server URL

    stframe = st.empty()

    # Create an asynchronous loop to fetch and process frames
    async def process_frames():
        async for frame in receive_frame(websocket_uri):
            # Convert frame (OpenCV) to PIL Image
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Detect faces
            img_with_boxes = detect_faces(frame)
            
            # Convert back to PIL Image
            img_pil = Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
            
            # Display the image
            stframe.image(img_pil, channels="RGB")

    # Run the asynchronous loop within Streamlit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_frames())

if __name__ == "__main__":
    app()
