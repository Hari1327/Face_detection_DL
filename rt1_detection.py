import streamlit as st
import numpy as np
import cv2
from PIL import Image
import asyncio
import websockets

st.title("Face Detection with WebSocket")

# WebSocket server URL
WEBSOCKET_URL = "ws://localhost:8765"

# Function to capture and send video frames
async def capture_frames():
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        while True:
            # Capture frame from webcam
            frame = st.camera_input("Capture")
            if frame:
                img_bytes = frame.read()
                base64_img = base64.b64encode(img_bytes).decode()
                
                # Send frame to WebSocket server
                await websocket.send(f"data:image/jpeg;base64,{base64_img}")

                # Receive processed frame from WebSocket server
                response = await websocket.recv()
                st.image(response, caption='Detected Faces', use_column_width=True)

# Button to start streaming
if st.button("Start Camera"):
    st.write("Streaming...")
    asyncio.run(capture_frames())
