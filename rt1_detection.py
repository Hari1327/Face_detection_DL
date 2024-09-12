import streamlit as st
import asyncio
import websockets
from PIL import Image
import io

# WebSocket server URI
WS_URI = "ws://localhost:8765"

async def receive_frame():
    async with websockets.connect(WS_URI) as websocket:
        while True:
            frame_bytes = await websocket.recv()
            return frame_bytes

def get_frame():
    # Run the asynchronous receive_frame function in a synchronous context
    return asyncio.run(receive_frame())

def app():
    st.title("Real-Time Face Detection with YOLOv8")

    # Display the video feed
    stframe = st.empty()

    while True:
        frame_bytes = get_frame()
        
        # Convert bytes to image
        img = Image.open(io.BytesIO(frame_bytes))
        stframe.image(img, caption='Live Video Feed', use_column_width=True)

if __name__ == "__main__":
    app()
