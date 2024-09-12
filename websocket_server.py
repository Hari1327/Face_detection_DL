import asyncio
import websockets
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO

# Load YOLOv8 model
model = YOLO('best_50.pt')  # Replace with your YOLOv8 model path

async def detect_faces(websocket, path):
    async for message in websocket:
        # Convert the received bytes to an image
        nparr = np.frombuffer(message, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform face detection
        results = model(img)
        
        # Draw bounding boxes on the image
        img_with_boxes = results.render()[0]
        
        # Convert image with bounding boxes to bytes
        _, img_encoded = cv2.imencode('.jpg', img_with_boxes)
        img_bytes = img_encoded.tobytes()
        
        # Send the image back to the client
        await websocket.send(img_bytes)

async def main():
    server = await websockets.serve(detect_faces, "localhost", 8765)
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
