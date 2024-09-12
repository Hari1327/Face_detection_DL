import asyncio
import websockets
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
import threading

# Load YOLOv8 model
model = YOLO('path/to/your/yolov8_model.pt')  # Replace with your YOLOv8 model path

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

def start_server():
    start_server = websockets.serve(detect_faces, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    # Run WebSocket server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.start()
