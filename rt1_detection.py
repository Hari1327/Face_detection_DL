# rt1_detection.py
import cv2
import numpy as np
import base64
import websockets
import asyncio

# Dummy face detection function (Replace with your actual model)
def dummy_face_detection(img):
    # Here you would use your model to detect faces and draw bounding boxes
    # For demonstration, we'll just return the image as is
    return img

async def process_frame(websocket, path):
    async for message in websocket:
        try:
            # Decode base64 to OpenCV image
            img_bytes = base64.b64decode(message.split(",")[1])
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Perform detection
            img = dummy_face_detection(img)

            # Encode the processed image back to base64
            _, buffer = cv2.imencode('.jpg', img)
            encoded_img = base64.b64encode(buffer).decode('utf-8')

            # Send back the processed image
            await websocket.send(f"data:image/jpeg;base64,{encoded_img}")
        except Exception as e:
            print(f"Error processing frame: {e}")

def start_websocket_server():
    start_server = websockets.serve(process_frame, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

def app():
    print("WebSocket server started")
    start_websocket_server()
