import asyncio
import websockets
import cv2
import numpy as np
from ultralytics import YOLO
import base64

model = YOLO("best_50.pt")

async def process_frame(websocket, path):
    async for message in websocket:
        try:
            # Decode the base64 image
            _, img_data = message.split(",")
            img_bytes = base64.b64decode(img_data)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Perform face detection
            img_resized = cv2.resize(img, (1280, 720))
            results = model(img_resized, imgsz=640, conf=0.5)

            # Draw bounding boxes
            if not results.pandas().xyxy[0].empty:
                for _, row in results.pandas().xyxy[0].iterrows():
                    x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    confidence = row['confidence']
                    cv2.rectangle(img_resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img_resized, f'{confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the image to base64
            _, img_encoded = cv2.imencode('.jpg', img_resized)
            img_base64 = base64.b64encode(img_encoded.tobytes()).decode()
            await websocket.send(f"data:image/jpeg;base64,{img_base64}")

        except Exception as e:
            await websocket.send(f"Error: {e}")

start_server = websockets.serve(process_frame, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
