# websocket_server.py
import asyncio
import websockets
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import io

model = YOLO("best.pt")

async def process_frame(websocket, path):
    async for message in websocket:
        # Decode base64 message
        try:
            base64_str = message.split(",")[1]
            img_bytes = base64.b64decode(base64_str)
            img_array = np.frombuffer(img_bytes, np.uint8)
            frame_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Perform face detection
            if frame_img is not None:
                results = model(frame_img)
                
                # Draw bounding boxes
                for result in results.pandas().xyxy[0].itertuples():
                    x_min, y_min, x_max, y_max = int(result.xmin), int(result.ymin), int(result.xmax), int(result.ymax)
                    confidence = result.confidence
                    if confidence > 0.5:
                        cv2.rectangle(frame_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame_img, f'{confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Convert image to base64
                _, buffer = cv2.imencode('.jpg', frame_img)
                base64_img = base64.b64encode(buffer).decode()
                await websocket.send(f"data:image/jpeg;base64,{base64_img}")

        except Exception as e:
            print(f"Error processing frame: {e}")

start_server = websockets.serve(process_frame, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
