import asyncio
import websockets
import cv2
import numpy as np
import base64

async def video_stream(websocket, path):
    video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        encoded_frame = base64.b64encode(frame_bytes).decode('utf-8')

        # Send frame data to WebSocket client
        await websocket.send(encoded_frame)

async def main():
    server = await websockets.serve(video_stream, '0.0.0.0', 8765)
    await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())
