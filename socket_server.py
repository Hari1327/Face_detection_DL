# websocket_server.py
import asyncio
import websockets
from rt1_detection import process_frame

async def main():
    async with websockets.serve(process_frame, "localhost", 8765):
        print("WebSocket server is running on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
