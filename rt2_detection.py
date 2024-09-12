# main.py
import streamlit as st
import base64
from PIL import Image
import numpy as np
import cv2

def app():
    st.title("Real-Time Face Detection with WebSocket")

    # WebSocket HTML and JavaScript for webcam and sending frames
    websocket_html = """
        <script>
            // Initialize WebSocket connection
            const ws = new WebSocket("ws://localhost:8765");

            ws.onopen = () => console.log("WebSocket connection opened");

            ws.onmessage = (event) => {
                // Update image source with the received data
                const img = new Image();
                img.src = event.data;
                document.getElementById('image').src = img.src;
            };

            function sendFrame(base64Frame) {
                ws.send(base64Frame);
            }

            async function startCamera() {
                try {
                    const video = document.getElementById('video');
                    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    video.srcObject = stream;

                    setInterval(() => {
                        const canvas = document.createElement('canvas');
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        const context = canvas.getContext('2d');
                        context.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const base64Frame = canvas.toDataURL('image/jpeg');
                        sendFrame(base64Frame);
                    }, 1000);  // Adjust frame capture interval as needed
                } catch (error) {
                    console.error('Error accessing the webcam: ', error);
                }
            }

            startCamera();
        </script>
        <video id="video" autoplay playsinline></video>
        <img id="image" />
    """

    st.components.v1.html(websocket_html, height=500)

if __name__ == "__main__":
    app()
