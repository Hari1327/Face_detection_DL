# main.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import time

def app():
    st.title("Real-Time Face Detection with WebSocket")

    # HTML and JavaScript for WebSocket connection
    websocket_html = """
        <script>
            let ws = new WebSocket("ws://localhost:8765");

            ws.onopen = function() {
                console.log("WebSocket connection opened");
            };

            ws.onmessage = function(event) {
                // Handle the received image
                let img = new Image();
                img.src = event.data;
                document.getElementById('image').src = img.src;
            };

            function sendFrame(base64Frame) {
                ws.send(base64Frame);
            }

            // Capture video frame and send to WebSocket
            function captureFrame() {
                let canvas = document.createElement('canvas');
                let video = document.getElementById('video');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                let context = canvas.getContext('2d');
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                let base64Frame = canvas.toDataURL('image/jpeg');
                sendFrame(base64Frame);
            }

            // Start capturing frames every second
            setInterval(captureFrame, 1000);
        </script>
        <video id="video" autoplay></video>
        <img id="image" />
    """

    st.components.v1.html(websocket_html, height=500)

if __name__ == "__main__":
    app()
