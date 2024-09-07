import streamlit as st
import streamlit.components.v1 as components

st.title("Real-Time Webcam Stream")

# HTML + JavaScript to access the webcam
html_string = """
<!DOCTYPE html>
<html>
<head>
    <title>Webcam Stream</title>
</head>
<body>
    <video id="webcam" width="640" height="480" autoplay></video>
    <canvas id="canvas" width="640" height="480"></canvas>
    <script>
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    video.play();
                    setInterval(() => {
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        // You can send the canvas data to the server here if needed
                    }, 1000 / 30); // 30 FPS
                };
            })
            .catch(error => console.error('Error accessing webcam:', error));
    </script>
</body>
</html>
"""

# Embed the HTML + JavaScript into the Streamlit app
components.html(html_string, height=500)
