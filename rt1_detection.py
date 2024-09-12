import streamlit as st
import streamlit.components.v1 as components

def app():
    st.title("Real-Time Face Detection with YOLO")

    # Add a slider for confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%.2f"
    )

    # WebSocket URL (ensure the WebSocket server is running)
    ws_url = "ws://localhost:8765"

    st.write(f"**Confidence Threshold:** {confidence_threshold}")

    # HTML and JavaScript for video capture and WebSocket communication
    video_html = f"""
        <video id="webcam" autoplay playsinline></video>
        <canvas id="canvas"></canvas>
        <script>
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');
            const ws = new WebSocket("{ws_url}");
            ws.onmessage = function(event) {
                const img = new Image(),
                img.src = event.data,
                img.onload = function() {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    context.drawImage(img, 0, 0);
                }
            }

            async function startCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    video.srcObject = stream;
                    video.play();
                    captureFrame();
                } catch (error) {
                    console.error('Error accessing the webcam: ', error);
                }
            }

            function captureFrame() {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const frameData = canvas.toDataURL('image/jpeg');
                ws.send(frameData);
                setTimeout(captureFrame, 100);  // Adjust frame capture interval as needed
            }

            startCamera();
        </script>
    """

    # Render the HTML video capture
    components.html(video_html, height=400)

if __name__ == "__main__":
    app()
