import streamlit as st
from ultralytics import YOLO
import PIL
import os

# Set this to avoid any library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def text_detection(file):
    model = YOLO("best.pt")
    
    # Open and resize the image to 640x640
    uploaded_image = PIL.Image.open(file)
    resized_image = uploaded_image.resize((640, 640))  # Resize the image

    # Run the detection on the resized image
    res = model.predict(resized_image, conf=0.5, save=True)
    
    # Extract bounding boxes and plot the results
    box = res[0].boxes.xyxy.tolist()  # List of bounding boxes
    res_plotted = res[0].plot()[:, :, ::-1]  # Convert to RGB

    # Display the detection result and number of detections
    st.image(res_plotted, caption='Detections', use_column_width=True)
    st.write(f"Number of Detections: {len(box)}")
    
    return resized_image

def app():
    st.title("Upload the Image and Click on Detect Button")
    
    # File uploader
    file = st.file_uploader("Upload an Image", type=("jpg", "jpeg", "png"))
    if file is not None:
        st.image(image=file, caption='Uploaded Image', use_column_width=True)
    
    # Detect button
    if st.button("Detect"):
        if file is not None:
            text_detection(file)
        else:
            st.write("Please upload an image file.")

# Run the app
if __name__ == '__main__':
    app()
