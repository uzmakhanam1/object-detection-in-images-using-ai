import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

# Title
st.title("🔍 YOLOv8 Object Detection App")
st.write("Upload an image to perform object detection using a pre-trained YOLOv8 model.")

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("\nProcessing...")

    # Convert image to numpy array
    image_np = np.array(image)

    # Perform object detection
    results = model.predict(source=image_np, show=False)

    # Draw bounding boxes on the image
    annotated_image = results[0].plot()

    # Save annotated image to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_file.name, annotated_image)

    # Display the output image
    st.image(annotated_image, caption='Detected Objects', use_column_width=True)

    # Option to download the annotated image
    with open(temp_file.name, "rb") as file:
        btn = st.download_button(
            label="Download Annotated Image",
            data=file,
            file_name="output_image.jpg",
            mime="image/jpeg"
        )