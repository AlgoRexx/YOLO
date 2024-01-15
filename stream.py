# Import required libraries
import tempfile
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

model_path = 'yolov8n.pt'

# Create a temporary directory to store uploaded files
temp_dir = tempfile.TemporaryDirectory()

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the YOLO model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Creating main page heading
st.title("Object Detection")
st.success("Model loaded successfully!", icon="👍")

# Creating sidebar
st.header("V I D E O  Config", divider="violet")
uploaded_video = st.file_uploader("Choose a video...", type=("mp4", "avi", "mov"))

# Model confidence slider
confi = float(st.slider("Select Model Confidence", 50, 100, 40)) / 100

# Display uploaded video and perform object detection
if uploaded_video is not None:
    video_path = temp_dir.name + "/" + uploaded_video.name

    with open(video_path, "wb") as video_file:
        video_file.write(uploaded_video.read())

    vid_cap = cv2.VideoCapture(video_path)
    vid_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = temp_dir.name + "/" + 'output.mp4'
    out = cv2.VideoWriter(output_path, fps=30, fourcc=fourcc, frameSize=(vid_width, vid_height))
    st_frame = st.empty()
    st_frame.info("Processing...")

    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = model.predict(image, conf=confi)
            result_tensor = res[0].boxes
            res_plotted = res[0].plot()
            st_frame.image(res_plotted, caption='Detected Video', use_column_width=True)
            out.write(cv2.cvtColor(np.array(res_plotted), cv2.COLOR_RGB2BGR))
        else:
            vid_cap.release()
            out.release()
            break
