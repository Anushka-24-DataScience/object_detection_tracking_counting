import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import cvzone
import streamlit as st
from tracker import Tracker
import tempfile

# Load the YOLO model
model_path = r"C:\project_detect_track_count\sheet_detction_tracking_counting\weights\detection\best .pt"
model = YOLO(model_path)

def detect_and_count_sheets(frame):
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    detected_sheets = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        detected_sheets.append([x1, y1, x2, y2])
    
    bbox_idx = tracker.update(detected_sheets)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
        cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if cy1 - offset < cy < cy1 + offset:
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            if id not in sheetcount:
                sheetcount.append(id)

    cv2.line(frame, (2, 485), (1018, 485), (0, 0, 255), 2)  # for more accuracy I changed the line coordinates
    counting = len(sheetcount)
    cvzone.putTextRect(frame, f'Sheetcount: {counting}', (50, 60), 2, 2)
    
    return frame, counting

# Streamlit setup
st.set_page_config(
    page_title="Sheet Detection Tracking Counting Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Sheet Stack Counting")

# Sidebar for model config
st.sidebar.header("DL Model Config")

task_type = st.sidebar.selectbox("Select Task", ["Detection"])

confidence = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

# Sidebar for image/video config
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox("Select Source", ["Image", "Video", "Webcam"])

# Load the class list
with open("coco1.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

tracker = Tracker()
cy1 = 485
offset = 10
sheetcount = []

# Define functions for Streamlit actions
def infer_uploaded_image():
    uploaded_image = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp"))

    if uploaded_image is not None:
        image = np.array(cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Sheets in Image"):
            with st.spinner("Processing..."):
                frame, count = detect_and_count_sheets(image)
                st.image(frame, caption=f"Detected Sheets: {count}", use_column_width=True)

def infer_uploaded_video():
    uploaded_video = st.sidebar.file_uploader("Choose a video...")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1020, 600))
            frame, count = detect_and_count_sheets(frame)
            stframe.image(frame, channels="BGR")
        
        cap.release()

def infer_uploaded_webcam():
    vid_cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while vid_cap.isOpened():
        ret, frame = vid_cap.read()
        if not ret:
            break

        frame, count = detect_and_count_sheets(frame)
        stframe.image(frame, channels="BGR")
        
        if st.button("Stop Webcam"):
            break

    vid_cap.release()

# Call appropriate function based on user selection
if source_selectbox == "Image":
    infer_uploaded_image()
elif source_selectbox == "Video":
    infer_uploaded_video()
else:
    infer_uploaded_webcam()
