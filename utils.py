from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import numpy as np
import config

def _display_detected_frames(conf, model, st_count, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLO model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLO): An instance of the `YOLO` class containing the YOLO model.
    :param st_count (Streamlit object): A Streamlit object to display the count of detected sheets.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Predict the objects in the image using YOLO model
    res = model.predict(image, conf=conf)

    sheet_count = 0
    for detection in res[0].boxes:
        if detection.label == 'sheet':  # Assuming 'sheet' is the label used for sheets
            sheet_count += 1
    
    # Display the count of sheets
    st_count.write(f'Sheet Count: {sheet_count}')
    
    # Plot the detected objects on the video frame
    res_plotted = np.array(res[0].plot())
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)

@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def infer_uploaded_image(conf, model):
    """
    Execute inference for an uploaded image.
    :param conf: Confidence of YOLO model
    :param model: An instance of the `YOLO` class containing the YOLO model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # Adding the uploaded image to the page with caption
            st.image(
                image=uploaded_image,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                uploaded_image = np.array(uploaded_image)
                res = model.predict(uploaded_image, conf=conf)
                boxes = res[0].boxes
                res_plotted = np.array(res[0].plot())[:, :, ::-1]

                with col2:
                    st.image(res_plotted, caption="Detected Image", use_column_width=True)
                    sheet_count = 0
                    with st.expander("Detection Results"):
                        for box in boxes:
                            if box.label == 'sheet':  # Assuming 'sheet' is the label used for sheets
                                sheet_count += 1
                            st.write(box.xywh)
                    st.write(f"Sheet Count: {sheet_count}")

def infer_uploaded_video(conf, model):
    """
    Execute inference for an uploaded video.
    :param conf: Confidence of YOLO model
    :param model: An instance of the `YOLO` class containing the YOLO model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    config.OBJECT_COUNTER1 = None
                    config.OBJECT_COUNTER = None
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(tfile.name)
                    st_count = st.empty()
                    st_frame = st.empty()
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf, model, st_count, st_frame, image)
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error processing video: {e}")

def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam input.
    :param conf: Confidence of YOLO model
    :param model: An instance of the `YOLO` class containing the YOLO model.
    :return: None
    """
    try:
        flag = st.button(label="Stop running")
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_count = st.empty()
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf, model, st_count, st_frame, image)
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error with webcam feed: {str(e)}")

if __name__ == "__main__":
    st.title("YOLO Sheet Counter")
    model_path = st.text_input("Enter model path", value="C:\project_detect_track_count\sheet_detction_tracking_counting\weights\detection\best.pt")
    confidence = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

    model = load_model(model_path)

    st.sidebar.title("Upload and Detect")
    infer_type = st.sidebar.selectbox("Select Inference Type", ("Image", "Video", "Webcam"))

    if infer_type == "Image":
        infer_uploaded_image(confidence, model)
    elif infer_type == "Video":
        infer_uploaded_video(confidence, model)
    else:
        infer_uploaded_webcam(confidence, model)
