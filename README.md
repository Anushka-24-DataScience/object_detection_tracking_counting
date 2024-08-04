
## Train the custom data
- First train custom data for object detection(I used YOLOV8) and for tracking (I used DeepSort) and download the trained weights in my case it is best.pt.
  ## Features
- Feature1: Object detection  tracking and counting task.
- Feature2: Multiple detection models. `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`,`best.pt`
- Feature3: Multiple input formats. `Image`, `Video`, `Webcam`
- Feature4: Multiple Object Tracking and Counting.

## Installation
### Create a virtual environment
```commandline
# create
conda creat -n your environment name python==3.9 or >

# activate
conda ctivate your environment name
```

### Clone repository
```commandline
git clone required ultralytics reporistory

```
## Setting the Directory.
- cd ultralytics/yolo/v8/detect 
- Downloading the DeepSORT Files
- After downloading the DeepSORT Zip file,unzip it go into the subfolders and place the deep_sort_pytorch folder into the yolo/v8/detect folder

### Install packages
```commandline
# Install requirements.txt
# Streamlit dependencies
pip install streamlit

# YOLOv8 dependecies
pip install -e '.[dev]'

```
### Download Pre-trained YOLOv8 Detection Weights
Create a directory named `weights` and create a subdirectory named `detection` and save the downloaded YOLOv8 object detection weights and deepsort weights inside this directory. 


## Run
```commandline
streamlit run app.py
```
Then will start the Streamlit server and open your web browser to the default Streamlit page automatically.
For Object Counting, you can choose "Video" from "Select Source" combo box and use "SampleVideo.mp3" inside videos folder as an example.Can Choose "Image" also.


  


