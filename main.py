import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import cvzone
from tracker import Tracker

# Load the YOLO model
model_path = r"C:\project_detect_track_count\sheet_detction_tracking_counting\weights\detection\best .pt"
model = YOLO(model_path)

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(r'C:\project_detect_track_count\SampleVideo.mp4')
with open("coco1.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

tracker = Tracker()
cy1 = 485
offset = 10

sheetcount = []

while True:
    ret, frame = cap.read()
    if not ret:
       break
    frame = cv2.resize(frame, (1020, 600))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
    
        list.append([x1, y1, x2, y2])
    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
        cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if cy1 < (cy + offset) and cy1 > (cy - offset):
           cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
           cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
           if sheetcount.count(id) == 0:
              sheetcount.append(id)

    cv2.line(frame, (2, 485), (1018, 485), (0, 0, 255), 2)  # for more accuracy I changed the line coordinates

    counting = len(sheetcount)
    cvzone.putTextRect(frame, f'Sheetcount: {counting}', (50, 60), 2, 2)
    
    cv2.imshow("RGB", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
