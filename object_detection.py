from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def say(text):
    engine.say(text)
    engine.runAndWait()
    

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0

def main():
    prev_frame_time = 0
    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                
                conf = math.ceil((box.conf[0] * 100)) / 100
                
                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    say(classNames[cls])

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__=='__main__':
    main()