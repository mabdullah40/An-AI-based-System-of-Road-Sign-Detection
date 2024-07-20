from ultralytics import YOLO
import cv2
import math
import cvzone
import numpy as np

model = YOLO(r'E:\Personal_Projects\Road_Sign_detection\Work\train3\weights\best.pt')

class_names = list(model.names.values())
print(class_names)

cap = cv2.VideoCapture(0)
cap.set(3, 720) 
cap.set(4, 720)

damage_threshold = 0.2

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            current_class = class_names[cls]
            
            if conf > damage_threshold:
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{current_class} {conf}', (max(0, x1), max(35, y1 - 15)), scale=1, thickness=1)
    
    cv2.imshow('Image', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
