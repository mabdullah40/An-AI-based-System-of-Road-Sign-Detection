from flask import Flask, request, render_template, jsonify
import cv2
import math
import cvzone
import numpy as np
from ultralytics import YOLO
import base64

app = Flask(__name__)

model = YOLO(r'Yolo_trained_model\weights\best.pt')
class_names = list(model.names.values())
sign_threshold = 0.2

def detect_road_signs(frame):
    results = model(frame, stream=True)
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = class_names[cls]

            if conf > sign_threshold:
                cvzone.cornerRect(frame, (x1, y1, w, h))
                cvzone.putTextRect(frame, f'{current_class} {conf}', (max(0, x1), max(35, y1 - 15)), scale=1, thickness=1)
                detections.append({
                    "class": current_class,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                })
    return frame, detections

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        frame, detections = detect_road_signs(img)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

        response = {
            "detections": detections,
            "frame": frame_base64
        }
        return jsonify(response), 200

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
