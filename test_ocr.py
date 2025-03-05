import cv2
import numpy as np
import pyttsx3
import pytesseract
from ultralytics import YOLO
from threading import Thread
import time

# Inisialisasi text-to-speech
engine = pyttsx3.init()

def speak_async(text):
    engine.say(text)
    engine.runAndWait()

# Load model YOLOv8 untuk deteksi uang
model = YOLO("yolov8n.pt")  # Ganti dengan model yang sudah dilatih untuk uang rupiah


# Konfigurasi OCR
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

classes = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]

def detect_text(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    return text

# Menggunakan kamera
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cv2.ocl.setUseOpenCL(True)

frame_count = 0
fps = 30  # Desired FPS
delay = 1 / fps

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi uang dengan YOLO
    results = model(frame)
    detected = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label = str(result.names[int(box.cls[0])])

            if confidence > 0.5 and label in classes and not detected:
                Thread(target=speak_async, args=(f"Terdeteksi uang {label} rupiah",)).start()
                detected = True

                # Gambar kotak deteksi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Run OCR every 30 frames
    if not detected and frame_count % 30 == 0:
        text_detected = detect_text(frame)
        for nominal in classes:
            if nominal in text_detected:
                Thread(target=speak_async, args=(f"OCR mendeteksi uang {nominal} rupiah",)).start()
                cv2.putText(frame, nominal, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                break

    frame_count += 1
    # Display frame
    cv2.imshow("Rupiah Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()