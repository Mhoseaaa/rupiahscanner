import cv2
import numpy as np
import pyttsx3
import pytesseract
import albumentations as A
from ultralytics import YOLO

# Inisialisasi text-to-speech
engine = pyttsx3.init()

def speak(text):
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
cap = cv2.VideoCapture(0)

while True:
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
            
            if confidence > 0.5 and label in classes:
                speak(f"Terdeteksi uang {label} rupiah")
                detected = True
                
                # Gambar kotak deteksi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if not detected:
        text_detected = detect_text(frame)
        for nominal in classes:
            if nominal in text_detected:
                speak(f"OCR mendeteksi uang {nominal} rupiah")
                cv2.putText(frame, nominal, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                break
    
    cv2.imshow("Rupiah Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

