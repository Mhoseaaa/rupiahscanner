import cv2
import numpy as np
from gtts import gTTS
import os
import pytesseract
from ultralytics import YOLO
from threading import Thread
import time

# Inisialisasi text-to-speech dengan gTTS
def speak_async(text):
    def run():
        tts = gTTS(text=text, lang="id")
        tts.save("rupiah.mp3")
        os.system("start rupiah.mp3")  # Untuk Windows

    Thread(target=run, daemon=True).start()

# Load model YOLOv8 untuk deteksi uang
model = YOLO("best.pt")  # Ganti dengan model yang sudah dilatih untuk uang rupiah

# Konfigurasi OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Daftar uang yang dikenali
classes = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]

# Fungsi OCR
def detect_text(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    return text

# Menggunakan kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
fps = 30  # Desired FPS
delay = 1 / fps

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi uang dengan YOLO
    results = model.predict(frame)
    detected = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label = str(result.names[int(box.cls[0])])

            if confidence > 0.5 and label in classes and not detected:
                t = Thread(target=speak_async, args=(f"Terdeteksi uang {label} rupiah",))
                t.daemon = True
                t.start()
                detected = True

                # Gambar kotak deteksi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Jalankan OCR setiap 30 frame
    if not detected and frame_count % 30 == 0:
        text_detected = detect_text(frame)
        for nominal in classes:
            if nominal in text_detected:
                t = Thread(target=speak_async, args=(f"OCR mendeteksi uang {nominal} rupiah",))
                t.daemon = True
                t.start()
                cv2.putText(frame, nominal, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                break

    frame_count += 1

    # Tampilkan frame
    cv2.imshow("Rupiah Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
