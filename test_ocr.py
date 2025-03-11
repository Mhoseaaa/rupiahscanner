import cv2
import numpy as np
<<<<<<< Updated upstream
from gtts import gTTS
import os
=======
>>>>>>> Stashed changes
import pytesseract
from ultralytics import YOLO
from gtts import gTTS
import os
import time
from threading import Thread

<<<<<<< Updated upstream
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
=======
# Menyimpan prediksi terakhir untuk menghindari pengulangan suara
last_detected_label = None
last_speak_time = 0  # Timestamp terakhir AI membaca nominal

def speak_async(text):
    """Fungsi untuk membaca suara hanya sekali dengan gTTS"""
    global last_speak_time
    if time.time() - last_speak_time > 3:
        last_speak_time = time.time()
        tts = gTTS(text=text, lang="id")  # Bahasa Indonesia
        tts.save("detected.mp3")
        os.system("start detected.mp3")  # Untuk Windows

# Load model YOLOv8
model = YOLO(r"D:\UKDW\Semester4_2025\KecerdasanBuatan_Gloria\RupiahScanner\rupiahscannerV2\runs\detect\rupiah_finetuned\weights\best.pt")

# Konfigurasi OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Daftar kelas nominal uang
classes = ['dua puluh ribu rupiah', 'dua ribu rupiah', 'lima puluh ribu rupiah', 
           'lima ribu rupiah', 'sepuluh ribu rupiah', 'seratus ribu rupiah', 'seribu rupiah']
>>>>>>> Stashed changes

# Fungsi OCR
def detect_text(frame):
    """Fungsi OCR untuk membaca teks pada uang."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    return text.strip()

<<<<<<< Updated upstream
# Menggunakan kamera
=======
# Inisialisasi kamera
>>>>>>> Stashed changes
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
fps = 15
delay = 1 / fps

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

<<<<<<< Updated upstream
    # Deteksi uang dengan YOLO
    results = model.predict(frame)
    detected = False
=======
    # Deteksi uang dengan YOLO (Gunakan confidence lebih tinggi)
    results = model.predict(frame, conf=0.8)
    detected_label = None
>>>>>>> Stashed changes

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label = str(model.names[int(box.cls[0])])

<<<<<<< Updated upstream
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
=======
            if confidence > 0.8 and label in classes:
                detected_label = label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Bacakan suara hanya jika label baru terdeteksi
    if detected_label and detected_label != last_detected_label:
        last_detected_label = detected_label
        Thread(target=speak_async, args=(f"Terdeteksi uang {detected_label}",)).start()

    frame_count += 1

    # Tampilkan hasil
>>>>>>> Stashed changes
    cv2.imshow("Rupiah Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
