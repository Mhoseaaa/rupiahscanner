import cv2
from gtts import gTTS
import os
import pytesseract
from ultralytics import YOLO
import time
from threading import Thread, Lock
import glob

# --- Bersihkan file mp3 temp lama saat program mulai ---
for f in glob.glob("rupiah_temp_*.mp3"):
    try:
        os.remove(f)
    except Exception:
        pass

# --- Konfigurasi Penting ---
# Ganti dengan PATH ABSOLUT atau RELATIF ke model YOLO Anda (YOLOv8 atau YOLOv11)
# Contoh: r"C:\Users\itsho\OneDrive\Dokumen\Semester 4\rupiahscanner\runs\detect\rupiah_detector_v19\weights\best.pt"
MODEL_PATH = r"C:\Users\itsho\OneDrive\Dokumen\Semester 4\rupiahscanner\runs\detect\train\weights\best.pt" 

# Konfigurasi Tesseract OCR
# Ganti dengan PATH ABSOLUT ke tesseract.exe Anda
TESSERACT_CMD_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Daftar uang yang dikenali (pastikan ini sesuai dengan kelas di model YOLO Anda)
CLASSES = ['dua puluh ribu rupiah', 'dua ribu rupiah', 'lima puluh ribu rupiah', 'lima ribu rupiah', 'sepuluhribu rupiah', 'seratusribu rupiah', 'seribu rupiah']
# Ambang batas keyakinan untuk deteksi YOLO (0.0 - 1.0)
YOLO_CONFIDENCE_THRESHOLD = 0.9 

# Waktu tunda (dalam detik) untuk pengulangan suara yang sama
SPEECH_DELAY_SECONDS = 3 

# --- Inisialisasi Pytesseract ---
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
except Exception as e:
    print(f"Error: Tidak dapat menemukan Tesseract OCR di '{TESSERACT_CMD_PATH}'.")
    print("Pastikan Tesseract terinstal dan path-nya benar.")
    exit()

# --- Inisialisasi Model YOLO ---
try:
    model = YOLO(MODEL_PATH)
    print(f"Model YOLO berhasil dimuat dari: {MODEL_PATH}")
except Exception as e:
    print(f"Error saat memuat model YOLO: {e}")
    print("Pastikan MODEL_PATH sudah benar dan file model ada.")
    exit()

# --- Variabel Global untuk Kontrol Suara dan Deteksi ---
last_spoken_label = None # Label uang terakhir yang diucapkan
last_spoken_time = 0     # Waktu terakhir suara diucapkan
speech_lock = Lock()     # Untuk mencegah beberapa thread suara berjalan bersamaan

# --- Fungsi Text-to-Speech Asinkron ---
def speak_async(text):
    global last_spoken_time, last_spoken_label

    if speech_lock.acquire(blocking=False):
        current_time = time.time()
        if (text != last_spoken_label) or (text == last_spoken_label and (current_time - last_spoken_time) > SPEECH_DELAY_SECONDS):
            last_spoken_label = text
            last_spoken_time = current_time
            print(f"Mengucapkan: {text}")

            audio_file = f"rupiah_temp_{int(time.time()*1000)}.mp3"
            try:
                tts = gTTS(text=text, lang="id")
                tts.save(audio_file)
                print("Memutar suara...")
                os.system(f'start {audio_file}')
                time.sleep(4)  # Tunggu agar suara selesai diputar
            except Exception as e:
                print(f"Gagal memutar suara: {e}")
            finally:
                # Coba hapus file, jika gagal tidak masalah
                try:
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                except Exception:
                    pass
        speech_lock.release()
    else:
        pass  # Suara sedang diputar, dilewati

# --- Fungsi OCR ---
def detect_text(frame):
    """Fungsi OCR untuk membaca teks pada uang."""
    # Konversi ke grayscale untuk Tesseract
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Anda bisa mencoba preprocessing tambahan seperti thresholding atau denoising
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Konfigurasi Tesseract: --psm 6 untuk single block of text
    # --oem 3 untuk default engine mode
    text = pytesseract.image_to_string(gray, config='--psm 6 --oem 3') 
    return text.strip()

# --- Inisialisasi Kamera ---
cap = cv2.VideoCapture(0) # 0 untuk webcam default
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera. Pastikan kamera terhubung dan tidak digunakan aplikasi lain.")
    exit()

print("Kamera berhasil dibuka. Tekan 'q' untuk keluar.")

# --- Variabel untuk Kontrol Frame Rate dan OCR ---
frame_count = 0
TARGET_FPS = 15
FRAME_DELAY = 1 / TARGET_FPS # Delay per frame untuk mencapai target FPS
OCR_FRAME_INTERVAL = 30 # Jalankan OCR setiap 30 frame

# --- Loop Deteksi Utama ---
while True:
    start_time = time.time() # Waktu mulai pemrosesan frame

    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil frame dari kamera.")
        break

    yolo_detected_this_frame = False # Flag untuk menandakan apakah YOLO mendeteksi sesuatu di frame ini

    # --- Deteksi Uang dengan YOLO ---
    # verbose=False untuk tidak menampilkan log deteksi di konsol
    results = model.predict(frame, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # Ambil deteksi dengan confidence tertinggi
            best_box_idx = boxes.conf.argmax()
            box = boxes[best_box_idx]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label = str(model.names[int(box.cls[0])])

            # Pastikan label ada dalam daftar kelas yang dikenali
            if label in CLASSES:
                # Panggil speak_async di thread terpisah
                Thread(target=speak_async, args=(f"Terdeteksi uang {label}",)).start()
                yolo_detected_this_frame = True

                # Gambar kotak deteksi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Hanya proses deteksi YOLO terbaik per frame untuk suara
                break 

    # --- Jalankan OCR jika YOLO tidak mendeteksi dan sesuai interval ---
    if not yolo_detected_this_frame and frame_count % OCR_FRAME_INTERVAL == 0:
        text_detected = detect_text(frame)
        print(f"OCR mendeteksi teks: '{text_detected}'") # Untuk debugging OCR
        for nominal in CLASSES:
            if nominal in text_detected:
                Thread(target=speak_async, args=(f"OCR mendeteksi uang {nominal} rupiah",)).start()
                # Gambar teks yang terdeteksi OCR
                cv2.putText(frame, f"OCR: {nominal}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                break # Hanya deteksi nominal pertama oleh OCR

    frame_count += 1

    # --- Tampilkan Frame dan Kontrol FPS ---
    cv2.imshow("Rupiah Detector", frame)

    # Hitung waktu yang dibutuhkan untuk memproses frame
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Jeda untuk mencapai target FPS
    if elapsed_time < FRAME_DELAY:
        time.sleep(FRAME_DELAY - elapsed_time)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Pembersihan ---
cap.release()
cv2.destroyAllWindows()
print("Program selesai.")
