import cv2
from gtts import gTTS
import os
import time
from threading import Thread, Lock
from ultralytics import YOLO
import glob

# --- Bersihkan file mp3 temp lama saat program mulai ---
for f in glob.glob("rupiah_temp_*.mp3"):
    try:
        os.remove(f)
    except Exception:
        pass

# --- Konfigurasi Penting ---
# Ganti dengan PATH ABSOLUT atau RELATIF ke model YOLOv11 Anda
MODEL_PATH = r"D:\CLONE_GITHUB\rupiahscanner\runs\detect\train4\weights\best.pt"

# Daftar kelas/kode nominal uang (harus sesuai dengan urutan 'names' saat training)
CLASSES = ['1000', '10000', '100000', '2000', '20000', '5000', '50000']

# Ambang batas confidence untuk YOLO (0.0 – 1.0). 
# Anda bisa turunkan jika deteksi terlalu sedikit (misalnya 0.5 atau 0.6).
YOLO_CONFIDENCE_THRESHOLD = 0.8

# Waktu tunda (detik) sebelum membacakan teks yang sama lagi
SPEECH_DELAY_SECONDS = 3

# --- Variabel Global untuk Kontrol Suara ---
last_spoken_label = None   # Label uang terakhir yang diucapkan
last_spoken_time = 0       # Waktu (epoch) terakhir suara diucapkan
speech_lock = Lock()       # Lock agar tidak ada lebih dari satu thread TTS berjalan bersamaan

# --- Inisialisasi Model YOLOv11 ---
try:
    model = YOLO(MODEL_PATH)
    print(f"Model YOLOv11 berhasil dimuat dari: {MODEL_PATH}")
except Exception as e:
    print(f"Error saat memuat model YOLOv11: {e}")
    print("Pastikan MODEL_PATH sudah benar dan file model ada.")
    exit()

# --- Fungsi Text-to-Speech Asinkron ---
def speak_async(text: str):
    """
    Membuat thread yang memanggil gTTS untuk membacakan 'text'.
    Hanya dipanggil jika label berubah, atau sudah melewati SPEECH_DELAY_SECONDS.
    """
    global last_spoken_label, last_spoken_time

    # Coba ambil lock. Jika gagal, berarti ada TTS lain yang sedang berjalan; lewati.
    if not speech_lock.acquire(blocking=False):
        return

    try:
        now = time.time()
        # Jika label berbeda, atau label sama tetapi sudah melewati delay
        if (text != last_spoken_label) or ((text == last_spoken_label) and ((now - last_spoken_time) > SPEECH_DELAY_SECONDS)):
            last_spoken_label = text
            last_spoken_time = now

            print(f"[TTS] Mengucapkan: {text}")
            audio_file = f"rupiah_temp_{int(time.time()*1000)}.mp3"
            try:
                tts = gTTS(text=text, lang="id")
                tts.save(audio_file)

                # Memutar audio. Di Windows, 'start' akan mengeksekusi default player.
                os.system(f'start /min {audio_file}')
                # Beri jeda untuk memastikan file sempat diputar sebelum dihapus
                time.sleep(3)

            except Exception as e:
                print(f"Gagal memutar suara: {e}")
            finally:
                # Hapus file mp3 jika masih ada
                if os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                    except:
                        pass
    finally:
        speech_lock.release()

# --- Inisialisasi Kamera ---
cap = cv2.VideoCapture(0)  # Index 0: webcam default
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera. Pastikan kamera terhubung.")
    exit()

print("Kamera berhasil dibuka. Tekan 'q' untuk keluar.")

# --- Loop Deteksi Utama ---
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Flag untuk menandakan apakah YOLO mendeteksi sesuatu di frame ini
    yolo_detected_this_frame = False

    # --- Deteksi dengan YOLOv11 ---
    # model.predict mengembalikan daftar result (biasanya 1 result karena satu input frame)
    # conf = YOLO_CONFIDENCE_THRESHOLD agar hanya prediksi dengan confidence >= threshold
    results = model.predict(frame, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)

    for result in results:
        boxes = result.boxes  # Semua bounding box yang terdeteksi pada satu frame
        if boxes is None or len(boxes) == 0:
            continue

        # Ambil deteksi dengan confidence tertinggi
        best_idx = boxes.conf.argmax().item()
        box = boxes[best_idx]

        # Koordinat kotak
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        cls_index = int(box.cls[0])
        label = model.names[cls_index]  # Contoh: '5000'

        # Pastikan label sesuai dengan daftar CLASSES
        if label in CLASSES:
            yolo_detected_this_frame = True

            # Gambar bounding box dan label di frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Panggil speak_async di thread terpisah
            Thread(target=speak_async, args=(f"Terdeteksi uang {label} rupiah",)).start()

            # Hanya proses satu deteksi terbaik per frame
            break

    # --- Tampilkan Frame Hasil Deteksi ---
    cv2.imshow("Rupiah Detector (YOLOv11)", frame)

    # Kontrol FPS agar tidak terlalu cepat loop-nya
    elapsed = time.time() - start_time
    if elapsed < 1 / 15:
        time.sleep((1 / 15) - elapsed)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Pembersihan Akhir ---
cap.release()
cv2.destroyAllWindows()
print("Program selesai.")