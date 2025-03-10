from ultralytics import YOLO
import cv2

# Muat model
model = YOLO("runs/detect/train14/weights/best.pt")

# Deteksi pada gambar
results = model("test_image.jpg")

# Tampilkan hasil
results[0].show()  # Tampilkan gambar dengan bounding box
results[0].save("hasil_deteksi.jpg")  # Simpan hasil