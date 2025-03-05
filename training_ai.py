from ultralytics import YOLO

# Load model YOLOv8 (gunakan model pre-trained sebagai starting point)
model = YOLO("yolov8n.pt")  # Anda bisa menggunakan 'yolov8s.pt', 'yolov8m.pt', dll.

# Train model
results = model.train(
    data="D:\\AI\\rupiahscanner\\rupiah_dataset\\rupiah.yaml",
    epochs=50,
    batch=8,
    imgsz=640,
    device="cpu"
)