from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    # Load model YOLOv8 (gunakan model pre-trained sebagai starting point)
    # Anda bisa menggunakan 'yolov8s.pt', 'yolov8m.pt', dll.

    # Train model
    results = model.train(
        data="C:\\Users\\itsho\\OneDrive\\Dokumen\\Semester 4\\rupiahscanner\\rupiah2_dataset\\data.yaml", #sesuaikan path ke rupiah.yaml
        epochs=150,
        patience=20,
        batch=8,
        imgsz=640,
        device="cuda", # Kalau ndak punya VGA ganti ke "cpu"
        augment=True,
        name="rupiah_detector_v1"
    )
