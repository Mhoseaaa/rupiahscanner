from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    # Load model YOLOv8 (gunakan model pre-trained sebagai starting point)
    # Anda bisa menggunakan 'yolov8s.pt', 'yolov8m.pt', dll.

    # Train model
    results = model.train(
        data="D:\\UKDW\\Semester4_2025\\KecerdasanBuatan_Gloria\\RupiahScanner\\rupiahscanner\\rupiah_dataset\\rupiah.yaml", #sesuaikan path ke rupiah.yaml
        epochs=50,
        batch=8,
        imgsz=640,
        device="cuda" # Kalau ndak punya VGA ganti ke "cpu"
    )