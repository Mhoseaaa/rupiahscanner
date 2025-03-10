from ultralytics import YOLO

def evaluate():
    # Muat model terbaik yang sudah dilatih
    model = YOLO("runs/detect/train14/weights/best.pt")

    # Evaluasi pada dataset validasi
    metrics = model.val()  # Hasil akan disimpan di runs/detect/val
    print(f"mAP50-95: {metrics.box.map}")  # Skor mAP
    print(f"mAP50: {metrics.box.map50}")   # Skor mAP@0.5
    print(f"mAP75: {metrics.box.map75}")   # Skor mAP@0.75

if __name__ == '__main__':
    evaluate()  # Pastikan pemanggilan fungsi dilakukan dengan benar
