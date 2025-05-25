from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11m.pt")
    max_epochs = 100
    patience = 3  # jumlah epoch berturut-turut menurun
    threshold = 0.9  # 90%
    best_acc = 0
    decrease_count = 0
    last_acc = None

    for epoch in range(max_epochs):
        results = model.train(
            data="rupiah2_dataset/data.yaml",
            epochs=100,
            imgsz=640,
            # resume=True,
            device="cuda",   # Ganti ke "cuda" jika GPU sudah terdeteksi
            batch=6        # Batch kecil agar hemat memori
        )
        # Ambil metrik akurasi, misal mAP50
        acc = results.metrics.get('mAP_0.5', 0)

        print(f"Epoch {epoch+1}: mAP50={acc:.4f}")

        if last_acc is not None and acc < last_acc:
            decrease_count += 1
        else:
            decrease_count = 0

        if acc >= threshold and decrease_count >= patience:
            print("Akurasi menurun terus hingga 90%, training dihentikan.")
            break

        last_acc = acc