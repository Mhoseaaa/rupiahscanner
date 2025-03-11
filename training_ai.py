from ultralytics import YOLO
<<<<<<< Updated upstream

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    # Load model YOLOv8 (gunakan model pre-trained sebagai starting point)
    # Anda bisa menggunakan 'yolov8s.pt', 'yolov8m.pt', dll.

    # Train model
    results = model.train(
        data="D:\\UKDW\Semester4_2025\\KecerdasanBuatan_Gloria\\RupiahScanner\\rupiahscanner\\rupiah2_dataset\\data.yaml", #sesuaikan path ke rupiah.yaml
        epochs=150,
        patience=20,
        batch=8,
        imgsz=640,
        device="cuda", # Kalau ndak punya VGA ganti ke "cpu"
        augment=True,
        name="rupiah_detector_v1"
    )
=======
import os
import torch

# Optimasi untuk GPU
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')


def train_model():
    # Gunakan model YOLOv8 pre-trained sebagai dasar training
    model = YOLO("yolov8n.pt")  # Bisa diganti dengan best.pt jika ingin fine-tuning

    # Training Model
    results = model.train(
        # Path dataset
        data=r"D:\UKDW\Semester4_2025\KecerdasanBuatan_Gloria\RupiahScanner\rupiahscannerV2\rupiah2_dataset\data.yaml",
        
        # Hyperparameter Training
        epochs=100,              # Jumlah epoch
        batch=8,                 # Batch size
        imgsz=640,               # Resolusi gambar
        device="cuda",           # Gunakan GPU
        name="rupiah_finetuned", # Nama hasil training
        patience=20,             # Early stopping jika tidak ada peningkatan dalam 20 epoch

        # Augmentasi
        hsv_h=0.015,    # Variasi hue
        hsv_s=0.7,      # Variasi saturasi
        hsv_v=0.4,      # Variasi kecerahan
        degrees=10,     # Rotasi kecil
        translate=0.1,  # Pergeseran kecil
        scale=0.5,      # Scaling agar model bisa mengenali dari jauh
        shear=0.01,     # Distorsi gambar kecil
        flipud=0.2,     # Flip atas bawah
        fliplr=0.5,     # Flip kiri kanan

        # Optimasi Hyperparameter
        lr0=0.001,      # Learning rate awal
        lrf=0.01,       # Learning rate final
        momentum=0.9,   # Momentum SGD
        weight_decay=0.0005,  # Regularisasi
        warmup_epochs=3,      # Waktu warm-up

        # Optimasi Memory
        cache="ram",   # Cache dataset ke RAM jika cukup besar
        workers=0,     # Hindari multiprocessing error di Windows

        # Simpan Model
        save=True,
        save_period=1,  # Simpan model setiap epoch
        val=True,       # Lakukan validasi setiap epoch
        plots=True,     # Simpan grafik hasil training
        exist_ok=True,
        verbose=True
    )

    # Evaluasi Model
    metrics = model.val()

    # Simpan log training
    with open("training_report.txt", "w") as f:
        f.write(str(metrics.results_dict))

    # Export Model ke ONNX untuk deployment
    model.export(format='onnx')

if __name__ == '__main__':
    train_model()
>>>>>>> Stashed changes
