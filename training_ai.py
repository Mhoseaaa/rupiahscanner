from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(results):
    """Visualisasi metrik training"""
    metrics = results.results_dict
    epochs = np.arange(1, len(metrics['train/box_loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plot Precision-Recall
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['metrics/precision'], label='Precision', marker='o')
    plt.plot(epochs, metrics['metrics/recall'], label='Recall', marker='o')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot mAP
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['metrics/mAP50'], label='mAP50', marker='o')
    plt.plot(epochs, metrics['metrics/mAP50-95'], label='mAP50-95', marker='o')
    plt.title('mAP Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

class CustomTrainer(BaseTrainer):
    def on_train_batch_end(self):
        """Fungsi yang dipanggil setiap akhir batch training"""
        if self.epoch_iter % 10 == 0:  # Setiap 10 iterasi
            self.save_checkpoint()  # Simpan checkpoint

    def save_checkpoint(self):
        """Simpan checkpoint model"""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_iter{self.epoch_iter}.pt")
        self.model.save(checkpoint_path)
        print(f"Checkpoint disimpan di: {checkpoint_path}")

def optimize_for_low_vram():
    """Optimasi pengaturan untuk hardware terbatas"""
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    optimize_for_low_vram()
    
    # Inisialisasi model
    model = YOLO("yolov8n.pt")
    
    # Daftarkan custom trainer
    model.trainer = CustomTrainer
    
    # Training dengan parameter standar
    results = model.train(
        # Dataset
        data="D://AI/rupiahscanner/rupiah_dataset/rupiah.yaml",
        epochs=100,
        
        # Optimasi Memori
        batch=4,
        imgsz=640,
        workers=2,
        device="0",
        half=True,
        cache=True,
        
        # Optimasi Training
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=2,
        close_mosaic=10,
        
        # Augmentasi
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=15,
        translate=0.05,
        scale=0.3,
        shear=0.02,
        flipud=0.3,
        fliplr=0.3,
        
        # System
        save=True,
        save_period=1,  # Simpan checkpoint setiap epoch
        val=True,
        plots=True,
        exist_ok=True,
        verbose=True,
    )
    
    # Evaluasi dan Recall
    metrics = model.val()
    
    # Print metrik per kelas
    print("\n=== Metrik per Kelas ===")
    for i, name in model.names.items():
        print(f"Kelas {name}:")
        print(f"  Precision: {metrics.box.p[i]:.4f}")
        print(f"  Recall:    {metrics.box.r[i]:.4f}")
        print(f"  mAP50:     {metrics.box.ap50[i]:.4f}")
        print("----------------------")
    
    # Visualisasi
    plot_metrics(results)
    
    # Simpan log
    with open("training_report.txt", "w") as f:
        f.write(str(metrics.results_dict))
    
    # Export model
    model.export(format='tflite')