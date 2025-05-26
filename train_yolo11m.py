from ultralytics import YOLO
import time

if __name__ == "__main__":
    model = YOLO("yolo11m.pt") # Menggunakan model medium

    max_epochs = 300 # Tingkatkan jumlah epoch total
    patience = 20 # Tingkatkan kesabaran: jumlah epoch berturut-turut mAP menurun sebelum berhenti
    # threshold = 0.9 # Hapus atau gunakan untuk logging saja, jangan untuk early stopping utama

    # Variabel untuk early stopping
    best_mAP = 0.0
    decrease_count = 0
    
    print("Memulai pelatihan model YOLOv11...")

    # Panggil model.train() sekali saja dengan jumlah epoch total
    results = model.train(
        data="rupiah2_dataset/data.yaml",
        epochs=max_epochs, # Jumlah epoch total
        imgsz=640,
        # resume=True, # Aktifkan jika ingin melanjutkan pelatihan dari checkpoint terakhir
        device="cuda", # Pastikan ini "cuda" jika GPU terdeteksi, atau "cpu"
        batch=6,       # Sesuaikan jika VRAM memungkinkan
        # val=True, # Pastikan validasi berjalan untuk memantau metrik
        # save_period=10, # Simpan checkpoint setiap 10 epoch
        # project="rupiah_detector_training", # Nama folder project
        # name="yolov11_rupiah_v1" # Nama run
    )

    # Setelah pelatihan selesai, Anda bisa mengakses metrik dari results.metrics
    # Atau, jika Anda ingin early stopping manual di luar loop train (tidak disarankan untuk Ultralytics)
    # Anda bisa memparsing output log Ultralytics atau menggunakan callbacks.

    # Catatan: Ultralytics memiliki early stopping bawaan yang bisa diatur via parameter 'patience'
    # di dalam model.train(). Jika Anda ingin menggunakan early stopping bawaan Ultralytics,
    # Anda bisa menghapus logika early stopping manual Anda dan cukup tambahkan:
    # results = model.train(..., patience=50) # Akan berhenti jika mAP tidak meningkat selama 50 epoch

    print("Pelatihan selesai.")
    print(f"Metrik akhir: {results.metrics}")

    # Jika Anda ingin memantau dan menghentikan secara manual (tidak disarankan untuk Ultralytics,
    # lebih baik gunakan fitur patience bawaan):
    # Untuk memantau metrik selama pelatihan, Anda perlu menggunakan callbacks atau
    # memparsing output log yang dicetak oleh Ultralytics saat pelatihan.
    # Contoh di atas adalah cara yang lebih umum untuk menjalankan pelatihan penuh.
