import cv2
from gtts import gTTS
import os
import time
from threading import Thread, Lock
from ultralytics import YOLO
import glob
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QWidget, QGroupBox, QTextEdit,
                            QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

# --- Clean up old temp mp3 files on startup ---
for f in glob.glob("rupiah_temp_*.mp3"):
    try:
        os.remove(f)
    except Exception:
        pass

class RupiahDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rupiah Currency Detector")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.init_config()
        self.init_model()
        self.init_camera()
        self.init_ui()
        
        # Detection variables
        self.frame_count = 0
        self.last_detection = ""
        self.detection_active = False
        self.last_spoken_time = 0
        self.last_spoken_label = None
        self.speech_lock = Lock()
        
        # Statistics
        self.total_detections = 0
        self.yolo_detections = 0
        self.ocr_detections = 0
        
    def init_config(self):
        """Initialize configuration"""
        # List of recognized currency classes
        self.CLASSES = ['1000', '10000', '100000', '2000', '20000', '5000', '50000']
        
        # Thresholds and intervals
        self.YOLO_CONFIDENCE_THRESHOLD = 0.82

        self.SPEECH_DELAY_SECONDS = 3
        
    def init_model(self):
        """Initialize YOLO model"""
        self.MODEL_PATH = r"D:/Kuliah/Sem 4/AI/rupiahscannerNew/runs/detect/train4/weights/best.pt"
        try:
            self.model = YOLO(self.MODEL_PATH)
            print(f"Model YOLO berhasil dimuat dari: {self.MODEL_PATH}")
        except Exception as e:
            self.show_error(f"Error saat memuat model YOLO: {e}\nPastikan MODEL_PATH sudah benar dan file model ada.")
            sys.exit()
    
    def init_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Tidak dapat membuka kamera. Pastikan kamera terhubung dan tidak digunakan aplikasi lain.")
            sys.exit()
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start video timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Camera and controls
        left_panel = QVBoxLayout()
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        left_panel.addWidget(self.camera_label)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Detection", self)
        self.start_btn.setStyleSheet("QPushButton { padding: 10px; font-weight: bold; }")
        self.start_btn.clicked.connect(self.start_detection)
        
        self.stop_btn = QPushButton("Stop Detection", self)
        self.stop_btn.setStyleSheet("QPushButton { padding: 10px; font-weight: bold; }")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        left_panel.addLayout(control_layout)
        
        main_layout.addLayout(left_panel)
        
        # Right panel - Information and logs
        right_panel = QVBoxLayout()
        
        # Detection information group
        info_group = QGroupBox("Detection Information")
        info_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        info_layout = QVBoxLayout()
        
        self.detection_label = QLabel("No detection yet")
        self.detection_label.setStyleSheet("font-size: 18px; color: #2c3e50;")
        self.detection_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.detection_label)
        
        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setStyleSheet("font-size: 14px;")
        info_layout.addWidget(self.confidence_label)
        
        self.method_label = QLabel("Method: -")
        self.method_label.setStyleSheet("font-size: 14px;")
        info_layout.addWidget(self.method_label)
        
        info_group.setLayout(info_layout)
        right_panel.addWidget(info_group)
        
        # Detection log group
        log_group = QGroupBox("Detection Log")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas; font-size: 12px;")
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)
        
        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        stats_layout = QVBoxLayout()
        
        self.total_detections_label = QLabel("Total Detections: 0")
        stats_layout.addWidget(self.total_detections_label)
        
        self.yolo_detections_label = QLabel("YOLO Detections: 0")
        stats_layout.addWidget(self.yolo_detections_label)
        
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        
        main_layout.addLayout(right_panel)
    
    def start_detection(self):
        """Start the detection process"""
        self.detection_active = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] Detection started")
    
    def stop_detection(self):
        """Stop the detection process"""
        self.detection_active = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] Detection stopped")
    
    def speak_async(self, text):
        """Text-to-speech in a separate thread"""
        if self.speech_lock.acquire(blocking=False):
            current_time = time.time()
            if (text != self.last_spoken_label) or (text == self.last_spoken_label and 
                                                  (current_time - self.last_spoken_time) > self.SPEECH_DELAY_SECONDS):
                self.last_spoken_label = text
                self.last_spoken_time = current_time
                
                audio_file = f"rupiah_temp_{int(time.time()*1000)}.mp3"
                try:
                    tts = gTTS(text=text, lang="id")
                    tts.save(audio_file)
                    os.system(f'start {audio_file}')
                    time.sleep(4)  # Wait for speech to finish
                except Exception as e:
                    self.log_text.append(f"[{time.strftime('%H:%M:%S')}] TTS Error: {str(e)}")
                finally:
                    try:
                        if os.path.exists(audio_file):
                            os.remove(audio_file)
                    except Exception:
                        pass
            self.speech_lock.release()
    
    def update_frame(self):
        """Update the camera frame and perform detection"""
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.log_text.append(f"[{time.strftime('%H:%M:%S')}] Failed to capture frame")
            return
        
        if self.detection_active:
            # Process detection
            frame, detection_info = self.process_detection(frame)
            
            # Update UI with detection info
            if detection_info:
                self.update_detection_info(detection_info)
        
        # Display frame
        self.display_image(frame)
    
    def process_detection(self, frame):
        """Process frame for currency detection"""
        detection_info = None
        
        # YOLO detection
        results = self.model.predict(frame, conf=self.YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                best_box_idx = boxes.conf.argmax()
                box = boxes[best_box_idx]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = str(self.model.names[int(box.cls[0])])
                
                if label in self.CLASSES:
                    detection_info = {
                        'label': label,
                        'confidence': confidence,
                        'method': 'YOLO',
                        'position': (x1, y1, x2, y2)
                    }
                    self.yolo_detections += 1
                    
                    # Draw detection box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 0), 2)
                    
                    # Speak detection
                    Thread(target=self.speak_async, 
                         args=(f"Terdeteksi uang {label} rupiah",)).start()
        
        self.frame_count += 1
        return frame, detection_info
    
    def update_detection_info(self, info):
        """Update UI with detection information"""
        if self.last_detection != info['label']:
            self.last_detection = info['label']
            self.total_detections += 1
            
            # Update detection info
            self.detection_label.setText(f"Detected: {info['label']} Rupiah")
            self.confidence_label.setText(f"Confidence: {info['confidence']:.2f}")
            self.method_label.setText(f"Method: {info['method']}")
            
            # Update statistics
            self.total_detections_label.setText(f"Total Detections: {self.total_detections}")
            self.yolo_detections_label.setText(f"YOLO Detections: {self.yolo_detections}")
            
            # Add to log
            log_entry = (f"[{time.strftime('%H:%M:%S')}] Detected {info['label']} Rupiah "
                        f"(Confidence: {info['confidence']:.2f}, Method: {info['method']})")
            self.log_text.append(log_entry)
    
    def display_image(self, frame):
        """Convert OpenCV image to QPixmap and display in QLabel"""
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.camera_label.setPixmap(pixmap.scaled(
                self.camera_label.width(), 
                self.camera_label.height(), 
                Qt.KeepAspectRatio
            ))
        except Exception as e:
            self.log_text.append(f"[{time.strftime('%H:%M:%S')}] Display Error: {str(e)}")
    
    def show_error(self, message):
        """Show error message dialog"""
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setText(message)
        error_box.setWindowTitle("Error")
        error_box.exec_()
    
    def closeEvent(self, event):
        """Cleanup when window is closed"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        
        # Clean up temp audio files
        for f in glob.glob("rupiah_temp_*.mp3"):
            try:
                os.remove(f)
            except Exception:
                pass
        
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = RupiahDetectorApp()
    window.show()
    sys.exit(app.exec_())