import sys
import cv2
import numpy as np
import pyttsx3
import pytesseract
from ultralytics import YOLO
from threading import Thread
import time
#INSTALL DULU DI CMD = pip install PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QWidget, QGroupBox, QTextEdit,
                            QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class RupiahDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rupiah Currency Detector")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.init_tts()
        self.init_model()
        self.init_camera()
        self.init_ui()
        
        # Detection variables
        self.frame_count = 0
        self.last_detection = ""
        self.detection_active = False
        
    def init_tts(self):
        """Initialize text-to-speech engine"""
        self.engine = pyttsx3.init()
        # Configure voice properties
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)  # Change index for different voice
        self.engine.setProperty('rate', 150)  # Speed of speech
        
    def init_model(self):
        """Initialize YOLO model and OCR configuration"""
        self.model = YOLO("yolov8n.pt")  # Replace with your trained model
        self.classes = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path to your Tesseract installation
        

    def init_camera(self):
        """Initialize camera with different backends"""
        self.cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            self.cap = cv2.VideoCapture(0, backend)
            if self.cap.isOpened():
                break
        
        if not self.cap or not self.cap.isOpened():
            self.show_error("Could not open camera. Please check your camera connection.")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cv2.ocl.setUseOpenCL(True)
        
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
        
        self.ocr_detections_label = QLabel("OCR Detections: 0")
        stats_layout.addWidget(self.ocr_detections_label)
        
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        
        main_layout.addLayout(right_panel)
        
        # Initialize counters
        self.total_detections = 0
        self.yolo_detections = 0
        self.ocr_detections = 0
    
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
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.log_text.append(f"[{time.strftime('%H:%M:%S')}] TTS Error: {str(e)}")
    
    def detect_text(self, frame):
        """Perform OCR on the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply some preprocessing for better OCR results
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(gray, config='--psm 6')
        return text
    
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
        detected = False
        
        # YOLO detection
        results = self.model(frame)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = str(result.names[int(box.cls[0])])
                
                if confidence > 0.5 and label in self.classes and not detected:
                    detection_info = {
                        'label': label,
                        'confidence': confidence,
                        'method': 'YOLO',
                        'position': (x1, y1, x2, y2)
                    }
                    detected = True
                    self.yolo_detections += 1
                    
                    # Draw detection box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 0), 2)
        
        # OCR detection (every 30 frames)
        if not detected and self.frame_count % 30 == 0:
            text_detected = self.detect_text(frame)
            for nominal in self.classes:
                if nominal in text_detected:
                    detection_info = {
                        'label': nominal,
                        'confidence': 0.8,  # OCR confidence estimate
                        'method': 'OCR',
                        'position': (50, 50)
                    }
                    detected = True
                    self.ocr_detections += 1
                    
                    # Draw OCR result
                    cv2.putText(frame, f"OCR: {nominal}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    break
        
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
            self.ocr_detections_label.setText(f"OCR Detections: {self.ocr_detections}")
            
            # Add to log
            log_entry = (f"[{time.strftime('%H:%M:%S')}] Detected {info['label']} Rupiah "
                        f"(Confidence: {info['confidence']:.2f}, Method: {info['method']})")
            self.log_text.append(log_entry)
            
            # Speak detection
            Thread(target=self.speak_async, 
                 args=(f"Terdeteksi uang {info['label']} rupiah",)).start()
    
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
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = RupiahDetectorApp()
    window.show()
    sys.exit(app.exec_())