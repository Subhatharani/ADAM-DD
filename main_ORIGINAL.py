"""
ADAM-DD: Attention-based Driver Anomaly Monitoring – Drowsiness Detection
Real-time video streaming with CNN + Dual Attention Mechanisms
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, Response, jsonify
import threading
from collections import deque
from datetime import datetime
import logging

# Import the drowsiness detection model
from model import DrowsinessDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global state management
class SystemState:
    def __init__(self):
        self.detector = None
        self.frame_buffer = deque(maxlen=30)
        self.alert_history = deque(maxlen=100)
        self.is_running = False
        self.drowsiness_score = 0
        self.cognitive_fatigue = 0
        self.alert_status = "ALERT"
        self.fps = 0
        self.frame_count = 0
        self.alert_triggered = False
        self.alert_timestamp = None
        
    def log_alert(self, drowsiness_score, cognitive_fatigue, fused_score, status):
        """Log alert with timestamp"""
        self.alert_history.append({
            'timestamp': datetime.now().isoformat(),
            'drowsiness_score': float(drowsiness_score),
            'cognitive_fatigue': float(cognitive_fatigue),
            'fused_score': float(fused_score),
            'status': status
        })

state = SystemState()

# =============================================================================
# FACE DETECTION & PREPROCESSING
# =============================================================================

def load_cascade_classifiers():
    """Load Haar cascade classifiers"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    return face_cascade, eye_cascade

face_cascade, eye_cascade = load_cascade_classifiers()

def detect_and_crop_face(frame):
    """
    Detect face using Haar Cascade and return cropped region
    Returns: (cropped_face, face_coords)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
    
    if len(faces) > 0:
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = frame[y:y+h, x:x+w]
        return face_roi, (x, y, w, h)
    return None, None

def preprocess_frame(face_roi, target_size=224):
    """
    Preprocess face region:
    - Resize to target size
    - Normalize pixel values
    - Apply histogram equalization for robustness
    """
    if face_roi is None:
        return None
    
    # Resize
    resized = cv2.resize(face_roi, (target_size, target_size))
    
    # Histogram equalization (CLAHE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to BGR and normalize
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    normalized = enhanced_bgr.astype(np.float32) / 255.0
    
    return normalized

# =============================================================================
# REAL-TIME VIDEO PROCESSING
# =============================================================================

class VideoProcessor:
    def __init__(self, detector):
        self.detector = detector
        self.cap = None
        self.running = False
        
    def initialize_camera(self, camera_index=0):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.running = True
        logger.info("Camera initialized successfully")
        
    def process_frame(self, frame):
        """
        Process single frame:
        1. Detect & crop face
        2. Preprocess
        3. Extract features
        4. Calculate drowsiness scores
        5. Adaptive risk fusion
        """
        start_time = cv2.getTickCount()
        
        # Face detection & cropping
        face_roi, face_coords = detect_and_crop_face(frame)
        if face_roi is None:
            return frame, None, None
        
        # Preprocessing
        processed_face = preprocess_frame(face_roi)
        if processed_face is None:
            return frame, None, None
        
        # CNN Feature Extraction + Dual Attention
        predictions = self.detector.predict(processed_face[np.newaxis, ...])
        drowsiness_score = float(predictions['drowsiness'][0, 0])
        cognitive_fatigue = float(predictions['cognitive_fatigue'][0, 0])
        
        # Adaptive Risk Fusion
        fused_score = self.adaptive_fusion(drowsiness_score, cognitive_fatigue)
        
        # Determine alert status
        if fused_score > 0.7:
            status = "CRITICAL - DROWSY"
            alert_triggered = True
            color = (0, 0, 255)  # Red
        elif fused_score > 0.5:
            status = "WARNING - FATIGUE DETECTED"
            alert_triggered = False
            color = (0, 165, 255)  # Orange
        else:
            status = "ALERT - NORMAL"
            alert_triggered = False
            color = (0, 255, 0)  # Green
        
        # Update state
        state.drowsiness_score = drowsiness_score
        state.cognitive_fatigue = cognitive_fatigue
        state.alert_status = status
        state.alert_triggered = alert_triggered
        state.log_alert(drowsiness_score, cognitive_fatigue, fused_score, status)
        
        # Draw annotations
        x, y, w, h = face_coords
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, status, (x, y-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw confidence scores
        info_y = 40
        cv2.putText(frame, f"Drowsiness: {drowsiness_score:.3f}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Cognitive Fatigue: {cognitive_fatigue:.3f}", (10, info_y+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Fused Risk Score: {fused_score:.3f}", (10, info_y+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # FPS calculation
        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        fps = 1 / elapsed if elapsed > 0 else 0
        state.fps = fps
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, info_y+75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        return frame, drowsiness_score, fused_score
    
    @staticmethod
    def adaptive_fusion(drowsiness, cognitive_fatigue, w1=0.6, w2=0.4):
        """
        Adaptive Risk Fusion:
        Combines physical drowsiness indicators and cognitive fatigue
        with weighted average (can be adaptive based on confidence)
        """
        fused = (w1 * drowsiness) + (w2 * cognitive_fatigue)
        return min(fused, 1.0)  # Clip to [0, 1]
    
    def generate_frames(self):
        """Generator for streaming frames"""
        self.initialize_camera()
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, drowsiness, fused = self.process_frame(frame)
            
            # Encode frame for streaming
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            state.frame_count += 1
    
    def stop(self):
        """Stop video processing"""
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("Video processing stopped")

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """Stream video frames"""
    return Response(video_processor.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/metrics')
def get_metrics():
    """Get current system metrics"""
    return jsonify({
        'drowsiness_score': state.drowsiness_score,
        'cognitive_fatigue': state.cognitive_fatigue,
        'alert_status': state.alert_status,
        'alert_triggered': state.alert_triggered,
        'fps': state.fps,
        'frame_count': state.frame_count
    })

@app.route('/api/alert_history')
def get_alert_history():
    """Get alert history"""
    return jsonify(list(state.alert_history))

@app.route('/api/system_status')
def system_status():
    """Get overall system status"""
    total_alerts = len([a for a in state.alert_history if 'DROWSY' in a['status']])
    
    return jsonify({
        'running': state.is_running,
        'total_frames_processed': state.frame_count,
        'total_alerts_triggered': total_alerts,
        'current_fps': state.fps,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/start')
def start_detection():
    """Start drowsiness detection"""
    state.is_running = True
    return jsonify({'status': 'started'})

@app.route('/api/stop')
def stop_detection():
    """Stop drowsiness detection"""
    state.is_running = False
    return jsonify({'status': 'stopped'})

# =============================================================================
# INITIALIZATION & MAIN
# =============================================================================

def initialize_system():
    """Initialize detection model and video processor"""
    global video_processor
    
    logger.info("Initializing ADAM-DD System...")
    try:
        # Load/create model
        detector = DrowsinessDetector()
        logger.info("Drowsiness detector initialized")
        
        # Initialize video processor
        video_processor = VideoProcessor(detector)
        logger.info("Video processor initialized")
        
        state.detector = detector
        state.is_running = True
        
        logger.info("System initialization complete!")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise

if __name__ == '__main__':
    initialize_system()
    
    # Run Flask app
    logger.info("Starting Flask server on http://localhost:5000")
    logger.info("Open your browser and navigate to http://localhost:5000")
    
    try:
        app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        video_processor.stop()
        cv2.destroyAllWindows()
