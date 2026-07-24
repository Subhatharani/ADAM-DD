# ADAM-DD: Attention-based Driver Anomaly Monitoring – Drowsiness Detection
## Complete Setup & Usage Guide
##hello there 
---

## 📋 PROJECT OVERVIEW

**ADAM-DD** is a real-time driver drowsiness detection system that combines:
- **CNN with Dual Attention Mechanisms** (Spatial + Channel Attention)
- **Real-time Video Processing** via webcam
- **Dual Output Heads** for comprehensive fatigue detection:
  - Physical drowsiness indicators (eye closure, facial posture)
  - Cognitive fatigue signals (microexpressions)
- **Adaptive Risk Fusion** combining both outputs
- **Flask-based Web Dashboard** for live monitoring

---

## 🔧 INSTALLATION

### Step 1: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv adam_dd_env
.\adam_dd_env\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python3 -m venv adam_dd_env
source adam_dd_env/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter issues with TensorFlow:
```bash
pip install tensorflow --upgrade
```

### Step 3: Project Structure

Organize files as follows:
```
project/
├── main.py                 (Main Flask application)
├── model.py               (CNN + Attention mechanisms)
├── requirements.txt       (Dependencies)
├── templates/
│   └── dashboard.html     (Web interface)
└── drowsiness_model.h5    (Trained model - auto-created on first run)
```

**Create templates folder:**
```powershell
mkdir templates
```

Then move `dashboard.html` into the `templates/` folder.

---

## 🚀 RUNNING THE APPLICATION

### Basic Usage

1. **Navigate to project directory:**
```bash
cd path\to\project  # Windows
cd path/to/project  # Mac/Linux
```

2. **Activate virtual environment:**
```powershell
.\adam_dd_env\Scripts\Activate.ps1  # Windows
source adam_dd_env/bin/activate      # Mac/Linux
```

3. **Run the application:**
```bash
python main.py
```

You'll see:
```
Initializing ADAM-DD System...
Drowsiness detector initialized
Video processor initialized
System initialization complete!
Starting Flask server on http://localhost:5000
Open your browser and navigate to http://localhost:5000
```

4. **Open in browser:**
- Navigate to: `http://localhost:5000`
- Allow camera access when prompted
- System will start real-time detection

---

## 📊 SYSTEM ARCHITECTURE

### 1. INPUT ACQUISITION
- Real-time webcam capture (640x480)
- Frame processing at ~30 FPS

### 2. PREPROCESSING
- **Face Detection:** Haar Cascade Classifier
- **Face Cropping:** Extract ROI
- **Normalization:** Pixel values to [0, 1]
- **Enhancement:** CLAHE (Contrast Limited Adaptive Histogram Equalization)

### 3. CNN FEATURE EXTRACTION WITH DUAL ATTENTION
```
Input (224×224×3)
    ↓
[ConvBlock 1] → MaxPool → Dropout
    ↓
[ConvBlock 2] → MaxPool → Dropout
    ↓
[ConvBlock 3] → [Dual Attention] → MaxPool → Dropout
    ↓
[ConvBlock 4] → [Dual Attention] → MaxPool → Dropout
    ↓
Global Average Pooling
    ↓
Dense Layers (512 → 256)
    ↓
┌─────────────────┬──────────────────┐
│                 │                  │
Head 1: Drowsiness    Head 2: Cognitive Fatigue
(Physical Signs)      (Microexpressions)
│                 │                  │
└─────────────────┴──────────────────┘
```

#### Spatial Attention Module
- Focuses on **WHERE** drowsiness cues appear
- Analyzes eye region, facial posture, head position
- Channel-wise mean & max pooling → Conv2D → Sigmoid

#### Channel Attention Module
- Focuses on **WHAT** features matter
- Identifies important feature maps
- Global pooling → Dense layers → Sigmoid

### 4. DECISION & FUSION
- **Drowsiness Score:** Output from physical indicators head (0-1)
- **Cognitive Fatigue:** Output from cognitive signals head (0-1)
- **Adaptive Fusion:** Weighted average (0.6 × drowsiness + 0.4 × cognitive)

### 5. ALERT GENERATION
- **Score > 0.7:** 🚨 CRITICAL - DROWSY (Red)
- **Score 0.5-0.7:** ⚠️ WARNING (Orange)
- **Score < 0.5:** ✓ ALERT - NORMAL (Green)

### 6. DATA LOGGING
- Timestamp-based alerts
- Score history with 100-alert buffer
- Real-time metrics (FPS, frame count)

---

## 🎮 WEB DASHBOARD FEATURES

### Real-Time Metrics
- **Live Video Stream:** Annotated with detection results
- **Drowsiness Score:** Visual bar + numerical value
- **Cognitive Fatigue:** Visual bar + numerical value
- **FPS:** Processing speed
- **Frames Processed:** Total count
- **Alerts Triggered:** Critical event count

### Alert Status Indicator
- **Animated pulse indicator** showing system status
- **Color-coded:** Green (normal) → Orange (warning) → Red (critical)
- **Status message** describing current state

### Alert History Log
- **Last 20 alerts** with timestamps
- **Detailed metrics:** Drowsiness, Cognitive, Risk scores
- **Color-coded entries:** Normal, Warning, Critical
- **Auto-scrolling:** New alerts appear at bottom

### Control Buttons
- **▶ Start:** Begin detection
- **⏹ Stop:** Pause detection

---

## 📈 UNDERSTANDING THE MODEL

### Why Dual Attention?

1. **Spatial Attention** = "Look here"
   - Weights different facial regions differently
   - Eyes → High importance for drowsiness
   - Mouth → Moderate importance for yawning
   - Overall posture → Secondary indicators

2. **Channel Attention** = "Focus on this"
   - Weights feature maps by importance
   - Some filters detect eye closure better
   - Others detect facial expressions
   - System learns which to prioritize

### Why Dual Outputs?

**Physical Drowsiness Indicators:**
- Eye closure (PERCLOS - % of eye closure)
- Blink frequency & duration
- Eyelid droop
- Head nod/tilt
- Facial asymmetry

**Cognitive Fatigue Signals:**
- Subtle microexpressions
- Facial muscle tension changes
- Reaction time patterns (from video patterns)
- Attention lapses (eye fixation changes)

### Why Adaptive Fusion?

Combines complementary signals:
- Physical indicators alone: False positives (blinking, yawning)
- Cognitive indicators alone: Harder to detect early
- Fusion: Better accuracy and earlier detection

---

## 🎯 MODEL PERFORMANCE EXPECTATIONS

### On First Run
- Model creates randomly initialized weights
- Scores will be random (~0.5 on average)
- Good for **proof-of-concept** and testing architecture

### Production Use
- Fine-tune on your dataset:
  - Alert faces: ~500 images
  - Drowsy faces: ~500 images
  - Various lighting, head poses, camera angles

Example fine-tuning:
```python
from model import train_drowsiness_model
import numpy as np

# Prepare your data
train_images = np.load('train_images.npy')  # (N, 224, 224, 3)
train_drowsiness_labels = np.load('train_drowsiness.npy')  # (N, 1)
train_cognitive_labels = np.load('train_cognitive.npy')  # (N, 1)

# Same for validation data
val_images = np.load('val_images.npy')
val_drowsiness_labels = np.load('val_drowsiness.npy')
val_cognitive_labels = np.load('val_cognitive.npy')

# Train
model, history = train_drowsiness_model(
    train_images, train_drowsiness_labels, train_cognitive_labels,
    val_images, val_drowsiness_labels, val_cognitive_labels,
    epochs=50, batch_size=32
)

# Save
model.save('drowsiness_model.h5')
```

---

## 🔍 TESTING & DEBUGGING

### Test Real-Time Detection Locally (Without Flask)

Create `test_detection.py`:
```python
import cv2
import numpy as np
from model import DrowsinessDetector
from main import detect_and_crop_face, preprocess_frame, VideoProcessor

# Initialize
detector = DrowsinessDetector()
processor = VideoProcessor(detector)

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process
    annotated, drowsiness, fused = processor.process_frame(frame)
    
    # Display
    cv2.imshow('ADAM-DD Detection', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Run:
```bash
python test_detection.py
```

### Test Model Directly

```python
from model import DrowsinessDetector
import numpy as np

detector = DrowsinessDetector()

# Test on random input
test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
predictions = detector.predict(test_input)

print(f"Drowsiness: {predictions['drowsiness'][0, 0]:.4f}")
print(f"Cognitive Fatigue: {predictions['cognitive_fatigue'][0, 0]:.4f}")
```

---

## 🐛 TROUBLESHOOTING

### Issue: Camera not detected
**Solution:**
- Check if webcam is enabled in system settings
- Try different camera index in `initialize_camera(camera_index=1)`
- Test with: `cv2.VideoCapture(0)` directly

### Issue: Model loading fails
**Solution:**
- Delete existing `drowsiness_model.h5` file
- System will create new model on restart
- First run will be slower (building model)

### Issue: High latency / Low FPS
**Solutions:**
1. Reduce input frame size (change `640×480` to `320×240`)
2. Use GPU acceleration (ensure CUDA installed for GPU TensorFlow)
3. Reduce model complexity (fewer conv layers)

### Issue: SSL/Certificate errors with Flask
**Solution:**
```bash
pip install --upgrade certifi
```

### Issue: Port 5000 already in use
**Solution:**
Change port in `main.py`:
```python
app.run(debug=True, port=8000)  # Use 8000 instead
```

---

## 📝 CODE STRUCTURE OVERVIEW

### main.py (Flask Application)
```
- SystemState: Global state management
- load_cascade_classifiers(): Load Haar cascades
- detect_and_crop_face(): Face detection
- preprocess_frame(): Image normalization
- VideoProcessor: Real-time frame processing
- Flask Routes: /video_feed, /api/metrics, etc.
```

### model.py (Deep Learning)
```
- SpatialAttention: Spatial attention layer
- ChannelAttention: Channel attention layer
- DualAttentionModule: Combined attention
- build_attention_cnn(): Model architecture
- DrowsinessDetector: High-level API
- train_drowsiness_model(): Training function
```

### dashboard.html (Frontend)
```
- Real-time video streaming display
- Animated status indicators
- Metrics cards with bar charts
- Alert history log
- Control buttons
- CSS animations & styling
```

---

## 📱 API ENDPOINTS

### GET /
Returns HTML dashboard

### GET /video_feed
MJPEG stream of annotated video frames

### GET /api/metrics
```json
{
  "drowsiness_score": 0.23,
  "cognitive_fatigue": 0.15,
  "alert_status": "ALERT - NORMAL",
  "alert_triggered": false,
  "fps": 28.5,
  "frame_count": 1543
}
```

### GET /api/alert_history
```json
[
  {
    "timestamp": "2024-01-15T14:30:45.123456",
    "drowsiness_score": 0.75,
    "cognitive_fatigue": 0.68,
    "fused_score": 0.72,
    "status": "CRITICAL - DROWSY"
  },
  ...
]
```

### GET /api/system_status
```json
{
  "running": true,
  "total_frames_processed": 1543,
  "total_alerts_triggered": 3,
  "current_fps": 28.5,
  "timestamp": "2024-01-15T14:30:50.123456"
}
```

### GET /api/start
Start detection system

### GET /api/stop
Stop detection system

---

## 🎓 LEARNING RESOURCES

### Understanding Attention Mechanisms
- Read about "Squeeze-and-Excitation Networks" (Channel Attention)
- Read about "CBAM: Convolutional Block Attention Module" (Spatial Attention)

### CNN Architecture Improvements
- Try MobileNet for faster inference
- Try EfficientNet for better accuracy/speed tradeoff
- Use transfer learning from pretrained ImageNet models

### Drowsiness Detection Datasets
- MPIIGaze (head pose)
- YawDD (yawning detection)
- NTHU Drowsy Driver (specific for drowsiness)
- CEW (Closed Eye in the Wild)

---

## 📞 SUPPORT

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in main.py and model.py
3. Verify all dependencies are installed: `pip list`
4. Check Flask is accessible: `curl http://localhost:5000`

---

## 🚀 NEXT STEPS

1. **Test the system** with your webcam
2. **Collect your own dataset** for fine-tuning
3. **Train the model** on real drowsy/alert samples
4. **Deploy** to edge devices (Raspberry Pi, NVIDIA Jetson)
5. **Integrate** with vehicle systems (CAN bus, etc.)

Enjoy building with ADAM-DD! 🎉
