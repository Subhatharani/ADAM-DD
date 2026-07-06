# 🚗 ADAM-DD: Attention-based Driver Drowsiness Detection

Real-time driver drowsiness detection system using CNN with dual attention mechanisms, adaptive risk fusion, and live web dashboard.

---

## 🎯 Key Features

✅ **CNN with Dual Attention Mechanisms**
- Spatial Attention: Focus on WHERE drowsiness cues appear (eyes, facial posture)
- Channel Attention: Focus on WHAT features matter (important feature maps)

✅ **Dual Output Heads**
- Physical Drowsiness Indicators (eye closure, facial posture, head nod)
- Cognitive Fatigue Signals (microexpressions, attention lapses)

✅ **Adaptive Risk Fusion**
- Combines physical + cognitive signals with weighted averaging
- Reduces false positives while improving early detection

✅ **Real-Time Processing**
- 640×480 video at ~30 FPS
- Haar Cascade face detection + preprocessing
- Live web dashboard with annotated video stream

✅ **Comprehensive Monitoring**
- Real-time drowsiness/cognitive fatigue scores
- Alert history with timestamps
- FPS tracking and frame counting
- Color-coded status indicators (Green/Orange/Red)

---

## 📦 Files Included

```
ADAM-DD/
├── main.py                        # Flask app + real-time video processing
├── model.py                       # CNN with dual attention mechanisms
├── dataset_utils.py               # Dataset preparation & fine-tuning utilities
├── requirements.txt               # Python dependencies
├── dashboard.html                 # Web interface (move to templates/)
├── SETUP_AND_USAGE_GUIDE.md      # Complete setup guide
├── run_windows.bat                # Quick start for Windows
├── run_unix.sh                    # Quick start for Mac/Linux
└── README.md                      # This file
```

---

## ⚡ Quick Start

### Windows (PowerShell)
```powershell
# Option 1: Use quick start script
.\run_windows.bat

# Option 2: Manual setup
python -m venv adam_dd_env
.\adam_dd_env\Scripts\Activate.ps1
pip install -r requirements.txt
mkdir templates
# Move dashboard.html to templates/
python main.py
```

Then open: **http://localhost:5000**

### Mac/Linux
```bash
# Option 1: Use quick start script
chmod +x run_unix.sh
./run_unix.sh

# Option 2: Manual setup
python3 -m venv adam_dd_env
source adam_dd_env/bin/activate
pip install -r requirements.txt
mkdir templates
# Move dashboard.html to templates/
python main.py
```

Then open: **http://localhost:5000**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT ACQUISITION                          │
│                  (Webcam 640×480 @ 30FPS)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Face Detection│  │ Face Cropping│  │  Resize &    │           │
│  │ (Haar)       │→ │ (ROI Extract)│→ │  Normalize   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                              │                  │
│                                              ▼                  │
│                                    ┌──────────────────┐          │
│                                    │ CLAHE Enhancement│          │
│                                    │ (Normalize [0,1])│          │
│                                    └──────────────────┘          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│         FEATURE EXTRACTION: CNN WITH DUAL ATTENTION             │
│                                                                 │
│  ConvBlock1 ──┐                                                 │
│   (32 filters)│                                                 │
│               ▼                                                 │
│  ConvBlock2 ──┐                                                 │
│   (64 filters)│                                                 │
│               ▼                                                 │
│  ConvBlock3 ──┬─► [DUAL ATTENTION MODULE] ◄──┐                 │
│  (128 filters)│    ├─ Spatial Attention      │                 │
│               │    └─ Channel Attention      │                 │
│               ▼                              │                 │
│  ConvBlock4 ──┬─► [DUAL ATTENTION MODULE] ◄──┐                 │
│  (256 filters)│    ├─ Spatial Attention      │                 │
│               └────└─ Channel Attention      │                 │
│                                              │                 │
│                     Global Avg Pool          │                 │
│                     Dense (512)              │                 │
│                     Dense (256)              │                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              DECISION & FUSION                                  │
│                                                                 │
│     ┌─────────────────┐      ┌────────────────────┐             │
│     │ Drowsiness Head │      │ Cognitive Fatigue  │             │
│     │ (Physical)      │      │ Head (Microexpr.)  │             │
│     └────────┬────────┘      └────────┬───────────┘             │
│              │ Score1 (0-1)           │ Score2 (0-1)           │
│              │                        │                        │
│              └────────────┬───────────┘                        │
│                           │                                    │
│            Adaptive Fusion (w1=0.6, w2=0.4)                   │
│         FusedScore = 0.6×Score1 + 0.4×Score2                 │
│                           │                                    │
│                           ▼                                    │
│                  Fused Risk Score [0,1]                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              ALERT GENERATION                                   │
│                                                                 │
│  Score > 0.7  ──► 🚨 CRITICAL - DROWSY (RED)                  │
│  0.5 < Score ≤ 0.7  ──► ⚠️ WARNING (ORANGE)                    │
│  Score ≤ 0.5  ──► ✓ ALERT - NORMAL (GREEN)                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│         DATA LOGGING & DISPLAY                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐   │
│  │  Web Dashboard   │  │  Alert History   │  │  Annotations │   │
│  │  (Real-time UI)  │  │  (Timestamped)   │  │  (Live Video)│   │
│  └──────────────────┘  └──────────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Understanding the Attention Mechanisms

### Spatial Attention
```
Input Feature Map (H × W × C)
        │
        ▼
Channel-wise Avg Pool (1 × 1 × 1)
Channel-wise Max Pool (1 × 1 × 1)
        │
        ├─ Concatenate (1 × 1 × 2)
        │
        ▼
Conv2D (1×1, kernel_size=7)
        │
        ▼
Sigmoid Activation
        │
        ▼
Scale Feature Map Element-wise
        │
        ▼
Output: Focus on SPATIAL regions (eyes, mouth, head)
```

**Benefit**: Learns WHERE in the face drowsiness indicators appear

### Channel Attention
```
Input Feature Map (H × W × C)
        │
    ┌───┴───┐
    ▼       ▼
 Avg Pool  Max Pool (Global)
    │       │
    └───┬───┘
        ▼
   Dense (C/16, ReLU)
        │
    ┌───┴───┐
    ▼       ▼
Dense(C)  Dense(C)
    │       │
    └───┬───┘
        ▼
   Sigmoid Activation
        │
        ▼
Scale Feature Map Channel-wise
        │
        ▼
Output: Focus on IMPORTANT features
```

**Benefit**: Learns WHICH feature maps are most relevant for drowsiness

---

## 📊 Model Performance

### First Run (Random Initialization)
- Predictions will be ~0.5 (random)
- Good for testing system architecture
- Perfect for proof-of-concept demos

### Production Use
- Fine-tune on your dataset:
  - Alert faces: 500+ images
  - Drowsy faces: 500+ images
  - Various lighting, head poses, angles

### Expected Metrics (with fine-tuned model)
- Accuracy: 85-95%
- Sensitivity: 90%+ (detecting drowsiness)
- Specificity: 85%+ (avoiding false positives)
- FPS: 20-30 (depending on GPU)

---

## 🎮 Web Dashboard

### Real-Time Display
- **Live Video Stream**: Annotated with detection box, status, and confidence scores
- **Status Indicator**: Animated pulse (Green→Orange→Red)
- **Confidence Scores**: Drowsiness (0-1) and Cognitive Fatigue (0-1)
- **Metrics**: FPS, frames processed, alerts triggered

### Controls
- **▶ Start**: Begin detection
- **⏹ Stop**: Pause detection

### Alert History
- Last 20 alerts with timestamps
- Detailed metrics for each alert
- Color-coded by severity

---

## 🔄 Real-Time Processing Pipeline

```python
# Pseudocode of main processing loop

while camera_is_running:
    # 1. Capture frame (640×480)
    frame = camera.read()
    
    # 2. Detect & crop face (using Haar Cascade)
    face_roi, coords = detect_and_crop_face(frame)
    
    # 3. Preprocess (resize 224×224, normalize, enhance)
    processed_face = preprocess_frame(face_roi)
    
    # 4. Feature extraction (CNN + Dual Attention)
    features = model.extract_features(processed_face)
    
    # 5. Predict drowsiness & cognitive fatigue
    drowsiness_score = model.predict_drowsiness(features)
    cognitive_score = model.predict_cognitive(features)
    
    # 6. Adaptive fusion
    fused_score = 0.6 * drowsiness_score + 0.4 * cognitive_score
    
    # 7. Alert generation
    if fused_score > 0.7:
        alert = "CRITICAL - DROWSY"
    elif fused_score > 0.5:
        alert = "WARNING"
    else:
        alert = "NORMAL"
    
    # 8. Annotate & display
    annotated_frame = draw_annotations(frame, alert, scores, coords)
    display(annotated_frame)
    
    # 9. Log
    log_alert(drowsiness_score, cognitive_score, fused_score, alert)
```

---

## 📈 Fine-Tuning on Your Dataset

### Step 1: Organize Data
```
data/
├── alert/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── drowsy/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Step 2: Prepare Dataset
```python
from dataset_utils import DatasetPreparer

preparer = DatasetPreparer(target_size=224)
data = preparer.create_dataset('data/alert', 'data/drowsy')
preparer.save_dataset(data, 'dataset')
```

### Step 3: Train Model
```python
from dataset_utils import train_on_custom_dataset

model, history = train_on_custom_dataset('data/alert', 'data/drowsy')
```

### Step 4: Use Trained Model
Replace `drowsiness_model.h5` with your trained model.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Check system settings, try different camera index |
| Model loading fails | Delete `drowsiness_model.h5`, restart (creates new) |
| Low FPS / High latency | Reduce frame size, enable GPU, reduce model size |
| Port 5000 in use | Change port in `main.py`: `app.run(port=8000)` |
| Face not detected | Improve lighting, get closer to camera, ensure face is visible |

---

## 📱 API Reference

### GET /
Returns HTML dashboard

### GET /video_feed
MJPEG video stream

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
  }
]
```

### GET /api/start
Start detection

### GET /api/stop
Stop detection

---

## 📚 References

### Attention Mechanisms
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (Channel Attention)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)

### Drowsiness Detection
- [MPIIGaze Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation/)
- [YawDD Dataset](https://www.cse.iitb.ac.in/~abhijit/yawdd/)
- [NTHU Drowsy Driver](http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/)

---

## 🚀 Next Steps

1. ✅ Run the system with your webcam
2. 📊 Collect your own drowsiness dataset
3. 🧠 Fine-tune the model on real data
4. 📱 Deploy to mobile/edge devices
5. 🚗 Integrate with vehicle systems

---

## 📞 Support

For issues:
1. Check **SETUP_AND_USAGE_GUIDE.md** for detailed troubleshooting
2. Review code comments in `main.py` and `model.py`
3. Verify all dependencies: `pip list`
4. Check Flask accessibility: `curl http://localhost:5000`

---

## 📄 License

This project is provided for educational and research purposes.

---

## 🎉 Get Started

```bash
# Quick setup
python -m venv env
source env/bin/activate  # or .\env\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
mkdir templates
# Move dashboard.html to templates/
python main.py

# Open browser to http://localhost:5000
```

**Happy building! 🚗💡**
