# ADAM-DD: Quick Reference Cheat Sheet

## ⚡ 60-Second Setup

### Windows PowerShell
```powershell
# 1. Create env
python -m venv adam_dd_env

# 2. Activate
.\adam_dd_env\Scripts\Activate.ps1

# 3. Install deps
pip install -r requirements.txt

# 4. Setup
mkdir templates
# Move dashboard.html to templates/

# 5. Run
python main.py

# 6. Open browser
# http://localhost:5000
```

### Mac/Linux
```bash
python3 -m venv adam_dd_env
source adam_dd_env/bin/activate
pip install -r requirements.txt
mkdir templates
# Move dashboard.html to templates/
python main.py
# http://localhost:5000
```

---

## 📋 File Structure (Required)

```
project_folder/
├── main.py                    ✓ REQUIRED
├── model.py                   ✓ REQUIRED
├── requirements.txt           ✓ REQUIRED
├── templates/
│   └── dashboard.html         ✓ REQUIRED (move here)
├── dataset_utils.py           (optional - for fine-tuning)
├── test_setup.py              (optional - for verification)
└── drowsiness_model.h5        (auto-created on first run)
```

---

## 🚀 Running the App

### Option 1: Windows Batch Script
```
Double-click: run_windows.bat
```

### Option 2: Manual Commands
```bash
python main.py
```

### Option 3: Test First
```bash
python test_setup.py
```

---

## 🔧 Key Commands Reference

| Task | Command |
|------|---------|
| Create virtual env | `python -m venv adam_dd_env` |
| Activate (Windows) | `.\adam_dd_env\Scripts\Activate.ps1` |
| Activate (Mac/Linux) | `source adam_dd_env/bin/activate` |
| Install deps | `pip install -r requirements.txt` |
| Run app | `python main.py` |
| Test setup | `python test_setup.py` |
| View model | `python -c "from model import DrowsinessDetector; DrowsinessDetector().get_model_summary()"` |
| Deactivate env | `deactivate` |

---

## 🎯 Expected Output

### After running `python main.py`:
```
Initializing ADAM-DD System...
Drowsiness detector initialized
Video processor initialized
System initialization complete!
Starting Flask server on http://localhost:5000
Open your browser and navigate to http://localhost:5000
```

### In Browser:
- Video stream from webcam
- Real-time drowsiness score (0.0 - 1.0)
- Real-time cognitive fatigue score (0.0 - 1.0)
- Alert status (Green/Orange/Red)
- FPS counter

---

## 🐛 Instant Troubleshooting

| Problem | Fix |
|---------|-----|
| "python not found" | Install Python 3.8+ from python.org |
| "No module named 'tensorflow'" | Run `pip install -r requirements.txt` |
| "Camera not working" | Check system settings, try different index in code |
| "Port 5000 in use" | Change in main.py: `app.run(port=8000)` |
| "dashboard.html not found" | Move file to `templates/` folder |
| "Model loading failed" | Delete `drowsiness_model.h5`, restart |

---

## 🎨 Dashboard Features

### Metrics Displayed
- **Drowsiness Score**: 0-1 (physical indicators)
- **Cognitive Fatigue**: 0-1 (microexpressions)
- **Fused Risk**: Weighted combination
- **FPS**: Processing speed
- **Alert Status**: Normal/Warning/Critical
- **Alert History**: Last 20 events

### Controls
- ▶ Start detection
- ⏹ Stop detection

---

## 📊 Understanding Scores

```
Score Range  │ Status        │ Color  │ Action
─────────────┼───────────────┼────────┼──────────────────
0.0 - 0.5    │ NORMAL        │ Green  │ Continue driving
0.5 - 0.7    │ WARNING       │ Orange │ Attention needed
0.7 - 1.0    │ CRITICAL      │ Red    │ Take immediate break
```

---

## 🧠 Model Overview

```
Input: 224×224×3 face image
  ↓
CNN with 4 Conv Blocks (32→64→128→256 filters)
  ↓
Dual Attention (Spatial + Channel) x2
  ↓
Global Avg Pool → Dense(512) → Dense(256)
  ↓
Output Head 1: Drowsiness (0-1)
Output Head 2: Cognitive Fatigue (0-1)
  ↓
Adaptive Fusion: 0.6×drowsiness + 0.4×cognitive
  ↓
Final Risk Score (0-1)
```

---

## 📈 Fine-Tuning Quick Start

```python
from dataset_utils import train_on_custom_dataset

# Organize your data first:
# data/
#   ├── alert/
#   │   ├── image1.jpg
#   │   └── ...
#   └── drowsy/
#       ├── image1.jpg
#       └── ...

# Then train:
model, history = train_on_custom_dataset('data/alert', 'data/drowsy')
```

---

## 🔗 API Endpoints (Quick Reference)

```
GET  http://localhost:5000/                 → Dashboard HTML
GET  http://localhost:5000/video_feed       → Live video stream
GET  http://localhost:5000/api/metrics      → Current scores (JSON)
GET  http://localhost:5000/api/alert_history → Alert logs (JSON)
GET  http://localhost:5000/api/system_status → System info (JSON)
GET  http://localhost:5000/api/start        → Start detection
GET  http://localhost:5000/api/stop         → Stop detection
```

---

## 💡 Pro Tips

1. **Better Detection**: Good lighting + face centered in frame
2. **Faster Speed**: Reduce frame resolution in main.py (640×480 → 320×240)
3. **GPU Acceleration**: Install GPU version of TensorFlow
4. **Custom Model**: Train on your own dataset for better accuracy
5. **Deployment**: Works on Raspberry Pi, Jetson Nano, etc.

---

## 📱 Testing Commands

```bash
# Test imports
python -c "import cv2, tensorflow, flask; print('OK')"

# Test model
python -c "from model import DrowsinessDetector; print('Model loaded')"

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Test Flask
python -c "from main import app; print('Flask OK')"
```

---

## 🎓 Next Steps After Setup

1. ✅ Run with default model (random weights)
2. 📸 Collect your own drowsy/alert face images (~1000 total)
3. 🧠 Fine-tune model using `dataset_utils.py`
4. 📊 Evaluate on test set
5. 🚀 Deploy with trained model

---

## 📞 Still Stuck?

1. Read: `SETUP_AND_USAGE_GUIDE.md`
2. Check: Code comments in `main.py` and `model.py`
3. Run: `python test_setup.py` (diagnostic)
4. Debug: `python -c "import cv2; print(cv2.__version__)"`

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created & activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `templates/` folder created
- [ ] `dashboard.html` moved to `templates/`
- [ ] Camera working (`python test_setup.py`)
- [ ] Flask starts without error (`python main.py`)
- [ ] Browser opens to `http://localhost:5000`
- [ ] Video stream displays
- [ ] Scores update in real-time

**You're ready to go! 🚗💡**

---

## 📄 File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Flask app + real-time processing |
| `model.py` | CNN + Attention mechanisms |
| `dashboard.html` | Web interface |
| `requirements.txt` | Python dependencies |
| `dataset_utils.py` | Dataset prep + fine-tuning |
| `test_setup.py` | System verification |
| `run_windows.bat` | One-click setup (Windows) |
| `run_unix.sh` | One-click setup (Mac/Linux) |

---

Generated: 2024 | ADAM-DD System | Quick Reference v1.0
