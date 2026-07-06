"""
ADAM-DD: System Verification Script
Tests all components before running the full application
"""

import sys
import importlib

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_import(module_name, display_name):
    """Test if module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {display_name}: OK")
        return True
    except ImportError as e:
        print(f"✗ {display_name}: FAILED - {e}")
        return False

def test_dependencies():
    """Test all required dependencies"""
    print_section("TESTING DEPENDENCIES")
    
    dependencies = [
        ('cv2', 'OpenCV'),
        ('tensorflow', 'TensorFlow'),
        ('keras', 'Keras'),
        ('flask', 'Flask'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('PIL', 'Pillow'),
    ]
    
    results = []
    for module, display in dependencies:
        results.append(test_import(module, display))
    
    return all(results)

def test_camera():
    """Test camera access"""
    print_section("TESTING CAMERA")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"✓ Camera: OK (Resolution: {frame.shape[1]}×{frame.shape[0]})")
                return True
            else:
                print("✗ Camera: Cannot capture frame")
                return False
        else:
            print("✗ Camera: Not accessible")
            return False
    except Exception as e:
        print(f"✗ Camera: ERROR - {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print_section("TESTING MODEL CREATION")
    
    try:
        from model import DrowsinessDetector
        import numpy as np
        
        print("Creating model...")
        detector = DrowsinessDetector()
        print("✓ Model created successfully")
        
        print("Testing inference...")
        dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
        predictions = detector.predict(dummy_input)
        
        drowsiness = predictions['drowsiness'][0, 0]
        cognitive = predictions['cognitive_fatigue'][0, 0]
        
        print(f"✓ Inference OK")
        print(f"  - Drowsiness Score: {drowsiness:.4f}")
        print(f"  - Cognitive Fatigue: {cognitive:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Model: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_detection():
    """Test face detection"""
    print_section("TESTING FACE DETECTION")
    
    try:
        import cv2
        from main import load_cascade_classifiers, detect_and_crop_face
        
        face_cascade, eye_cascade = load_cascade_classifiers()
        print("✓ Cascades loaded")
        
        # Create dummy image with face-like features
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # This won't detect anything, but we're testing the function works
        face_roi, coords = detect_and_crop_face(dummy_frame)
        print("✓ Face detection function OK")
        
        return True
    except Exception as e:
        print(f"✗ Face Detection: ERROR - {e}")
        return False

def test_flask_server():
    """Test Flask app creation"""
    print_section("TESTING FLASK APP")
    
    try:
        from main import app
        print("✓ Flask app created successfully")
        
        # Test basic route
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("✓ Dashboard route accessible")
                return True
            else:
                print(f"✗ Dashboard route failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"✗ Flask App: ERROR - {e}")
        return False

def test_directory_structure():
    """Test if required directories exist"""
    print_section("TESTING DIRECTORY STRUCTURE")
    
    import os
    
    required_dirs = ['templates']
    results = []
    
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✓ Directory '{dir_name}': EXISTS")
            results.append(True)
        else:
            print(f"✗ Directory '{dir_name}': MISSING")
            results.append(False)
    
    # Check for key files
    required_files = ['dashboard.html']
    for file_name in required_files:
        file_path = os.path.join('templates', file_name)
        if os.path.isfile(file_path):
            print(f"✓ File '{file_path}': EXISTS")
            results.append(True)
        else:
            print(f"✗ File '{file_path}': MISSING")
            print(f"  → Run: mkdir templates && mv dashboard.html templates/")
            results.append(False)
    
    return all(results)

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "ADAM-DD: System Verification Script" + " "*13 + "║")
    print("║" + " "*58 + "║")
    print("║" + " "*8 + "Attention-based Driver Drowsiness Detection" + " "*7 + "║")
    print("╚" + "="*58 + "╝")
    
    results = {
        'Dependencies': test_dependencies(),
        'Camera': test_camera(),
        'Directory Structure': test_directory_structure(),
        'Face Detection': test_face_detection(),
        'Model Creation': test_model_creation(),
        'Flask Server': test_flask_server(),
    }
    
    # Summary
    print_section("TEST SUMMARY")
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print(f"\n{'Overall Status:':<40} {'✓ READY TO RUN' if all_passed else '✗ NEEDS SETUP'}")
    
    if all_passed:
        print_section("NEXT STEPS")
        print("""
1. Start the application:
   python main.py

2. Open your browser:
   http://localhost:5000

3. Allow camera access when prompted

4. System will begin real-time drowsiness detection

Enjoy! 🚗💡
        """)
    else:
        print_section("SETUP REQUIRED")
        print("""
Fix the failed tests above:

1. Install missing dependencies:
   pip install -r requirements.txt

2. Create templates directory:
   mkdir templates

3. Move dashboard.html:
   mv dashboard.html templates/

4. Ensure camera is accessible in system settings

Then run this script again to verify.
        """)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
