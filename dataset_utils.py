"""
ADAM-DD: Dataset Preparation & Model Fine-tuning Utilities
For training on custom drowsiness/alert datasets
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DATASET PREPARATION
# =============================================================================

class DatasetPreparer:
    """Prepare image dataset for model training"""
    
    def __init__(self, target_size=224):
        self.target_size = target_size
    
    def load_images_from_directory(self, directory, label, max_images=None):
        """
        Load images from directory
        
        Args:
            directory: Path to directory containing images
            label: Binary label (0 for alert, 1 for drowsy)
            max_images: Maximum images to load (for testing)
        
        Returns:
            images: List of preprocessed images
            labels: List of labels
        """
        images = []
        labels = []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        image_files = [f for f in os.listdir(directory) 
                      if Path(f).suffix.lower() in valid_extensions]
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Loading {len(image_files)} images from {directory}...")
        
        for idx, filename in enumerate(image_files):
            if (idx + 1) % 50 == 0:
                logger.info(f"  Loaded {idx + 1}/{len(image_files)}")
            
            try:
                img_path = os.path.join(directory, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    logger.warning(f"  Could not read {filename}")
                    continue
                
                # Preprocess
                img = self.preprocess_image(img)
                images.append(img)
                labels.append(label)
                
            except Exception as e:
                logger.warning(f"  Error processing {filename}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(images)} images")
        return np.array(images), np.array(labels)
    
    def preprocess_image(self, img):
        """
        Preprocess image:
        - Resize to target size
        - Normalize to [0, 1]
        - Apply CLAHE enhancement
        """
        # Resize
        img = cv2.resize(img, (self.target_size, self.target_size))
        
        # CLAHE enhancement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def create_dataset(self, alert_dir, drowsy_dir, val_split=0.2, 
                      test_split=0.1, max_per_class=None):
        """
        Create train/val/test split from directories
        
        Directory structure:
        data/
        ├── alert/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── drowsy/
            ├── image1.jpg
            ├── image2.jpg
            └── ...
        
        Args:
            alert_dir: Directory with alert face images
            drowsy_dir: Directory with drowsy face images
            val_split: Validation split ratio
            test_split: Test split ratio
            max_per_class: Maximum images per class (for testing)
        
        Returns:
            train_images, train_labels
            val_images, val_labels
            test_images, test_labels
        """
        logger.info("Creating dataset from directories...")
        
        # Load images
        alert_images, alert_labels = self.load_images_from_directory(
            alert_dir, label=0, max_images=max_per_class)
        drowsy_images, drowsy_labels = self.load_images_from_directory(
            drowsy_dir, label=1, max_images=max_per_class)
        
        # Combine
        X = np.vstack([alert_images, drowsy_images])
        y = np.concatenate([alert_labels, drowsy_labels])
        
        logger.info(f"Total images: {len(X)}")
        logger.info(f"  Alert: {np.sum(y == 0)}")
        logger.info(f"  Drowsy: {np.sum(y == 1)}")
        
        # First split: train + val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y)
        
        # Second split: train / val
        val_size = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp)
        
        logger.info(f"\nDataset split:")
        logger.info(f"  Train: {len(X_train)} (Alert: {np.sum(y_train == 0)}, Drowsy: {np.sum(y_train == 1)})")
        logger.info(f"  Val:   {len(X_val)} (Alert: {np.sum(y_val == 0)}, Drowsy: {np.sum(y_val == 1)})")
        logger.info(f"  Test:  {len(X_test)} (Alert: {np.sum(y_test == 0)}, Drowsy: {np.sum(y_test == 1)})")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save_dataset(self, data, output_dir='dataset'):
        """Save preprocessed dataset to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
        
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        logger.info(f"Dataset saved to {output_dir}/")
    
    @staticmethod
    def load_dataset(data_dir='dataset'):
        """Load preprocessed dataset"""
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

class DataAugmenter:
    """Advanced data augmentation for training"""
    
    @staticmethod
    def apply_augmentations(X, num_augmentations=2):
        """
        Apply random augmentations to increase dataset size
        
        Args:
            X: Input images (N, 224, 224, 3)
            num_augmentations: Number of augmented versions per image
        
        Returns:
            X_augmented: Original + augmented images
        """
        X_augmented = [X]
        
        for aug_idx in range(num_augmentations):
            X_aug = X.copy()
            
            # Random rotation (-10 to +10 degrees)
            for i in range(len(X_aug)):
                if np.random.rand() > 0.5:
                    angle = np.random.uniform(-10, 10)
                    X_aug[i] = DataAugmenter._rotate_image(X_aug[i], angle)
            
            # Random brightness adjustment
            if np.random.rand() > 0.5:
                brightness_factor = np.random.uniform(0.8, 1.2)
                X_aug = np.clip(X_aug * brightness_factor, 0, 1)
            
            # Random horizontal flip
            if np.random.rand() > 0.5:
                X_aug = np.fliplr(X_aug)
            
            # Random zoom
            if np.random.rand() > 0.5:
                zoom_factor = np.random.uniform(0.9, 1.1)
                X_aug_zoomed = []
                for img in X_aug:
                    X_aug_zoomed.append(
                        DataAugmenter._zoom_image(img, zoom_factor)
                    )
                X_aug = np.array(X_aug_zoomed)
            
            X_augmented.append(X_aug)
            logger.info(f"  Augmentation {aug_idx + 1}/{num_augmentations} complete")
        
        return np.vstack(X_augmented)
    
    @staticmethod
    def _rotate_image(img, angle):
        """Rotate image"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated
    
    @staticmethod
    def _zoom_image(img, zoom_factor):
        """Zoom image"""
        h, w = img.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        if zoom_factor > 1:
            resized = cv2.resize(img, (new_w, new_h))
            y1, x1 = (new_h - h) // 2, (new_w - w) // 2
            return resized[y1:y1+h, x1:x1+w]
        else:
            resized = cv2.resize(img, (new_w, new_h))
            y1, x1 = (h - new_h) // 2, (w - new_w) // 2
            padded = np.ones_like(img)
            padded[y1:y1+new_h, x1:x1+new_w] = resized
            return padded

# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_on_custom_dataset(alert_dir, drowsy_dir, output_model='drowsiness_model_trained.h5'):
    """
    Complete training pipeline
    
    Usage:
    train_on_custom_dataset('path/to/alert_images', 'path/to/drowsy_images')
    """
    from model import DrowsinessDetector, train_drowsiness_model
    
    logger.info("="*60)
    logger.info("ADAM-DD: Custom Dataset Training Pipeline")
    logger.info("="*60)
    
    # Step 1: Prepare dataset
    logger.info("\n[STEP 1] Preparing dataset...")
    preparer = DatasetPreparer(target_size=224)
    data = preparer.create_dataset(alert_dir, drowsy_dir, max_per_class=None)
    preparer.save_dataset(data, 'dataset')
    
    # Step 2: Load data
    logger.info("\n[STEP 2] Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    
    # Step 3: Data augmentation (optional)
    logger.info("\n[STEP 3] Applying data augmentation...")
    augmenter = DataAugmenter()
    X_train_aug = augmenter.apply_augmentations(X_train, num_augmentations=1)
    logger.info(f"Augmented training set: {len(X_train_aug)} images")
    
    # Step 4: Create labels for dual outputs
    logger.info("\n[STEP 4] Preparing dual-output labels...")
    # For simplicity: y_cognitive = y_drowsiness (can be different in practice)
    y_train_drowsiness = y_train_aug
    y_train_cognitive = y_train_aug
    y_val_drowsiness = y_train
    y_val_cognitive = y_val
    
    # Step 5: Train model
    logger.info("\n[STEP 5] Training model...")
    model, history = train_drowsiness_model(
        X_train_aug, y_train_drowsiness, y_train_cognitive,
        X_val, y_val_drowsiness, y_val_cognitive,
        epochs=50, batch_size=32
    )
    
    # Step 6: Evaluate
    logger.info("\n[STEP 6] Evaluating on test set...")
    test_loss, test_acc_drowsiness, test_acc_cognitive = model.evaluate(
        X_test,
        {'drowsiness': y_test, 'cognitive_fatigue': y_test},
        verbose=0
    )
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Drowsiness Accuracy: {test_acc_drowsiness:.4f}")
    logger.info(f"Cognitive Fatigue Accuracy: {test_acc_cognitive:.4f}")
    
    # Step 7: Save
    logger.info("\n[STEP 7] Saving model...")
    model.save(output_model)
    logger.info(f"Model saved to {output_model}")
    
    logger.info("\n" + "="*60)
    logger.info("Training complete!")
    logger.info("="*60)
    
    return model, history

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    """
    Example: Prepare and train on your custom dataset
    
    Expected directory structure:
    data/
    ├── alert/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── drowsy/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
    """
    
    # Option 1: Just prepare dataset (no training)
    print("\n" + "="*60)
    print("Dataset Preparation Example")
    print("="*60)
    
    preparer = DatasetPreparer(target_size=224)
    # data = preparer.create_dataset('data/alert', 'data/drowsy', max_per_class=100)
    # preparer.save_dataset(data, 'dataset')
    
    # Option 2: Full training pipeline
    # Uncomment below and provide your dataset paths
    # model, history = train_on_custom_dataset('data/alert', 'data/drowsy')
    
    print("To train on your dataset, uncomment the last line and provide paths")
    print("to directories containing alert and drowsy face images.")
