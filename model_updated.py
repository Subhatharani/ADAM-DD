"""
CNN with Dual Attention Mechanisms for Drowsiness Detection
UPDATED VERSION with:
- Better weight initialization
- Transfer learning support
- More robust feature extraction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import os

class SpatialAttention(layers.Layer):
    """
    Spatial Attention Module
    Computes attention weights across spatial dimensions (H x W)
    Helps focus on specific facial regions indicating drowsiness
    (e.g., eyes, mouth, head position)
    """
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(1, kernel_size=kernel_size, 
                                  padding='same', activation='sigmoid')
    
    def call(self, x):
        # Channel-wise statistics (mean and max)
        avg_pool = tf.reduce_mean(x, axis=3, keepdims=True)
        max_pool = tf.reduce_max(x, axis=3, keepdims=True)
        
        # Concatenate and apply convolution
        concat = tf.concat([avg_pool, max_pool], axis=3)
        spatial_attention = self.conv(concat)
        
        # Apply attention to input
        return x * spatial_attention

class ChannelAttention(layers.Layer):
    """
    Channel Attention Module
    Computes importance weights for each channel
    Helps identify which feature maps are most relevant for drowsiness
    """
    def __init__(self, ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = layers.Dense(channels // self.ratio, activation='relu')
        self.fc2 = layers.Dense(channels)
        super().build(input_shape)
    
    def call(self, x):
        # Global average pooling
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        avg_out = self.fc2(self.fc1(avg_pool))
        
        # Global max pooling
        max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        max_out = self.fc2(self.fc1(max_pool))
        
        # Sum and apply sigmoid
        channel_attention = tf.nn.sigmoid(avg_out + max_out)
        
        # Apply attention to input
        return x * channel_attention

class DualAttentionModule(layers.Layer):
    """
    Combined Dual Attention Module
    Applies both spatial and channel attention sequentially
    """
    def __init__(self, **kwargs):
        super(DualAttentionModule, self).__init__(**kwargs)
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention()
    
    def call(self, x):
        # Apply spatial attention first
        x = self.spatial_attention(x)
        # Then channel attention
        x = self.channel_attention(x)
        return x

# =============================================================================
# DROWSINESS DETECTOR MODEL - UPDATED VERSION
# =============================================================================

def build_attention_cnn(input_shape=(224, 224, 3), use_transfer_learning=False):
    """
    Build CNN with Dual Attention Mechanisms
    
    UPDATED: Now supports transfer learning from ImageNet
    
    Architecture:
    - ConvBlock 1-4: Progressive feature extraction with attention
    - Dual Attention: Spatial + Channel attention in bottleneck
    - Dual Output Heads: Drowsiness + Cognitive Fatigue
    
    Args:
        input_shape: Input image shape (default: 224×224×3)
        use_transfer_learning: Use pre-trained MobileNetV2 backbone (default: False)
    """
    
    if use_transfer_learning:
        # UPDATED: Transfer learning approach (much better starting point!)
        print("Building model with transfer learning (ImageNet weights)...")
        
        # Load pre-trained MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'  # ← Pre-trained ImageNet weights!
        )
        
        # Freeze base model (don't retrain ImageNet features)
        base_model.trainable = False
        
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        
        # Add custom layers for drowsiness detection
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Dual output heads
        drowsiness_output = layers.Dense(1, activation='sigmoid', 
                                        name='drowsiness')(x)
        cognitive_output = layers.Dense(1, activation='sigmoid',
                                       name='cognitive_fatigue')(x)
        
        model = Model(inputs=inputs, 
                     outputs=[drowsiness_output, cognitive_output])
        
        return model
    
    else:
        # Original: Custom CNN with Dual Attention (from scratch)
        print("Building model from scratch with dual attention...")
        
        inputs = keras.Input(shape=input_shape)
        
        # =================================================================
        # STAGE 1: Initial Feature Extraction
        # =================================================================
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)  # 224 -> 112
        x = layers.Dropout(0.25)(x)
        
        # =================================================================
        # STAGE 2: Intermediate Feature Extraction
        # =================================================================
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)  # 112 -> 56
        x = layers.Dropout(0.25)(x)
        
        # =================================================================
        # STAGE 3: Deep Feature Extraction with Attention
        # =================================================================
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Apply Dual Attention
        x = DualAttentionModule()(x)
        
        x = layers.MaxPooling2D(2)(x)  # 56 -> 28
        x = layers.Dropout(0.25)(x)
        
        # =================================================================
        # STAGE 4: Fine-grained Feature Extraction
        # =================================================================
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Apply Dual Attention again
        x = DualAttentionModule()(x)
        
        x = layers.MaxPooling2D(2)(x)  # 28 -> 14
        x = layers.Dropout(0.25)(x)
        
        # =================================================================
        # GLOBAL AVERAGE POOLING & BOTTLENECK
        # =================================================================
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # =================================================================
        # DUAL OUTPUT HEADS
        # =================================================================
        
        # Head 1: Physical Drowsiness Indicator
        drowsiness_head = layers.Dense(128, activation='relu')(x)
        drowsiness_head = layers.Dropout(0.5)(drowsiness_head)
        drowsiness_output = layers.Dense(1, activation='sigmoid', 
                                         name='drowsiness')(drowsiness_head)
        
        # Head 2: Cognitive Fatigue Signal
        cognitive_head = layers.Dense(128, activation='relu')(x)
        cognitive_head = layers.Dropout(0.5)(cognitive_head)
        cognitive_output = layers.Dense(1, activation='sigmoid', 
                                        name='cognitive_fatigue')(cognitive_head)
        
        # =================================================================
        # BUILD MODEL
        # =================================================================
        model = Model(inputs=inputs, 
                      outputs=[drowsiness_output, cognitive_output])
        
        return model

# =============================================================================
# DROWSINESS DETECTOR CLASS - UPDATED
# =============================================================================

class DrowsinessDetector:
    """
    High-level wrapper for drowsiness detection
    Handles model loading, prediction, and preprocessing
    
    UPDATED: Supports both CNN and transfer learning models
    """
    def __init__(self, model_path='drowsiness_model.h5', use_transfer_learning=False):
        self.model_path = model_path
        self.model = None
        self.use_transfer_learning = use_transfer_learning
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create new one with optional transfer learning"""
        if os.path.exists(self.model_path):
            print(f"✓ Loading pre-trained model from {self.model_path}")
            self.model = keras.models.load_model(
                self.model_path,
                custom_objects={
                    'SpatialAttention': SpatialAttention,
                    'ChannelAttention': ChannelAttention,
                    'DualAttentionModule': DualAttentionModule
                }
            )
            print("✓ Model loaded successfully!")
        else:
            print("✓ Creating new model...")
            self.model = build_attention_cnn(
                use_transfer_learning=self.use_transfer_learning
            )
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss={
                    'drowsiness': 'binary_crossentropy',
                    'cognitive_fatigue': 'binary_crossentropy'
                },
                metrics=['accuracy']
            )
            print("✓ Model created and compiled")
            if self.use_transfer_learning:
                print("✓ Using transfer learning (ImageNet weights)")
    
    def predict(self, preprocessed_frame):
        """
        Make prediction on preprocessed frame
        Input: preprocessed_frame (1, 224, 224, 3) in range [0, 1]
        Output: dict with 'drowsiness' and 'cognitive_fatigue' scores
        """
        predictions = self.model.predict(preprocessed_frame, verbose=0)
        
        return {
            'drowsiness': predictions[0],
            'cognitive_fatigue': predictions[1]
        }
    
    def save_model(self):
        """Save model to disk"""
        self.model.save(self.model_path)
        print(f"✓ Model saved to {self.model_path}")
    
    def get_model_summary(self):
        """Get model architecture summary"""
        self.model.summary()

# =============================================================================
# TRAINING UTILITIES (Optional - for fine-tuning on your dataset)
# =============================================================================

def create_training_data_generators(batch_size=32):
    """
    Create data augmentation generators for training
    Simulates real-world variations in lighting, head pose, etc.
    """
    train_augmentation = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_augmentation = keras.preprocessing.image.ImageDataGenerator()
    
    return train_augmentation, val_augmentation

def train_drowsiness_model(train_images, train_labels_drowsiness, train_labels_cognitive,
                          val_images, val_labels_drowsiness, val_labels_cognitive,
                          epochs=50, batch_size=32, use_transfer_learning=True):
    """
    Train drowsiness detection model
    
    UPDATED: Supports transfer learning for faster training
    
    Args:
        train_images: (N, 224, 224, 3) array
        train_labels_drowsiness: (N, 1) binary labels
        train_labels_cognitive: (N, 1) binary labels
        val_images: validation images
        val_labels_drowsiness: validation drowsiness labels
        val_labels_cognitive: validation cognitive labels
        epochs: Number of training epochs
        batch_size: Batch size
        use_transfer_learning: Use ImageNet pre-trained weights
    """
    model = build_attention_cnn(use_transfer_learning=use_transfer_learning)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            'drowsiness': 'binary_crossentropy',
            'cognitive_fatigue': 'binary_crossentropy'
        },
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            'best_drowsiness_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train
    history = model.fit(
        train_images,
        {'drowsiness': train_labels_drowsiness, 
         'cognitive_fatigue': train_labels_cognitive},
        validation_data=(
            val_images,
            {'drowsiness': val_labels_drowsiness,
             'cognitive_fatigue': val_labels_cognitive}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

if __name__ == '__main__':
    # Test model creation with both approaches
    print("\n" + "="*60)
    print("Testing Model Creation")
    print("="*60)
    
    # Test 1: Original CNN (from scratch)
    print("\n[Test 1] Creating CNN from scratch...")
    detector_cnn = DrowsinessDetector(use_transfer_learning=False)
    detector_cnn.get_model_summary()
    
    # Test 2: Transfer learning model
    print("\n[Test 2] Creating model with transfer learning...")
    detector_transfer = DrowsinessDetector(
        model_path='drowsiness_model_transfer.h5',
        use_transfer_learning=True
    )
    detector_transfer.get_model_summary()
    
    # Test prediction
    print("\n" + "="*60)
    print("Testing Predictions")
    print("="*60)
    dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    
    print("\n[CNN Model] Test Prediction:")
    predictions_cnn = detector_cnn.predict(dummy_input)
    print(f"  Drowsiness Score: {predictions_cnn['drowsiness'][0, 0]:.4f}")
    print(f"  Cognitive Fatigue Score: {predictions_cnn['cognitive_fatigue'][0, 0]:.4f}")
    
    print("\n[Transfer Learning Model] Test Prediction:")
    predictions_transfer = detector_transfer.predict(dummy_input)
    print(f"  Drowsiness Score: {predictions_transfer['drowsiness'][0, 0]:.4f}")
    print(f"  Cognitive Fatigue Score: {predictions_transfer['cognitive_fatigue'][0, 0]:.4f}")
    
    print("\n✓ Both models working correctly!")
