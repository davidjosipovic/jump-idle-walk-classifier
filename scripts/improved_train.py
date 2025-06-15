#!/usr/bin/env python3
"""
Improved Video Game Character State Classifier - Training
========================================================

An improved training script that addresses overfitting and class bias:
- Better data augmentation
- Lower learning rate
- More regularization
- Better validation monitoring

Features:
- Uses MobileNetV2 with improved setup
- Strong regularization to prevent overfitting
- Better data augmentation
- Lower learning rate for stable training
"""

import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import logging

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Improved Configuration
CLASS_NAMES = ['idle', 'jumping', 'walking']
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16  # Smaller batch size for better training
EPOCHS = 25
LEARNING_RATE = 0.0001  # Much lower learning rate

# Directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def get_class_counts():
    """Count images in each class."""
    train_dir = DATA_DIR / 'train'
    counts = {}
    
    for class_name in CLASS_NAMES:
        class_path = train_dir / class_name
        if class_path.exists():
            counts[class_name] = len(list(class_path.glob('*')))
        else:
            counts[class_name] = 0
    
    return counts


def create_improved_data_generators():
    """Create improved data generators with better augmentation."""
    
    # Training data augmentation - more conservative
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,      # Reduced rotation
        width_shift_range=0.1,  # Reduced shifts
        height_shift_range=0.1,
        shear_range=0.1,        # Reduced shear
        zoom_range=0.1,         # Reduced zoom
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data - no augmentation
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR / 'train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        DATA_DIR / 'validation',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )
    
    return train_generator, validation_generator


def create_improved_model():
    """Create improved model with better regularization."""
    
    # Load base model
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=IMAGE_SIZE + (3,)
    )
    
    # Freeze fewer layers - allow more fine-tuning
    for layer in base_model.layers[:-30]:  # Freeze all but last 30 layers
        layer.trainable = False
    
    # Build improved model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),  # Add dropout for regularization
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # More dropout
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    # Compile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history, filename='logs/improved_training_history.png'):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Training plot saved: {filename}")


def main():
    print("🎮 Improved Character State Classifier Training")
    print("=" * 50)
    
    # Check dataset
    counts = get_class_counts()
    total_images = sum(counts.values())
    print("📊 Dataset Info:")
    for class_name, count in counts.items():
        print(f"   {class_name}: {count} images")
    print(f"   Total: {total_images} images")
    
    if total_images == 0:
        print("❌ No training data found!")
        return
    
    # Compute class weights
    y = []
    for i, (class_name, count) in enumerate(counts.items()):
        y.extend([i] * count)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"⚖️  Class weights: {class_weight_dict}")
    
    # Create data generators
    print("\n📁 Loading datasets...")
    train_gen, val_gen = create_improved_data_generators()
    
    # Create model
    print("🏗️  Creating improved model...")
    model = create_improved_model()
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,  # Reduce learning rate when stuck
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODELS_DIR / 'improved_character_classifier.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n🚀 Training for {EPOCHS} epochs...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n📊 Evaluating model...")
    val_gen.reset()  # Reset generator
    predictions = model.predict(val_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    
    # Classification report
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=CLASS_NAMES,
        digits=2
    )
    print("\nClassification Report:")
    print(report)
    
    # Plot results
    plot_training_history(history)
    
    # Final accuracy
    final_accuracy = max(history.history['val_accuracy'])
    
    print("\n" + "=" * 50)
    print("🏆 IMPROVED TRAINING COMPLETED!")
    print("=" * 50)
    print(f"📈 Best Validation Accuracy: {final_accuracy:.1%}")
    print(f"📁 Model saved: improved_character_classifier.keras")
    print(f"📊 Training plot: improved_training_history.png")
    
    if final_accuracy > 0.8:
        print("✅ Great results!")
    elif final_accuracy > 0.6:
        print("⚠️ Decent results - might need more data or tuning")
    else:
        print("❌ Poor results - check data quality and model setup")
    
    print("\n🎯 Test your improved model:")
    print("   python simple_predict.py path/to/image.png")


if __name__ == "__main__":
    main()
