#!/usr/bin/env python3
"""
Ultimate training script optimized for 80% accuracy on balanced boosted dataset.
Uses state-of-the-art techniques with the massively augmented balanced dataset.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict
import matplotlib.pyplot as plt

from config.config import CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE, MODELS_DIR, LOGS_DIR
from src.trainer import save_model, plot_training_history, evaluate_model_detailed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "ultimate_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_boosted_datasets(data_dir: Path) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load the perfectly balanced boosted datasets."""
    logger.info("Loading massively augmented balanced datasets...")
    
    # Load training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        shuffle=True,
        seed=42
    )
    
    # Load validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "validation", 
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        shuffle=False
    )
    
    # Load test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        shuffle=False
    )
    
    # Normalize pixel values to [0,1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds


def create_ultimate_model() -> tf.keras.Model:
    """Create optimized model architecture for 80% accuracy."""
    logger.info("Creating ultimate model architecture...")
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    # Advanced data augmentation layer (for extra variation during training)
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    x = augment(inputs)
    
    # Efficient feature extraction backbone
    # Block 1 - Initial feature extraction
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # Block 2 - Mid-level features
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # Block 3 - High-level features
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # Block 4 - Deep features
    x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax', name='predictions')(x)
    
    model = tf.keras.Model(inputs, outputs, name='ultimate_character_classifier')
    
    # Compile with optimized settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_advanced_callbacks(patience: int = 15) -> list:
    """Create optimized callbacks for ultimate training."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Monitor accuracy instead of loss
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='max'  # Maximize accuracy
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODELS_DIR / "checkpoints" / "ultimate_best.keras",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]


def ultimate_training_workflow():
    """Complete ultimate training workflow for 80% accuracy."""
    print("🎮 ULTIMATE TRAINING FOR 80% ACCURACY")
    print("🚀 Using Massively Augmented Balanced Dataset")
    print("="*60)
    
    try:
        # Load boosted datasets
        print("\n📊 Loading boosted balanced datasets...")
        boosted_dir = Path("data/boosted")
        train_ds, val_ds, test_ds = load_boosted_datasets(boosted_dir)
        
        # Verify dataset sizes
        train_size = len(list(train_ds))
        val_size = len(list(val_ds))
        test_size = len(list(test_ds))
        
        print(f"   Training batches: {train_size}")
        print(f"   Validation batches: {val_size}")
        print(f"   Test batches: {test_size}")
        
        # Create ultimate model
        print(f"\n🧠 Creating ultimate model architecture...")
        model = create_ultimate_model()
        
        print(f"\n📋 Ultimate Model Summary:")
        model.summary()
        
        total_params = model.count_params()
        print(f"📊 Total parameters: {total_params:,}")
        
        # Setup callbacks
        callbacks = create_advanced_callbacks(patience=15)
        
        # Ultimate training
        print(f"\n🚀 Starting ultimate training...")
        print(f"   Target: 80%+ accuracy")
        print(f"   Dataset: 4,500 balanced training samples")
        print(f"   Architecture: Deep CNN with regularization")
        
        history = model.fit(
            train_ds,
            epochs=50,  # More epochs for convergence
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate final performance
        print(f"\n📊 Final Model Evaluation...")
        
        # Validation performance
        val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
        print(f"   Validation Accuracy: {val_accuracy:.3f} ({val_accuracy*100:.1f}%)")
        print(f"   Validation Loss: {val_loss:.3f}")
        
        # Test performance (ultimate test)
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
        print(f"   Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        print(f"   Test Loss: {test_loss:.3f}")
        
        # Detailed evaluation
        print(f"\n🔍 Detailed Performance Analysis...")
        detailed_results = evaluate_model_detailed(model, test_ds, CLASS_NAMES)
        
        # Save ultimate model
        print(f"\n💾 Saving ultimate model...")
        model_path = MODELS_DIR / "ultimate_character_classifier.keras"
        save_model(model, model_path)
        
        # Plot training history
        plot_path = LOGS_DIR / "ultimate_training_history.png"
        plot_training_history(history, plot_path)
        
        # Final results summary
        print(f"\n" + "="*70)
        print(f"🏆 ULTIMATE TRAINING COMPLETED!")
        print(f"="*70)
        
        accuracy_achieved = test_accuracy >= 0.80
        status_emoji = "✅" if accuracy_achieved else "⚠️"
        
        print(f"📈 Final Results:")
        print(f"   {status_emoji} Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        print(f"   📊 Validation Accuracy: {val_accuracy:.3f} ({val_accuracy*100:.1f}%)")
        print(f"   💾 Model saved: {model_path}")
        print(f"   📈 Training plot: {plot_path}")
        
        if accuracy_achieved:
            print(f"\n🎉 SUCCESS! Achieved target 80%+ accuracy!")
            print(f"🔥 The model is ready for production use!")
        else:
            print(f"\n🔄 Close to target! Consider:")
            print(f"   • Extended training with lower learning rate")
            print(f"   • Fine-tuning hyperparameters")
            print(f"   • Additional data augmentation")
        
        print(f"\n📋 Usage:")
        print(f"   python scripts/predict.py <image_path> --model {model_path}")
        
        return model, history, detailed_results
        
    except Exception as e:
        logger.error(f"Ultimate training failed: {str(e)}")
        raise


def main():
    """Main entry point."""
    ultimate_training_workflow()


if __name__ == "__main__":
    main()