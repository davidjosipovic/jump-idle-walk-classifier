#!/usr/bin/env python3
"""
Main training script for video game character state classification.
Handles data imbalance and overfitting with proven techniques.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config.config import DATA_DIR, MODELS_DIR, LOGS_DIR, CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE
from src.data_loader import get_class_counts
from src.trainer import save_model, plot_training_history


def setup_logging() -> logging.Logger:
    """Setup comprehensive logging."""
    LOGS_DIR.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / "training.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_datasets(data_dir: Path) -> tuple:
    """Load and preprocess datasets with validation split."""
    train_dir = data_dir / "train"
    
    # Create train/validation split from training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES
    )
    
    # Normalize and optimize datasets
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds


def create_model() -> tf.keras.Model:
    """Create CNN model with regularization to prevent overfitting."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
        
        # Data augmentation (training only)
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        
        # Convolutional blocks with regularization
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        
        # Global pooling and classification
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def calculate_class_weights(data_dir: Path) -> dict:
    """Calculate balanced class weights to handle data imbalance."""
    train_dir = data_dir / "train"
    class_counts = get_class_counts(train_dir)
    
    class_names = list(class_counts.keys())
    counts = list(class_counts.values())
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(class_names)),
        y=np.repeat(np.arange(len(class_names)), counts)
    )
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"\n⚖️  Class Weights Applied:")
    for i, (name, weight) in enumerate(zip(class_names, class_weights)):
        print(f"   {name}: {weight:.3f}")
    
    return class_weight_dict


def evaluate_model_predictions(model: tf.keras.Model, val_ds: tf.data.Dataset) -> dict:
    """Evaluate model and check prediction distribution."""
    y_true, y_pred = [], []
    
    for batch_images, batch_labels in val_ds:
        predictions = model.predict(batch_images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(batch_labels.numpy())
    
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    
    # Check prediction distribution
    print(f"\n🎯 Prediction Distribution on Validation Set:")
    unique, counts = np.unique(y_pred, return_counts=True)
    total = len(y_pred)
    
    for i, count in enumerate(counts):
        class_name = CLASS_NAMES[unique[i]] if i < len(unique) else f"Class_{unique[i]}"
        percentage = count / total * 100
        print(f"   {class_name}: {count}/{total} ({percentage:.1f}%)")
    
    max_percentage = max(counts) / total * 100
    if max_percentage > 80:
        print(f"   ⚠️  Still heavily biased toward one class!")
    elif max_percentage > 60:
        print(f"   ⚠️  Some bias remaining")
    else:
        print(f"   ✅ More balanced predictions!")
    
    return {'accuracy': accuracy}


def main():
    """Main training workflow."""
    logger = setup_logging()
    
    print("🎮 VIDEO GAME CHARACTER STATE CLASSIFICATION")
    print("🔧 Training with Data Balance and Overfitting Prevention")
    print("="*60)
    
    try:
        # Analyze data distribution
        print(f"\n📊 Analyzing data distribution...")
        train_dir = DATA_DIR / "processed" / "train"
        class_counts = get_class_counts(train_dir)
        
        total = sum(class_counts.values())
        print(f"\n📈 Training Data:")
        for class_name, count in class_counts.items():
            percentage = count / total * 100
            print(f"   {class_name}: {count} images ({percentage:.1f}%)")
        
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        ratio = max_count / min_count
        print(f"\n⚖️  Imbalance Ratio: {ratio:.1f}:1")
        
        # Calculate class weights and load data
        class_weights = calculate_class_weights(DATA_DIR / "processed")
        print(f"\n📂 Loading datasets...")
        train_ds, val_ds = load_datasets(DATA_DIR / "processed")
        
        # Create and train model
        print(f"\n🧠 Creating model...")
        model = create_model()
        print(f"\n📋 Model Summary:")
        model.summary()
        
        print(f"\n🚀 Starting training...")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        history = model.fit(
            train_ds,
            epochs=30,
            validation_data=val_ds,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate and save
        print(f"\n📊 Evaluating model...")
        results = evaluate_model_predictions(model, val_ds)
        
        print(f"\n💾 Saving model...")
        model_path = MODELS_DIR / "character_state_classifier.keras"
        save_model(model, model_path)
        
        plot_path = LOGS_DIR / "training_history.png"
        plot_training_history(history, plot_path)
        
        # Final summary
        print(f"\n" + "="*60)
        print(f"✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"="*60)
        print(f"📈 Results:")
        print(f"   Validation Accuracy: {results['accuracy']:.1%}")
        print(f"   Model saved: {model_path}")
        print(f"   Training plot: {plot_path}")
        
        print(f"\n📋 Next Steps:")
        print(f"   Test model: python scripts/predict.py <image_path>")
        print(f"   Run diagnostics: python scripts/predict.py --diagnose")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()