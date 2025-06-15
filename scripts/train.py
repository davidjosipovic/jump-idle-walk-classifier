#!/usr/bin/env python3

from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = ['idle', 'jumping', 'walking']
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 35
LEARNING_RATE = 0.00005

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' 
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def get_class_counts():
    train_dir = DATA_DIR / 'train'
    counts = {}
    
    for class_name in CLASS_NAMES:
        class_path = train_dir / class_name
        if class_path.exists():
            counts[class_name] = len(list(class_path.glob('*')))
        else:
            counts[class_name] = 0
    
    return counts


def create_data_generators():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
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


def create_model():
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=IMAGE_SIZE + (3,)
    )
    
    for layer in base_model.layers[:-15]:
        layer.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history, filename='logs/training_history.png'):
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
    print(f"Training plot saved: {filename}")


def main():
    print("Character State Classifier Training")
    print("=" * 40)
    
    counts = get_class_counts()
    total_images = sum(counts.values())
    print("Dataset Info:")
    for class_name, count in counts.items():
        print(f"   {class_name}: {count} images")
    print(f"   Total: {total_images} images")
    
    if total_images == 0:
        print("No training data found!")
        return
    
    y = []
    for i, (class_name, count) in enumerate(counts.items()):
        y.extend([i] * count)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    jumping_idx = CLASS_NAMES.index('jumping')
    class_weight_dict[jumping_idx] *= 4.0
    
    print(f"Final class weights: {class_weight_dict}")

    validation_dir = DATA_DIR / 'validation'
    if not validation_dir.exists():
        print("Validation data directory not found!")
        return

    logger.info("Loading training and validation datasets...")
    train_gen, val_gen = create_data_generators()
    logger.info("Datasets loaded successfully.")
    
    print("Creating model...")
    model = create_model()
    
    print(f"Model created with {model.count_params():,} parameters")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODELS_DIR / 'character_classifier.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print(f"\nTraining for {EPOCHS} epochs...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nEvaluating model...")
    val_gen.reset()
    predictions = model.predict(val_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=CLASS_NAMES,
        digits=2
    )
    print("\nClassification Report:")
    print(report)
    
    misclassified = [(true, pred) for true, pred in zip(true_classes, predicted_classes) if true != pred]
    logger.info(f"Misclassified images: {len(misclassified)}")
    for true, pred in misclassified[:10]:
        logger.info(f"True: {CLASS_NAMES[true]}, Predicted: {CLASS_NAMES[pred]}")
    
    plot_training_history(history)
    
    final_accuracy = max(history.history['val_accuracy'])
    
    print("\n" + "=" * 40)
    print("TRAINING COMPLETED")
    print("=" * 40)
    print(f"Best Validation Accuracy: {final_accuracy:.1%}")
    print(f"Model saved: character_classifier.keras")
    print(f"Training plot: training_history.png")
    
    if final_accuracy > 0.8:
        print("Great results!")
    elif final_accuracy > 0.6:
        print("Decent results - might need more data or tuning")
    else:
        print("Poor results - check data quality and model setup")
    
    print("\nTest your model:")
    print("   python predict.py path/to/image.png")


if __name__ == "__main__":
    main()
