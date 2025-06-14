#!/usr/bin/env python3
"""
Elite training script for video game character state classification.
Designed to achieve 80%+ accuracy using state-of-the-art techniques.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from config.config import (
    DATA_DIR, MODELS_DIR, LOGS_DIR, CLASS_NAMES, 
    IMAGE_SIZE, BATCH_SIZE, EPOCHS
)
from src.data_loader import load_advanced_image_datasets, load_elite_image_datasets, get_class_counts
from src.model import (
    create_efficientnet_model, create_vision_transformer_model,
    create_hybrid_cnn_transformer_model, create_cnn_model, compile_model
)
from src.trainer import save_model, plot_training_history, load_saved_model


def setup_elite_logging() -> logging.Logger:
    """Setup comprehensive logging for elite training."""
    LOGS_DIR.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / "elite_training.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_cosine_annealing_scheduler(initial_lr: float = 0.001, 
                                    min_lr: float = 1e-7,
                                    cycle_length: int = 50) -> tf.keras.callbacks.Callback:
    """
    Create cosine annealing learning rate scheduler.
    
    Args:
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate
        cycle_length: Length of the cosine cycle
        
    Returns:
        Keras callback for cosine annealing
    """
    def cosine_annealing(epoch):
        if epoch < cycle_length:
            return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * epoch / cycle_length))
        else:
            # Restart cycle
            epoch_in_cycle = epoch % cycle_length
            return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * epoch_in_cycle / cycle_length))
    
    return tf.keras.callbacks.LearningRateScheduler(cosine_annealing, verbose=1)


def create_advanced_callbacks(
    model_name: str,
    patience: int = 15,
    monitor: str = 'val_accuracy'
) -> List[tf.keras.callbacks.Callback]:
    """
    Create advanced callback suite for training.
    
    Args:
        model_name: Name for checkpoint files
        patience: Early stopping patience
        monitor: Metric to monitor
        
    Returns:
        List of advanced callbacks
    """
    checkpoint_dir = MODELS_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    callbacks = [
        # Advanced early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='max',
            min_delta=0.001
        ),
        
        # Model checkpointing
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / f"{model_name}_best.keras"),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max'
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=8,
            min_lr=1e-8,
            verbose=1,
            mode='max',
            cooldown=3
        ),
        
        # Cosine annealing scheduler
        create_cosine_annealing_scheduler(initial_lr=0.001, min_lr=1e-7),
        
        # CSV logger for detailed metrics
        tf.keras.callbacks.CSVLogger(
            str(LOGS_DIR / f"{model_name}_training_log.csv"),
            append=True
        ),
        
        # Terminate on NaN
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    
    return callbacks


def create_ensemble_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    models_config: List[Dict[str, Any]]
) -> tf.keras.Model:
    """
    Create ensemble model combining multiple architectures.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        models_config: Configuration for each model in ensemble
        
    Returns:
        Ensemble model
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    model_outputs = []
    
    for i, config in enumerate(models_config):
        model_type = config['type']
        weight = config.get('weight', 1.0)
        
        if model_type == 'efficientnet':
            model = create_efficientnet_model(input_shape, num_classes, config.get('size', 'B0'))
        elif model_type == 'vit':
            model = create_vision_transformer_model(input_shape, num_classes)
        elif model_type == 'hybrid':
            model = create_hybrid_cnn_transformer_model(input_shape, num_classes)
        elif model_type == 'cnn':
            model = create_cnn_model(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get the model output (without the input layer)
        model_output = model(inputs)
        
        # Apply weight to this model's contribution
        weighted_output = tf.keras.layers.Lambda(lambda x: x * weight, name=f'weight_{i}')(model_output)
        model_outputs.append(weighted_output)
    
    # Average the outputs
    if len(model_outputs) > 1:
        ensemble_output = tf.keras.layers.Average(name='ensemble_average')(model_outputs)
    else:
        ensemble_output = model_outputs[0]
    
    ensemble_model = tf.keras.Model(inputs, ensemble_output, name='EnsembleModel')
    
    return ensemble_model


def progressive_training_strategy(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Dict[int, float],
    logger: logging.Logger
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Implement progressive training strategy for optimal results.
    
    Args:
        model: Model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        class_weights: Class weights for imbalanced data
        logger: Logger instance
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    logger.info("🚀 Starting Progressive Training Strategy")
    
    # Stage 1: Warmup with lower learning rate
    logger.info("📚 Stage 1: Warmup Training (20 epochs)")
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    warmup_callbacks = create_advanced_callbacks("warmup", patience=10)
    
    history_stage1 = model.fit(
        train_ds,
        epochs=20,
        validation_data=val_ds,
        callbacks=warmup_callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    stage1_acc = max(history_stage1.history['val_accuracy'])
    logger.info(f"✅ Stage 1 completed. Best validation accuracy: {stage1_acc:.3f}")
    
    # Stage 2: Main training with optimal learning rate
    if stage1_acc > 0.4:  # Only proceed if warmup was successful
        logger.info("📈 Stage 2: Main Training (40 epochs)")
        
        # Increase learning rate for main training
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        main_callbacks = create_advanced_callbacks("main", patience=15)
        
        history_stage2 = model.fit(
            train_ds,
            epochs=40,
            validation_data=val_ds,
            callbacks=main_callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        stage2_acc = max(history_stage2.history['val_accuracy'])
        logger.info(f"✅ Stage 2 completed. Best validation accuracy: {stage2_acc:.3f}")
        
        # Combine histories
        combined_history = tf.keras.callbacks.History()
        combined_history.history = {}
        
        for key in history_stage1.history:
            combined_history.history[key] = (
                history_stage1.history[key] + history_stage2.history[key]
            )
        
        # Stage 3: Fine-tuning with very low learning rate
        if stage2_acc > 0.6:  # Only proceed if main training was successful
            logger.info("🎯 Stage 3: Fine-tuning (30 epochs)")
            
            # Very low learning rate for fine-tuning
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate=0.00005, weight_decay=0.00005),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            finetune_callbacks = create_advanced_callbacks("finetune", patience=12)
            
            history_stage3 = model.fit(
                train_ds,
                epochs=30,
                validation_data=val_ds,
                callbacks=finetune_callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            stage3_acc = max(history_stage3.history['val_accuracy'])
            logger.info(f"✅ Stage 3 completed. Best validation accuracy: {stage3_acc:.3f}")
            
            # Add stage 3 to combined history
            for key in history_stage3.history:
                combined_history.history[key].extend(history_stage3.history[key])
        
        return model, combined_history
    
    else:
        logger.warning("⚠️ Stage 1 accuracy too low. Stopping progressive training.")
        return model, history_stage1


def cross_validation_training(
    data_paths: List[str],
    labels: List[int],
    model_config: Dict[str, Any],
    logger: logging.Logger,
    n_folds: int = 3
) -> List[float]:
    """
    Perform cross-validation training for robust evaluation.
    
    Args:
        data_paths: List of image paths
        labels: List of corresponding labels
        model_config: Model configuration
        logger: Logger instance
        n_folds: Number of CV folds
        
    Returns:
        List of validation accuracies for each fold
    """
    logger.info(f"🔄 Starting {n_folds}-Fold Cross-Validation")
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_paths, labels)):
        logger.info(f"📁 Training Fold {fold + 1}/{n_folds}")
        
        # Create fold datasets
        train_paths_fold = [data_paths[i] for i in train_idx]
        train_labels_fold = [labels[i] for i in train_idx]
        val_paths_fold = [data_paths[i] for i in val_idx]
        val_labels_fold = [labels[i] for i in val_idx]
        
        # Create datasets (simplified version for CV)
        def create_fold_dataset(paths, labels, is_training=True):
            def load_and_preprocess(path, label):
                image = tf.io.read_file(path)
                image = tf.image.decode_image(image, channels=3)
                image = tf.image.resize(image, IMAGE_SIZE)
                image = tf.cast(image, tf.float32) / 255.0
                return image, label
            
            dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
            dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            
            if is_training:
                dataset = dataset.shuffle(1000)
                # Apply light augmentation
                augmentation = tf.keras.Sequential([
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(0.1),
                    tf.keras.layers.RandomBrightness(0.1),
                ])
                dataset = dataset.map(
                    lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            
            dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            return dataset
        
        train_ds_fold = create_fold_dataset(train_paths_fold, train_labels_fold, True)
        val_ds_fold = create_fold_dataset(val_paths_fold, val_labels_fold, False)
        
        # Create and train model
        if model_config['type'] == 'efficientnet':
            model = create_efficientnet_model((*IMAGE_SIZE, 3), len(CLASS_NAMES))
        elif model_config['type'] == 'hybrid':
            model = create_hybrid_cnn_transformer_model((*IMAGE_SIZE, 3), len(CLASS_NAMES))
        else:
            model = create_cnn_model((*IMAGE_SIZE, 3), len(CLASS_NAMES))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Calculate class weights for this fold
        fold_class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels_fold),
            y=train_labels_fold
        )
        fold_class_weights = {i: weight for i, weight in enumerate(fold_class_weights)}
        
        # Train model
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.5, patience=5
            )
        ]
        
        history = model.fit(
            train_ds_fold,
            epochs=30,  # Reduced epochs for CV
            validation_data=val_ds_fold,
            callbacks=callbacks,
            class_weight=fold_class_weights,
            verbose=0  # Reduced verbosity for CV
        )
        
        # Get best validation accuracy
        best_acc = max(history.history['val_accuracy'])
        cv_scores.append(best_acc)
        
        logger.info(f"✅ Fold {fold + 1} completed. Validation accuracy: {best_acc:.3f}")
    
    logger.info(f"🎯 Cross-Validation Results:")
    logger.info(f"  Mean CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    logger.info(f"  Individual Scores: {[f'{score:.3f}' for score in cv_scores]}")
    
    return cv_scores


def main():
    """Elite training pipeline designed to achieve 80%+ accuracy."""
    logger = setup_elite_logging()
    
    print("🏆 ELITE VIDEO GAME CHARACTER STATE CLASSIFICATION")
    print("🎯 Target: 80%+ Accuracy with State-of-the-Art Techniques")
    print("="*80)
    
    try:
        # Load and analyze data
        print(f"\n📊 Loading and analyzing dataset...")
        data_dir = DATA_DIR / "processed"
        
        # Load datasets with advanced preprocessing
        train_ds, val_ds, test_ds, class_counts = load_elite_image_datasets(
            data_dir,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            validation_split=0.2 if not (data_dir / "validation").exists() else None,
            apply_augmentation=True
        )
        
        print(f"\n📈 Dataset Analysis:")
        total_samples = sum(class_counts.values())
        print(f"   Total samples: {total_samples}")
        print(f"   Class distribution: {class_counts}")
        
        # Calculate imbalance ratio
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Calculate class weights
        class_labels = []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_labels.extend([class_idx] * class_counts[class_name])
        
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(class_labels),
            y=class_labels
        )
        class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        
        print(f"\n⚖️ Class weights: {class_weights}")
        
        # Determine optimal model configuration based on dataset size
        if total_samples < 500:
            model_configs = [
                {'type': 'efficientnet', 'size': 'B0', 'weight': 0.4},
                {'type': 'cnn', 'weight': 0.6}
            ]
            use_ensemble = False
            print(f"\n🧠 Using single model approach (small dataset)")
        elif total_samples < 2000:
            model_configs = [
                {'type': 'efficientnet', 'size': 'B0', 'weight': 0.5},
                {'type': 'hybrid', 'weight': 0.5}
            ]
            use_ensemble = True
            print(f"\n🧠 Using ensemble approach (medium dataset)")
        else:
            model_configs = [
                {'type': 'efficientnet', 'size': 'B1', 'weight': 0.4},
                {'type': 'vit', 'weight': 0.3},
                {'type': 'hybrid', 'weight': 0.3}
            ]
            use_ensemble = True
            print(f"\n🧠 Using advanced ensemble approach (large dataset)")
        
        # Create and train model(s)
        if use_ensemble and len(model_configs) > 1:
            print(f"\n🎭 Creating ensemble model with {len(model_configs)} architectures")
            model = create_ensemble_model((*IMAGE_SIZE, 3), len(CLASS_NAMES), model_configs)
        else:
            print(f"\n🎯 Creating single {model_configs[0]['type']} model")
            config = model_configs[0]
            if config['type'] == 'efficientnet':
                model = create_efficientnet_model((*IMAGE_SIZE, 3), len(CLASS_NAMES), config.get('size', 'B0'))
            elif config['type'] == 'vit':
                model = create_vision_transformer_model((*IMAGE_SIZE, 3), len(CLASS_NAMES))
            elif config['type'] == 'hybrid':
                model = create_hybrid_cnn_transformer_model((*IMAGE_SIZE, 3), len(CLASS_NAMES))
            else:
                model = create_cnn_model((*IMAGE_SIZE, 3), len(CLASS_NAMES))
        
        # Compile model with advanced optimization
        model = compile_model(model, class_weights, learning_rate=0.001)
        
        print(f"\n📋 Model Summary:")
        print(f"   Total parameters: {model.count_params():,}")
        
        # Count trainable parameters manually
        trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
        print(f"   Trainable parameters: {trainable_count:,}")
        
        # Progressive training strategy
        model, history = progressive_training_strategy(
            model, train_ds, val_ds, class_weights, logger
        )
        
        # Final evaluation
        print(f"\n📊 Final Evaluation...")
        final_loss, final_accuracy = model.evaluate(val_ds, verbose=0)
        
        # Test on test set if available
        if test_ds is not None:
            test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
            print(f"   Test Accuracy: {test_accuracy:.3f}")
            print(f"   Test Loss: {test_loss:.3f}")
        
        # Save the best model
        final_model_path = MODELS_DIR / "elite_character_classifier.keras"
        save_model(model, final_model_path)
        
        # Plot training history
        plot_path = LOGS_DIR / "elite_training_history.png"
        plot_training_history(history, plot_path)
        
        # Results summary
        print(f"\n" + "="*80)
        print(f"🏆 ELITE TRAINING COMPLETED!")
        print(f"="*80)
        print(f"📈 Final Results:")
        print(f"   Validation Accuracy: {final_accuracy:.1%}")
        print(f"   Validation Loss: {final_loss:.3f}")
        print(f"   Model saved: {final_model_path}")
        
        if final_accuracy >= 0.8:
            print(f"   🎉 TARGET ACHIEVED! Excellent performance (≥80%)")
        elif final_accuracy >= 0.75:
            print(f"   🌟 Outstanding performance! Very close to target")
        elif final_accuracy >= 0.7:
            print(f"   ✅ Excellent performance! Good progress toward target")
        elif final_accuracy >= 0.6:
            print(f"   📈 Good performance! Consider additional techniques")
        else:
            print(f"   ⚠️ Needs improvement - check data quality and model configuration")
        
        print(f"\n📋 Usage:")
        print(f"   python scripts/predict.py <image_path> --model {final_model_path}")
        
        logger.info(f"Elite training completed. Final accuracy: {final_accuracy:.3f}")
        
    except Exception as e:
        logger.error(f"Elite training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()