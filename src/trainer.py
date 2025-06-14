"""
Training, evaluation, and prediction utilities for video game character state classification.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports when running directly
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from config.config import EPOCHS, CLASS_NAMES, MODELS_DIR
from src.data_loader import load_single_image

logger = logging.getLogger(__name__)


def focal_loss_fn(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss implementation to handle class imbalance.
    Compatible with TensorFlow model serialization.
    """
    # Convert to one-hot if needed
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true, depth=len(CLASS_NAMES))
    
    # Clip predictions to prevent log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Compute focal loss
    ce_loss = -y_true_one_hot * tf.math.log(y_pred)
    pt = tf.where(tf.equal(y_true_one_hot, 1), y_pred, 1 - y_pred)
    focal_weight = alpha * tf.pow(1 - pt, gamma)
    focal_loss = focal_weight * ce_loss
    
    return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    epochs: int = EPOCHS,
    checkpoint_dir: Path = MODELS_DIR / "checkpoints",
    class_weights: Optional[Dict[int, float]] = None,
    steps_per_epoch: Optional[int] = None
) -> tf.keras.callbacks.History:
    """
    Train the model with training and validation datasets.
    
    Args:
        model: Compiled Keras model
        train_dataset: Training dataset (can be infinite with .repeat())
        validation_dataset: Validation dataset
        epochs: Number of training epochs
        checkpoint_dir: Directory to save model checkpoints
        class_weights: Optional class weights for handling imbalanced data
        steps_per_epoch: Number of steps per epoch (required for infinite datasets)
        
    Returns:
        Training history object
    """
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate steps_per_epoch if not provided
    if steps_per_epoch is None:
        try:
            # Try to estimate from validation dataset size
            val_size = 0
            for _ in validation_dataset:
                val_size += 1
            # Use 4x validation size as a reasonable estimate for training steps
            steps_per_epoch = max(50, val_size * 4)  # Minimum 50 steps
            logger.info(f"Estimated steps_per_epoch: {steps_per_epoch} (based on validation size: {val_size})")
        except:
            # Fallback to a reasonable default
            steps_per_epoch = 100
            logger.info(f"Using default steps_per_epoch: {steps_per_epoch}")
    
    # Enhanced callbacks for better overfitting prevention
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor='val_accuracy',  # Monitor validation accuracy
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max'  # Maximize validation accuracy
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Monitor validation accuracy instead of loss
            patience=20,  # Increased patience for longer training with imbalanced data
            restore_best_weights=True,
            verbose=1,
            mode='max',
            min_delta=0.005  # Require at least 0.5% improvement (reduced threshold)
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',  # Monitor validation accuracy
            factor=0.3,  # More aggressive reduction for imbalanced data
            patience=10,  # Reduced patience for faster adaptation
            min_lr=1e-8,  # Lower minimum learning rate
            verbose=1,
            mode='max',
            min_delta=0.003  # Require at least 0.3% improvement
        ),
        # Add learning rate scheduling for better convergence with imbalanced data
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.0001 * (0.95 ** epoch) if epoch < 15 else 0.0001 * (0.98 ** (epoch - 15)),
            verbose=0
        )
    ]
    
    logger.info(f"Starting training for {epochs} epochs with enhanced overfitting prevention")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    if class_weights is not None:
        logger.info("Using class weights to handle imbalanced dataset")
        # Log class weights for debugging
        for class_idx, weight in class_weights.items():
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Class_{class_idx}"
            logger.info(f"  {class_name} (class {class_idx}): weight = {weight:.3f}")
    
    # Train the model with specified steps per epoch
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,  # Important for infinite datasets
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    logger.info("Training completed")
    
    return history


def evaluate_model_detailed(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: list = CLASS_NAMES
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with detailed metrics.
    
    Args:
        model: Trained Keras model
        test_dataset: Test dataset
        class_names: List of class names
        
    Returns:
        Dictionary containing detailed evaluation metrics
    """
    
    logger.info("Performing detailed model evaluation...")
    
    # Collect all predictions and true labels
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for batch_images, batch_labels in test_dataset:
        predictions = model.predict(batch_images, verbose=0)
        y_pred_proba.extend(predictions)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(batch_labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate basic metrics
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    
    # Generate classification report
    class_report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Print detailed results
    print(f"\n📊 DETAILED EVALUATION RESULTS:")
    print(f"=" * 50)
    print(f"Overall Test Accuracy: {test_accuracy:.1%}")
    print(f"Overall Test Loss: {test_loss:.3f}")
    
    print(f"\n📋 Per-Class Performance:")
    for i, class_name in enumerate(class_names):
        precision = class_report[class_name]['precision']
        recall = class_report[class_name]['recall']
        f1 = class_report[class_name]['f1-score']
        support = class_report[class_name]['support']
        
        print(f"  {class_name:>8}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f} (n={support})")
    
    # Check for prediction bias
    print(f"\n🎯 Prediction Distribution:")
    unique, counts = np.unique(y_pred, return_counts=True)
    total_predictions = len(y_pred)
    for i, count in enumerate(counts):
        class_name = class_names[unique[i]] if i < len(unique) else f"Class_{unique[i]}"
        percentage = count / total_predictions * 100
        print(f"  {class_name:>8}: {count:3d} predictions ({percentage:.1f}%)")
    
    # Identify problematic patterns
    if len(set(y_pred)) < len(class_names):
        missing_classes = [class_names[i] for i in range(len(class_names)) if i not in y_pred]
        print(f"\n⚠️  WARNING: Model never predicts: {missing_classes}")
        print(f"   This indicates severe bias - consider retraining with more balanced data")
    
    logger.info("Detailed evaluation completed")
    
    return results


def evaluate_model(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset
) -> Dict[str, float]:
    """
    Simple model evaluation (backwards compatibility).
    """
    results = evaluate_model_detailed(model, test_dataset)
    return {
        'test_loss': results['test_loss'],
        'test_accuracy': results['test_accuracy']
    }


def predict_single_image(
    model: tf.keras.Model,
    image_path: Path,
    class_names: list = CLASS_NAMES,
    confidence_threshold: float = 0.5
) -> Tuple[str, float, np.ndarray]:
    """
    Make prediction on a single image with enhanced output.
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
        class_names: List of class names
        confidence_threshold: Minimum confidence for reliable prediction
        
    Returns:
        Tuple of (predicted_class, confidence, probabilities)
    """
    
    # Load and preprocess the image
    image = load_single_image(image_path)
    
    # Make prediction
    predictions = model.predict(image, verbose=0)
    probabilities = predictions[0]
    
    # Get the predicted class
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]
    
    logger.info(f"Prediction for {image_path.name}: {predicted_class} ({confidence:.3f})")
    
    print(f"\n🔍 PREDICTION RESULTS for {image_path.name}:")
    print(f"=" * 50)
    print(f"🎯 Predicted State: {predicted_class}")
    print(f"📊 Confidence: {confidence:.1%}")
    
    # Confidence assessment
    if confidence >= 0.8:
        confidence_level = "Very High 🟢"
    elif confidence >= confidence_threshold:
        confidence_level = "Good 🟡"
    else:
        confidence_level = "Low 🔴"
    
    print(f"🎯 Confidence Level: {confidence_level}")
    
    print(f"\n📈 All Class Probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        bar_length = int(prob * 20)  # Scale to 20 characters
        bar = "█" * bar_length + "░" * (20 - bar_length)
        marker = " ← PREDICTED" if i == predicted_class_idx else ""
        print(f"  {class_name:>8}: {prob:.1%} |{bar}|{marker}")
    
    # Prediction reliability warning
    if confidence < confidence_threshold:
        print(f"\n⚠️  LOW CONFIDENCE WARNING:")
        print(f"   The model is not very confident about this prediction.")
        print(f"   Consider improving the model or collecting more training data.")
    
    # Check for uncertain predictions (close probabilities)
    sorted_probs = sorted([float(p) for p in probabilities], reverse=True)  # Convert to float
    if len(sorted_probs) > 1 and sorted_probs[0] - sorted_probs[1] < 0.2:
        print(f"\n⚠️  UNCERTAIN PREDICTION:")
        print(f"   Top two classes have similar probabilities.")
        # Fix the array indexing issue by converting to list first
        prob_list = [float(p) for p in probabilities]
        top_two_indices = sorted(range(len(prob_list)), key=lambda i: prob_list[i])[-2:]
        top_two_classes = [class_names[idx] for idx in top_two_indices]
        print(f"   The model is unsure between {top_two_classes}.")
    
    return predicted_class, confidence, probabilities


def save_model(
    model: tf.keras.Model,
    model_path: Path = MODELS_DIR / "character_state_classifier.keras",
    save_format: str = "keras"
) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained Keras model
        model_path: Path where to save the model
        save_format: Format to save the model ('keras' or 'h5')
    """
    
    # Create models directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_format == "keras":
        # Save in Keras native format (recommended)
        model.save(str(model_path))
        logger.info(f"Model saved in Keras format: {model_path}")
    elif save_format == "h5":
        # Save in HDF5 format
        h5_path = model_path.with_suffix('.h5')
        model.save(str(h5_path))
        logger.info(f"Model saved in HDF5 format: {h5_path}")
    else:
        raise ValueError("save_format must be 'keras' or 'h5'")
    
    print(f"Model successfully saved to: {model_path}")


def load_saved_model(model_path: Path) -> tf.keras.Model:
    """
    Load a previously saved model with proper handling of custom objects.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded Keras model
    """
    try:
        # Try loading with custom objects
        custom_objects = {
            'focal_loss_fn': focal_loss_fn,
        }
        model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
        logger.info(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        logger.warning(f"Failed to load model with focal loss: {str(e)}")
        logger.info("Attempting to load model without custom loss function...")
        
        try:
            # Try loading without custom objects and recompile
            model = tf.keras.models.load_model(str(model_path), compile=False)
            
            # Recompile with standard loss
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Model loaded from: {model_path} (recompiled with standard loss)")
            print("⚠️  Note: Model loaded with standard categorical crossentropy loss instead of focal loss")
            print("   The model should still work fine for predictions!")
            return model
            
        except Exception as e2:
            logger.error(f"Failed to load model: {str(e2)}")
            raise ValueError(f"Cannot load model from {model_path}. Error: {str(e2)}")


def plot_training_history(history: tf.keras.callbacks.History, save_path: Path = None) -> None:
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Training history object
        save_path: Optional path to save the plot
    """
    
    # Check what metrics are available in history
    available_metrics = list(history.history.keys())
    logger.info(f"Available metrics in training history: {available_metrics}")
    
    # Determine which metrics to plot
    has_accuracy = 'accuracy' in history.history
    has_val_accuracy = 'val_accuracy' in history.history
    has_loss = 'loss' in history.history
    has_val_loss = 'val_loss' in history.history
    
    # Create subplots based on available metrics
    if has_accuracy or has_loss:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax1, ax2 = axes
    else:
        logger.warning("No standard metrics found in training history")
        return
    
    # Plot accuracy if available
    if has_accuracy:
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        if has_val_accuracy:
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add accuracy range
        if has_val_accuracy:
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            ax1.text(0.02, 0.98, f'Final Training: {final_train_acc:.3f}\nFinal Validation: {final_val_acc:.3f}', 
                    transform=ax1.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax1.text(0.5, 0.5, 'No accuracy data available', 
                transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title('Model Accuracy (No Data)')
    
    # Plot loss if available
    if has_loss:
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        if has_val_loss:
            ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add loss values
        if has_val_loss:
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            ax2.text(0.02, 0.98, f'Final Training: {final_train_loss:.3f}\nFinal Validation: {final_val_loss:.3f}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No loss data available', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Model Loss (No Data)')
    
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot: {str(e)}")
    
    # Also save as PNG and show
    try:
        plt.show()
    except Exception:
        # In case display is not available
        logger.info("Display not available, plot saved to file only")
    
    plt.close()  # Clean up memory