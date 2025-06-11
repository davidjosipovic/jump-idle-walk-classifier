"""
CNN model architecture for video game character state classification.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports when running directly
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import tensorflow as tf
from typing import Tuple, Dict, Optional
import logging
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from config.config import (
    IMAGE_SIZE, NUM_CLASSES, CONV_FILTERS, DENSE_UNITS, 
    DROPOUT_RATE, LEARNING_RATE
)

logger = logging.getLogger(__name__)


def calculate_class_weights(class_counts: Dict[str, int]) -> Dict[int, float]:
    """
    Calculate class weights to handle imbalanced datasets.
    
    Args:
        class_counts: Dictionary mapping class names to their counts
        
    Returns:
        Dictionary mapping class indices to their weights
    """
    class_names = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Calculate weights using sklearn's balanced approach
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(class_names)),
        y=np.repeat(np.arange(len(class_names)), counts)
    )
    
    # Convert to dictionary format expected by Keras
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    logger.info("Calculated class weights for imbalanced dataset:")
    for i, (name, weight) in enumerate(zip(class_names, class_weights)):
        logger.info(f"  {name}: {weight:.3f}")
    
    return class_weight_dict


def create_cnn_model(
    input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3),
    num_classes: int = NUM_CLASSES
) -> tf.keras.Model:
    """
    Create a CNN model for character state classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # Rescaling layer for normalization
        tf.keras.layers.Rescaling(1./255),
        
        # First convolutional block
        tf.keras.layers.Conv2D(CONV_FILTERS[0], (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(CONV_FILTERS[1], (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(CONV_FILTERS[2], (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten layer
        tf.keras.layers.Flatten(),
        
        # Dense layers for classification
        tf.keras.layers.Dense(DENSE_UNITS[0], activation='relu'),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        
        tf.keras.layers.Dense(DENSE_UNITS[1], activation='relu'),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        
        # Output layer with softmax activation for multi-class classification
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    logger.info(f"Created CNN model with input shape: {input_shape}")
    logger.info(f"Model has {num_classes} output classes")
    
    return model


def compile_model(
    model: tf.keras.Model, 
    class_weights: Optional[Dict[int, float]] = None
) -> tf.keras.Model:
    """
    Compile the model with appropriate optimizer, loss function, and metrics.
    
    Args:
        model: Keras model to compile
        class_weights: Optional class weights for handling imbalanced data
        
    Returns:
        Compiled Keras model
    """
    
    # Use standard loss function - class weights will be passed to fit()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    if class_weights is not None:
        logger.info("Model compiled with class weights support")
    else:
        logger.info("Model compiled with standard configuration")
    
    logger.info(f"Model compiled with Adam optimizer (lr={LEARNING_RATE})")
    
    return model


def create_and_compile_model(
    input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    class_weights: Optional[Dict[int, float]] = None
) -> tf.keras.Model:
    """
    Create and compile a CNN model in one step.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        class_weights: Optional class weights for handling imbalanced data
        
    Returns:
        Compiled Keras model ready for training
    """
    model = create_cnn_model(input_shape, num_classes)
    model = compile_model(model, class_weights)
    
    return model