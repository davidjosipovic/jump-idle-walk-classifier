"""
Data loading utilities for video game character state classification.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports when running directly
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import tensorflow as tf
from typing import Tuple, Optional, Dict
import logging

from config.config import IMAGE_SIZE, BATCH_SIZE, CLASS_NAMES

logger = logging.getLogger(__name__)


def get_class_counts(data_dir: Path) -> Dict[str, int]:
    """
    Count the number of images in each class directory.
    
    Args:
        data_dir: Directory containing class subdirectories
        
    Returns:
        Dictionary mapping class names to their counts
    """
    class_counts = {}
    
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if class_dir.exists():
            # Count images with common extensions
            count = 0
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.PNG', '*.webp', '*.avif', '*.bmp', '*.gif']:
                count += len(list(class_dir.glob(ext)))
            class_counts[class_name] = count
        else:
            class_counts[class_name] = 0
            
    logger.info("Class distribution:")
    total_images = sum(class_counts.values())
    for class_name, count in class_counts.items():
        percentage = (count / total_images * 100) if total_images > 0 else 0
        logger.info(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    return class_counts


def create_data_augmentation() -> tf.keras.Sequential:
    """
    Create data augmentation pipeline to increase dataset diversity.
    
    Returns:
        Data augmentation sequential model
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])


def augment_minority_classes(
    dataset: tf.data.Dataset, 
    class_counts: Dict[str, int], 
    target_count: Optional[int] = None
) -> tf.data.Dataset:
    """
    Augment minority classes to balance the dataset.
    
    Args:
        dataset: Original dataset
        class_counts: Dictionary of class counts
        target_count: Target number of samples per class (uses max if None)
        
    Returns:
        Balanced dataset with augmented minority classes
    """
    if target_count is None:
        target_count = max(class_counts.values())
    
    # Create augmentation pipeline
    augmentation = create_data_augmentation()
    
    # Calculate how many times to repeat each class
    class_multipliers = {}
    for i, class_name in enumerate(CLASS_NAMES):
        current_count = class_counts[class_name]
        if current_count > 0:
            multiplier = max(1, target_count // current_count)
            class_multipliers[i] = multiplier
        else:
            class_multipliers[i] = 1
    
    logger.info("Class augmentation multipliers:")
    for i, class_name in enumerate(CLASS_NAMES):
        logger.info(f"  {class_name}: {class_multipliers[i]}x")
    
    # Convert multipliers to tensors for TensorFlow operations
    multiplier_tensor = tf.constant([class_multipliers[i] for i in range(len(CLASS_NAMES))], dtype=tf.int32)
    
    def augment_sample(image, label):
        # Get multiplier for this class using tf.gather
        multiplier = tf.gather(multiplier_tensor, label)
        
        # Only augment if multiplier > 1
        should_augment = tf.greater(multiplier, 1)
        
        # Apply augmentation conditionally
        augmented_image = tf.cond(
            should_augment,
            lambda: augmentation(image, training=True),
            lambda: image
        )
        
        return augmented_image, label
    
    # Apply augmentation and repeat minority classes
    balanced_datasets = []
    
    for i, class_name in enumerate(CLASS_NAMES):
        # Filter dataset for this class
        class_dataset = dataset.filter(lambda x, y: tf.equal(y, i))
        
        # Repeat the class dataset if needed
        if class_multipliers[i] > 1:
            class_dataset = class_dataset.repeat(class_multipliers[i])
            # Apply augmentation to repeated samples
            class_dataset = class_dataset.map(augment_sample, num_parallel_calls=tf.data.AUTOTUNE)
        
        balanced_datasets.append(class_dataset)
    
    # Combine all class datasets
    balanced_dataset = balanced_datasets[0]
    for ds in balanced_datasets[1:]:
        balanced_dataset = balanced_dataset.concatenate(ds)
    
    # Shuffle the combined dataset
    balanced_dataset = balanced_dataset.shuffle(buffer_size=1000)
    
    return balanced_dataset


def load_image_datasets(
    data_dir: Path,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: Optional[float] = None,
    apply_augmentation: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
    """
    Load image datasets from directory structure using TensorFlow's image_dataset_from_directory.
    
    Args:
        data_dir: Base directory containing train, validation, and test subdirectories
        image_size: Target size for images (height, width)
        batch_size: Batch size for datasets
        validation_split: If provided, split training data for validation
        apply_augmentation: Whether to apply data augmentation to balance classes
        
    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset, class_counts)
    """
    
    train_dir = data_dir / "train"
    
    # Get class counts for imbalance handling
    class_counts = get_class_counts(train_dir)
    
    if validation_split is not None:
        # Use automatic validation split from train directory
        logger.info(f"Using automatic validation split ({validation_split:.0%}) from training data")
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=None,  # We'll batch later after augmentation
            class_names=CLASS_NAMES
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            class_names=CLASS_NAMES
        )
        
        # For test set, use a small portion of validation data
        test_ds = val_ds.take(max(1, len(list(val_ds)) // 3))
        val_ds = val_ds.skip(max(1, len(list(val_ds)) // 3))
        
    else:
        # Use separate directories
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=image_size,
            batch_size=None,  # We'll batch later after augmentation
            class_names=CLASS_NAMES
        )
        
        # Load validation dataset
        val_dir = data_dir / "validation"
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=image_size,
            batch_size=batch_size,
            class_names=CLASS_NAMES
        )
        
        # Load test dataset
        test_dir = data_dir / "test"
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=image_size,
            batch_size=batch_size,
            class_names=CLASS_NAMES
        )
    
    # Apply augmentation to balance classes in training data
    if apply_augmentation:
        logger.info("Applying data augmentation to balance classes...")
        train_ds = augment_minority_classes(train_ds, class_counts)
    
    # Batch the training dataset after augmentation
    train_ds = train_ds.batch(batch_size)
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    logger.info(f"Loaded datasets with image size: {image_size}, batch size: {batch_size}")
    logger.info(f"Class names: {CLASS_NAMES}")
    
    return train_ds, val_ds, test_ds, class_counts


def load_single_image(image_path: Path, image_size: Tuple[int, int] = IMAGE_SIZE) -> tf.Tensor:
    """
    Load and preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        image_size: Target size for the image
        
    Returns:
        Preprocessed image tensor ready for model prediction
    """
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    image = tf.expand_dims(image, 0)  # Add batch dimension
    
    return image