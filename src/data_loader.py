"""
Enhanced data loading utilities for video game character state classification.
Optimized for 80% accuracy target with advanced augmentation and preprocessing.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports when running directly
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
import cv2
import albumentations as A

from config.config import IMAGE_SIZE, BATCH_SIZE, CLASS_NAMES

logger = logging.getLogger(__name__)


def get_class_counts(data_dir: Path) -> Dict[str, int]:
    """Get count of images per class."""
    class_counts = {}
    
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if class_dir.exists():
            # Count all image files
            image_files = (
                list(class_dir.glob("*.jpg")) + 
                list(class_dir.glob("*.jpeg")) + 
                list(class_dir.glob("*.png")) + 
                list(class_dir.glob("*.PNG")) +
                list(class_dir.glob("*.JPG")) +
                list(class_dir.glob("*.JPEG"))
            )
            class_counts[class_name] = len(image_files)
        else:
            class_counts[class_name] = 0
    
    return class_counts


def create_advanced_augmentation() -> tf.keras.Sequential:
    """
    Create advanced data augmentation pipeline with TensorFlow.
    """
    return tf.keras.Sequential([
        # Geometric augmentations
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15, fill_mode='reflect'),
        tf.keras.layers.RandomZoom(0.15, fill_mode='reflect'),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='reflect'),
        
        # Color augmentations
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        
        # Additional augmentations for better generalization
        tf.keras.layers.RandomRotation(0.05),  # Additional slight rotation
    ], name="advanced_augmentation")


def create_albumentations_augmentation():
    """
    Create advanced augmentation pipeline using Albumentations.
    This provides more sophisticated augmentations.
    """
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=0.7
        ),
        A.ElasticTransform(
            alpha=1, sigma=50, alpha_affine=50,
            border_mode=cv2.BORDER_REFLECT,
            p=0.3
        ),
        
        # Color and lighting augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        
        # Coarse dropout for better generalization
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3
        ),
        
        # Ensure final image is in correct format
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ])


def apply_albumentations_augmentation(image: np.ndarray, augmentation_pipeline) -> np.ndarray:
    """Apply Albumentations augmentation to a single image."""
    # Convert from [0,1] to [0,255] if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Apply augmentation
    augmented = augmentation_pipeline(image=image)
    augmented_image = augmented['image']
    
    # Ensure output is float32 in [0,1] range
    if augmented_image.dtype != np.float32:
        augmented_image = augmented_image.astype(np.float32)
    
    if augmented_image.max() > 1.0:
        augmented_image = augmented_image / 255.0
    
    return augmented_image


def create_mixup_augmentation(alpha: float = 0.2):
    """
    Create MixUp augmentation for better generalization.
    
    Args:
        alpha: Beta distribution parameter for mixing ratio
        
    Returns:
        Function that applies MixUp to a batch
    """
    def mixup(images, labels):
        batch_size = tf.shape(images)[0]
        
        # Sample mixing ratio from Beta distribution
        mix_weight = tf.random.gamma([batch_size, 1, 1, 1], alpha, 1.0)
        mix_weight = tf.maximum(mix_weight, 1.0 - mix_weight)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Mix images
        mixed_images = (mix_weight * images + 
                       (1.0 - mix_weight) * tf.gather(images, indices))
        
        # Mix labels (one-hot encoded)
        labels_onehot = tf.one_hot(labels, depth=len(CLASS_NAMES))
        mixed_labels = (tf.squeeze(mix_weight) * labels_onehot + 
                       (1.0 - tf.squeeze(mix_weight)) * tf.gather(labels_onehot, indices))
        
        return mixed_images, mixed_labels
    
    return mixup


def create_cutmix_augmentation(alpha: float = 1.0):
    """
    Create CutMix augmentation for better generalization.
    
    Args:
        alpha: Beta distribution parameter for mixing ratio
        
    Returns:
        Function that applies CutMix to a batch
    """
    def cutmix(images, labels):
        batch_size = tf.shape(images)[0]
        image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]
        
        # Sample mixing ratio
        mix_weight = tf.random.beta([batch_size], alpha, alpha)
        
        # Calculate cut area
        cut_ratio = tf.sqrt(1.0 - mix_weight)
        cut_width = tf.cast(cut_ratio * tf.cast(image_width, tf.float32), tf.int32)
        cut_height = tf.cast(cut_ratio * tf.cast(image_height, tf.float32), tf.int32)
        
        # Random center point
        center_x = tf.random.uniform([batch_size], 0, image_width, dtype=tf.int32)
        center_y = tf.random.uniform([batch_size], 0, image_height, dtype=tf.int32)
        
        # Calculate bounding box
        x1 = tf.maximum(0, center_x - cut_width // 2)
        y1 = tf.maximum(0, center_y - cut_height // 2)
        x2 = tf.minimum(image_width, center_x + cut_width // 2)
        y2 = tf.minimum(image_height, center_y + cut_height // 2)
        
        # Shuffle indices for mixing
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        
        # Create mask and apply CutMix
        # This is a simplified version - full implementation would require more complex masking
        mixed_images = images  # Placeholder for actual CutMix implementation
        
        # Mix labels based on area ratio
        area_ratio = tf.cast((x2 - x1) * (y2 - y1), tf.float32) / tf.cast(image_height * image_width, tf.float32)
        labels_onehot = tf.one_hot(labels, depth=len(CLASS_NAMES))
        shuffled_labels_onehot = tf.one_hot(shuffled_labels, depth=len(CLASS_NAMES))
        
        mixed_labels = ((1.0 - tf.expand_dims(area_ratio, 1)) * labels_onehot + 
                       tf.expand_dims(area_ratio, 1) * shuffled_labels_onehot)
        
        return mixed_images, mixed_labels
    
    return cutmix


def balanced_sampling_augmentation(
    dataset: tf.data.Dataset, 
    class_counts: Dict[str, int], 
    target_count: Optional[int] = None
) -> tf.data.Dataset:
    """
    Apply balanced sampling with sophisticated augmentation for minority classes.
    
    Args:
        dataset: Input dataset
        class_counts: Count of samples per class
        target_count: Target number of samples per class
        
    Returns:
        Balanced dataset with augmentation
    """
    if target_count is None:
        target_count = max(class_counts.values())
    
    logger.info(f"Balancing dataset to {target_count} samples per class")
    
    # Create augmentation pipeline
    augmentation = create_advanced_augmentation()
    
    # Separate dataset by class
    class_datasets = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        current_count = class_counts[class_name]
        
        # Filter dataset for current class
        class_dataset = dataset.filter(lambda image, label: tf.equal(label, class_idx))
        
        if current_count < target_count:
            # Calculate how many augmented samples we need
            augment_factor = target_count // current_count
            remainder = target_count % current_count
            
            logger.info(f"Augmenting {class_name}: {current_count} -> {target_count} samples")
            
            # Create multiple augmented versions
            augmented_datasets = [class_dataset]  # Original data
            
            for _ in range(augment_factor - 1):
                augmented_data = class_dataset.map(
                    lambda image, label: (augmentation(image, training=True), label),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
                augmented_datasets.append(augmented_data)
            
            # Add partial augmentation for remainder
            if remainder > 0:
                partial_augmented = class_dataset.take(remainder).map(
                    lambda image, label: (augmentation(image, training=True), label),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
                augmented_datasets.append(partial_augmented)
            
            # Combine all augmented versions
            balanced_class_dataset = augmented_datasets[0]
            for aug_ds in augmented_datasets[1:]:
                balanced_class_dataset = balanced_class_dataset.concatenate(aug_ds)
                
        else:
            # If class has enough samples, just take what we need
            balanced_class_dataset = class_dataset.take(target_count)
        
        class_datasets.append(balanced_class_dataset)
    
    # Combine all balanced class datasets
    balanced_dataset = class_datasets[0]
    for ds in class_datasets[1:]:
        balanced_dataset = balanced_dataset.concatenate(ds)
    
    # Shuffle the final balanced dataset
    balanced_dataset = balanced_dataset.shuffle(buffer_size=target_count * len(CLASS_NAMES))
    
    return balanced_dataset


def create_advanced_preprocessing_pipeline():
    """
    Create advanced preprocessing pipeline with multiple techniques.
    """
    def preprocess_image(image, label):
        # Ensure image is float32
        image = tf.cast(image, tf.float32)
        
        # Normalize to [0, 1] range
        image = image / 255.0
        
        # Apply histogram equalization for better contrast
        # Convert to grayscale for histogram equalization, then back to RGB
        gray = tf.image.rgb_to_grayscale(image)
        
        # Apply CLAHE-like enhancement
        # This is a simplified version - full CLAHE would require more complex implementation
        enhanced = tf.image.adjust_contrast(gray, 1.2)
        enhanced = tf.image.adjust_brightness(enhanced, 0.1)
        
        # Convert back to RGB by replicating channels
        enhanced_rgb = tf.image.grayscale_to_rgb(enhanced)
        
        # Blend with original image
        image = 0.7 * image + 0.3 * enhanced_rgb
        
        # Ensure values are in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    return preprocess_image


def load_advanced_image_datasets(
    data_dir: Path,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: Optional[float] = None,
    apply_augmentation: bool = True,
    use_mixup: bool = True,
    augmentation_type: str = "tensorflow"  # "tensorflow" or "albumentations"
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
    """
    Load image datasets with advanced preprocessing and augmentation.
    
    Args:
        data_dir: Directory containing train/validation/test subdirectories
        image_size: Target image size
        batch_size: Batch size for training
        validation_split: Validation split ratio if no separate validation dir
        apply_augmentation: Whether to apply data augmentation
        use_mixup: Whether to apply MixUp augmentation
        augmentation_type: Type of augmentation to use
        
    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset, class_counts)
    """
    train_dir = data_dir / "train"
    class_counts = get_class_counts(train_dir)
    
    logger.info(f"Loading datasets from {data_dir}")
    logger.info(f"Class distribution: {class_counts}")
    
    # Load datasets based on directory structure
    if validation_split is not None and (data_dir / "validation").exists() is False:
        # Use automatic validation split from training data
        logger.info(f"Using automatic validation split ({validation_split:.0%}) from training data")
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=None,  # We'll batch later
            class_names=CLASS_NAMES
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=None,
            class_names=CLASS_NAMES
        )
        
        # Use a portion of validation for testing
        val_size = len(list(val_ds))
        test_size = max(1, val_size // 3)
        test_ds = val_ds.take(test_size)
        val_ds = val_ds.skip(test_size)
        
    else:
        # Use separate directories
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=image_size,
            batch_size=None,
            class_names=CLASS_NAMES
        )
        
        val_dir = data_dir / "validation"
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=image_size,
            batch_size=None,
            class_names=CLASS_NAMES
        )
        
        test_dir = data_dir / "test"
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=image_size,
            batch_size=None,
            class_names=CLASS_NAMES
        )
    
    # Apply advanced preprocessing
    preprocessing = create_advanced_preprocessing_pipeline()
    train_ds = train_ds.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation and balancing to training data
    if apply_augmentation:
        logger.info("Applying advanced augmentation and class balancing...")
        train_ds = balanced_sampling_augmentation(train_ds, class_counts)
        
        if augmentation_type == "albumentations":
            # Note: Albumentations integration would require tf.py_function
            # For now, use TensorFlow augmentation
            logger.info("Using TensorFlow-based augmentation")
        
        # Apply additional augmentations
        augmentation = create_advanced_augmentation()
        train_ds = train_ds.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Batch the datasets
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    
    # Apply MixUp to training data
    if use_mixup and apply_augmentation:
        logger.info("Applying MixUp augmentation to training data")
        mixup_fn = create_mixup_augmentation(alpha=0.2)
        train_ds = train_ds.map(mixup_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    logger.info(f"Loaded datasets with advanced preprocessing:")
    logger.info(f"  Image size: {image_size}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Augmentation: {apply_augmentation}")
    logger.info(f"  MixUp: {use_mixup}")
    logger.info(f"  Class names: {CLASS_NAMES}")
    
    return train_ds, val_ds, test_ds, class_counts


def load_single_image(image_path: Path, image_size: Tuple[int, int] = IMAGE_SIZE) -> tf.Tensor:
    """
    Load and preprocess a single image for prediction with advanced preprocessing.
    
    Args:
        image_path: Path to the image file
        image_size: Target size for the image
        
    Returns:
        Preprocessed image tensor ready for model prediction
    """
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    
    # Resize with high-quality method
    image = tf.image.resize(image, image_size, method='lanczos3')
    image = tf.cast(image, tf.float32) / 255.0
    
    # Apply the same preprocessing as training
    preprocessing = create_advanced_preprocessing_pipeline()
    image, _ = preprocessing(image, 0)  # Dummy label
    
    # Add batch dimension
    image = tf.expand_dims(image, 0)
    
    return image


def load_elite_image_datasets(
    data_dir: Path,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: Optional[float] = None,
    apply_augmentation: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
    """
    Load image datasets with elite preprocessing but without MixUp for class weights compatibility.
    
    Args:
        data_dir: Directory containing train/validation/test subdirectories
        image_size: Target image size
        batch_size: Batch size for training
        validation_split: Validation split ratio if no separate validation dir
        apply_augmentation: Whether to apply data augmentation
        
    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset, class_counts)
    """
    train_dir = data_dir / "train"
    class_counts = get_class_counts(train_dir)
    
    logger.info(f"Loading elite datasets from {data_dir}")
    logger.info(f"Class distribution: {class_counts}")
    
    # Load datasets based on directory structure
    if validation_split is not None and (data_dir / "validation").exists() is False:
        # Use automatic validation split from training data
        logger.info(f"Using automatic validation split ({validation_split:.0%}) from training data")
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=None,  # We'll batch later
            class_names=CLASS_NAMES
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=None,
            class_names=CLASS_NAMES
        )
        
        # Use a portion of validation for testing
        val_size = len(list(val_ds))
        test_size = max(1, val_size // 3)
        test_ds = val_ds.take(test_size)
        val_ds = val_ds.skip(test_size)
        
    else:
        # Use separate directories
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=image_size,
            batch_size=None,
            class_names=CLASS_NAMES
        )
        
        val_dir = data_dir / "validation"
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=image_size,
            batch_size=None,
            class_names=CLASS_NAMES
        )
        
        test_dir = data_dir / "test"
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=image_size,
            batch_size=None,
            class_names=CLASS_NAMES
        )
    
    # Apply advanced preprocessing
    def elite_preprocess(image, label):
        # Ensure image is float32
        image = tf.cast(image, tf.float32) / 255.0
        
        # Apply contrast enhancement
        enhanced = tf.image.adjust_contrast(image, 1.1)
        enhanced = tf.image.adjust_brightness(enhanced, 0.05)
        
        # Blend with original
        image = 0.8 * image + 0.2 * enhanced
        
        # Ensure values are in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    train_ds = train_ds.map(elite_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(elite_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(elite_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation and balancing to training data
    if apply_augmentation:
        logger.info("Applying elite augmentation and class balancing...")
        train_ds = balanced_sampling_augmentation(train_ds, class_counts)
        
        # Apply additional augmentations
        augmentation = create_advanced_augmentation()
        train_ds = train_ds.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Batch the datasets
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    logger.info(f"Loaded elite datasets:")
    logger.info(f"  Image size: {image_size}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Augmentation: {apply_augmentation}")
    logger.info(f"  Class names: {CLASS_NAMES}")
    
    return train_ds, val_ds, test_ds, class_counts