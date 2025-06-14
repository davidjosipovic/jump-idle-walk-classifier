"""
Configuration file for video game character state classification project.
"""

from pathlib import Path
from typing import Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data directories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

# Model parameters
IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE: int = 8  # Reduced batch size for better gradient updates with small dataset
NUM_CLASSES: int = 3
CLASS_NAMES = ["idle", "walking", "jumping"]

# Training parameters - Enhanced for longer training
EPOCHS: int = 50  # Increased from 20 to 50 for better convergence with imbalanced data
LEARNING_RATE: float = 0.0001  # Reduced for more stable training with severe imbalance
VALIDATION_SPLIT: float = 0.2

# Model architecture
CONV_FILTERS = [32, 64, 128]
DENSE_UNITS = [128, 64]
DROPOUT_RATE: float = 0.3  # Reduced dropout for small dataset to prevent underfitting