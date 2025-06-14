"""
Prediction script for video game character state classification.
Includes diagnostics and bias detection capabilities.
"""

import argparse
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import MODELS_DIR, CLASS_NAMES, DATA_DIR
from src.trainer import load_saved_model, predict_single_image


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_model_on_samples(model: tf.keras.Model, test_dir: Path, num_samples: int = 3) -> None:
    """Test model on sample images from each class."""
    print(f"\n🔍 TESTING MODEL ON SAMPLE IMAGES")
    print("="*50)
    
    for class_name in CLASS_NAMES:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            print(f"⚠️  Directory not found: {class_dir}")
            continue
            
        # Get sample images
        image_files = list(class_dir.glob("*.PNG")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        
        if not image_files:
            print(f"⚠️  No images found in {class_dir}")
            continue
            
        sample_files = image_files[:num_samples]
        
        print(f"\n📁 Testing {class_name} images:")
        correct_predictions = 0
        
        for img_path in sample_files:
            try:
                predicted_class, confidence, _ = predict_single_image(
                    model, img_path, CLASS_NAMES, confidence_threshold=0.3
                )
                
                is_correct = predicted_class.lower() == class_name.lower()
                status = "✅" if is_correct else "❌"
                print(f"   {status} {img_path.name}: {predicted_class} ({confidence:.1%})")
                
                if is_correct:
                    correct_predictions += 1
                    
            except Exception as e:
                print(f"   ❌ Error processing {img_path.name}: {str(e)}")
        
        # Summary for this class
        if sample_files:
            accuracy = correct_predictions / len(sample_files)
            print(f"   📊 Class Accuracy: {correct_predictions}/{len(sample_files)} ({accuracy:.1%})")


def diagnose_model_bias(model: tf.keras.Model) -> None:
    """Run comprehensive bias diagnosis on training and validation data."""
    print(f"\n🔬 MODEL BIAS DIAGNOSIS")
    print("="*50)
    
    data_dir = DATA_DIR / "processed"
    
    # Test on training data
    train_dir = data_dir / "train"
    if train_dir.exists():
        print(f"\n📋 Testing on training data:")
        test_model_on_samples(model, train_dir, num_samples=2)
    
    # Test on validation data if available
    val_dir = data_dir / "validation"
    if val_dir.exists():
        print(f"\n📋 Testing on validation data:")
        test_model_on_samples(model, val_dir, num_samples=2)


def main() -> None:
    """Main prediction workflow."""
    parser = argparse.ArgumentParser(description='Character state prediction with diagnostics')
    parser.add_argument('image_path', type=str, nargs='?', 
                       help='Path to the image file (optional for diagnostics mode)')
    parser.add_argument('--model_path', type=str, 
                       default=str(MODELS_DIR / "ultra_enhanced_classifier.keras"),
                       help='Path to the trained model')
    parser.add_argument('--diagnose', action='store_true', 
                       help='Run bias diagnosis on training data')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='Minimum confidence threshold for reliable predictions')
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load the model
        model_path = Path(args.model_path)
        if not model_path.exists():
            # Try fallback to balanced model
            fallback_path = MODELS_DIR / "balanced_character_classifier.keras"
            if fallback_path.exists():
                print(f"⚠️  Model {model_path.name} not found, using fallback: {fallback_path.name}")
                model_path = fallback_path
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
        logger.info(f"Loading model from {model_path}")
        model = load_saved_model(model_path)
        
        print(f"\n🧠 Model loaded: {model_path.name}")
        print(f"📊 Model summary:")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output classes: {CLASS_NAMES}")
        
        # Run diagnostics if requested
        if args.diagnose:
            diagnose_model_bias(model)
            return
        
        # Single image prediction
        if args.image_path:
            image_path = Path(args.image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            logger.info(f"Making prediction for {image_path}")
            predicted_class, confidence, probabilities = predict_single_image(
                model, image_path, CLASS_NAMES, confidence_threshold=args.confidence_threshold
            )
            
            logger.info("Prediction completed successfully!")
            
        else:
            print(f"\n📋 Usage Examples:")
            print(f"   Single prediction: python scripts/predict.py path/to/image.png")
            print(f"   Run diagnostics: python scripts/predict.py --diagnose")
            print(f"   Ultra-enhanced model: python scripts/predict.py path/to/image.png --model_path models/ultra_enhanced_classifier.keras")
            print(f"   Custom confidence: python scripts/predict.py path/to/image.png --confidence_threshold 0.8")

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


if __name__ == "__main__":
    main()