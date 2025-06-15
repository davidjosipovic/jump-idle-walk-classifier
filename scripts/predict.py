#!/usr/bin/env python3
"""
Improved Video Game Character State Classifier - Prediction
==========================================================

Improved prediction script with better diagnostics and fixed bugs.

Usage:
    python improved_predict.py image.png           # Single image
    python improved_predict.py folder/             # All images in folder
    python improved_predict.py --test              # Test on validation data
"""

import sys
import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
CLASS_NAMES = ['idle', 'jumping', 'walking']
IMAGE_SIZE = (224, 224)

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'


class ImprovedPredictor:
    """Improved character state predictor with better diagnostics."""
    
    def __init__(self, model_path=None):
        self.class_names = CLASS_NAMES
        self.image_size = IMAGE_SIZE
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model."""
        if model_path is None:
            # Try improved model first, then fallback to simple
            possible_paths = [
                MODELS_DIR / 'improved_character_classifier.keras',
                MODELS_DIR / 'simple_character_classifier.keras',
                MODELS_DIR / 'character_classifier_final.keras',
                MODELS_DIR / 'GameCharacterStateClassifier_final.keras'
            ]
            
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError("No trained model found! Please train a model first.")
        
        print(f"🧠 Loading model: {Path(model_path).name}")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction."""
        try:
            # Load and resize image
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.image_size)
            
            # Convert to array and normalize
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
        except Exception as e:
            print(f"❌ Error preprocessing {image_path}: {e}")
            return None
    
    def predict_single(self, image_path, show_details=True):
        """Predict a single image."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"❌ File not found: {image_path}")
            return None
        
        # Preprocess
        image_array = self.preprocess_image(image_path)
        if image_array is None:
            return None
        
        # Predict
        predictions = self.model.predict(image_array, verbose=0)
        probabilities = predictions[0]
        
        # Get results
        predicted_index = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_index]
        confidence = probabilities[predicted_index] * 100
        
        if show_details:
            print(f"\n🎯 PREDICTION: {image_path.name}")
            print("=" * 40)
            print(f"🏷️  Class: {predicted_class.upper()}")
            print(f"📊 Confidence: {confidence:.1f}%")
            
            if confidence > 80:
                print("✅ High confidence")
            elif confidence > 60:
                print("⚠️ Medium confidence")
            else:
                print("❌ Low confidence")
            
            print(f"\n📈 All probabilities:")
            for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities)):
                bar = "█" * int(prob * 20)  # Visual bar
                print(f"   {class_name:>8}: {prob*100:5.1f}% {bar}")
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'all_classes': self.class_names
        }
    
    def predict_folder(self, folder_path):
        """Predict all images in a folder."""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"❌ Folder not found: {folder_path}")
            return
        
        # Find image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ No image files found in {folder_path}")
            return
        
        print(f"🔍 Found {len(image_files)} images in {folder_path}")
        print("=" * 50)
        
        results = {}
        for image_file in sorted(image_files):
            result = self.predict_single(image_file, show_details=False)
            if result:
                results[image_file.name] = result
                
                # Simple one-line output
                class_name = result['class']
                confidence = result['confidence']
                status = "✅" if confidence > 60 else "⚠️"
                print(f"{status} {image_file.name}: {class_name} ({confidence:.1f}%)")
        
        # Summary
        if results:
            class_counts = {}
            for result in results.values():
                class_name = result['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print(f"\n📊 Summary:")
            for class_name, count in class_counts.items():
                print(f"   {class_name}: {count} images")
    
    def test_validation_data(self):
        """Test on validation dataset."""
        val_dir = DATA_DIR / 'test'
        
        if not val_dir.exists():
            print(f"❌ Validation directory not found: {val_dir}")
            return
        
        print("🧪 Testing on validation data...")
        
        overall_correct = 0
        overall_total = 0
        
        for class_name in self.class_names:
            class_dir = val_dir / class_name
            
            if not class_dir.exists():
                print(f"⚠️ No validation data for class: {class_name}")
                continue
            
            # Find images
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(class_dir.glob(f'*{ext}'))
                image_files.extend(class_dir.glob(f'*{ext.upper()}'))
            
            if not image_files:
                print(f"⚠️ No images found for class: {class_name}")
                continue
            
            print(f"\n📁 Testing {class_name} class ({len(image_files)} images):")
            
            correct = 0
            for image_file in sorted(image_files):
                result = self.predict_single(image_file, show_details=False)
                if result:
                    predicted = result['class']
                    confidence = result['confidence']
                    
                    if predicted == class_name:
                        status = "✅"
                        correct += 1
                        overall_correct += 1
                    else:
                        status = "❌"
                    
                    print(f"   {status} {image_file.name}: {predicted} ({confidence:.1f}%)")
                    overall_total += 1
            
            if len(image_files) > 0:
                accuracy = (correct / len(image_files)) * 100
                print(f"   📊 {class_name} accuracy: {accuracy:.1f}%")
        
        if overall_total > 0:
            overall_accuracy = (overall_correct / overall_total) * 100
            print(f"\n🎯 Overall accuracy: {overall_accuracy:.1f}% ({overall_correct}/{overall_total})")


def main():
    parser = argparse.ArgumentParser(description='Improved Character State Prediction')
    parser.add_argument('input', nargs='?', help='Image file or folder path')
    parser.add_argument('--test', action='store_true', help='Test on validation data')
    parser.add_argument('--model', help='Path to model file')
    
    args = parser.parse_args()
    
    print("🎮 Improved Character State Predictor")
    print("=" * 40)
    
    try:
        predictor = ImprovedPredictor(args.model)
        
        if args.test:
            predictor.test_validation_data()
        elif args.input:
            input_path = Path(args.input)
            
            if input_path.is_file():
                predictor.predict_single(input_path)
            elif input_path.is_dir():
                predictor.predict_folder(input_path)
            else:
                print(f"❌ File/folder not found: {input_path}")
        else:
            print("❌ Please provide an image file, folder, or use --test")
            print("\nUsage examples:")
            print("  python improved_predict.py image.png")
            print("  python improved_predict.py folder/")
            print("  python improved_predict.py --test")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
