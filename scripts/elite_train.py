#!/usr/bin/env python3
"""
Elite Training Script - Advanced techniques to achieve 80%+ accuracy
Implements state-of-the-art techniques for video game character state classification
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from config.config import Config

class EliteTrainer:
    def __init__(self):
        self.config = Config()
        self.setup_logging()
        self.setup_gpu()
        
    def setup_logging(self):
        """Setup enhanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/elite_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_gpu(self):
        """Optimize GPU usage"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"GPU acceleration enabled: {len(gpus)} GPU(s)")
            except RuntimeError as e:
                self.logger.warning(f"GPU setup failed: {e}")
        else:
            self.logger.info("Running on CPU")

    def create_elite_data_generators(self):
        """Create elite data generators with advanced augmentation"""
        
        # Elite training augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.15,
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1,
            fill_mode='nearest'
        )
        
        # Validation/test without augmentation
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load data with optimal settings
        train_generator = train_datagen.flow_from_directory(
            'data/boosted/train',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            'data/boosted/validation',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            'data/boosted/test',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator

    def create_ensemble_model(self):
        """Create an ensemble of pre-trained models"""
        input_tensor = layers.Input(shape=(224, 224, 3))
        
        # Branch 1: EfficientNetB0
        base1 = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )
        base1.trainable = False
        x1 = layers.GlobalAveragePooling2D()(base1.output)
        x1 = layers.Dropout(0.3)(x1)
        
        # Branch 2: ResNet50V2  
        base2 = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )
        base2.trainable = False
        x2 = layers.GlobalAveragePooling2D()(base2.output)
        x2 = layers.Dropout(0.3)(x2)
        
        # Branch 3: MobileNetV3Large
        base3 = MobileNetV3Large(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor,
            include_preprocessing=False
        )
        base3.trainable = False
        x3 = layers.GlobalAveragePooling2D()(base3.output)
        x3 = layers.Dropout(0.3)(x3)
        
        # Combine features
        combined = layers.Concatenate()([x1, x2, x3])
        combined = layers.Dense(512, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.5)(combined)
        
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.4)(combined)
        
        # Output layer
        outputs = layers.Dense(3, activation='softmax', name='predictions')(combined)
        
        model = models.Model(inputs=input_tensor, outputs=outputs)
        
        # Store base models for fine-tuning
        self.base_models = [base1, base2, base3]
        
        return model

    def compile_model(self, model, learning_rate=1e-4):
        """Compile model with optimal settings"""
        optimizer = optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-5
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        return model

    def get_elite_callbacks(self, stage="stage1"):
        """Get elite callbacks for training"""
        callbacks_list = [
            callbacks.ModelCheckpoint(
                f'models/checkpoints/elite_{stage}.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.CSVLogger(f'logs/elite_{stage}_training.csv'),
        ]
        
        return callbacks_list

    def train_stage1(self, model, train_gen, val_gen):
        """Stage 1: Train with frozen base models"""
        self.logger.info("🚀 Starting Stage 1: Training with frozen backbones...")
        
        history1 = model.fit(
            train_gen,
            epochs=60,
            validation_data=val_gen,
            callbacks=self.get_elite_callbacks("stage1"),
            verbose=1
        )
        
        return history1

    def train_stage2(self, model, train_gen, val_gen):
        """Stage 2: Fine-tune top layers of base models"""
        self.logger.info("🔥 Starting Stage 2: Fine-tuning top layers...")
        
        # Unfreeze top layers of each base model
        for base_model in self.base_models:
            base_model.trainable = True
            # Freeze all but last few layers
            for layer in base_model.layers[:-20]:
                layer.trainable = False
        
        # Recompile with lower learning rate
        model = self.compile_model(model, learning_rate=1e-5)
        
        history2 = model.fit(
            train_gen,
            epochs=40,
            validation_data=val_gen,
            callbacks=self.get_elite_callbacks("stage2"),
            verbose=1
        )
        
        return history2

    def train_stage3(self, model, train_gen, val_gen):
        """Stage 3: Full fine-tuning with very low learning rate"""
        self.logger.info("⚡ Starting Stage 3: Full fine-tuning...")
        
        # Unfreeze all layers
        for base_model in self.base_models:
            base_model.trainable = True
        
        # Recompile with very low learning rate
        model = self.compile_model(model, learning_rate=5e-6)
        
        history3 = model.fit(
            train_gen,
            epochs=30,
            validation_data=val_gen,
            callbacks=self.get_elite_callbacks("stage3"),
            verbose=1
        )
        
        return history3

    def evaluate_model(self, model, test_gen):
        """Comprehensive model evaluation"""
        self.logger.info("📊 Performing comprehensive evaluation...")
        
        # Get predictions
        test_gen.reset()
        predictions = model.predict(test_gen, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = test_gen.classes
        class_labels = list(test_gen.class_indices.keys())
        
        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        accuracy = np.mean(predicted_classes == true_classes)
        report = classification_report(true_classes, predicted_classes, 
                                     target_names=class_labels, digits=3)
        
        self.logger.info(f"\n🎯 ELITE MODEL EVALUATION:")
        self.logger.info(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        self.logger.info(f"\nDetailed Report:\n{report}")
        
        return accuracy, predictions

    def plot_training_history(self, histories, save_path):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Combine all histories
        all_acc = []
        all_val_acc = []
        all_loss = []
        all_val_loss = []
        
        for hist in histories:
            all_acc.extend(hist.history['accuracy'])
            all_val_acc.extend(hist.history['val_accuracy'])
            all_loss.extend(hist.history['loss'])
            all_val_loss.extend(hist.history['val_loss'])
        
        epochs = range(1, len(all_acc) + 1)
        
        # Accuracy
        axes[0,0].plot(epochs, all_acc, 'b-', label='Training Accuracy')
        axes[0,0].plot(epochs, all_val_acc, 'r-', label='Validation Accuracy')
        axes[0,0].set_title('Model Accuracy')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Loss
        axes[0,1].plot(epochs, all_loss, 'b-', label='Training Loss')
        axes[0,1].plot(epochs, all_val_loss, 'r-', label='Validation Loss')
        axes[0,1].set_title('Model Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Final accuracy comparison
        final_acc = [max(h.history['val_accuracy']) for h in histories]
        stage_names = ['Stage 1', 'Stage 2', 'Stage 3']
        axes[1,0].bar(stage_names, final_acc[:len(stage_names)])
        axes[1,0].set_title('Best Validation Accuracy by Stage')
        axes[1,0].set_ylabel('Accuracy')
        
        # Final metrics
        axes[1,1].text(0.1, 0.8, f'Final Validation Accuracy: {max(all_val_acc):.3f}', 
                      transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].text(0.1, 0.6, f'Best Stage: {stage_names[np.argmax(final_acc[:len(stage_names)])]}', 
                      transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].text(0.1, 0.4, f'Total Epochs: {len(epochs)}', 
                      transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title('Training Summary')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"📈 Training history saved: {save_path}")

    def run_elite_training(self):
        """Run the complete elite training pipeline"""
        self.logger.info("🏆 Starting ELITE Training Pipeline...")
        
        # Create data generators
        train_gen, val_gen, test_gen = self.create_elite_data_generators()
        
        # Create ensemble model
        model = self.create_ensemble_model()
        model = self.compile_model(model)
        
        self.logger.info(f"Model created with {model.count_params():,} parameters")
        
        # Multi-stage training
        histories = []
        
        # Stage 1: Frozen backbones
        history1 = self.train_stage1(model, train_gen, val_gen)
        histories.append(history1)
        
        # Stage 2: Fine-tune top layers
        history2 = self.train_stage2(model, train_gen, val_gen)
        histories.append(history2)
        
        # Stage 3: Full fine-tuning
        history3 = self.train_stage3(model, train_gen, val_gen)
        histories.append(history3)
        
        # Final evaluation
        accuracy, predictions = self.evaluate_model(model, test_gen)
        
        # Save final model
        model_path = 'models/elite_character_classifier.keras'
        model.save(model_path)
        self.logger.info(f"💾 Elite model saved: {model_path}")
        
        # Plot training history
        self.plot_training_history(histories, 'logs/elite_training_history.png')
        
        # Final results
        print("\n" + "="*70)
        print("🏆 ELITE TRAINING COMPLETED!")
        print("="*70)
        print(f"🎯 Final Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        if accuracy >= 0.80:
            print("🎉 TARGET ACHIEVED! 80%+ accuracy reached!")
        else:
            print(f"📈 Progress made! {(accuracy*100):.1f}% accuracy achieved")
        print(f"💾 Model saved: {model_path}")
        print(f"📈 Training plot: logs/elite_training_history.png")
        print("="*70)
        
        return model, accuracy

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)
    
    # Run elite training
    trainer = EliteTrainer()
    model, accuracy = trainer.run_elite_training()