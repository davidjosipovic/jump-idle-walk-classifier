# Video Game Character State Classification

This is a machine learning project focused on classifying video game character states using computer vision and deep learning techniques.

## 🎯 Project Overview

The system can classify three character states:
- **Idle**: Character standing still
- **Walking**: Character in walking motion  
- **Jumping**: Character in jumping motion

## 🚀 Quick Start

### Training a Model

```bash
python scripts/train.py
```

This will:
1. Analyze data distribution and handle class imbalance
2. Create a CNN model with regularization
3. Train with data augmentation and class weights
4. Save the trained model and training plots

### Making Predictions

```bash
# Single image prediction
python scripts/predict.py path/to/your/image.png

# Run model diagnostics
python scripts/predict.py --diagnose

# Use custom model
python scripts/predict.py image.png --model_path models/my_model.keras
```

## 📁 Project Structure

```
├── README.md
├── requirements.txt
├── config/
│   └── config.py            # Configuration settings
├── data/
│   └── processed/           # Training datasets
│       ├── train/           # Training images by class
│       └── validation/      # Validation images by class
├── models/
│   └── character_state_classifier.keras  # Trained model
├── logs/                    # Training logs and plots
├── scripts/
│   ├── train.py            # Main training script
│   └── predict.py          # Prediction and diagnostics
└── src/
    ├── data_loader.py      # Data loading utilities
    ├── model.py            # CNN model architecture
    └── trainer.py          # Training and evaluation utilities
```

## 🛠️ Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the package:
   ```bash
   pip install -e .
   ```

## 📊 Model Performance

The model uses advanced techniques to handle data imbalance:
- **Class weights** to penalize minority class errors
- **Data augmentation** for better generalization
- **Regularization** to prevent overfitting
- **Early stopping** for optimal training

## 🧪 Testing

Run model diagnostics to check for bias:
```bash
python scripts/predict.py --diagnose
```

This will test the model on sample images from each class and report prediction distribution.

## 📋 Requirements

- Python 3.11+
- TensorFlow 2.x
- NumPy, Pandas
- scikit-learn
- Matplotlib, Seaborn
- OpenCV (optional)

## 🔧 Configuration

Modify `config/config.py` to adjust:
- Image size and batch size
- Model architecture parameters
- Training hyperparameters
- File paths

## 📈 Training Tips

1. **Balanced data**: Collect similar amounts of each class
2. **Quality images**: Use clear, well-lit character images
3. **Consistent format**: Keep image sizes and formats uniform
4. **Validation**: Always validate on unseen data

## 🐛 Troubleshooting

- **Low accuracy**: Check data quality and balance
- **Overfitting**: Increase dropout or reduce model complexity
- **Bias**: Use class weights or collect more balanced data
