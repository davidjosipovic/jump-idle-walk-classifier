<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Klasifikacija stanja likova u videoigrama (Video Game Character State Classification)

This is a machine learning project focused on classifying video game character states using computer vision and deep learning techniques.

## Project Context
- **Language**: Python 3.11
- **Domain**: Computer Vision, Deep Learning, Video Game Analysis
- **Main Libraries**: TensorFlow, PyTorch, OpenCV, scikit-learn

## Coding Guidelines
- Use type hints for all function parameters and return values
- Follow PEP 8 style guidelines (enforced by black and flake8)
- Write comprehensive docstrings for classes and functions
- Include error handling and input validation
- Use descriptive variable names in English (avoid Serbian/Croatian variable names in code)
- Prefer numpy arrays and pandas DataFrames for data manipulation
- Use pathlib for file path operations
- Include logging for debugging and monitoring

## Project Structure
- `src/`: Main source code
- `data/`: Dataset storage (raw, processed, interim)
- `models/`: Trained models and model artifacts
- `notebooks/`: Jupyter notebooks for experimentation
- `scripts/`: Utility scripts and data processing
- `tests/`: Unit tests and integration tests

## ML Best Practices
- Always split data into train/validation/test sets
- Use cross-validation for model evaluation
- Save model checkpoints during training
- Log experiment parameters and results
- Implement data preprocessing pipelines
- Use configuration files for hyperparameters