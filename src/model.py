"""
CNN model architecture for video game character state classification.
Enhanced with state-of-the-art techniques for 80% accuracy target.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports when running directly
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import tensorflow as tf
from typing import Tuple, Dict, Optional, List
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


def create_efficientnet_model(
    input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    model_size: str = "B0"
) -> tf.keras.Model:
    """
    Create EfficientNet-based model with transfer learning.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        model_size: EfficientNet model size (B0, B1, etc.)
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained EfficientNet
    if model_size == "B0":
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif model_size == "B1":
        base_model = tf.keras.applications.EfficientNetB1(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    
    # Freeze initial layers, unfreeze last few for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # Preprocessing
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Attention mechanism
    attention = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='attention_weights')(x)
    x = tf.keras.layers.Multiply(name='attention_applied')([x, attention])
    
    # Global pooling with both average and max
    global_avg = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    global_max = tf.keras.layers.GlobalMaxPooling2D(name='global_max_pool')(x)
    
    # Combine pooled features
    x = tf.keras.layers.Concatenate(name='pooled_features')([global_avg, global_max])
    
    # Feature refinement
    x = tf.keras.layers.Dense(512, activation='relu', name='feature_dense1')(x)
    x = tf.keras.layers.BatchNormalization(name='feature_bn1')(x)
    x = tf.keras.layers.Dropout(0.3, name='feature_dropout1')(x)
    
    x = tf.keras.layers.Dense(256, activation='relu', name='feature_dense2')(x)
    x = tf.keras.layers.BatchNormalization(name='feature_bn2')(x)
    x = tf.keras.layers.Dropout(0.5, name='feature_dropout2')(x)
    
    # Final classification layer
    outputs = tf.keras.layers.Dense(
        num_classes, 
        activation='softmax',
        name='predictions'
    )(x)
    
    model = tf.keras.Model(inputs, outputs, name='EfficientNet_Character_Classifier')
    
    return model


def create_vision_transformer_model(
    input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    patch_size: int = 16,
    num_layers: int = 8,
    d_model: int = 256,
    num_heads: int = 8
) -> tf.keras.Model:
    """
    Create Vision Transformer model for character state classification.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        patch_size: Size of image patches
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        
    Returns:
        Compiled Keras model
    """
    
    def patch_embedding_layer(patch_size: int, d_model: int):
        """Create patch embedding layer."""
        def patches(images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches
        
        return patches
    
    def positional_encoding(num_patches: int, d_model: int):
        """Add positional encoding to patches."""
        def add_position_emb(patch_embeddings):
            positions = tf.range(start=0, limit=num_patches, delta=1)
            pos_encoding = tf.keras.layers.Embedding(
                input_dim=num_patches, output_dim=d_model
            )(positions)
            encoded = patch_embeddings + pos_encoding
            return encoded
        
        return add_position_emb
    
    def transformer_block(d_model: int, num_heads: int, dff: int = None):
        """Create transformer block."""
        if dff is None:
            dff = 4 * d_model
            
        def block(x):
            # Multi-head attention
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model
            )(x, x)
            attn_output = tf.keras.layers.Dropout(0.1)(attn_output)
            out1 = tf.keras.layers.LayerNormalization()(x + attn_output)
            
            # Feed forward
            ffn_output = tf.keras.layers.Dense(dff, activation="gelu")(out1)
            ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
            ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
            out2 = tf.keras.layers.LayerNormalization()(out1 + ffn_output)
            
            return out2
        
        return block
    
    # Calculate number of patches
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    # Model architecture
    inputs = tf.keras.Input(shape=input_shape)
    
    # Create patches
    patches = patch_embedding_layer(patch_size, d_model)(inputs)
    
    # Linear projection of patches
    patch_embeddings = tf.keras.layers.Dense(d_model)(patches)
    
    # Add positional encoding
    encoded_patches = positional_encoding(num_patches, d_model)(patch_embeddings)
    
    # Add class token
    class_token = tf.Variable(
        tf.random.normal([1, 1, d_model]), trainable=True, name="class_token"
    )
    class_tokens = tf.tile(class_token, [tf.shape(encoded_patches)[0], 1, 1])
    encoded_patches = tf.concat([class_tokens, encoded_patches], axis=1)
    
    # Transformer layers
    x = encoded_patches
    for _ in range(num_layers):
        x = transformer_block(d_model, num_heads)(x)
    
    # Extract class token and classify
    x = tf.keras.layers.LayerNormalization()(x)
    class_token_output = x[:, 0]  # Extract class token
    
    # Classification head
    x = tf.keras.layers.Dense(512, activation='gelu')(class_token_output)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name='ViT_Character_Classifier')
    
    return model


def create_hybrid_cnn_transformer_model(
    input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3),
    num_classes: int = NUM_CLASSES
) -> tf.keras.Model:
    """
    Create hybrid CNN-Transformer model combining the best of both architectures.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # CNN feature extraction
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    
    # Initial CNN layers for low-level features
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    def residual_block(x, filters, stride=1):
        shortcut = x
        
        x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride)(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x
    
    # CNN feature extraction layers
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    # Reshape for transformer
    batch_size = tf.shape(x)[0]
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    x = tf.reshape(x, [batch_size, height * width, channels])
    
    # Transformer layers
    def transformer_block(x, d_model=256, num_heads=8):
        # Multi-head attention
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )(x, x)
        attn = tf.keras.layers.Dropout(0.1)(attn)
        x = tf.keras.layers.LayerNormalization()(x + attn)
        
        # Feed forward
        ffn = tf.keras.layers.Dense(d_model * 4, activation='gelu')(x)
        ffn = tf.keras.layers.Dense(d_model)(ffn)
        ffn = tf.keras.layers.Dropout(0.1)(ffn)
        x = tf.keras.layers.LayerNormalization()(x + ffn)
        
        return x
    
    # Apply transformer blocks
    x = transformer_block(x)
    x = transformer_block(x)
    
    # Global attention pooling
    attention_weights = tf.keras.layers.Dense(1, activation='softmax')(x)
    x = tf.reduce_sum(x * attention_weights, axis=1)
    
    # Classification head
    x = tf.keras.layers.Dense(512, activation='gelu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(256, activation='gelu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name='Hybrid_CNN_Transformer')
    
    return model


def create_cnn_model(
    input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3),
    num_classes: int = NUM_CLASSES
) -> tf.keras.Model:
    """
    Enhanced CNN model with modern techniques.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Preprocessing
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    
    # Feature extraction with depthwise separable convolutions
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SeparableConv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Second block
    x = tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Third block with attention
    x = tf.keras.layers.SeparableConv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Channel attention
    channel_attention = tf.keras.layers.GlobalAveragePooling2D()(x)
    channel_attention = tf.keras.layers.Dense(256//8, activation='relu')(channel_attention)
    channel_attention = tf.keras.layers.Dense(256, activation='sigmoid')(channel_attention)
    channel_attention = tf.keras.layers.Reshape((1, 1, 256))(channel_attention)
    x = tf.keras.layers.Multiply()([x, channel_attention])
    
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name='Enhanced_CNN')
    
    return model


def compile_model(
    model: tf.keras.Model, 
    class_weights: Optional[Dict[int, float]] = None,
    learning_rate: float = LEARNING_RATE
) -> tf.keras.Model:
    """
    Compile model with advanced optimization and loss functions.
    
    Args:
        model: Keras model to compile
        class_weights: Optional class weights for imbalanced datasets
        learning_rate: Initial learning rate
        
    Returns:
        Compiled model
    """
    # Use AdamW optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Use focal loss for imbalanced datasets
    def focal_loss(alpha=0.25, gamma=2.0):
        def focal_loss_fn(y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)
            y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
            
            ce_loss = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)
            p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
            alpha_t = alpha
            focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
            focal_loss = focal_weight * ce_loss
            
            return tf.reduce_mean(focal_loss)
        
        return focal_loss_fn
    
    # Compile with focal loss and multiple metrics
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(alpha=0.25, gamma=2.0),
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    logger.info(f"Model compiled with AdamW optimizer (lr={learning_rate}) and focal loss")
    
    return model