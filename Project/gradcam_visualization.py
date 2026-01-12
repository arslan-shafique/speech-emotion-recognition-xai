"""
TESS Emotion Recognition - Grad-CAM Visualization Module

This script implements Grad-CAM (Gradient-weighted Class Activation Mapping)
to visualize what the CNN "sees" when making predictions.

Learning Points:
- Grad-CAM shows which parts of input the model focuses on
- Uses gradients to weight feature maps
- Helps understand CNN decision-making
- Validates that model learned correct patterns

Author: Educational Implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import cv2
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters"""
    # Files
    FEATURES_FILE = "tess_features.csv"
    MODEL_PATH = "models/cnn_model_best.h5"  # Use CNN model
    
    # Parameters
    NUM_SAMPLES = 5
    OUTPUT_DIR = "gradcam_output"
    
    # Visualization
    COLORMAP = cv2.COLORMAP_JET  # Color scheme for heatmap
    ALPHA = 0.4  # Transparency for overlay
    
    # Random seed
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)


# ==================== GRAD-CAM IMPLEMENTATION ====================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    
    Learning Point:
        Grad-CAM works by:
        1. Forward pass: Get feature maps from target layer
        2. Backward pass: Compute gradients of class score w.r.t. feature maps
        3. Weight feature maps by gradient importance
        4. Average weighted maps to get attention heatmap
        5. Overlay heatmap on input to see what model "sees"
        
        This reveals which input features are most important for prediction!
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained CNN model
            layer_name: Name of conv layer to visualize (None = last conv layer)
        """
        self.model = model
        
        # Find convolutional layers
        conv_layers = [layer for layer in model.layers 
                      if 'conv' in layer.name.lower()]
        
        if not conv_layers:
            raise ValueError("No convolutional layers found in model!")
        
        # Use specified layer or last conv layer
        if layer_name:
            self.target_layer = model.get_layer(layer_name)
        else:
            self.target_layer = conv_layers[-1]
        
        print(f"\n🎯 Using layer for Grad-CAM: {self.target_layer.name}")
        print(f"   Layer output shape: {self.target_layer.output.shape}")
        
        # Create gradient model
        self.grad_model = models.Model(
            inputs=model.input,
            outputs=[self.target_layer.output, model.output]
        )
    
    def compute_heatmap(self, instance, class_idx=None):
        """
        Compute Grad-CAM heatmap for an instance.
        
        Learning Point:
            The heatmap shows "importance" of each position in feature maps:
            - Red/hot areas = important for prediction
            - Blue/cold areas = less important
            - This tells us which input features drive the decision
        
        Args:
            instance: Input features (single sample)
            class_idx: Class to explain (None = predicted class)
        
        Returns:
            heatmap: 2D attention map
            prediction: Model prediction
        """
        # Convert to tensor
        instance_tensor = tf.cast(tf.expand_dims(instance, 0), tf.float32)
        
        # Watch gradients
        with tf.GradientTape() as tape:
            # Forward pass
            conv_outputs, predictions = self.grad_model(instance_tensor)
            
            # Get prediction
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get class score
            class_channel = predictions[:, class_idx]
        
        # Compute gradients of class score w.r.t. feature maps
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        # This gives importance weight for each feature map
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        
        # Weight feature maps by importance
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        
        # Multiply each feature map by its weight
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, i] *= pooled_grads[i]
        
        # Average weighted feature maps
        heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()
        
        # Normalize heatmap to [0, 1]
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        return heatmap, predictions[0].numpy()
    
    def visualize_heatmap(self, heatmap, input_features=None, 
                         true_label=None, pred_label=None, 
                         confidence=None, save_path=None):
        """
        Visualize Grad-CAM heatmap.
        
        Args:
            heatmap: Computed heatmap
            input_features: Original input features for overlay
            true_label: True emotion label
            pred_label: Predicted emotion label
            confidence: Prediction confidence
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Original heatmap
        im1 = axes[0].imshow(heatmap.reshape(1, -1), cmap='jet', aspect='auto')
        axes[0].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Feature Position')
        axes[0].set_ylabel('Attention')
        plt.colorbar(im1, ax=axes[0], label='Importance')
        
        # 2. Heatmap as bar plot
        axes[1].bar(range(len(heatmap)), heatmap, color='red', alpha=0.7)
        axes[1].set_title('Feature Importance', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Feature Position')
        axes[1].set_ylabel('Importance')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Overlay on input (if provided)
        if input_features is not None:
            # Resize heatmap to match input size
            heatmap_resized = cv2.resize(heatmap, (len(input_features), 1))
            heatmap_resized = heatmap_resized.flatten()
            
            # Normalize input for visualization
            input_norm = (input_features - input_features.min()) / (input_features.max() - input_features.min() + 1e-10)
            
            # Create overlay
            axes[2].plot(input_norm, label='Input Features', linewidth=2, alpha=0.7)
            axes[2].plot(heatmap_resized, label='Attention', linewidth=2, alpha=0.7)
            axes[2].fill_between(range(len(heatmap_resized)), heatmap_resized, alpha=0.3)
            axes[2].set_title('Input with Attention Overlay', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Feature Index')
            axes[2].set_ylabel('Value (normalized)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Add prediction info
        if true_label and pred_label:
            title = f'True: {true_label} | Predicted: {pred_label}'
            if confidence:
                title += f' ({confidence*100:.1f}%)'
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved visualization to: {save_path}")
        
        plt.show()


# ==================== FEATURE MAP VISUALIZATION ====================

def visualize_feature_maps(model, instance, layer_names=None):
    """
    Visualize feature maps from convolutional layers.
    
    Learning Point:
        Feature maps show what patterns the CNN detects:
        - Early layers: simple patterns (edges, basic features)
        - Middle layers: combinations of patterns
        - Later layers: complex, emotion-specific patterns
        
        This hierarchical learning is key to CNN success!
    
    Args:
        model: Trained model
        instance: Input sample
        layer_names: List of layer names to visualize
    """
    print("\n" + "="*70)
    print("VISUALIZING FEATURE MAPS")
    print("="*70)
    
    # Get all conv layers if not specified
    if layer_names is None:
        layer_names = [layer.name for layer in model.layers 
                      if 'conv' in layer.name.lower()]
    
    print(f"\n📊 Visualizing {len(layer_names)} layers:")
    for name in layer_names:
        print(f"   - {name}")
    
    # Create models for each layer
    for layer_name in layer_names:
        try:
            layer_model = models.Model(
                inputs=model.input,
                outputs=model.get_layer(layer_name).output
            )
            
            # Get feature maps
            feature_maps = layer_model.predict(instance.reshape(1, -1), verbose=0)
            
            # Plot feature maps
            n_features = min(16, feature_maps.shape[-1])  # Show up to 16 feature maps
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*3))
            axes = axes.flatten()
            
            for i in range(n_features):
                feature_map = feature_maps[0, :, i]
                axes[i].plot(feature_map)
                axes[i].set_title(f'Filter {i+1}', fontsize=10)
                axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'Feature Maps: {layer_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{Config.OUTPUT_DIR}/feature_maps_{layer_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Visualized {layer_name}: {feature_maps.shape}")
        
        except Exception as e:
            print(f"⚠️  Could not visualize {layer_name}: {e}")


# ==================== DATA PREPARATION ====================

def load_data_and_model():
    """Load features and model"""
    print("\n" + "="*70)
    print("LOADING DATA AND MODEL")
    print("="*70)
    
    # Load features
    df = pd.read_csv(Config.FEATURES_FILE)
    print(f"\n✅ Loaded {len(df)} samples")
    
    # Prepare features
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    # Separate features and labels
    exclude_cols = ['file_path', 'emotion', 'actress', 'word', 'filename', 'duration']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['emotion'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_encoded,
        test_size=0.15,
        random_state=Config.RANDOM_STATE,
        stratify=y_encoded
    )
    
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Load model
    print(f"\n🔄 Loading model: {Config.MODEL_PATH}")
    try:
        model = keras.models.load_model(Config.MODEL_PATH)
        print("✅ CNN model loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Make sure you've trained the CNN model first!")
        print("   Run model_training.py to create the model.")
        raise
    
    return model, X_test, y_test, feature_cols, label_encoder


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    
    print("\n" + "🔥"*35)
    print("TESS EMOTION RECOGNITION - GRAD-CAM VISUALIZATION")
    print("🔥"*35)
    
    # Create output directory
    import os
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Load data and model
    try:
        model, X_test, y_test, feature_names, label_encoder = load_data_and_model()
    except Exception as e:
        print(f"\n❌ Failed to load data/model: {e}")
        return
    
    class_names = label_encoder.classes_
    
    # Initialize Grad-CAM
    print("\n" + "="*70)
    print("INITIALIZING GRAD-CAM")
    print("="*70)
    
    try:
        gradcam = GradCAM(model)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Grad-CAM requires a CNN model with convolutional layers.")
        print("   Make sure you're using the CNN model, not Dense or LSTM.")
        return
    
    # Select random samples
    sample_indices = np.random.choice(len(X_test), Config.NUM_SAMPLES, replace=False)
    
    # Generate Grad-CAM for each sample
    for i, idx in enumerate(sample_indices, 1):
        print("\n" + "="*70)
        print(f"GRAD-CAM VISUALIZATION {i}/{len(sample_indices)}")
        print("="*70)
        
        instance = X_test[idx]
        true_label_idx = y_test[idx]
        true_label = class_names[true_label_idx]
        
        # Compute heatmap
        print(f"\n🔄 Computing Grad-CAM heatmap...")
        heatmap, predictions = gradcam.compute_heatmap(instance)
        
        pred_label_idx = np.argmax(predictions)
        pred_label = class_names[pred_label_idx]
        confidence = predictions[pred_label_idx]
        
        print(f"✅ Heatmap computed!")
        print(f"\n📊 Prediction:")
        print(f"   True: {true_label}")
        print(f"   Predicted: {pred_label} ({confidence*100:.1f}% confidence)")
        
        # Visualize
        save_path = f"{Config.OUTPUT_DIR}/gradcam_sample_{i}.png"
        gradcam.visualize_heatmap(
            heatmap,
            input_features=instance,
            true_label=true_label,
            pred_label=pred_label,
            confidence=confidence,
            save_path=save_path
        )
        
        # Show prediction distribution
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(class_names)), predictions, color='skyblue', alpha=0.7)
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.xlabel('Emotion')
        plt.ylabel('Probability')
        plt.title(f'Prediction Confidence Distribution\nTrue: {true_label}',
                 fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{Config.OUTPUT_DIR}/confidence_sample_{i}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Visualize feature maps for one sample
    print("\n" + "="*70)
    print("FEATURE MAP VISUALIZATION")
    print("="*70)
    visualize_feature_maps(model, X_test[sample_indices[0]])
    
    # Summary
    print("\n" + "="*70)
    print("GRAD-CAM ANALYSIS COMPLETE")
    print("="*70)
    print("\n🎓 What you learned:")
    print("   1. Grad-CAM - Visualizing CNN attention")
    print("   2. Feature maps - What patterns CNNs detect")
    print("   3. Hierarchical learning - From simple to complex")
    print("   4. Model interpretability for CNNs")
    print("\n💡 Key insights:")
    print("   - Heatmaps show which features drive predictions")
    print("   - Different layers detect different patterns")
    print("   - Red/hot areas = most important for decision")
    print("   - Helps validate model learned correct patterns")
    print(f"\n📁 All visualizations saved in: {Config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
