"""
TESS Emotion Recognition - Explainable AI Module

This script implements Explainable AI (XAI) techniques:
1. LIME - Local Interpretable Model-agnostic Explanations
2. SHAP - SHapley Additive exPlanations

Learning Points:
- XAI helps us understand WHY a model makes predictions
- LIME explains individual predictions locally
- SHAP provides global and local interpretability
- Understanding models builds trust and finds improvements

Author: Educational Implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Try importing XAI libraries
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️  LIME not available. Install with: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")


# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters"""
    # Files
    FEATURES_FILE = "tess_features.csv"
    MODEL_PATH = "models/hybrid_model_best.h5"  # Use best model
    
    # Parameters
    NUM_SAMPLES_EXPLAIN = 5  # Number of samples to explain
    OUTPUT_DIR = "xai_output"
    
    # Random seed
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)


# ==================== DATA PREPARATION ====================

def load_data_and_model():
    """
    Load features, model, and prepare data.
    
    Returns:
        model, X_test, y_test, feature_names, label_encoder
    """
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
    
    # Split (use same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_encoded,
        test_size=0.15,
        random_state=Config.RANDOM_STATE,
        stratify=y_encoded
    )
    
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Load model
    print(f"\n🔄 Loading model: {Config.MODEL_PATH}")
    model = keras.models.load_model(Config.MODEL_PATH)
    print("✅ Model loaded!")
    
    return model, X_train, X_test, y_test, feature_cols, label_encoder, scaler


# ==================== LIME EXPLANATIONS ====================

class LIMEExplainer:
    """
    LIME - Local Interpretable Model-agnostic Explanations
    
    Learning Point:
        LIME explains individual predictions by:
        1. Generating perturbed samples around the instance
        2. Getting predictions for perturbed samples
        3. Training a simple linear model locally
        4. Using linear model to explain prediction
        
        Result: Which features most influenced THIS prediction?
    """
    
    def __init__(self, model, X_train, feature_names, class_names):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained neural network
            X_train: Training data for reference
            feature_names: Names of features
            class_names: Emotion labels
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME not available. Install with: pip install lime")
        
        print("\n🔍 Initializing LIME explainer...")
        
        # Create prediction function for LIME
        def predict_fn(X):
            """Prediction function for LIME"""
            return model.predict(X, verbose=0)
        
        self.predict_fn = predict_fn
        
        # Create LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            random_state=Config.RANDOM_STATE
        )
        
        self.feature_names = feature_names
        self.class_names = class_names
        
        print("✅ LIME explainer ready!")
    
    def explain_instance(self, instance, true_label_idx, save_path=None):
        """
        Explain a single prediction.
        
        Args:
            instance: Feature vector to explain
            true_label_idx: True label index
            save_path: Path to save explanation plot
        
        Returns:
            Explanation object
        """
        # Get explanation
        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.predict_fn,
            num_features=10,  # Top 10 features
            num_samples=5000  # Samples for local model
        )
        
        # Get prediction
        pred_probs = self.predict_fn(instance.reshape(1, -1))[0]
        pred_label_idx = np.argmax(pred_probs)
        pred_label = self.class_names[pred_label_idx]
        true_label = self.class_names[true_label_idx]
        
        print(f"\n📊 LIME Explanation:")
        print(f"   True Label: {true_label}")
        print(f"   Predicted: {pred_label} ({pred_probs[pred_label_idx]*100:.1f}% confidence)")
        
        # Visualize
        fig = exp.as_pyplot_figure(label=pred_label_idx)
        plt.title(f'LIME Explanation\\nTrue: {true_label} | Predicted: {pred_label}',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved explanation to: {save_path}")
        
        plt.show()
        
        return exp


# ==================== SHAP EXPLANATIONS ====================

class SHAPExplainer:
    """
    SHAP - SHapley Additive exPlanations
    
    Learning Point:
        SHAP is based on game theory (Shapley values):
        - Assigns each feature an importance value for a prediction
        - Considers all possible feature combinations
        - Provides consistent and fair attributions
        - Works globally AND locally
        
        Result: Comprehensive understanding of feature importance
    """
    
    def __init__(self, model, X_train):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained neural network
            X_train: Training data for background
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        print("\n🔍 Initializing SHAP explainer...")
        print("   (This may take a minute...)")
        
        # Use a subset of training data as background
        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        
        # Create SHAP explainer
        # DeepExplainer works with neural networks
        self.explainer = shap.DeepExplainer(model, background)
        
        print("✅ SHAP explainer ready!")
    
    def explain_instance(self, instance, feature_names, true_label, pred_label, save_path=None):
        """
        Explain a single prediction with SHAP.
        
        Args:
            instance: Feature vector
            feature_names: Names of features
            true_label: True emotion label
            pred_label: Predicted emotion label
            save_path: Path to save plot
        """
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))
        
        # Create force plot
        print(f"\n📊 SHAP Explanation:")
        print(f"   True: {true_label} | Predicted: {pred_label}")
        
        # Waterfall plot (shows how features push prediction)
        plt.figure(figsize=(12, 6))
        
        # Get SHAP values for predicted class
        if isinstance(shap_values, list):
            # Multi-class: get values for predicted class
            pred_class_idx = 0  # This would be the predicted class index
            values = shap_values[pred_class_idx][0]
        else:
            values = shap_values[0]
        
        # Plot top features
        top_n = 10
        feature_importance = np.abs(values)
        top_indices = np.argsort(feature_importance)[-top_n:]
        
        plt.barh(range(top_n), values[top_indices])
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel('SHAP Value (impact on prediction)')
        plt.title(f'SHAP Feature Importance\\nTrue: {true_label} | Predicted: {pred_label}',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved explanation to: {save_path}")
        
        plt.show()
    
    def plot_global_importance(self, X_test, feature_names, save_path=None):
        """
        Plot global feature importance across all samples.
        
        Learning Point:
            Global importance shows which features are most important
            across ALL predictions, not just one instance.
            This helps understand the model's overall behavior.
        """
        print("\n📊 Calculating global feature importance...")
        print("   (This may take several minutes...)")
        
        # Calculate SHAP values for test set (use subset)
        n_samples = min(100, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test[sample_indices]
        
        shap_values = self.explainer.shap_values(X_sample)
        
        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            # Average across all classes
            mean_abs_shap = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Plot top features
        top_n = 20
        top_indices = np.argsort(mean_abs_shap)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), mean_abs_shap[top_indices])
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Global Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved plot to: {save_path}")
        
        plt.show()
        
        return mean_abs_shap


# ==================== FEATURE IMPORTANCE ANALYSIS ====================

def analyze_feature_importance(model, X_test, feature_names):
    """
    Analyze which features are most important for the model.
    
    Learning Point:
        Feature importance helps us understand:
        - Which audio characteristics matter most for emotion recognition
        - Whether the model learned reasonable patterns
        - How to improve feature extraction
    """
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Get predictions
    print("\n🔄 Calculating predictions...")
    predictions = model.predict(X_test, verbose=0)
    
    # Calculate feature statistics
    feature_means = np.mean(X_test, axis=0)
    feature_stds = np.std(X_test, axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean': feature_means,
        'std': feature_stds,
        'coef_variation': feature_stds / (np.abs(feature_means) + 1e-10)
    })
    
    importance_df = importance_df.sort_values('coef_variation', ascending=False)
    
    print("\n📊 Top 10 most variable features:")
    print(importance_df.head(10)[['feature', 'coef_variation']])
    
    # Plot
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['coef_variation'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Coefficient of Variation')
    plt.title('Feature Variability', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{Config.OUTPUT_DIR}/feature_variability.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    
    print("\n" + "🔍"*35)
    print("TESS EMOTION RECOGNITION - EXPLAINABLE AI")
    print("🔍"*35)
    
    # Create output directory
    import os
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Check availability
    print("\n" + "="*70)
    print("XAI LIBRARIES STATUS")
    print("="*70)
    print(f"LIME: {'✅ Available' if LIME_AVAILABLE else '❌ Not available'}")
    print(f"SHAP: {'✅ Available' if SHAP_AVAILABLE else '❌ Not available'}")
    
    if not (LIME_AVAILABLE or SHAP_AVAILABLE):
        print("\n⚠️  No XAI libraries available. Install with:")
        print("   pip install lime shap")
        return
    
    # Load data and model
    model, X_train, X_test, y_test, feature_names, label_encoder, scaler = load_data_and_model()
    class_names = label_encoder.classes_
    
    # Analyze feature importance
    analyze_feature_importance(model, X_test, feature_names)
    
    # Select random samples to explain
    sample_indices = np.random.choice(len(X_test), Config.NUM_SAMPLES_EXPLAIN, replace=False)
    
    # Initialize explainers
    if LIME_AVAILABLE:
        lime_explainer = LIMEExplainer(model, X_train, feature_names, class_names)
    
    if SHAP_AVAILABLE:
        shap_explainer = SHAPExplainer(model, X_train)
    
    # Explain each sample
    for i, idx in enumerate(sample_indices, 1):
        print("\n" + "="*70)
        print(f"EXPLAINING SAMPLE {i}/{len(sample_indices)}")
        print("="*70)
        
        instance = X_test[idx]
        true_label_idx = y_test[idx]
        true_label = class_names[true_label_idx]
        
        # Get prediction
        pred_probs = model.predict(instance.reshape(1, -1), verbose=0)[0]
        pred_label_idx = np.argmax(pred_probs)
        pred_label = class_names[pred_label_idx]
        
        print(f"\nSample {i}:")
        print(f"  True emotion: {true_label}")
        print(f"  Predicted: {pred_label} ({pred_probs[pred_label_idx]*100:.1f}% confidence)")
        
        # LIME explanation
        if LIME_AVAILABLE:
            lime_path = f"{Config.OUTPUT_DIR}/lime_explanation_{i}.png"
            lime_explainer.explain_instance(instance, true_label_idx, save_path=lime_path)
        
        # SHAP explanation
        if SHAP_AVAILABLE:
            shap_path = f"{Config.OUTPUT_DIR}/shap_explanation_{i}.png"
            shap_explainer.explain_instance(
                instance, feature_names, true_label, pred_label, save_path=shap_path
            )
    
    # Global SHAP importance
    if SHAP_AVAILABLE:
        print("\n" + "="*70)
        print("GLOBAL FEATURE IMPORTANCE")
        print("="*70)
        global_path = f"{Config.OUTPUT_DIR}/shap_global_importance.png"
        shap_explainer.plot_global_importance(X_test, feature_names, save_path=global_path)
    
    # Summary
    print("\n" + "="*70)
    print("XAI ANALYSIS COMPLETE")
    print("="*70)
    print("\n🎓 What you learned:")
    print("   1. LIME - Local explanations for individual predictions")
    print("   2. SHAP - Game theory-based feature importance")
    print("   3. Global vs local interpretability")
    print("   4. Which features drive emotion recognition")
    print("\n💡 Why XAI matters:")
    print("   - Builds trust in AI systems")
    print("   - Helps debug model errors")
    print("   - Ensures fairness and accountability")
    print("   - Guides feature engineering")
    print(f"\n📁 All explanations saved in: {Config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
