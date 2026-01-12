"""
TESS Emotion Recognition - Model Training Module

This script builds and trains deep learning models for emotion classification.

Models Implemented:
1. CNN - Convolutional Neural Network
2. LSTM - Long Short-Term Memory
3. CNN-LSTM Hybrid
4. Transformer (optional)

Learning Points:
- CNNs learn spatial patterns in spectrograms
- LSTMs learn temporal dependencies
- Hybrid models combine both strengths
- Transformers use attention mechanisms

Author: Educational Implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters for model training"""
    # Data
    FEATURES_FILE = "tess_features.csv"
    
    # Training parameters
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # Model parameters
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.3
    
    # Output
    MODEL_SAVE_PATH = "models"
    BEST_MODEL_NAME = "best_emotion_model.h5"
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


# ==================== DATA PREPARATION ====================

def load_and_prepare_data(features_file):
    """
    Load features and prepare for training.
    
    Learning Point:
        Proper data preparation is crucial:
        - Separate features from labels
        - Encode categorical labels to numbers
        - Normalize features to same scale
        - Split into train/val/test sets
    
    Returns:
        X_train, X_val, X_test: Feature arrays
        y_train, y_val, y_test: Label arrays
        label_encoder: For inverse transforming predictions
        scaler: For denormalizing if needed
    """
    print("\n" + "="*70)
    print("LOADING AND PREPARING DATA")
    print("="*70)
    
    # Load features
    df = pd.read_csv(features_file)
    print(f"\n✅ Loaded {len(df)} samples")
    print(f"   Columns: {len(df.columns)}")
    
    # Separate features and labels
    # Exclude non-feature columns
    exclude_cols = ['file_path', 'emotion', 'actress', 'word', 'filename', 'duration']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['emotion'].values
    
    print(f"\n📊 Data shape:")
    print(f"   Features: {X.shape}")
    print(f"   Labels: {y.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n🏷️  Emotion labels:")
    for idx, emotion in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == idx)
        print(f"   {idx}: {emotion} ({count} samples)")
    
    # Normalize features
    print(f"\n🔄 Normalizing features...")
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Split data: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_normalized, y_encoded,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=Config.VAL_SIZE / (1 - Config.TEST_SIZE),
        random_state=Config.RANDOM_STATE,
        stratify=y_temp
    )
    
    print(f"\n📂 Data split:")
    print(f"   Training:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"   Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"   Testing:    {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Convert labels to categorical (one-hot encoding)
    num_classes = len(label_encoder.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    return (X_train, X_val, X_test, 
            y_train_cat, y_val_cat, y_test_cat,
            y_train, y_val, y_test,  # Keep original for evaluation
            label_encoder, scaler, num_classes)


# ==================== MODEL ARCHITECTURES ====================

def build_dense_model(input_shape, num_classes):
    """
    Build a simple fully-connected (Dense) neural network.
    
    Learning Point:
        Dense layers connect every neuron to every other neuron.
        This is the simplest architecture but works well for our aggregated features.
        - Multiple hidden layers learn increasingly abstract representations
        - Dropout prevents overfitting
        - BatchNormalization stabilizes training
    
    Architecture:
        Input -> Dense(512) -> Dropout -> Dense(256) -> Dropout -> Output
    """
    print("\n🏗️  Building Dense Model...")
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # First hidden layer
        layers.Dense(512, activation='relu', name='dense_1'),
        layers.BatchNormalization(),
        layers.Dropout(Config.DROPOUT_RATE),
        
        # Second hidden layer
        layers.Dense(256, activation='relu', name='dense_2'),
        layers.BatchNormalization(),
        layers.Dropout(Config.DROPOUT_RATE),
        
        # Third hidden layer
        layers.Dense(128, activation='relu', name='dense_3'),
        layers.BatchNormalization(),
        layers.Dropout(Config.DROPOUT_RATE),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def build_cnn_model(input_shape, num_classes):
    """
    Build a 1D CNN model.
    
    Learning Point:
        CNNs are excellent at detecting local patterns:
        - Conv1D layers learn filters that detect specific patterns
        - MaxPooling reduces dimensionality
        - Multiple conv layers learn hierarchical features
        
        For audio: early layers detect basic patterns, 
        later layers detect complex emotional signatures
    
    Architecture:
        Input -> Conv1D blocks -> GlobalPooling -> Dense -> Output
    """
    print("\n🏗️  Building 1D CNN Model...")
    
    # Reshape for 1D convolution (samples, timesteps, features)
    # We'll treat our features as a 1D sequence
    input_layer = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(input_layer)
    
    # First conv block
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Second conv block
    x = layers.Conv1D(256, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Third conv block
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Output
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=output)
    return model


def build_lstm_model(input_shape, num_classes):
    """
    Build an LSTM model.
    
    Learning Point:
        LSTMs are designed for sequential data:
        - They maintain a "memory" of previous inputs
        - Learn long-term dependencies
        - Bidirectional LSTMs process sequence both ways
        
        For audio: capture temporal evolution of emotions
    
    Architecture:
        Input -> Bidirectional LSTM layers -> Dense -> Output
    """
    print("\n🏗️  Building LSTM Model...")
    
    input_layer = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(input_layer)
    
    # First LSTM layer (return sequences for next LSTM)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Second LSTM layer
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Output
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=output)
    return model


def build_hybrid_model(input_shape, num_classes):
    """
    Build a hybrid CNN-LSTM model.
    
    Learning Point:
        Combining CNN and LSTM gets the best of both:
        - CNN extracts spatial/local patterns
        - LSTM captures temporal dependencies
        - Together they model complex audio patterns
        
        This is often the best architecture for audio tasks!
    
    Architecture:
        Input -> CNN blocks -> LSTM layers -> Dense -> Output
    """
    print("\n🏗️  Building Hybrid CNN-LSTM Model...")
    
    input_layer = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(input_layer)
    
    # CNN feature extraction
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    x = layers.Conv1D(256, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # LSTM temporal modeling
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Dense classification
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Output
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=output)
    return model


# ==================== TRAINING ====================

def compile_and_train(model, X_train, y_train, X_val, y_val, model_name):
    """
    Compile and train a model.
    
    Learning Point:
        Training process:
        - Optimizer (Adam) adjusts weights to minimize loss
        - Loss function measures prediction error
        - Metrics track performance
        - Callbacks improve training:
          * EarlyStopping prevents overfitting
          * ReduceLROnPlateau adjusts learning rate
          * ModelCheckpoint saves best model
    
    Returns:
        Trained model and training history
    """
    print(f"\n🎯 Training {model_name}...")
    print("="*70)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    callbacks_list = [
        # Stop if validation loss doesn't improve
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate if stuck
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Save best model
        callbacks.ModelCheckpoint(
            filepath=f'{Config.MODEL_SAVE_PATH}/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    return model, history


def plot_training_history(history, model_name):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{Config.MODEL_SAVE_PATH}/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Training plot saved: {Config.MODEL_SAVE_PATH}/{model_name}_training_history.png")


# ==================== EVALUATION ====================

def evaluate_model(model, X_test, y_test, y_test_orig, label_encoder, model_name):
    """
    Evaluate model performance.
    
    Learning Point:
        Evaluation metrics:
        - Accuracy: Overall correct predictions
        - Precision: Of predicted positives, how many are correct
        - Recall: Of actual positives, how many we found
        - F1-Score: Harmonic mean of precision and recall
        - Confusion Matrix: Shows which emotions are confused
    """
    print(f"\n📊 Evaluating {model_name}...")
    print("="*70)
    
    # Make predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_orig, y_pred)
    f1 = f1_score(y_test_orig, y_pred, average='weighted')
    
    print(f"\n🎯 Test Accuracy: {accuracy*100:.2f}%")
    print(f"🎯 Weighted F1-Score: {f1:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test_orig, y_pred, 
                                target_names=label_encoder.classes_,
                                digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_orig, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{Config.MODEL_SAVE_PATH}/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Confusion matrix saved: {Config.MODEL_SAVE_PATH}/{model_name}_confusion_matrix.png")
    
    return accuracy, f1


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    
    print("\n" + "🧠"*35)
    print("TESS EMOTION RECOGNITION - MODEL TRAINING")
    print("🧠"*35)
    
    # Create models directory
    import os
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Load and prepare data
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     y_train_orig, y_val_orig, y_test_orig,
     label_encoder, scaler, num_classes) = load_and_prepare_data(Config.FEATURES_FILE)
    
    input_shape = (X_train.shape[1],)
    
    # Dictionary to store results
    results = {}
    
    # Train Dense Model
    print("\n" + "="*70)
    print("MODEL 1: DENSE NEURAL NETWORK")
    print("="*70)
    dense_model = build_dense_model(input_shape, num_classes)
    dense_model, dense_history = compile_and_train(
        dense_model, X_train, y_train, X_val, y_val, "dense_model"
    )
    plot_training_history(dense_history, "Dense Model")
    acc, f1 = evaluate_model(dense_model, X_test, y_test, y_test_orig, 
                             label_encoder, "Dense Model")
    results['Dense'] = {'accuracy': acc, 'f1': f1}
    
    # Train CNN Model
    print("\n" + "="*70)
    print("MODEL 2: 1D CONVOLUTIONAL NEURAL NETWORK")
    print("="*70)
    cnn_model = build_cnn_model(input_shape, num_classes)
    cnn_model, cnn_history = compile_and_train(
        cnn_model, X_train, y_train, X_val, y_val, "cnn_model"
    )
    plot_training_history(cnn_history, "CNN Model")
    acc, f1 = evaluate_model(cnn_model, X_test, y_test, y_test_orig,
                             label_encoder, "CNN Model")
    results['CNN'] = {'accuracy': acc, 'f1': f1}
    
    # Train LSTM Model
    print("\n" + "="*70)
    print("MODEL 3: LSTM NEURAL NETWORK")
    print("="*70)
    lstm_model = build_lstm_model(input_shape, num_classes)
    lstm_model, lstm_history = compile_and_train(
        lstm_model, X_train, y_train, X_val, y_val, "lstm_model"
    )
    plot_training_history(lstm_history, "LSTM Model")
    acc, f1 = evaluate_model(lstm_model, X_test, y_test, y_test_orig,
                            label_encoder, "LSTM Model")
    results['LSTM'] = {'accuracy': acc, 'f1': f1}
    
    # Train Hybrid Model
    print("\n" + "="*70)
    print("MODEL 4: HYBRID CNN-LSTM MODEL")
    print("="*70)
    hybrid_model = build_hybrid_model(input_shape, num_classes)
    hybrid_model, hybrid_history = compile_and_train(
        hybrid_model, X_train, y_train, X_val, y_val, "hybrid_model"
    )
    plot_training_history(hybrid_history, "Hybrid Model")
    acc, f1 = evaluate_model(hybrid_model, X_test, y_test, y_test_orig,
                            label_encoder, "Hybrid Model")
    results['Hybrid'] = {'accuracy': acc, 'f1': f1}
    
    # Compare results
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('accuracy', ascending=False)
    print("\n", results_df)
    
    # Plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    results_df.plot(kind='bar', ax=ax, rot=0)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(['Accuracy', 'F1-Score'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{Config.MODEL_SAVE_PATH}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Determine best model
    best_model_name = results_df.index[0]
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {results_df.iloc[0]['accuracy']*100:.2f}%")
    print(f"   F1-Score: {results_df.iloc[0]['f1']:.4f}")
    
    print("\n" + "✅"*35)
    print("MODEL TRAINING COMPLETE!")
    print("✅"*35)
    print(f"\n📁 All models saved in: {Config.MODEL_SAVE_PATH}/")
    print(f"\n🎯 Next Steps:")
    print("   1. Use best model for predictions")
    print("   2. Apply Explainable AI (LIME, SHAP)")
    print("   3. Visualize with Grad-CAM")
    print("   4. Deploy for real-time emotion recognition")


if __name__ == "__main__":
    main()
