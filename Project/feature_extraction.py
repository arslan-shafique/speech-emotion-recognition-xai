"""
TESS Emotion Recognition - Feature Extraction Module

This script extracts comprehensive audio features from the TESS dataset.

Learning Points:
- MFCC: Mel-Frequency Cepstral Coefficients capture spectral envelope
- Mel-Spectrogram: Time-frequency representation
- Chroma: Pitch class information
- ZCR: Zero Crossing Rate measures signal noisiness
- Spectral features: Describe frequency distribution

Author: Educational Implementation
"""

import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters for feature extraction"""
    # Dataset path - UPDATE THIS!
    DATASET_PATH = "/path/to/your/TESS-data"
    
    # Audio parameters
    SAMPLE_RATE = 22050  # Standard sample rate for speech
    DURATION = 3.0  # Maximum duration in seconds
    
    # Feature extraction parameters
    N_MFCC = 40  # Number of MFCC coefficients
    N_MELS = 128  # Number of mel bands
    N_FFT = 2048  # FFT window size
    HOP_LENGTH = 512  # Samples between frames
    
    # Output
    OUTPUT_CSV = "tess_features.csv"
    OUTPUT_PKL = "tess_features.pkl"


# ==================== HELPER FUNCTIONS ====================

def load_audio_file(file_path, sr=Config.SAMPLE_RATE, duration=Config.DURATION):
    """
    Load and normalize an audio file.
    
    Learning Point:
        We standardize all audio to the same length for consistent features.
        This ensures all feature vectors have the same dimensions.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        duration: Maximum duration
    
    Returns:
        audio: Normalized audio array
        sr: Sample rate
    """
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
        
        # Ensure consistent length
        target_length = int(sr * duration)
        if len(audio) < target_length:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            # Truncate if too long
            audio = audio[:target_length]
        
        return audio, sr
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def extract_mfcc(audio, sr, n_mfcc=Config.N_MFCC):
    """
    Extract MFCC features.
    
    Learning Point:
        MFCCs are the most popular features for speech recognition!
        They mimic human auditory perception and capture spectral envelope.
        The spectral envelope contains information about phonemes and emotions.
    
    Returns:
        Dictionary with mean and std of each MFCC coefficient
    """
    mfccs = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=n_mfcc, 
        n_fft=Config.N_FFT, 
        hop_length=Config.HOP_LENGTH
    )
    
    # Aggregate over time: mean and std for each coefficient
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # Create feature dictionary
    features = {}
    for i in range(n_mfcc):
        features[f'mfcc_{i}_mean'] = mfcc_mean[i]
        features[f'mfcc_{i}_std'] = mfcc_std[i]
    
    return features


def extract_mel_spectrogram(audio, sr, n_mels=Config.N_MELS):
    """
    Extract Mel-spectrogram features.
    
    Learning Point:
        Mel-spectrograms show energy distribution across frequencies over time.
        The mel scale matches human pitch perception - we're better at 
        distinguishing low frequencies than high frequencies.
    
    Returns:
        Dictionary with mel-spectrogram statistics
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Aggregate statistics
    mel_mean = np.mean(mel_spec_db, axis=1)
    mel_std = np.std(mel_spec_db, axis=1)
    
    features = {}
    for i in range(n_mels):
        features[f'mel_{i}_mean'] = mel_mean[i]
        features[f'mel_{i}_std'] = mel_std[i]
    
    return features


def extract_chroma(audio, sr):
    """
    Extract Chroma features.
    
    Learning Point:
        Chroma features represent the 12 pitch classes (C, C#, D, ..., B).
        They capture harmonic and melodic content, which is important for
        distinguishing emotions - happy speech often has higher pitch!
    
    Returns:
        Dictionary with chroma feature statistics
    """
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH
    )
    
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    features = {}
    for i in range(12):
        features[f'chroma_{i}_mean'] = chroma_mean[i]
        features[f'chroma_{i}_std'] = chroma_std[i]
    
    return features


def extract_zero_crossing_rate(audio):
    """
    Extract Zero Crossing Rate.
    
    Learning Point:
        ZCR measures how often the signal changes sign (crosses zero).
        High ZCR = noisy/unvoiced sounds (like 's', 'f', angry speech)
        Low ZCR = tonal/voiced sounds (like vowels, calm speech)
    
    Returns:
        Dictionary with ZCR statistics
    """
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=Config.HOP_LENGTH)
    
    return {
        'zcr_mean': np.mean(zcr),
        'zcr_std': np.std(zcr),
        'zcr_max': np.max(zcr),
        'zcr_min': np.min(zcr)
    }


def extract_spectral_features(audio, sr):
    """
    Extract spectral features: centroid, bandwidth, rolloff.
    
    Learning Point:
        Spectral features describe frequency distribution:
        - Centroid: "center of mass" of spectrum (brightness)
        - Bandwidth: spread around centroid (tonality)
        - Rolloff: frequency below which 85% of energy is contained
        
        These help distinguish:
        - Bright vs dark sounds (happy vs sad)
        - Sharp vs smooth sounds (angry vs neutral)
    
    Returns:
        Dictionary with spectral feature statistics
    """
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH
    )
    
    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH
    )
    
    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH
    )
    
    return {
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_std': np.std(spectral_centroid),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'spectral_bandwidth_std': np.std(spectral_bandwidth),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'spectral_rolloff_std': np.std(spectral_rolloff)
    }


def extract_all_features(file_path):
    """
    Extract all features from an audio file.
    
    Returns:
        Dictionary containing all extracted features
    """
    # Load audio
    audio, sr = load_audio_file(file_path)
    
    if audio is None:
        return None
    
    # Initialize feature dictionary
    features = {}
    
    # Extract all features
    features.update(extract_mfcc(audio, sr))
    features.update(extract_mel_spectrogram(audio, sr))
    features.update(extract_chroma(audio, sr))
    features.update(extract_zero_crossing_rate(audio))
    features.update(extract_spectral_features(audio, sr))
    
    # Add metadata
    features['file_path'] = file_path
    features['duration'] = len(audio) / sr
    
    return features


# ==================== DATASET PROCESSING ====================

def build_dataset_inventory(dataset_path):
    """
    Build a DataFrame with all audio files and their metadata.
    
    Returns:
        DataFrame with file paths, emotions, actresses, and words
    """
    data = []
    
    print("\n" + "="*70)
    print("BUILDING DATASET INVENTORY")
    print("="*70)
    
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Parse folder name: "OAF_angry" -> actress=OAF, emotion=angry
        parts = folder.split('_')
        actress = parts[0]
        emotion = '_'.join(parts[1:])  # Handle multi-word emotions
        
        # Get all WAV files
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                word = file.replace('.wav', '').split('_')[-1]
                
                data.append({
                    'file_path': file_path,
                    'actress': actress,
                    'emotion': emotion,
                    'word': word,
                    'filename': file
                })
    
    df = pd.DataFrame(data)
    
    print(f"\n✅ Found {len(df)} audio files")
    print(f"\n📊 Emotions: {sorted(df['emotion'].unique())}")
    print(f"👤 Actresses: {sorted(df['actress'].unique())}")
    print(f"\n📈 Emotion distribution:")
    print(df['emotion'].value_counts().sort_index())
    
    return df


def extract_features_from_dataset(dataset_df):
    """
    Extract features from all audio files in the dataset.
    
    Returns:
        DataFrame with all features and labels
    """
    print("\n" + "="*70)
    print("STARTING FEATURE EXTRACTION")
    print("="*70)
    print(f"\nProcessing {len(dataset_df)} audio files...")
    print("This may take several minutes. Please be patient!\n")
    
    features_list = []
    
    # Process each file with progress bar
    for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc="Extracting features"):
        file_path = row['file_path']
        
        # Extract features
        features = extract_all_features(file_path)
        
        if features is not None:
            # Add labels
            features['emotion'] = row['emotion']
            features['actress'] = row['actress']
            features['word'] = row['word']
            features['filename'] = row['filename']
            
            features_list.append(features)
    
    # Create DataFrame
    features_df = pd.DataFrame(features_list)
    
    print("\n" + "="*70)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\n✅ Extracted features from {len(features_df)} files")
    print(f"   Total features per file: {len(features_df.columns) - 5}")
    print(f"   DataFrame shape: {features_df.shape}")
    
    return features_df


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    
    print("\n" + "🎵"*35)
    print("TESS EMOTION RECOGNITION - FEATURE EXTRACTION")
    print("🎵"*35)
    
    # Check if dataset path exists
    if not os.path.exists(Config.DATASET_PATH):
        print(f"\n❌ ERROR: Dataset path not found!")
        print(f"   Path: {Config.DATASET_PATH}")
        print(f"\n   Please update the DATASET_PATH in the Config class.")
        print(f"   Download TESS dataset from:")
        print(f"   https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess")
        return
    
    # Build dataset inventory
    dataset_df = build_dataset_inventory(Config.DATASET_PATH)
    
    # Extract features
    features_df = extract_features_from_dataset(dataset_df)
    
    # Save features
    print("\n" + "="*70)
    print("SAVING FEATURES")
    print("="*70)
    
    # Save as CSV
    features_df.to_csv(Config.OUTPUT_CSV, index=False)
    print(f"\n✅ Saved to CSV: {Config.OUTPUT_CSV}")
    print(f"   File size: {os.path.getsize(Config.OUTPUT_CSV) / 1024 / 1024:.2f} MB")
    
    # Save as pickle (faster loading)
    features_df.to_pickle(Config.OUTPUT_PKL)
    print(f"✅ Saved to Pickle: {Config.OUTPUT_PKL}")
    print(f"   (Pickle files load faster than CSV!)")
    
    # Summary statistics
    print("\n" + "="*70)
    print("FEATURE SUMMARY")
    print("="*70)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    print(f"\nTotal numeric features: {len(numeric_cols)}")
    print(f"Samples per emotion:")
    print(features_df['emotion'].value_counts().sort_index())
    
    # Check for missing values
    missing = features_df.isnull().sum().sum()
    if missing > 0:
        print(f"\n⚠️ Warning: {missing} missing values found")
    else:
        print(f"\n✅ No missing values!")
    
    print("\n" + "✅"*35)
    print("FEATURE EXTRACTION COMPLETE!")
    print("✅"*35)
    print("\n🎯 Next Steps:")
    print("   1. Use these features to train ML/DL models")
    print("   2. Try different model architectures")
    print("   3. Apply XAI techniques to understand predictions")
    print("   4. Visualize with Grad-CAM")


if __name__ == "__main__":
    main()
