"""
TESS Emotion Recognition - Main Execution Script

This script provides a command-line interface to run all modules
of the emotion recognition project.

Usage:
    python main.py --step [step_name]
    
Steps:
    1. extract    - Extract features from audio files
    2. train      - Train emotion recognition models
    3. speech     - Speech-to-text and text-to-speech conversion
    4. xai        - Explainable AI analysis (LIME & SHAP)
    5. gradcam    - Grad-CAM visualization
    6. all        - Run all steps in sequence

Author: Educational Implementation
"""

import argparse
import sys
import os

def print_banner():
    """Print project banner"""
    print("\n" + "="*70)
    print("🎭 TESS EMOTION RECOGNITION PROJECT 🎭")
    print("="*70)
    print("\nA comprehensive deep learning project for audio emotion recognition")
    print("with educational explanations at every step!")
    print("="*70 + "\n")


def run_feature_extraction():
    """Run feature extraction module"""
    print("\n" + "🎵"*35)
    print("STEP 1: FEATURE EXTRACTION")
    print("🎵"*35 + "\n")
    
    try:
        from feature_extraction import main
        main()
        return True
    except Exception as e:
        print(f"\n❌ Feature extraction failed: {e}")
        return False


def run_model_training():
    """Run model training module"""
    print("\n" + "🧠"*35)
    print("STEP 2: MODEL TRAINING")
    print("🧠"*35 + "\n")
    
    try:
        from model_training import main
        main()
        return True
    except Exception as e:
        print(f"\n❌ Model training failed: {e}")
        return False


def run_speech_conversion():
    """Run speech conversion module"""
    print("\n" + "🎙️"*35)
    print("STEP 3: SPEECH CONVERSION")
    print("🎙️"*35 + "\n")
    
    try:
        from speech_conversion import main
        main()
        return True
    except Exception as e:
        print(f"\n❌ Speech conversion failed: {e}")
        return False


def run_explainable_ai():
    """Run XAI module"""
    print("\n" + "🔍"*35)
    print("STEP 4: EXPLAINABLE AI")
    print("🔍"*35 + "\n")
    
    try:
        from explainable_ai import main
        main()
        return True
    except Exception as e:
        print(f"\n❌ XAI analysis failed: {e}")
        return False


def run_gradcam():
    """Run Grad-CAM module"""
    print("\n" + "🔥"*35)
    print("STEP 5: GRAD-CAM VISUALIZATION")
    print("🔥"*35 + "\n")
    
    try:
        from gradcam_visualization import main
        main()
        return True
    except Exception as e:
        print(f"\n❌ Grad-CAM visualization failed: {e}")
        return False


def run_all_steps():
    """Run all steps in sequence"""
    print("\n" + "🚀"*35)
    print("RUNNING ALL STEPS")
    print("🚀"*35 + "\n")
    
    steps = [
        ("Feature Extraction", run_feature_extraction),
        ("Model Training", run_model_training),
        ("Speech Conversion", run_speech_conversion),
        ("Explainable AI", run_explainable_ai),
        ("Grad-CAM", run_gradcam),
    ]
    
    results = []
    for step_name, step_func in steps:
        print(f"\n{'='*70}")
        print(f"Running: {step_name}")
        print(f"{'='*70}")
        
        success = step_func()
        results.append((step_name, success))
        
        if not success:
            print(f"\n⚠️  {step_name} failed. Continuing with next step...")
    
    # Print summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    for step_name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{step_name:30s} {status}")


def print_help():
    """Print help information"""
    help_text = """
╔═══════════════════════════════════════════════════════════════════╗
║          TESS EMOTION RECOGNITION - HELP GUIDE                    ║
╚═══════════════════════════════════════════════════════════════════╝

USAGE:
    python main.py --step [step_name]

AVAILABLE STEPS:
    extract     Extract audio features (MFCC, Mel-spectrogram, etc.)
    train       Train emotion recognition models (CNN, LSTM, Hybrid)
    speech      Speech-to-text and text-to-speech conversion
    xai         Explainable AI analysis (LIME & SHAP)
    gradcam     Grad-CAM visualization for CNNs
    all         Run all steps in sequence

EXAMPLES:
    # Extract features
    python main.py --step extract
    
    # Train models
    python main.py --step train
    
    # Run everything
    python main.py --step all

BEFORE RUNNING:
    1. Download TESS dataset from Kaggle:
       https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
    
    2. Update dataset path in each module's Config class
    
    3. Install dependencies:
       pip install -r requirements.txt

PROJECT STRUCTURE:
    feature_extraction.py    - Extract features from audio
    model_training.py        - Train deep learning models
    speech_conversion.py     - Speech-to-text & text-to-speech
    explainable_ai.py        - LIME & SHAP explanations
    gradcam_visualization.py - Grad-CAM for CNN interpretation
    main.py                  - This file (entry point)

For detailed documentation, see README.md
"""
    print(help_text)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='TESS Emotion Recognition Project',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=['extract', 'train', 'speech', 'xai', 'gradcam', 'all', 'help'],
        help='Step to execute'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Execute requested step
    if args.step is None or args.step == 'help':
        print_help()
    
    elif args.step == 'extract':
        run_feature_extraction()
    
    elif args.step == 'train':
        run_model_training()
    
    elif args.step == 'speech':
        run_speech_conversion()
    
    elif args.step == 'xai':
        run_explainable_ai()
    
    elif args.step == 'gradcam':
        run_gradcam()
    
    elif args.step == 'all':
        run_all_steps()
    
    print("\n" + "="*70)
    print("🎉 THANK YOU FOR USING TESS EMOTION RECOGNITION! 🎉")
    print("="*70)
    print("\n💡 Learning never stops! Keep exploring and experimenting!\n")


if __name__ == "__main__":
    main()
