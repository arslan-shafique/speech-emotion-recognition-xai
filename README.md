# TESS Emotion Recognition Project 🎭🔊

A comprehensive Deep Learning project for emotion recognition from speech using the Toronto Emotional Speech Set (TESS) dataset. This project includes educational explanations to help you understand every concept!

## 📋 Project Overview

This project implements an end-to-end emotion recognition system with:
- **Feature Extraction** from audio files
- **Deep Learning Models** (CNN, LSTM, Transformer)
- **Speech-to-Text and Text-to-Speech** conversion
- **Explainable AI (XAI)** techniques
- **Grad-CAM** visualizations for model interpretability

## 🎯 Learning Objectives

By working through this project, you will learn:
1. How audio data is processed and represented
2. Feature extraction techniques (MFCC, Mel-spectrogram, etc.)
3. Building and training deep learning models for audio
4. Making models interpretable with XAI
5. Understanding what neural networks learn with Grad-CAM
6. Working with speech recognition and synthesis

## 📊 Dataset

**TESS (Toronto Emotional Speech Set)**
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- **Size**: 2,800 audio files
- **Speakers**: 2 actresses (ages 26 and 64)
- **Emotions**: 7 emotions (angry, disgust, fear, happy, neutral, pleasant surprise, sad)
- **Words**: 200 target words per actress per emotion

### Download Instructions

1. Download the dataset from Kaggle
2. Extract to your preferred location
3. Update the `dataset_path` variable in the notebooks

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA for GPU acceleration

### Setup

```bash
# Clone the repository
git clone https://github.com/arslan-shafique/speech-emotion-recognition-xai.git
cd speech-emotion-recognition-xai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
Deep-Learning-NLP/
├── Project/
│   ├── EDA.ipynb                          # Exploratory Data Analysis
│   ├── feature-extraction.ipynb           # Feature extraction from audio
│   ├── model-training.ipynb               # Train ML/DL models
│   ├── speech-conversion.ipynb            # Speech-to-Text & Text-to-Speech
│   ├── explainable-ai.ipynb               # XAI with LIME & SHAP
│   ├── grad-cam.ipynb                     # Grad-CAM visualizations
│   ├── tess_features.csv                  # Extracted features
│   └── models/                            # Saved trained models
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── .gitignore                            # Git ignore rules
```

## 📚 Notebook Guide

### 1. EDA.ipynb - Exploratory Data Analysis
**What you'll learn:**
- Dataset structure and organization
- Audio waveform visualization
- Spectrogram analysis
- How emotions differ in audio signals

**Key Concepts:**
- Sample rate and duration
- Time domain vs frequency domain
- Spectrograms as visual representations

### 2. feature-extraction.ipynb - Feature Extraction
**What you'll learn:**
- MFCC (Mel-Frequency Cepstral Coefficients)
- Mel-spectrograms
- Chroma features
- Zero Crossing Rate (ZCR)
- Spectral features (centroid, bandwidth, rolloff)

**Key Concepts:**
- Why we need features for ML models
- What each feature represents
- How to normalize and prepare data

### 3. model-training.ipynb - Model Training
**What you'll learn:**
- Building CNN models for audio classification
- Using LSTM for sequence modeling
- Hybrid CNN-LSTM architectures
- (Optional) Transformer models
- Training, validation, and testing

**Key Concepts:**
- Model architecture design
- Loss functions and optimizers
- Overfitting prevention
- Model evaluation metrics

### 4. speech-conversion.ipynb - Speech Conversion
**What you'll learn:**
- Speech-to-Text using Whisper
- Text-to-Speech synthesis
- Audio format conversion
- Pipeline creation

**Key Concepts:**
- ASR (Automatic Speech Recognition)
- TTS (Text-to-Speech) synthesis
- Audio file manipulation

### 5. explainable-ai.ipynb - Explainable AI
**What you'll learn:**
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Feature importance visualization
- Model decision interpretation

**Key Concepts:**
- Why XAI matters
- Local vs global explanations
- Trust and transparency in ML

### 6. grad-cam.ipynb - Grad-CAM Visualization
**What you'll learn:**
- Gradient-weighted Class Activation Mapping
- Visualizing CNN attention
- Understanding what the model "sees"
- Overlaying attention on spectrograms

**Key Concepts:**
- Activation maps
- Gradient flow
- Visual interpretation of deep learning

## 🔬 Methodology

### Feature Extraction Process
1. Load audio files using librosa
2. Extract multiple features:
   - **MFCC**: Captures spectral envelope
   - **Mel-spectrogram**: Time-frequency representation
   - **Chroma**: Pitch class information
   - **ZCR**: Signal noisiness
   - **Spectral features**: Frequency characteristics
3. Aggregate features (mean, std, etc.)
4. Normalize for model training

### Model Architecture
- **Input**: Extracted audio features
- **CNN Layers**: Learn spatial patterns in spectrograms
- **LSTM Layers**: Learn temporal dependencies
- **Dense Layers**: Final classification
- **Output**: 7 emotion classes

### Training Strategy
- Train/Validation/Test split: 70/15/15
- Data augmentation (pitch shift, time stretch)
- Early stopping and learning rate scheduling
- Cross-validation for robustness

## 📈 Expected Results

- **Accuracy**: 85-95% (depending on model)
- **F1-Score**: High across all emotion classes
- **Confusion Matrix**: Clear emotion separation
- **XAI Insights**: Understandable feature importance

## 🎓 Educational Approach

Each notebook includes:
- 📖 **Detailed Comments**: Every code block explained
- 💡 **Learning Points**: Key concepts highlighted
- 📊 **Visualizations**: See what's happening
- ❓ **Why This Matters**: Real-world context
- ✅ **Checkpoints**: Verify your understanding

## 🛠️ Usage

### Quick Start

```python
# 1. Run EDA notebook first
jupyter notebook Project/EDA.ipynb

# 2. Extract features
jupyter notebook Project/feature-extraction.ipynb

# 3. Train models
jupyter notebook Project/model-training.ipynb

# 4. Explore XAI
jupyter notebook Project/explainable-ai.ipynb

# 5. Visualize with Grad-CAM
jupyter notebook Project/grad-cam.ipynb
```

### Custom Dataset Path

Update the dataset path in each notebook:
```python
dataset_path = "/your/path/to/TESS-data"
```

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- TESS Dataset creators
- librosa and audio processing community
- Deep learning framework developers
- Open source XAI libraries

## 📧 Contact

For questions or discussions about this project, feel free to open an issue!

---

**Happy Learning! 🚀**

Remember: The goal is not just to build a model, but to **understand** every step of the process!
