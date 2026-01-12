# TESS Emotion Recognition - Quick Start Guide

Welcome! This guide will help you get started with the emotion recognition project.

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- (Optional) NVIDIA GPU with CUDA for faster training

## 🚀 Quick Setup (5 minutes)

### Step 1: Clone Repository
```bash
git clone https://github.com/umairva7/Deep-Learning-NLP.git
cd Deep-Learning-NLP
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

⏰ This may take 5-10 minutes depending on your internet speed.

### Step 4: Download TESS Dataset

1. Go to: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
2. Download the dataset (about 550 MB)
3. Extract to a folder (e.g., `/path/to/TESS-data`)

### Step 5: Update Configuration

Edit the dataset path in each module:

**In `feature_extraction.py`:**
```python
DATASET_PATH = "/your/path/to/TESS-data"  # Line 29
```

**In `speech_conversion.py`:**
```python
DATASET_PATH = "/your/path/to/TESS-data"  # Line 61
```

## 🎯 Running the Project

### Option 1: Run Everything (Recommended for First Time)

```bash
cd Project
python main.py --step all
```

This will:
1. Extract features (~10 minutes)
2. Train all models (~30-60 minutes depending on hardware)
3. Demonstrate speech conversion (~5 minutes)
4. Generate XAI explanations (~10 minutes)
5. Create Grad-CAM visualizations (~5 minutes)

**Total time: ~1-2 hours**

### Option 2: Run Individual Steps

```bash
# Step 1: Extract features
python main.py --step extract

# Step 2: Train models
python main.py --step train

# Step 3: Speech conversion demo
python main.py --step speech

# Step 4: Explainable AI
python main.py --step xai

# Step 5: Grad-CAM visualization
python main.py --step gradcam
```

### Option 3: Run Modules Directly

```bash
# Feature extraction
python feature_extraction.py

# Model training
python model_training.py

# Speech conversion
python speech_conversion.py

# Explainable AI
python explainable_ai.py

# Grad-CAM
python gradcam_visualization.py
```

## 📊 What to Expect

### After Feature Extraction:
- `tess_features.csv` (extracted features)
- `tess_features.pkl` (faster loading format)

### After Model Training:
- `models/` directory with saved models
- Training plots showing accuracy/loss
- Confusion matrices for each model
- Model comparison chart

### After Speech Conversion:
- `speech_conversion_output/` with synthesized audio
- Text transcriptions
- Comparison statistics

### After XAI:
- `xai_output/` with LIME and SHAP visualizations
- Feature importance plots
- Individual prediction explanations

### After Grad-CAM:
- `gradcam_output/` with attention heatmaps
- Feature map visualizations
- Confidence distributions

## 🎓 Learning Path

### For Beginners:
1. Start with `README.md` to understand the project
2. Run feature extraction and observe the outputs
3. Read the educational comments in each script
4. Try different parameters and see what changes

### For Intermediate:
1. Experiment with different model architectures
2. Try data augmentation techniques
3. Tune hyperparameters
4. Add new features to extract

### For Advanced:
1. Implement transformer models
2. Try transfer learning with pre-trained models
3. Implement ensemble methods
4. Deploy as a web service

## 🐛 Common Issues

### Issue: "CUDA out of memory"
**Solution:** Reduce `BATCH_SIZE` in `model_training.py`

### Issue: "Dataset path not found"
**Solution:** Update `DATASET_PATH` in the config of each module

### Issue: "Module not found"
**Solution:** Make sure virtual environment is activated and dependencies installed:
```bash
pip install -r requirements.txt
```

### Issue: "TensorFlow not using GPU"
**Solution:** Install CUDA-enabled TensorFlow:
```bash
pip install tensorflow-gpu
```

## 💡 Tips

1. **Start Small**: Use a subset of data first to verify everything works
2. **Monitor Resources**: Training can be resource-intensive
3. **Save Often**: Models are automatically saved during training
4. **Experiment**: Try different parameters and architectures
5. **Ask Questions**: Open an issue on GitHub if stuck

## 📚 Additional Resources

- **TensorFlow Tutorial**: https://www.tensorflow.org/tutorials
- **Librosa Documentation**: https://librosa.org/doc/latest/
- **LIME Paper**: https://arxiv.org/abs/1602.04938
- **SHAP Paper**: https://arxiv.org/abs/1705.07874
- **Grad-CAM Paper**: https://arxiv.org/abs/1610.02391

## 🤝 Getting Help

- Check `README.md` for detailed documentation
- Read code comments for explanations
- Open an issue on GitHub
- Review the educational content in each module

## ✅ Success Checklist

- [ ] Environment set up correctly
- [ ] Dependencies installed
- [ ] Dataset downloaded and path configured
- [ ] Feature extraction completed successfully
- [ ] At least one model trained
- [ ] Visualizations generated

## 🎉 Next Steps

Once you've completed the basic workflow:

1. **Improve Performance**: Try different features, models, or hyperparameters
2. **Deploy**: Create a web app or API
3. **Extend**: Add more emotion categories
4. **Share**: Contribute improvements back to the project

---

**Happy Learning! 🚀**

Remember: The goal is not just to build a model, but to **understand** every step!
