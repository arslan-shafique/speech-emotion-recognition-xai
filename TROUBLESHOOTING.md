# Troubleshooting Guide - TESS Emotion Recognition

## Common Issues and Solutions

### 🔴 Installation Issues

#### Issue: `pip install` fails with "ERROR: Could not build wheels"

**Possible Causes**:
- Missing system dependencies
- Incompatible Python version
- Missing build tools

**Solutions**:

**For Windows**:
```bash
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then retry installation
pip install -r requirements.txt
```

**For Mac**:
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required libraries
brew install ffmpeg portaudio

# Retry installation
pip install -r requirements.txt
```

**For Linux**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev ffmpeg

# Fedora/RHEL
sudo dnf install python3-devel portaudio-devel ffmpeg

# Retry installation
pip install -r requirements.txt
```

#### Issue: TensorFlow installation fails

**Solution**:
```bash
# For CPU only
pip install tensorflow

# For GPU (CUDA required)
pip install tensorflow-gpu

# If conflicts, uninstall first
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow
```

#### Issue: "No module named 'librosa'"

**Solution**:
```bash
# Install librosa with dependencies
pip install librosa soundfile

# If still fails, try:
pip install --upgrade librosa
```

---

### 🔴 Dataset Issues

#### Issue: "Dataset path not found"

**Solution**:
1. Verify dataset is downloaded and extracted
2. Check the path in config:
```python
# In each module, update this line:
DATASET_PATH = "/absolute/path/to/TESS-data"
```
3. Use absolute paths, not relative paths
4. On Windows, use forward slashes or escape backslashes:
```python
# Good
DATASET_PATH = "C:/Users/YourName/TESS-data"
# Or
DATASET_PATH = "C:\\Users\\YourName\\TESS-data"
```

#### Issue: "No WAV files found"

**Possible Causes**:
- Dataset structure is different
- Files not extracted properly
- Path points to wrong directory

**Solution**:
```bash
# Check directory structure should be:
TESS-data/
    OAF_angry/
        OAF_back_angry.wav
        OAF_bar_angry.wav
        ...
    OAF_disgust/
        ...
    YAF_angry/
        ...

# If structure is different, adjust code in feature_extraction.py
```

---

### 🔴 Feature Extraction Issues

#### Issue: "Error loading audio file"

**Possible Causes**:
- Corrupted audio file
- Unsupported format
- Missing audio codec

**Solution**:
```bash
# Install ffmpeg for audio format support
# Windows: Download from https://ffmpeg.org/
# Mac: brew install ffmpeg
# Linux: sudo apt-get install ffmpeg

# Then reinstall audioread
pip install --upgrade audioread
```

#### Issue: "Memory Error during feature extraction"

**Solution**:
```python
# Process in smaller batches
# In feature_extraction.py, modify:

# Instead of processing all at once, process in chunks
CHUNK_SIZE = 100
for i in range(0, len(dataset_df), CHUNK_SIZE):
    chunk = dataset_df.iloc[i:i+CHUNK_SIZE]
    # Process chunk
```

#### Issue: Features contain NaN values

**Solution**:
```python
# Add this after feature extraction:
features_df = features_df.fillna(0)  # Replace NaN with 0
# Or
features_df = features_df.dropna()  # Remove rows with NaN
```

---

### 🔴 Model Training Issues

#### Issue: "Model accuracy not improving"

**Possible Causes**:
- Learning rate too high/low
- Model too simple/complex
- Data not normalized
- Features not informative

**Solutions**:

**Try different learning rates**:
```python
# In model_training.py
LEARNING_RATE = 0.0001  # Lower (was 0.001)
# or
LEARNING_RATE = 0.01    # Higher
```

**Add more data augmentation**:
```python
# Add noise to training data
def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(*data.shape)
    return data + noise_factor * noise

X_train_augmented = add_noise(X_train)
```

**Check feature normalization**:
```python
# Ensure features are normalized
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

#### Issue: "CUDA out of memory"

**Solutions**:

**Reduce batch size**:
```python
# In model_training.py
BATCH_SIZE = 16  # Reduce from 32
# or
BATCH_SIZE = 8
```

**Use mixed precision training**:
```python
# Add to model_training.py
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

**Clear GPU memory**:
```python
# Add between model trainings
import tensorflow as tf
tf.keras.backend.clear_session()
```

#### Issue: Model overfitting (high train, low val accuracy)

**Solutions**:

**Increase dropout**:
```python
DROPOUT_RATE = 0.5  # Increase from 0.3
```

**Add regularization**:
```python
from tensorflow.keras import regularizers

layers.Dense(128, 
    activation='relu',
    kernel_regularizer=regularizers.l2(0.01)
)
```

**Use early stopping**:
```python
callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,  # Reduce from 15
    restore_best_weights=True
)
```

#### Issue: Training very slow

**Solutions**:

**Enable GPU**:
```python
# Check GPU availability
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# If GPU available but not used, try:
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

**Reduce model complexity**:
```python
# Use smaller model
layers.Dense(256)  # Instead of 512
```

**Use smaller dataset for testing**:
```python
# Test with 10% of data first
X_train_small = X_train[:len(X_train)//10]
y_train_small = y_train[:len(y_train)//10]
```

---

### 🔴 Speech Conversion Issues

#### Issue: "Whisper model download fails"

**Solution**:
```python
# Download manually and specify path
import whisper
model = whisper.load_model("base", download_root="/path/to/save")
```

#### Issue: TTS model too large / slow

**Solution**:
```python
# Use faster model
TTS_MODEL = "tts_models/en/ljspeech/glow-tts"  # Faster than tacotron2
```

#### Issue: "No audio input/output device"

**Solution**:
```bash
# Install portaudio
# Windows: pip install pyaudio
# Mac: brew install portaudio && pip install pyaudio
# Linux: sudo apt-get install portaudio19-dev python3-pyaudio
```

---

### 🔴 XAI Issues

#### Issue: LIME/SHAP very slow

**Solutions**:

**Reduce number of samples**:
```python
# In explainable_ai.py
NUM_SAMPLES_EXPLAIN = 3  # Reduce from 5

# For SHAP global importance
n_samples = 50  # Reduce from 100
```

**Use simpler explainer**:
```python
# Instead of DeepExplainer, use KernelExplainer (faster)
explainer = shap.KernelExplainer(model.predict, background_data)
```

#### Issue: "SHAP values dimension mismatch"

**Solution**:
```python
# Ensure correct input shape
instance = instance.reshape(1, -1)  # Add batch dimension
shap_values = explainer.shap_values(instance)
```

---

### 🔴 Grad-CAM Issues

#### Issue: "No convolutional layers found"

**Cause**: Using Dense or LSTM model instead of CNN

**Solution**:
```python
# Use CNN or Hybrid model for Grad-CAM
MODEL_PATH = "models/cnn_model_best.h5"
# or
MODEL_PATH = "models/hybrid_model_best.h5"
```

#### Issue: Heatmap all zeros

**Solution**:
```python
# Ensure using correct layer
# Print all layer names
for layer in model.layers:
    print(layer.name, layer.output_shape)

# Specify layer explicitly
gradcam = GradCAM(model, layer_name='conv1d_2')
```

---

### 🔴 General Python Issues

#### Issue: "ImportError: cannot import name 'X'"

**Solution**:
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# If specific package issues:
pip install --upgrade tensorflow keras numpy pandas
```

#### Issue: Jupyter Notebook kernel dies

**Solutions**:

**Increase memory limit**:
```bash
# Start jupyter with more memory
jupyter notebook --NotebookApp.iopub_data_rate_limit=1e10
```

**Process data in chunks**:
```python
# Don't load all data at once
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    process(chunk)
```

---

## 🛠️ Debugging Tips

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Data Shapes

```python
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {len(np.unique(y_train))}")
```

### Validate Model Input/Output

```python
# Check model expects correct input
print(model.input_shape)  # Should match your data

# Test prediction
test_input = X_test[0:1]  # Single sample
prediction = model.predict(test_input)
print(f"Prediction shape: {prediction.shape}")  # Should be (1, num_classes)
```

### Monitor Resource Usage

```python
import psutil
import GPUtil

# CPU and RAM
print(f"CPU: {psutil.cpu_percent()}%")
print(f"RAM: {psutil.virtual_memory().percent}%")

# GPU (if available)
GPUs = GPUtil.getGPUs()
if GPUs:
    gpu = GPUs[0]
    print(f"GPU Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
```

---

## 📞 Getting More Help

### Before Asking for Help

1. ✅ Read error message carefully
2. ✅ Check this troubleshooting guide
3. ✅ Search error message online
4. ✅ Try solutions from similar issues
5. ✅ Isolate the problem (minimal example)

### When Asking for Help

Include:
1. **Error message** (full traceback)
2. **Code snippet** (relevant part)
3. **Environment** (OS, Python version, library versions)
4. **What you tried** (solutions attempted)
5. **Expected vs actual behavior**

### Useful Commands for Debugging

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Check TensorFlow/CUDA
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

# Check librosa
python -c "import librosa; print(librosa.__version__)"

# System info
python -m platform
```

---

## 🔍 Quick Checklist

Before running the project:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded and extracted
- [ ] Dataset path updated in config
- [ ] Sufficient disk space (2GB+ free)
- [ ] Sufficient RAM (4GB+ recommended)

If all checked and still issues, refer to specific sections above!

---

**Still stuck?** Open an issue on GitHub with details! We're here to help! 🚀
