"""
TESS Emotion Recognition - Speech Conversion Module

This script handles:
1. Speech-to-Text (Audio -> Text transcription)
2. Text-to-Speech (Text -> Audio synthesis)

Learning Points:
- ASR (Automatic Speech Recognition) converts audio to text
- TTS (Text-to-Speech) synthesizes speech from text
- Pipeline: Audio -> Text -> Audio (voice conversion practice)

Author: Educational Implementation
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try importing speech recognition libraries
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("⚠️  speech_recognition not available. Install with: pip install SpeechRecognition")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  whisper not available. Install with: pip install openai-whisper")

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  TTS not available. Install with: pip install TTS")


# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters"""
    # Paths
    DATASET_PATH = "/path/to/your/TESS-data"
    OUTPUT_DIR = "speech_conversion_output"
    
    # Audio parameters
    SAMPLE_RATE = 22050
    
    # Models
    WHISPER_MODEL = "base"  # tiny, base, small, medium, large
    TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"


# ==================== SPEECH TO TEXT ====================

class SpeechToText:
    """
    Convert speech audio to text.
    
    Learning Point:
        ASR systems work by:
        1. Extracting features from audio (like MFCCs)
        2. Using acoustic models to map features to phonemes
        3. Using language models to form words/sentences
        
        Modern ASR (like Whisper) uses transformers end-to-end!
    """
    
    def __init__(self, method='whisper'):
        """
        Initialize STT with chosen method.
        
        Args:
            method: 'whisper' or 'google' (requires internet)
        """
        self.method = method
        
        if method == 'whisper':
            if not WHISPER_AVAILABLE:
                raise ImportError("Whisper not available. Install with: pip install openai-whisper")
            print(f"\n🎤 Loading Whisper model: {Config.WHISPER_MODEL}")
            self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
            print("✅ Whisper model loaded!")
        
        elif method == 'google':
            if not SPEECH_RECOGNITION_AVAILABLE:
                raise ImportError("SpeechRecognition not available")
            self.recognizer = sr.Recognizer()
            print("\n🎤 Using Google Speech Recognition (requires internet)")
    
    def transcribe_with_whisper(self, audio_path):
        """
        Transcribe audio using Whisper.
        
        Learning Point:
            Whisper is a robust ASR model from OpenAI:
            - Trained on 680,000 hours of multilingual data
            - Works well on noisy audio
            - Doesn't require internet
            - Returns confidence scores
        """
        try:
            result = self.whisper_model.transcribe(audio_path)
            return {
                'text': result['text'].strip(),
                'language': result.get('language', 'en'),
                'segments': result.get('segments', [])
            }
        except Exception as e:
            print(f"Error transcribing with Whisper: {e}")
            return None
    
    def transcribe_with_google(self, audio_path):
        """
        Transcribe audio using Google Speech Recognition.
        
        Learning Point:
            Google's ASR is cloud-based:
            - Requires internet connection
            - Very accurate
            - Free tier available
            - Supports many languages
        """
        try:
            # Load audio
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            # Save as temporary WAV file (sr.Recognizer needs file)
            temp_wav = "temp_audio.wav"
            sf.write(temp_wav, audio_data, sr)
            
            # Recognize
            with sr.AudioFile(temp_wav) as source:
                audio = self.recognizer.record(source)
            
            text = self.recognizer.recognize_google(audio)
            
            # Clean up
            os.remove(temp_wav)
            
            return {
                'text': text,
                'language': 'en'
            }
        
        except Exception as e:
            print(f"Error transcribing with Google: {e}")
            return None
    
    def transcribe(self, audio_path):
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with transcription results
        """
        print(f"\n🎤 Transcribing: {os.path.basename(audio_path)}")
        
        if self.method == 'whisper':
            result = self.transcribe_with_whisper(audio_path)
        elif self.method == 'google':
            result = self.transcribe_with_google(audio_path)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if result:
            print(f"✅ Transcribed: \"{result['text']}\"")
        else:
            print("❌ Transcription failed")
        
        return result


# ==================== TEXT TO SPEECH ====================

class TextToSpeech:
    """
    Convert text to speech audio.
    
    Learning Point:
        TTS systems typically have two components:
        1. Text-to-Phoneme: Convert text to pronunciation
        2. Phoneme-to-Audio: Generate speech waveform
        
        Modern TTS (like Tacotron2) uses neural networks end-to-end!
    """
    
    def __init__(self, model_name=None):
        """
        Initialize TTS.
        
        Args:
            model_name: TTS model to use
        """
        if not TTS_AVAILABLE:
            raise ImportError("TTS library not available. Install with: pip install TTS")
        
        model_name = model_name or Config.TTS_MODEL
        
        print(f"\n🔊 Loading TTS model: {model_name}")
        print("   (This may take a minute for first-time download...)")
        
        try:
            self.tts = TTS(model_name=model_name, progress_bar=True)
            print("✅ TTS model loaded!")
        except Exception as e:
            print(f"❌ Error loading TTS model: {e}")
            print("\n💡 Try these alternative models:")
            print("   - tts_models/en/ljspeech/tacotron2-DDC")
            print("   - tts_models/en/ljspeech/glow-tts")
            print("   - tts_models/en/vctk/vits")
            raise
    
    def synthesize(self, text, output_path):
        """
        Synthesize speech from text.
        
        Learning Point:
            Neural TTS works by:
            1. Converting text to linguistic features
            2. Predicting mel-spectrogram from features
            3. Using vocoder to generate waveform from spectrogram
            
            Result: Natural-sounding synthetic speech!
        
        Args:
            text: Text to synthesize
            output_path: Where to save audio file
        
        Returns:
            Path to generated audio file
        """
        print(f"\n🔊 Synthesizing: \"{text}\"")
        
        try:
            # Generate speech
            self.tts.tts_to_file(
                text=text,
                file_path=output_path
            )
            
            print(f"✅ Saved to: {output_path}")
            return output_path
        
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
            return None


# ==================== CONVERSION PIPELINE ====================

class SpeechConversionPipeline:
    """
    Complete pipeline: Audio -> Text -> Audio
    
    Learning Point:
        This demonstrates the full speech processing cycle:
        1. Transcribe original audio to text (ASR)
        2. Generate new audio from text (TTS)
        3. Compare original vs synthesized
        
        Real applications:
        - Voice conversion
        - Speech enhancement
        - Accent modification
        - Voice cloning
    """
    
    def __init__(self, stt_method='whisper'):
        """Initialize pipeline with STT and TTS"""
        print("\n" + "="*70)
        print("INITIALIZING SPEECH CONVERSION PIPELINE")
        print("="*70)
        
        # Initialize STT
        self.stt = SpeechToText(method=stt_method)
        
        # Initialize TTS
        self.tts = TextToSpeech()
        
        print("\n✅ Pipeline ready!")
    
    def process_audio(self, input_audio_path, output_audio_path):
        """
        Process: Input Audio -> Text -> Output Audio
        
        Args:
            input_audio_path: Original audio file
            output_audio_path: Path for synthesized audio
        
        Returns:
            Dictionary with results
        """
        print("\n" + "="*70)
        print(f"PROCESSING: {os.path.basename(input_audio_path)}")
        print("="*70)
        
        # Step 1: Transcribe
        print("\n📝 Step 1: Transcribing audio to text...")
        transcription = self.stt.transcribe(input_audio_path)
        
        if not transcription:
            print("❌ Transcription failed. Aborting.")
            return None
        
        text = transcription['text']
        
        # Step 2: Synthesize
        print(f"\n🔊 Step 2: Synthesizing new audio from text...")
        output_path = self.tts.synthesize(text, output_audio_path)
        
        if not output_path:
            print("❌ Synthesis failed. Aborting.")
            return None
        
        # Step 3: Compare
        print(f"\n📊 Step 3: Comparing original and synthesized...")
        comparison = self.compare_audio(input_audio_path, output_audio_path)
        
        result = {
            'input_file': input_audio_path,
            'output_file': output_audio_path,
            'transcription': text,
            'comparison': comparison
        }
        
        print("\n✅ Processing complete!")
        return result
    
    def compare_audio(self, original_path, synthesized_path):
        """
        Compare original and synthesized audio.
        
        Learning Point:
            We can compare:
            - Duration
            - Energy/amplitude
            - Spectral characteristics
            - Pitch
        """
        try:
            # Load both audio files
            orig_audio, orig_sr = librosa.load(original_path, sr=Config.SAMPLE_RATE)
            synth_audio, synth_sr = librosa.load(synthesized_path, sr=Config.SAMPLE_RATE)
            
            # Calculate statistics
            comparison = {
                'original_duration': len(orig_audio) / orig_sr,
                'synthesized_duration': len(synth_audio) / synth_sr,
                'original_rms': np.sqrt(np.mean(orig_audio**2)),
                'synthesized_rms': np.sqrt(np.mean(synth_audio**2)),
            }
            
            print(f"\n   Original duration: {comparison['original_duration']:.2f}s")
            print(f"   Synthesized duration: {comparison['synthesized_duration']:.2f}s")
            print(f"   Original RMS energy: {comparison['original_rms']:.4f}")
            print(f"   Synthesized RMS energy: {comparison['synthesized_rms']:.4f}")
            
            return comparison
        
        except Exception as e:
            print(f"⚠️  Comparison failed: {e}")
            return None


# ==================== BATCH PROCESSING ====================

def batch_process_dataset(dataset_path, output_dir, num_samples=10):
    """
    Process multiple audio files from dataset.
    
    Args:
        dataset_path: Path to TESS dataset
        output_dir: Where to save results
        num_samples: Number of samples to process
    """
    print("\n" + "="*70)
    print("BATCH PROCESSING DATASET")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = SpeechConversionPipeline(stt_method='whisper')
    
    # Get audio files
    audio_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    # Sample random files
    import random
    audio_files = random.sample(audio_files, min(num_samples, len(audio_files)))
    
    print(f"\n📁 Processing {len(audio_files)} files...")
    
    # Process each file
    results = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'='*70}")
        print(f"FILE {i}/{len(audio_files)}")
        print(f"{'='*70}")
        
        # Generate output path
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        output_path = os.path.join(output_dir, f"{basename}_synthesized.wav")
        
        # Process
        result = pipeline.process_audio(audio_file, output_path)
        if result:
            results.append(result)
    
    # Save results summary
    if results:
        results_df = pd.DataFrame(results)
        results_csv = os.path.join(output_dir, "conversion_results.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"\n✅ Results saved to: {results_csv}")
    
    print("\n" + "✅"*35)
    print("BATCH PROCESSING COMPLETE!")
    print("✅"*35)
    print(f"\n📁 Output directory: {output_dir}")


# ==================== DEMONSTRATION ====================

def demonstrate_tts():
    """Demonstrate TTS with custom text"""
    print("\n" + "="*70)
    print("TEXT-TO-SPEECH DEMONSTRATION")
    print("="*70)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Sample texts with different emotions
    texts = [
        "I am so happy to see you today!",
        "This makes me very angry and frustrated.",
        "I am feeling quite sad and lonely.",
        "What a wonderful surprise this is!",
        "This situation fills me with fear.",
    ]
    
    # Initialize TTS
    tts = TextToSpeech()
    
    # Synthesize each text
    for i, text in enumerate(texts, 1):
        output_path = os.path.join(Config.OUTPUT_DIR, f"demo_tts_{i}.wav")
        tts.synthesize(text, output_path)
    
    print("\n✅ Demonstration complete!")
    print(f"📁 Audio files saved in: {Config.OUTPUT_DIR}")


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    
    print("\n" + "🎙️"*35)
    print("TESS EMOTION RECOGNITION - SPEECH CONVERSION")
    print("🎙️"*35)
    
    print("\n" + "="*70)
    print("AVAILABLE FEATURES")
    print("="*70)
    print(f"Speech Recognition: {'✅' if SPEECH_RECOGNITION_AVAILABLE else '❌'}")
    print(f"Whisper (ASR):      {'✅' if WHISPER_AVAILABLE else '❌'}")
    print(f"TTS:                {'✅' if TTS_AVAILABLE else '❌'}")
    
    if not (WHISPER_AVAILABLE and TTS_AVAILABLE):
        print("\n⚠️  Missing required libraries. Install with:")
        if not WHISPER_AVAILABLE:
            print("   pip install openai-whisper")
        if not TTS_AVAILABLE:
            print("   pip install TTS")
        return
    
    # Demonstrate TTS
    demonstrate_tts()
    
    # If dataset exists, process some samples
    if os.path.exists(Config.DATASET_PATH):
        print("\n" + "="*70)
        print("PROCESSING DATASET SAMPLES")
        print("="*70)
        batch_process_dataset(Config.DATASET_PATH, Config.OUTPUT_DIR, num_samples=5)
    else:
        print(f"\n⚠️  Dataset not found at: {Config.DATASET_PATH}")
        print("   Update Config.DATASET_PATH to process TESS dataset")
    
    print("\n" + "="*70)
    print("LEARNING SUMMARY")
    print("="*70)
    print("\n🎓 What you learned:")
    print("   1. Speech-to-Text (ASR) - Converting audio to text")
    print("   2. Text-to-Speech (TTS) - Synthesizing speech from text")
    print("   3. Audio conversion pipeline - Full cycle processing")
    print("   4. Comparing original and synthesized audio")
    print("\n💡 Real-world applications:")
    print("   - Voice assistants (Siri, Alexa)")
    print("   - Audiobook narration")
    print("   - Accessibility tools")
    print("   - Voice cloning")
    print("   - Speech translation")


if __name__ == "__main__":
    main()
