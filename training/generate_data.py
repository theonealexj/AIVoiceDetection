from gtts import gTTS
import os
import random
import numpy as np
import soundfile as sf
import librosa

DATA_DIR = r"c:\Users\Alex\Desktop\Antigravity\AIVoiceDetection\data"
HUMAN_DIR = os.path.join(DATA_DIR, "human")
AI_DIR = os.path.join(DATA_DIR, "ai")

TEXT_SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Voice detection capabilities are improving every day.",
    "Hello current user, this is a test sample.",
    "Machine learning models require good data.",
    "FastAPI is a modern, fast (high-performance), web framework for building APIs.",
    "Deep learning is a subset of machine learning."
]

def generate_ai_samples(count=10):
    print(f"Generating {count} AI samples...")
    if not os.path.exists(AI_DIR):
        os.makedirs(AI_DIR)
        
    for i in range(count):
        text = random.choice(TEXT_SAMPLES)
        try:
            tts = gTTS(text=text, lang='en')
            filename = os.path.join(AI_DIR, f"ai_gen_{i}.mp3")
            tts.save(filename)
            # Convert to wav for consistency if needed, but preprocessing handles loading
        except Exception as e:
            print(f"Failed to generate sample {i}: {e}")

def download_human_samples_placeholder(count=10):
    print(f"Generating {count} placeholder Human samples (Noise/Tone)...")
    if not os.path.exists(HUMAN_DIR):
        os.makedirs(HUMAN_DIR)
        
    sr = 16000
    duration = 4
    for i in range(count):
        # Generate synthetic 'human' audio (just noise/tone for testing pipeline)
        # In a real scenario, download from Common Voice
        t = np.linspace(0, duration, int(sr * duration))
        freq = random.randint(200, 500)
        y = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.normal(0, 1, len(t))
        
        filename = os.path.join(HUMAN_DIR, f"human_placeholder_{i}.wav")
        sf.write(filename, y, sr)

if __name__ == "__main__":
    generate_ai_samples(20)
    download_human_samples_placeholder(20)
    print("Data generation complete.")
