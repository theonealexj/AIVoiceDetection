import librosa
import numpy as np
import io
import soundfile as sf
import torch

# Constants
SAMPLE_RATE = 16000
DURATION = 4  # seconds
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    """
    Decodes audio bytes, resamples to 16kHz, converts to mono,
    normalizes, and pads/trims to fixed duration.
    """
    try:
        # Load audio from bytes
        # librosa.load supports file-like objects
        with io.BytesIO(audio_bytes) as f:
            y, sr = librosa.load(f, sr=SAMPLE_RATE, mono=True)
        
        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
            
        # Fix length
        target_len = SAMPLE_RATE * DURATION
        if len(y) > target_len:
            y = y[:target_len]
        else:
            padding = target_len - len(y)
            y = np.pad(y, (0, padding), 'constant')
            
        return y
    except Exception as e:
        raise ValueError(f"Error processing audio: {e}")

def extract_features(y: np.ndarray) -> np.ndarray:
    """
    Extracts Mel Spectrogram features from the waveform.
    Returns a numpy array of shape (N_MELS, TimeFrames).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=SAMPLE_RATE, 
        n_mels=N_MELS, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def feature_to_tensor(feature: np.ndarray) -> torch.Tensor:
    """
    Converts numpy feature array to PyTorch tensor with channel dimension.
    Shape: (1, N_MELS, TimeFrames)
    """
    tensor = torch.tensor(feature, dtype=torch.float32)
    return tensor.unsqueeze(0) # Add channel dim
