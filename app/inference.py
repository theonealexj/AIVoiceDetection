import torch
import numpy as np
import base64
from .model import VoiceDetectorCNN
from .audio_utils import preprocess_audio, extract_features, feature_to_tensor

# Load model
MODEL_PATH = r"c:\Users\Alex\Desktop\Antigravity\AIVoiceDetection\model.pth"
device = torch.device("cpu") # CPU inference requirement
model = VoiceDetectorCNN().to(device)

def load_model():
    """
    Loads the model weights. Should be called on startup.
    """
    global model
    try:
        if hasattr(torch, 'load'):
            # simple check if path exists
            import os
            if os.path.exists(MODEL_PATH):
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                model.eval()
                print("Model loaded successfully.")
            else:
                print(f"Warning: Model file not found at {MODEL_PATH}. Using random weights.")
                model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")

def predict_voice(base64_audio: str):
    """
    Pipeline: Base64 -> Bytes -> Preprocess -> Feature -> Model -> Output
    Returns: classification (str), confidence (float), debug_metrics (dict)
    """
    try:
        # Decode base64
        try:
            audio_bytes = base64.b64decode(base64_audio)
        except Exception:
            raise ValueError("Invalid Base64 string")

        if not audio_bytes:
            raise ValueError("Empty audio content")

        # Preprocess
        y = preprocess_audio(audio_bytes)
        
        # Features
        features = extract_features(y)
        input_tensor = feature_to_tensor(features).to(device).unsqueeze(0) # Add batch dim: (1, 1, F, T)
        
        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()
            
        # Classification
        label = "AI_GENERATED" if prob > 0.5 else "HUMAN"
        # Confidence: if AI (prob > 0.5), conf = prob. If Human (prob <= 0.5), conf = 1 - prob.
        confidence = prob if label == "AI_GENERATED" else 1.0 - prob
        
        # Explainability (Simple metrics computed from features)
        # Spectral smoothness: std dev of mel bands
        # Pitch variance: approximate from simple spectral centroid variance (proxy)
        spec_smoothness = float(np.std(features))
        pitch_variance = 0.0 # Placeholder as pitch extraction requires more complex logic/libraries not in simplified utils
        
        return {
            "classification": label,
            "confidence": round(confidence, 4),
            "explainability": {
                "spectral_smoothness": round(spec_smoothness, 4),
                "pitch_variance": 0.0
            }
        }
        
    except Exception as e:
        print(f"Inference error: {e}")
        raise e
