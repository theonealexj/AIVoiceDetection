import sys
import os
import base64
from app.inference import load_model, predict_voice

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file>")
        return

    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Loading model and analyzing: {file_path}...")
    
    # 1. Load Model
    load_model()
    
    # 2. Read File & Encode to Base64 (because inference.py expects base64)
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
        base64_str = base64.b64encode(audio_bytes).decode("utf-8")
        
    # 3. Predict
    try:
        result = predict_voice(base64_str)
        
        print("\n" + "="*30)
        print(f"RESULT: {result['classification']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print("="*30 + "\n")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
