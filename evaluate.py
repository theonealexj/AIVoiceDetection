import torch
from torch.utils.data import DataLoader
import os
import sys
import glob
import random

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import VoiceDetectorCNN
from training.dataset import VoiceDataset

# Config (Must match train.py logic for reproduction)
BATCH_SIZE = 8
DATA_DIR = r"c:\Users\Alex\Desktop\Antigravity\AIVoiceDetection\data"
MODEL_PATH = r"c:\Users\Alex\Desktop\Antigravity\AIVoiceDetection\model.pth"

def evaluate():
    print("Initializing evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Reproduce Data Split
    random.seed(42)
    torch.manual_seed(42)
    
    human_dir = os.path.join(DATA_DIR, "human")
    ai_dir = os.path.join(DATA_DIR, "ai")
    
    # Matches the updated train.py logic
    supported_exts = ('.wav', '.mp3', '.flac', '.ogg')
    
    human_files = []
    for root, dirs, files in os.walk(human_dir):
        for file in files:
            if file.lower().endswith(supported_exts):
                human_files.append(os.path.join(root, file))
                
    ai_files = []
    for root, dirs, files in os.walk(ai_dir):
        for file in files:
            if file.lower().endswith(supported_exts):
                ai_files.append(os.path.join(root, file))
                
    if not human_files or not ai_files:
        print("Error: Could not find data files.")
        return

    all_files = [(f, 0) for f in human_files] + [(f, 1) for f in ai_files]
    random.shuffle(all_files)
    
    val_count = int(0.2 * len(all_files))
    val_files = all_files[:val_count]  # This is the test set
    
    print(f"Evaluating on {len(val_files)} held-out test samples.")
    
    # 2. Setup Loader
    val_dataset = VoiceDataset(file_list=val_files, train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Load Model
    if not os.path.exists(MODEL_PATH):
        print("Error: model.pth not found. Train first.")
        return
        
    model = VoiceDetectorCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 4. Evaluation Loop
    correct = 0
    total = 0
    
    print("Running inference...")
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(1)
            
            output = model(data)
            preds = torch.sigmoid(output) > 0.5
            correct += (preds == target).sum().item()
            total += target.size(0)
            
    acc = correct / total if total > 0 else 0
    print("\n" + "="*30)
    print(f"TEST RESULTS")
    print(f"Total Samples: {total}")
    print(f"Correct:       {correct}")
    print(f"Accuracy:      {acc*100:.2f}%")
    print("="*30 + "\n")

if __name__ == "__main__":
    evaluate()
