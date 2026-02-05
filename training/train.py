import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import glob
import random

# Add parent dir to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import VoiceDetectorCNN
from dataset import VoiceDataset

# Config
BATCH_SIZE = 16
EPOCHS = 25
LR = 0.001
MODEL_SAVE_PATH = r"c:\Users\Alex\Desktop\Antigravity\AIVoiceDetection\model.pth"
DATA_DIR = r"c:\Users\Alex\Desktop\Antigravity\AIVoiceDetection\data"

def train():
    # Set seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    human_dir = os.path.join(DATA_DIR, "human")
    ai_dir = os.path.join(DATA_DIR, "ai")
    
    if not os.path.exists(human_dir) or not os.path.exists(ai_dir):
        print("Data directories not found. Please run generate_data.py first.")
        return

    # Prepare file lists
    human_files = glob.glob(os.path.join(human_dir, "*"))
    ai_files = glob.glob(os.path.join(ai_dir, "*"))
    
    print(f"Found {len(human_files)} Human samples and {len(ai_files)} AI samples.")
    
    if len(human_files) == 0 or len(ai_files) == 0:
        print("Insufficient data.")
        return

    all_files = [(f, 0) for f in human_files] + [(f, 1) for f in ai_files]
    random.shuffle(all_files)
    
    # Split 80/20
    val_count = int(0.2 * len(all_files))
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]
    
    print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")
    
    # Datasets
    train_dataset = VoiceDataset(file_list=train_files, train=True)
    val_dataset = VoiceDataset(file_list=val_files, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model Setup
    model = VoiceDetectorCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.sigmoid(output) > 0.5
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                target = target.unsqueeze(1)
                
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                preds = torch.sigmoid(output) > 0.5
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Acc: {val_acc:.4f}")
        
        # Scheduler step
        scheduler.step(avg_val_loss)
        
        # Save Best Model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  > New Best Model Saved (Loss: {best_loss:.4f})")
            
    print(f"Training Complete. Best Validation Loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()
