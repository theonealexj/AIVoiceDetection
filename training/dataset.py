import os
import glob
import torch
from torch.utils.data import Dataset
import sys
import torchaudio.transforms as T

# Add parent dir to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.audio_utils import preprocess_audio, extract_features, feature_to_tensor, SAMPLE_RATE

class VoiceDataset(Dataset):
    def __init__(self, human_dir=None, ai_dir=None, file_list=None, train=True):
        """
        Args:
            human_dir: Path to human audio directory
            ai_dir: Path to AI audio directory
            file_list: Optional list of tuples (file_path, label) to use directly. 
                       If provided, human_dir and ai_dir are ignored.
            train: Boolean, whether to apply augmentation
        """
        self.train = train
        
        if file_list is not None:
             self.all_files = file_list
        else:
            self.human_files = glob.glob(os.path.join(human_dir, "*"))
            self.ai_files = glob.glob(os.path.join(ai_dir, "*"))
            self.all_files = [(f, 0) for f in self.human_files] + [(f, 1) for f in self.ai_files]
        
        # SpecAugment transforms
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)
        
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path, label = self.all_files[idx]
        
        try:
            # Read file bytes
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            
            # Preprocess
            y = preprocess_audio(audio_bytes)
            
            # Feature extract
            features = extract_features(y)
            
            # To Tensor (1, N_MELS, Time)
            x = feature_to_tensor(features)
            
            # Apply Augmentation only during training
            if self.train:
                x = self.freq_mask(x)
                x = self.time_mask(x)
            
            return x, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a zero tensor in case of error
            return torch.zeros((1, 128, 126)), torch.tensor(label, dtype=torch.float32)
