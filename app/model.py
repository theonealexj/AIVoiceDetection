import torch
import torch.nn as nn
from torchvision import models

class VoiceDetectorCNN(nn.Module):
    def __init__(self):
        super(VoiceDetectorCNN, self).__init__()
        # Use ResNet18 as the backbone
        # pretrained=False because spectrograms are quite different from ImageNet images
        # and we have a specific domain.
        self.resnet = models.resnet18(weights=None)
        
        # Modify first layer to accept 1 channel (Mel Spectrogram) instead of 3 (RGB)
        # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the fully connected layer for binary classification (1 output)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)
