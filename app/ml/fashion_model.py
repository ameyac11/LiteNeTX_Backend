"""FashionMNIST model definition and loader."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class LiteGrayCNN(nn.Module):
    """Lightweight CNN for grayscale images (FashionMNIST)."""
    
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 14 * 14, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        x = self.fc4(x)
        
        return x


# Global model instance
_fashion_model = None


def load_fashion_model():
    """Load the FashionMNIST model (singleton pattern)."""
    global _fashion_model
    
    if _fashion_model is not None:
        return _fashion_model
    
    device = torch.device("cpu")
    model = LiteGrayCNN(num_classes=10)
    
    # Load model weights
    model_path = Path(__file__).parent.parent.parent / "models" / "LiteGrayCNN.pth"
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel wrapper if present
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    _fashion_model = model
    return _fashion_model


def get_fashion_model():
    """Get the loaded FashionMNIST model."""
    if _fashion_model is None:
        return load_fashion_model()
    return _fashion_model
