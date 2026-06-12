"""LiteNeTX FashionMNIST model core."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from collections import OrderedDict


class LiteNeTX_Base_CNN_FashionMNIST(nn.Module):
    """FashionMNIST CNN model."""

    def __init__(self, num_classes=10):
        super().__init__()

        # First down block
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_down1 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn_down1 = nn.BatchNorm2d(64)
        self.dropout2d_1 = nn.Dropout2d(0.1)

        # Second down block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv_down2 = nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False)
        self.bn_down2 = nn.BatchNorm2d(128)
        self.dropout2d_2 = nn.Dropout2d(0.15)

        # Final classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.dropout2d_1(x)
        x = F.relu(self.bn_down1(self.conv_down1(x)), inplace=True)

        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.dropout2d_2(x)
        x = F.relu(self.bn_down2(self.conv_down2(x)), inplace=True)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


_model = None


def load_fashion_model():
    """Load FashionMNIST model."""
    global _model
    if _model is not None:
        return _model

    device = torch.device("cpu")
    model = LiteNeTX_Base_CNN_FashionMNIST(num_classes=10)

    model_path = Path(__file__).parent.parent.parent / "models" / "LiteNeTX_Base_CNN_FashionMNIST.safetensors"
    state_dict = load_file(str(model_path))

    clean = OrderedDict()
    for k, v in state_dict.items():
        clean[k.replace('module.', '') if k.startswith('module.') else k] = v

    model.load_state_dict(clean)
    model.to(device)
    model.eval()

    _model = model
    return _model


def get_fashion_model():
    if _model is None:
        return load_fashion_model()
    return _model
