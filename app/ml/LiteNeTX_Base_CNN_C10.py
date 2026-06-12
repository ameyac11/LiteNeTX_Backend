"""LiteNeTX CIFAR-10 model core."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from collections import OrderedDict


class ResidualBlock(nn.Module):
    """Residual CIFAR-10 block."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class LiteNeTX_Base_CNN_C10(nn.Module):
    """CIFAR-10 CNN model."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.stage1 = self._make_stage(32, 64, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(128, 192, num_blocks=2, stride=2)

        self.dropout2d = nn.Dropout2d(0.1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(192, num_classes)

        self._initialize_weights()

    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

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

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.dropout2d(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


_model = None


def load_cifar_model():
    """Load CIFAR-10 model."""
    global _model
    if _model is not None:
        return _model

    device = torch.device("cpu")
    model = LiteNeTX_Base_CNN_C10(num_classes=10)

    model_path = Path(__file__).parent.parent.parent / "models" / "LiteNeTX_Base_CNN_C10.safetensors"
    state_dict = load_file(str(model_path))

    clean = OrderedDict()
    for k, v in state_dict.items():
        clean[k.replace('module.', '') if k.startswith('module.') else k] = v

    model.load_state_dict(clean)
    model.to(device)
    model.eval()

    _model = model
    return _model


def get_cifar_model():
    if _model is None:
        return load_cifar_model()
    return _model
