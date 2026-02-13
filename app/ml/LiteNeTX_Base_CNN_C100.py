"""LiteNeTX-3 CIFAR-100 model definition and loader."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from collections import OrderedDict


def _drop_path(x, drop_prob, training):
    """Stochastic depth — randomly drops residual branches during training."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = (torch.rand(shape, dtype=x.dtype, device=x.device) >= drop_prob).float()
    return x.div(keep_prob) * mask


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, ch, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class PreActBottleneckSE(nn.Module):
    """Pre-activation bottleneck with SE attention and stochastic depth."""
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1, drop=0.0, drop_path_rate=0.0):
        super().__init__()
        out_ch = mid_ch * self.expansion

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(mid_ch)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)

        self.se = SEBlock(out_ch)
        self.drop_path_rate = drop_path_rate

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        out = F.relu(self.bn3(out), inplace=True)
        out = self.drop(out)
        out = self.conv3(out)
        out = self.se(out)

        if self.training and self.drop_path_rate > 0:
            out = _drop_path(out, self.drop_path_rate, True)

        return out + self.shortcut(x)


class LiteNeTX_Base_CNN_C100(nn.Module):
    """Pre-Act Bottleneck SE-ResNet for CIFAR-100 — 14.6M params, 23 blocks."""

    def __init__(self, num_classes=100, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        block_counts = [3, 12, 8]
        max_drop_path = 0.2
        total_blocks = sum(block_counts)
        dpr = [x.item() for x in torch.linspace(0, max_drop_path, total_blocks)]

        # Stem: 3 -> 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Stages with bottleneck expansion=4
        cur = 0
        self.stage1 = self._make_stage(64, 64, block_counts[0], stride=1, drop=0.05,
                                        dprs=dpr[cur:cur+block_counts[0]])
        cur += block_counts[0]
        self.stage2 = self._make_stage(256, 128, block_counts[1], stride=2, drop=0.10,
                                        dprs=dpr[cur:cur+block_counts[1]])
        cur += block_counts[1]
        self.stage3 = self._make_stage(512, 256, block_counts[2], stride=2, drop=0.15,
                                        dprs=dpr[cur:cur+block_counts[2]])

        # Head
        self.final_bn = nn.BatchNorm2d(1024)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, num_classes)

        self._init_weights()

    def _make_stage(self, in_ch, mid_ch, blocks, stride, drop, dprs):
        out_ch = mid_ch * PreActBottleneckSE.expansion
        layers = [PreActBottleneckSE(in_ch, mid_ch, stride, drop, dprs[0])]
        for i in range(1, blocks):
            layers.append(PreActBottleneckSE(out_ch, mid_ch, 1, drop, dprs[i]))
        return nn.Sequential(*layers)

    def _init_weights(self):
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

        x = F.relu(self.final_bn(x), inplace=True)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


_model = None


def load_cifar100_model():
    """Load LiteNeTX-3 CIFAR-100 model from safetensors."""
    global _model
    if _model is not None:
        return _model

    device = torch.device("cpu")
    model = LiteNeTX_Base_CNN_C100(num_classes=100, use_checkpoint=False)

    model_path = Path(__file__).parent.parent.parent / "models" / "LiteNeTX_Base_CNN_C100.safetensors"
    state_dict = load_file(str(model_path))

    clean = OrderedDict()
    for k, v in state_dict.items():
        clean[k.replace('module.', '') if k.startswith('module.') else k] = v

    model.load_state_dict(clean)
    model.to(device)
    model.eval()

    _model = model
    return _model


def get_cifar100_model():
    if _model is None:
        return load_cifar100_model()
    return _model
