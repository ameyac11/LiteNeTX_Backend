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


class PreActBasicSE(nn.Module):
    """Pre-activation basic residual block with SE attention and stochastic depth.
    BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> SE -> DropPath + Shortcut

    Wider basic blocks provide better feature learning than narrow bottleneck blocks
    for CIFAR-scale tasks, while being faster on CPU (fewer sequential ops).
    """
    def __init__(self, in_ch, out_ch, stride=1, drop_path_rate=0.0, se_reduction=16):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        self.se = SEBlock(out_ch, se_reduction)
        self.drop_path_rate = drop_path_rate

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(x)

        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        out = self.se(out)

        if self.training and self.drop_path_rate > 0:
            out = _drop_path(out, self.drop_path_rate, True)

        return out + shortcut


class LiteNeTX_Base_CNN_C100(nn.Module):
    """
    LiteNeTX CIFAR-100: ~18.9M params, PreAct Wide SE-ResNet.
    Wider-than-deep design with SE attention and stochastic depth.
    Channels: 64 -> 128 -> 256 -> 512
    Blocks: 4 + 4 + 3 = 11 basic blocks (23 conv layers total)
    Optimized for efficient CPU inference and high accuracy.
    """
    def __init__(self, num_classes=100):
        super().__init__()

        block_counts = [4, 4, 3]
        channels = [128, 256, 512]
        max_drop_path = 0.15
        total_blocks = sum(block_counts)
        dpr = [x.item() for x in torch.linspace(0, max_drop_path, total_blocks)]

        # stem: 3 -> 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # stages: wide basic blocks with SE attention
        cur = 0
        self.stage1 = self._make_stage(64, channels[0], block_counts[0], stride=1,
                                        dprs=dpr[cur:cur+block_counts[0]])     # 32x32, 128ch
        cur += block_counts[0]
        self.stage2 = self._make_stage(channels[0], channels[1], block_counts[1], stride=2,
                                        dprs=dpr[cur:cur+block_counts[1]])     # 16x16, 256ch
        cur += block_counts[1]
        self.stage3 = self._make_stage(channels[1], channels[2], block_counts[2], stride=2,
                                        dprs=dpr[cur:cur+block_counts[2]])     # 8x8, 512ch

        # head
        self.final_bn = nn.BatchNorm2d(channels[2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(channels[2], num_classes)

        self._init_weights()

    def _make_stage(self, in_ch, out_ch, blocks, stride, dprs):
        layers = [PreActBasicSE(in_ch, out_ch, stride, dprs[0])]
        for i in range(1, blocks):
            layers.append(PreActBasicSE(out_ch, out_ch, 1, dprs[i]))
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
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 32x32, 64ch
        x = self.stage1(x)   # 32x32, 128ch
        x = self.stage2(x)   # 16x16, 256ch
        x = self.stage3(x)   # 8x8, 512ch

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
    model = LiteNeTX_Base_CNN_C100(num_classes=100)

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
