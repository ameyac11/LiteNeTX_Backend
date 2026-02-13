"""Prediction service for all three LiteNeTX models."""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict

from .LiteNeTX_Base_CNN_FashionMNIST import get_fashion_model
from .LiteNeTX_Base_CNN_C10 import get_cifar_model
from .LiteNeTX_Base_CNN_C100 import get_cifar100_model
from .preprocessing import preprocess_fashion, preprocess_cifar10, preprocess_cifar100
from .labels import FASHION_LABELS, CIFAR10_LABELS, CIFAR100_LABELS


def predict_fashion(image: Image.Image) -> Dict:
    """Run inference on LiteNeTX-1 FashionMNIST."""
    model = get_fashion_model()
    tensor = preprocess_fashion(image).to("cpu")

    with torch.inference_mode():
        output = model(tensor)
        probs = F.softmax(output[0], dim=0)

    top_probs, top_idx = torch.topk(probs, k=3)
    top3 = [{"label": FASHION_LABELS[idx], "confidence": float(prob)} for prob, idx in zip(top_probs, top_idx)]

    return {"model": "LiteNeTX_Base_CNN_FashionMNIST", "top1": top3[0], "top3": top3}


def predict_cifar10(image: Image.Image) -> Dict:
    """Run inference on LiteNeTX-2 CIFAR-10."""
    model = get_cifar_model()
    tensor = preprocess_cifar10(image).to("cpu")

    with torch.inference_mode():
        output = model(tensor)
        probs = F.softmax(output[0], dim=0)

    top_probs, top_idx = torch.topk(probs, k=3)
    top3 = [{"label": CIFAR10_LABELS[idx], "confidence": float(prob)} for prob, idx in zip(top_probs, top_idx)]

    return {"model": "LiteNeTX_Base_CNN_C10", "top1": top3[0], "top3": top3}


def predict_cifar100(image: Image.Image) -> Dict:
    """Run inference on LiteNeTX-3 CIFAR-100."""
    model = get_cifar100_model()
    tensor = preprocess_cifar100(image).to("cpu")

    with torch.inference_mode():
        output = model(tensor)
        probs = F.softmax(output[0], dim=0)

    top_probs, top_idx = torch.topk(probs, k=5)
    top5 = [{"label": CIFAR100_LABELS[idx], "confidence": float(prob)} for prob, idx in zip(top_probs, top_idx)]

    return {"model": "LiteNeTX_Base_CNN_C100", "top1": top5[0], "top3": top5[:3], "top5": top5}
