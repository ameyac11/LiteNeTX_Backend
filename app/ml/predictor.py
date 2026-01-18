"""Prediction service for both models."""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List

from .fashion_model import get_fashion_model
from .cifar_model import get_cifar_model
from .preprocessing import preprocess_fashion, preprocess_cifar
from .labels import FASHION_LABELS, CIFAR_LABELS


def predict_fashion(image: Image.Image) -> Dict:
    """
    Predict FashionMNIST class for the given image.
    
    Args:
        image: PIL Image to classify
        
    Returns:
        Dict with model name, top1 prediction, and top3 predictions
    """
    model = get_fashion_model()
    device = torch.device("cpu")
    
    # Preprocess
    tensor = preprocess_fashion(image).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output[0], dim=0)
    
    # Get top 3 predictions
    probs, indices = torch.topk(probabilities, k=3)
    
    top3 = [
        {
            "label": FASHION_LABELS[idx],
            "confidence": float(prob)
        }
        for prob, idx in zip(probs, indices)
    ]
    
    return {
        "model": "LiteNet-Fashion",
        "top1": top3[0],
        "top3": top3
    }


def predict_cifar(image: Image.Image) -> Dict:
    """
    Predict CIFAR-10 class for the given image.
    
    Args:
        image: PIL Image to classify
        
    Returns:
        Dict with model name, top1 prediction, and top3 predictions
    """
    model = get_cifar_model()
    device = torch.device("cpu")
    
    # Preprocess
    tensor = preprocess_cifar(image).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output[0], dim=0)
    
    # Get top 3 predictions
    probs, indices = torch.topk(probabilities, k=3)
    
    top3 = [
        {
            "label": CIFAR_LABELS[idx],
            "confidence": float(prob)
        }
        for prob, idx in zip(probs, indices)
    ]
    
    return {
        "model": "LiteNet-CIFAR",
        "top1": top3[0],
        "top3": top3
    }
