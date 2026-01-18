"""Image preprocessing utilities for both models."""

import torch
from torchvision import transforms
from PIL import Image


def preprocess_fashion(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for FashionMNIST model.
    
    Args:
        image: PIL Image to preprocess
        
    Returns:
        Preprocessed tensor of shape (1, 1, 28, 28)
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def preprocess_cifar(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for CIFAR-10 model.
    
    Args:
        image: PIL Image to preprocess
        
    Returns:
        Preprocessed tensor of shape (1, 3, 32, 32)
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor
