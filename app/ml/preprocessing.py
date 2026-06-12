"""LiteNeTX preprocessing helpers."""

import torch
from torchvision import transforms
from PIL import Image


def preprocess_fashion(image: Image.Image) -> torch.Tensor:
    """Prep FashionMNIST input."""
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform(image).unsqueeze(0)


def preprocess_cifar10(image: Image.Image) -> torch.Tensor:
    """Prep CIFAR-10 input."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    return transform(image).unsqueeze(0)


def preprocess_cifar100(image: Image.Image) -> torch.Tensor:
    """Prep CIFAR-100 input."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    return transform(image).unsqueeze(0)
