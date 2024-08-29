import torch
import torch.nn.functional as F
import random
import numpy as np


def augment_data(text, audio, vision):
    """
    Apply data augmentation to multimodal data.

    Args:
    text (torch.Tensor): Text data of shape (batch_size, seq_len, feature_dim)
    audio (torch.Tensor): Audio data of shape (batch_size, seq_len, feature_dim)
    vision (torch.Tensor): Vision data of shape (batch_size, seq_len, feature_dim)

    Returns:
    tuple: Augmented text, audio, and vision data
    """
    augmented_text = augment_text(text)
    augmented_audio = augment_audio(audio)
    augmented_vision = augment_vision(vision)

    return augmented_text, augmented_audio, augmented_vision


def augment_text(text):
    """
    Augment text data using techniques like word dropout and shuffling.
    """
    batch_size, seq_len, feature_dim = text.shape

    # Word dropout
    dropout_mask = torch.bernoulli(torch.full((batch_size, seq_len, 1), 0.9)).to(text.device)
    augmented_text = text * dropout_mask

    # Word shuffling (within each sequence)
    for i in range(batch_size):
        perm = torch.randperm(seq_len)
        augmented_text[i] = augmented_text[i][perm]

    return augmented_text


def augment_audio(audio):
    """
    Augment audio data using techniques like adding noise and time stretching.
    """
    batch_size, seq_len, feature_dim = audio.shape

    # Add Gaussian noise
    noise = torch.randn_like(audio) * 0.01
    augmented_audio = audio + noise

    # Time stretching (simulate by interpolation)
    stretch_factor = random.uniform(0.8, 1.2)
    new_len = int(seq_len * stretch_factor)
    augmented_audio = F.interpolate(augmented_audio.transpose(1, 2), size=new_len, mode='linear',
                                    align_corners=False).transpose(1, 2)

    # Pad or crop to original length
    if new_len > seq_len:
        augmented_audio = augmented_audio[:, :seq_len, :]
    else:
        augmented_audio = F.pad(augmented_audio, (0, 0, 0, seq_len - new_len))

    return augmented_audio


def augment_vision(vision):
    """
    Augment vision data using techniques like random cropping and flipping.
    """
    batch_size, seq_len, feature_dim = vision.shape

    # Random cropping (simulate by zeroing out parts of the sequence)
    crop_length = int(seq_len * random.uniform(0.8, 1.0))
    start = random.randint(0, seq_len - crop_length)
    mask = torch.zeros((batch_size, seq_len, 1), device=vision.device)
    mask[:, start:start + crop_length] = 1
    augmented_vision = vision * mask

    # Horizontal flipping (assuming the feature dimension represents spatial information)
    flip_mask = torch.bernoulli(torch.full((batch_size, 1, 1), 0.5)).to(vision.device)
    flipped_vision = torch.flip(vision, [2])
    augmented_vision = flip_mask * flipped_vision + (1 - flip_mask) * augmented_vision

    return augmented_vision


# Additional utility functions for more advanced augmentations

def mixup(x1, x2, y1, y2, alpha=0.2):
    """
    Perform mixup augmentation on the input data and labels.
    """
    lambda_ = np.random.beta(alpha, alpha)
    mixed_x = lambda_ * x1 + (1 - lambda_) * x2
    mixed_y = lambda_ * y1 + (1 - lambda_) * y2
    return mixed_x, mixed_y


def cutmix(x1, x2, y1, y2):
    """
    Perform cutmix augmentation on the input data and labels.
    """
    batch_size, seq_len, feature_dim = x1.shape
    lambda_ = np.random.beta(1, 1)
    cut_length = int(seq_len * lambda_)
    cut_start = random.randint(0, seq_len - cut_length)

    mixed_x = x1.clone()
    mixed_x[:, cut_start:cut_start + cut_length] = x2[:, cut_start:cut_start + cut_length]

    mixed_y = (1 - lambda_) * y1 + lambda_ * y2
    return mixed_x, mixed_y