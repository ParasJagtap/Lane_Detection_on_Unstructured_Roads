import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from config import device, model_config, train_config


def create_model():
    """Create and initialize the U-Net model"""
    model = smp.Unet(
        encoder_name=model_config['encoder_name'],
        encoder_weights=model_config['encoder_weights'],
        in_channels=model_config['in_channels'],
        classes=model_config['classes'],
        activation=model_config['activation']
    )
    model = model.to(device)
    return model


def create_optimizer(model):
    """Create optimizer for the model"""
    return torch.optim.RAdam(model.parameters(), lr=train_config['learning_rate'])


def combined_loss(output, target):
    """Combined BCE and Dice loss with proper sigmoid handling"""
    output_probs = torch.sigmoid(output)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = smp.losses.DiceLoss(mode='binary')

    bce = criterion_bce(output, target)  #e raw logits for BCE
    dice = criterion_dice(output_probs, target)  # Use probabilities for Dice
    return 0.5 * bce + 0.5 * dice  # Equal weighting