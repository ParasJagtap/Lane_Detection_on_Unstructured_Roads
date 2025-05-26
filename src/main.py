import torch
from dataset import get_data_loaders
from model import create_model, create_optimizer
from train import train_model
from test import test_model
from config import device, checkpoint_path


def main():
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders()

    # Create model and optimizer
    model = create_model()
    optimizer = create_optimizer(model)

    # Train the model
    print("Starting training...")
    train_model(model, optimizer, train_loader, val_loader)

    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("Evaluating on test set...")
    test_model(model, test_loader)


if __name__ == "__main__":
    main()