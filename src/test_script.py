import os

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Disable update check

import torch
from dataset import get_data_loaders
from model import create_model
from test import test_model, visualize_segmentation_results
from config import device, checkpoint_path, results_path


def main():
    print("Starting test script...")

    # Get data loaders
    print("Getting data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"Test loader size: {len(test_loader.dataset)} images")

    # Create model and load best weights
    print("Creating model...")
    model = create_model()
    print(f"Loading model from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    print("Model loaded successfully")

    # Test the model
    print("\nTesting model...")
    test_model(model, test_loader)

    # Visualize results
    print("\nVisualizing results...")
    visualize_segmentation_results(model, test_loader, device)
    print("Results saved to:", results_path)


if __name__ == "__main__":
    main()