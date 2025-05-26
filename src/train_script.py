from dataset import get_data_loaders
from model import create_model, create_optimizer
from train import train_model
from config import device, checkpoint_path


def main():
    print("Starting training script...")
    print("Getting data loaders...")

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders()

    print("Creating model and optimizer...")
    # Create model and optimizer
    model = create_model()
    optimizer = create_optimizer(model)

    # Train the model
    print("Starting training...")
    train_model(model, optimizer, train_loader, val_loader)
    print("Training completed! Best model saved to:", checkpoint_path)


if __name__ == "__main__":
    main()