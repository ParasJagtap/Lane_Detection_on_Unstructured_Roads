import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from config import device, train_config, checkpoint_path, training_plot_path
from model import combined_loss
import sys


def train_model(model, optimizer, train_loader, val_loader):
    """Train the model with early stopping and learning rate scheduling"""
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=train_config['lr_patience'])
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Lists to track losses
    train_losses = []
    val_losses = []

    for epoch in range(train_config['num_epochs']):
        print(f"\nStarting epoch {epoch + 1}/{train_config['num_epochs']}...")
        sys.stdout.flush()  # Force flush the output

        # Training phase
        model.train()
        running_loss = 0.0
        train_iterator = tqdm(train_loader,
                              desc=f"Epoch {epoch + 1}/{train_config['num_epochs']} [Train]",
                              unit="batch",
                              file=sys.stdout,
                              leave=True)

        for images, masks in train_iterator:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = model(images)
                loss = combined_loss(outputs, masks)

            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            train_iterator.set_postfix(loss=f"{loss.item():.4f}")
            sys.stdout.flush()  # Force flush the output

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_iterator = tqdm(val_loader,
                            desc=f"Epoch {epoch + 1}/{train_config['num_epochs']} [Val]",
                            unit="batch",
                            file=sys.stdout,
                            leave=True)

        with torch.no_grad():
            for images, masks in val_iterator:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss = combined_loss(outputs, masks)
                running_val_loss += val_loss.item() * images.size(0)
                val_iterator.set_postfix(val_loss=f"{val_loss.item():.4f}")
                sys.stdout.flush()  # Force flush the output

        validation_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(validation_loss)

        print(f'\nEpoch {epoch + 1} completed.')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {validation_loss:.4f}')
        sys.stdout.flush()  # Force flush the output


        scheduler.step(validation_loss)

        # Save the best model
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(
                f"Validation loss did not improve. Early stop counter: {early_stop_counter}/{train_config['patience']}")
        sys.stdout.flush()  # Force flush the output

        # Early stopping
        if early_stop_counter >= train_config['patience']:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Plot training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(training_plot_path)
    plt.close()

    return train_losses, val_losses