import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.segmentation import MeanIoU
from config import device, results_path
import os
import cv2


def calculate_metrics(pred, target):
    """Calculate precision, recall, and F1 score for binary segmentation"""
    # Convert to binary predictions
    pred = pred.bool()
    target = target.bool()

    # Calculate true positives, false positives, and false negatives
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()

    # Calculate precision and recall
    precision = tp / (tp + fp + 1e-7)  # Add small epsilon to avoid division by zero
    recall = tp / (tp + fn + 1e-7)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    return precision.item(), recall.item(), f1.item()


def test_model(model, test_loader):
    """Evaluate the model on the test set"""
    model.eval()
    dice_metric = MeanIoU(num_classes=2).to(device)

    # Initialize metrics
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_batches = 0

    with torch.no_grad():
        test_iterator = tqdm(test_loader, desc="Testing", unit="batch")
        for batch_idx, (images, masks) in enumerate(test_iterator):
            images, masks = images.to(device), masks.to(device)

            # Debug: print detailed information about masks and predictions
            print(f"\nBatch {batch_idx + 1}:")
            print(f"Mask stats before processing:")
            print(f"  Shape: {masks.shape}")
            print(f"  Min: {masks.min().item():.4f}")
            print(f"  Max: {masks.max().item():.4f}")
            print(f"  Mean: {masks.mean().item():.4f}")
            print(f"  Unique values: {torch.unique(masks).tolist()}")

            # Normalize masks if max value > 1
            if masks.max() > 1:
                masks = masks / 255.0
                print(f"Mask stats after normalization:")
                print(f"  Min: {masks.min().item():.4f}")
                print(f"  Max: {masks.max().item():.4f}")
                print(f"  Mean: {masks.mean().item():.4f}")
                print(f"  Unique values: {torch.unique(masks).tolist()}")

            outputs = model(images)

            # Debug: print information about raw model outputs
            print(f"Raw output stats:")
            print(f"  Shape: {outputs.shape}")
            print(f"  Min: {outputs.min().item():.4f}")
            print(f"  Max: {outputs.max().item():.4f}")
            print(f"  Mean: {outputs.mean().item():.4f}")

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)
            print(f"Probability stats:")
            print(f"  Min: {probs.min().item():.4f}")
            print(f"  Max: {probs.max().item():.4f}")
            print(f"  Mean: {probs.mean().item():.4f}")

            # Convert to binary predictions with a lower threshold
            outputs = (probs > 0.001).bool()  # Lower threshold for better road detection
            masks = (masks > 0.001).bool()  # Match threshold for masks

            print(f"Binary prediction stats:")
            print(f"  Prediction positive pixels: {outputs.sum().item()}")
            print(f"  Mask positive pixels: {masks.sum().item()}")
            print(f"  Prediction positive %: {(outputs.sum().item() / outputs.numel() * 100):.2f}%")
            print(f"  Mask positive %: {(masks.sum().item() / masks.numel() * 100):.2f}%")

            # Update IoU metric
            dice_metric.update(outputs, masks)

            # Calculate precision, recall, and F1 for this batch
            precision, recall, f1 = calculate_metrics(outputs, masks)

            print(f"Batch metrics:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")

            # Accumulate metrics
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            num_batches += 1

    # Compute average metrics
    avg_iou = dice_metric.compute()
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1 = total_f1 / num_batches

    # Print all metrics
    print("\nFinal Test Metrics:")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

    return avg_iou, avg_precision, avg_recall, avg_f1


def visualize_segmentation_results(model, test_loader, device, num_images=5):
    """Visualize segmentation results for random test samples"""
    model.eval()
    test_samples = random.sample(list(test_loader), num_images)

    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))  # 3 columns: original, confidence, prediction
    if num_images == 1:
        axes = axes.reshape(1, -1)  # Ensure axes is 2D even for a single image

    for idx, (images, masks) in enumerate(test_samples):
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Get probabilities
            predictions = (outputs > 0.001).float()  # Binary predictions with lower threshold

        # Get the first image in the batch
        original_image = images[0].cpu().permute(1, 2, 0).numpy()
        original_image = (original_image * 255).astype(np.uint8)
        confidence = outputs[0].squeeze(0).cpu().numpy()

        # Create prediction overlay with blending
        overlay = original_image.copy()
        red_mask = np.zeros_like(overlay)
        red_mask[predictions[0].squeeze(0).cpu().numpy() == 1] = [255, 0, 0]
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1, red_mask, alpha, 0)

        # Original image
        axes[idx, 0].imshow(original_image)
        axes[idx, 0].set_title("Original Image")
        axes[idx, 0].axis("off")

        # Confidence map
        conf_map = axes[idx, 1].imshow(confidence, cmap="viridis")
        axes[idx, 1].set_title("Confidence Map")
        axes[idx, 1].axis("off")
        plt.colorbar(conf_map, ax=axes[idx, 1])

        # Prediction overlay
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Prediction Overlay")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()
    print("Visualization complete")