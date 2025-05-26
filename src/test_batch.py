import os
import sys
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_tensor
from config import device
from model import create_model


def predict_and_save(model, image_path):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get original dimensions
    orig_height, orig_width = image.shape[:2]

    # Calculate padding to make dimensions divisible by 32
    pad_height = (32 - orig_height % 32) % 32
    pad_width = (32 - orig_width % 32) % 32

    # Pad the image
    if pad_height > 0 or pad_width > 0:
        image = cv2.copyMakeBorder(
            image,
            0, pad_height,
            0, pad_width,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

    # Convert to tensor and move to device
    image = to_tensor(image).unsqueeze(0).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        confidence = torch.sigmoid(output)  # Get confidence scores
        prediction = (confidence > 0.001).float()  # Using lower threshold for better road detection

    # Convert to numpy for visualization and remove padding
    original = image[0].cpu().permute(1, 2, 0).numpy()
    original = (original * 255).astype(np.uint8)
    original = original[:orig_height, :orig_width]  # Remove padding

    mask = prediction[0].squeeze(0).cpu().numpy()
    mask = mask[:orig_height, :orig_width]  # Remove padding

    # Create a region mask to ignore top area
    region_mask = np.ones_like(mask)
    # Define the region to ignore (top portion of the image)
    ignore_height = orig_height // 4  # Top quarter of the image
    # Ignore the entire top portion
    region_mask[:ignore_height, :] = 0  # Set top region to 0

    # Apply region mask to prediction
    mask = mask * region_mask

    # Create overlay
    overlay = original.copy()
    overlay[mask == 1] = [255, 0, 0]  # Red overlay for road

    # Display results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    print(f"Displayed visualization for: {image_path}")


def main():
    # Hardcoded directory path
    image_dir = r"C:\Users\Paras Jagtap\PycharmProjects\PythonProject3333\unseen_images"
    if not os.path.isdir(image_dir):
        print(f"Directory not found: {image_dir}")
        return

    # Load model
    model = create_model()
    model.load_state_dict(torch.load("unet_model.pth", map_location=device))
    model.eval()

    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No images found in directory.")
        return

    print(f"Found {len(image_files)} images. Running predictions...")
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        print(f"Processing: {img_path}")
        predict_and_save(model, img_path)
    print("Batch prediction complete!")


if __name__ == "__main__":
    main() 