print("Starting script...")
import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_tensor
from config import device
from model import create_model
from tkinter import Tk, filedialog
import matplotlib.colors as mcolors

print("Imports successful.")


def select_image():
    print("Opening file dialog...")
    # Create and hide the root window
    root = Tk()
    root.withdraw()  # Hide the main window

    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )
    print(f"File dialog returned: {file_path}")

    # Destroy the root window
    root.destroy()
    return file_path


def predict_and_show(model, image_path, confidence_threshold=0.001, save_output=False):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get original dimensions
    orig_height, orig_width = image.shape[:2]
    print(f"Image dimensions: {orig_width}x{orig_height}")

    # Calculate padding to make dimensions divisible by 32
    pad_height = (32 - orig_height % 32) % 32
    pad_width = (32 - orig_width % 32) % 32
    print(f"Padding added: {pad_width}px width, {pad_height}px height")

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
        confidence = torch.sigmoid(output)
        prediction = (confidence > confidence_threshold).float()

        # Print confidence statistics
        conf_np = confidence[0].squeeze(0).cpu().numpy()
        print(f"\nConfidence statistics:")
        print(f"  Min: {conf_np.min():.4f}")
        print(f"  Max: {conf_np.max():.4f}")
        print(f"  Mean: {conf_np.mean():.4f}")
        print(f"  Median: {np.median(conf_np):.4f}")
        print(f"  Threshold: {confidence_threshold:.4f}")

        # Calculate percentiles
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"  {p}th percentile: {np.percentile(conf_np, p):.4f}")

        # Calculate percentage of pixels in different confidence ranges
        ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        for low, high in ranges:
            pct = np.mean((conf_np >= low) & (conf_np < high)) * 100
            print(f"  {low:.1f}-{high:.1f} confidence: {pct:.2f}%")

        print(f"  Pixels above threshold: {(prediction[0].sum().item() / prediction[0].numel() * 100):.2f}%")

    # Convert to numpy for visualization and remove padding
    original = image[0].cpu().permute(1, 2, 0).numpy()
    original = (original * 255).astype(np.uint8)
    original = original[:orig_height, :orig_width]

    mask = prediction[0].squeeze(0).cpu().numpy()
    mask = mask[:orig_height, :orig_width]

    # Get confidence map (without padding)
    conf_map = conf_np[:orig_height, :orig_width]

    # Create a region mask to ignore top area
    region_mask = np.ones_like(mask)
    ignore_height = orig_height // 4  # Back to ignoring top quarter
    region_mask[:ignore_height, :] = 0

    # Apply region mask to prediction
    mask = mask * region_mask
    print(f"Final road pixels after masking: {(mask.sum() / mask.size * 100):.2f}%")

    # Create overlay
    overlay = original.copy()
    overlay[mask == 1] = [255, 0, 0]  # Red overlay for road

    # Create figure with 3 subplots (removed confidence heatmap)
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")

    # Binary prediction
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Binary Prediction")
    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()

    # Show the plot instead of saving
    plt.show()
    plt.close()  # Close the figure to free up memory


def main():
    # Load model
    print("Loading model...")
    model = create_model()
    model.load_state_dict(torch.load("unet_model.pth", map_location=device))
    model.eval()
    print("Model loaded successfully!")

    print("Starting main prediction loop.")
    while True:
        print("\nPlease select an image file...")
        image_path = select_image()

        # If user cancels the file dialog, image_path will be empty
        if not image_path:
            print("No file selected. Exiting program...")
            break

        print(f"\nProcessing image: {image_path}")

        # Remove save output question since we're showing instead of saving
        predict_and_show(model, image_path, confidence_threshold=0.001, save_output=False)

        # Ask if user wants to process another image
        while True:
            choice = input("\nWould you like to process another image? (y/n): ").lower()
            if choice in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'")

        if choice == 'n':
            print("Exiting program...")
            break


if __name__ == "__main__":
    main() 