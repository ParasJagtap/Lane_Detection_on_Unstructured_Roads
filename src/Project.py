import os
import cv2
import json
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


# Configuration
class Config:
    IMAGE_DIR = r"C:\Users\Paras Jagtap\PycharmProjects\PythonProject\data\idd20kII\leftImg8bit\train"
    LABEL_DIR = r"C:\Users\Paras Jagtap\PycharmProjects\PythonProject\data\idd20kII\gtFine\train"
    BATCH_SIZE = 8
    NUM_EPOCHS = 25
    LR = 0.0001
    IMG_SIZE = (512, 512)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LaneDataset(Dataset):
    def __init__(self, image_paths, label_paths):
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, Config.IMG_SIZE) / 255.0

        # Load mask from JSON
        with open(self.label_paths[idx], 'r') as f:
            label_data = json.load(f)

        mask = np.zeros(Config.IMG_SIZE, dtype=np.float32)
        lane_keywords = {'lane', 'road', 'drivable', 'carriageway'}

        for obj in label_data["objects"]:
            if any(kw in obj["label"].lower() for kw in lane_keywords):
                try:
                    # Convert polygon to proper OpenCV format
                    polygon = np.array(obj["polygon"], np.int32)
                    if len(polygon) < 3:  # Skip invalid polygons
                        continue

                    # Reshape to (N, 1, 2) as required by fillPoly
                    polygon = polygon.reshape((-1, 1, 2))

                    # Handle potential coordinate overflow
                    polygon[:, :, 0] = np.clip(polygon[:, :, 0], 0, Config.IMG_SIZE[0] - 1)
                    polygon[:, :, 1] = np.clip(polygon[:, :, 1], 0, Config.IMG_SIZE[1] - 1)

                    cv2.fillPoly(mask, [polygon], 1.0)
                except Exception as e:
                    print(f"Error processing polygon: {e}")
                    continue

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask


def create_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(Config.DEVICE)


def train_model():
    # Dataset setup (same as verifier)
    paired_files = []
    for root, _, files in os.walk(Config.IMAGE_DIR):
        for file in files:
            if file.endswith("_leftImg8bit.jpg"):
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(img_path, Config.IMAGE_DIR)
                label_path = os.path.join(Config.LABEL_DIR, rel_path) \
                    .replace("_leftImg8bit.jpg", "_gtFine_polygons.json")

                if os.path.exists(label_path):
                    paired_files.append((img_path, label_path))

    image_paths, label_paths = zip(*paired_files)
    dataset = LaneDataset(image_paths, label_paths)
    train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)

    # Model setup
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)

    # Custom loss wrapper to work with SMP
    def bce_loss(y_pred, y_true):
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(Config.DEVICE))
        return loss_fn(y_pred, y_true)

    bce_loss.__name__ = "bce_loss"  # Add name attribute for logging

    # Manual training loop
    best_iou = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        iou_scores = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        for images, masks in progress_bar:
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)

            # Forward pass
            outputs = model(images)
            loss = bce_loss(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            epoch_loss += loss.item()

            # Convert to binary masks
            preds = torch.sigmoid(outputs) > 0.5
            targets = masks > 0.5  # Convert float masks to boolean

            # Calculate IoU
            intersection = (preds & targets).float().sum()
            union = (preds | targets).float().sum()
            iou = intersection / (union + 1e-8)  # Add epsilon to avoid division by zero

            iou_scores.append(iou.item())

            progress_bar.set_postfix({
                "loss": loss.item(),
                "iou": np.mean(iou_scores)
            })

        # Save best model
        mean_iou = np.mean(iou_scores)
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best IoU: {best_iou:.4f}")


if __name__ == "__main__":
    train_model()