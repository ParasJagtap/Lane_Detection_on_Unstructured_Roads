import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.transforms.functional import to_tensor
import albumentations as A
from config import image_folder, mask_folder, train_config
from PIL import Image


class RoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

        # Verify that image and mask files match after removing prefixes
        for img_file, mask_file in zip(self.image_files, self.mask_files):
            img_num = img_file.replace('Image_', '')
            mask_num = mask_file.replace('Mask_', '')
            if img_num != mask_num:
                raise ValueError(
                    f"Image and mask files do not match after removing prefixes: {img_file} vs {mask_file}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)

        # Apply transformations if specified
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to tensors and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        # Convert mask to tensor (already binary 0/1)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


def get_data_loaders():
    # Define augmentations
    train_augmentations = A.Compose([
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.3),
    ])

    # Initialize full dataset
    full_dataset = RoadSegmentationDataset(
        image_folder,
        mask_folder,
        transform=None
    )


    train_size = int(0.75 * len(full_dataset))  # 75% training
    val_size = int(0.1 * len(full_dataset))  # 10% validation
    test_size = len(full_dataset) - train_size - val_size  # 15% testing

    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        range(len(full_dataset)),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create datasets
    train_dataset = RoadSegmentationDataset(
        image_folder,
        mask_folder,
        transform=train_augmentations
    )
    train_dataset = Subset(train_dataset, train_indices)

    val_dataset = RoadSegmentationDataset(
        image_folder,
        mask_folder,
        transform=None
    )
    val_dataset = Subset(val_dataset, val_indices)

    test_dataset = RoadSegmentationDataset(
        image_folder,
        mask_folder,
        transform=None
    )
    test_dataset = Subset(test_dataset, test_indices)

    # Create data loaders with optimized settings
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=4,  # Use multiple workers for data loading
        pin_memory=True  # Keep data in pinned memory for faster transfer to GPU
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader