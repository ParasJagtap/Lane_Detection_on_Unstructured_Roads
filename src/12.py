# dataset_verifier.py
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# CONFIGURATION
# =================================================================
IMAGE_DIR = r"C:\Users\Paras Jagtap\PycharmProjects\PythonProject\data\idd20kII\leftImg8bit\train"
LABEL_DIR = r"C:\Users\Paras Jagtap\PycharmProjects\PythonProject\data\idd20kII\gtFine\train"


# =================================================================

def verify_dataset():
    print("\n🔍 Starting Dataset Verification...")

    paired_files = []
    missing_labels = []

    # Walk through image directory structure
    for root, dirs, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.endswith("_leftImg8bit.jpg"):
                # Construct image path
                img_path = os.path.join(root, file)

                # Convert to label path
                rel_path = os.path.relpath(img_path, IMAGE_DIR)
                label_path = os.path.join(LABEL_DIR, rel_path) \
                    .replace("_leftImg8bit.jpg", "_gtFine_polygons.json")

                if os.path.exists(label_path):
                    paired_files.append((img_path, label_path))
                else:
                    missing_labels.append(label_path)

    if not paired_files:
        print("❌ No valid pairs found")
        if missing_labels:
            print(f"First missing label: {missing_labels[0]}")
        return

    print(f"\n✅ Found {len(paired_files)} valid pairs")
    print(f"❌ Missing {len(missing_labels)} labels")

    # Verify sample pair
    img_path, lbl_path = paired_files[0]
    print("\n🔎 Sample Pair:")
    print(f"Image: {os.path.relpath(img_path, IMAGE_DIR)}")
    print(f"Label: {os.path.relpath(lbl_path, LABEL_DIR)}")

    # Label content check
    try:
        with open(lbl_path, 'r') as f:
            data = json.load(f)

        print("\n📄 Label Content:")
        print(f"Objects: {len(data['objects'])}")
        lane_keywords = {'lane', 'road', 'drivable', 'carriageway'}
        lane_count = sum(1 for obj in data["objects"] if any(kw in obj["label"].lower() for kw in lane_keywords))
        print(f"Lane-related objects: {lane_count}")

    except Exception as e:
        print(f"❌ Label error: {str(e)}")


if __name__ == "__main__":
    verify_dataset()