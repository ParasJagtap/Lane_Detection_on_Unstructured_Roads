Lane Detection on Unstructured Roads
A PyTorch-based pipeline for pixel-wise drivable-area segmentation on unmarked, poorly structured roads. This project reframes lane detection as a binary segmentation task, using a U-Net decoder with an EfficientNet-B3 encoder, mixed-precision training, and a composite loss (BCEWithLogits + Dice) for robust performance across dirt, gravel, and semi-urban scenes.

Features
U-Net + EfficientNet-B3 architecture for multi-scale feature extraction
Mixed-precision RAdam optimizer for fast, stable convergence
Composite loss: 0.5·BCEWithLogits + 0.5·Dice to balance pixel accuracy and region overlap
Albumentations-powered data augmentations (flips, brightness/contrast)
Deterministic 80/10/10 train/val/test split with fixed seed
Post-processing: thresholding, morphological opening, connected-component filtering
Quantitative metrics (Precision, Recall, F1, IoU) and qualitative visualizations
Single-image GUI inference via predict_single.py

Getting Started

Usage
1. Training
Configure hyperparameters and paths in config.py, then run:

bash
python src/train_script.py
– Trains for 50 epochs with early stopping (patience = 10).
– Saves best model to unet_model.pth and training loss plot to training_plot.png.

2. Evaluation
After training or loading a checkpoint, evaluate on the test set:

bash
python src/test_script.py
– Prints batch-wise and average Precision, Recall, F1, IoU.
– Displays side-by-side triplets (original, confidence map, overlay).

3. Single-Image Prediction
Run an interactive GUI to select and segment any image:

bash
python src/predict_single.py
– Applies padding, thresholding (default 0.001), and ignores top-quarter region.
– Shows original, binary mask, and red overlay panels.

Project Structure
text
Lane_Detection_on_Unstructured_Roads/
├── .idea/                            # PyCharm config
├── src/
│   ├── config.py                    # Paths & hyperparameters
│   ├── dataset.py                   # RoadSegmentationDataset class
│   ├── model.py                     # U-Net/EfficientNet-B3 definition
│   ├── train_script.py              # Training entry point
│   ├── test_script.py               # Evaluation & visualization
│   └── predict_single.py            # GUI-based single-image inference
├── requirements.txt                 # Python dependencies
└── README.md                        # Project overview
