import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset paths
image_folder = r'C:\Users\Paras Jagtap\PycharmProjects\PythonProject3333\data\image_archive'
mask_folder = r'C:\Users\Paras Jagtap\PycharmProjects\PythonProject3333\data\mask_archive'


# Model configuration
model_config = {
    'encoder_name': 'efficientnet-b3',
    'encoder_weights': 'imagenet',
    'in_channels': 3,
    'classes': 1,
    'activation': None  # No activation in the model, we'll use sigmoid in loss
}

# Training configuration
train_config = {
    'num_epochs': 50,  # More epochs
    'batch_size': 8,
    'learning_rate': 0.001,  # Lower learning rate
    'patience': 10,  # More patience for early stopping
    'lr_patience': 3,  # More patience for learning rate reduction
    'num_images': 6993  # Using all available images
}

# Paths
checkpoint_path = 'unet_model.pth'
training_plot_path = 'training_plot.png'
results_path = 'segmentation_results.png'