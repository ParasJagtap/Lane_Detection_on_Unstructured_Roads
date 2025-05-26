import matplotlib.pyplot as plt
from config import Config
from dataset import LaneDataset, get_data_pairs


def visualize_samples(split_file):
    images, xmls = get_data_pairs(split_file)
    dataset = LaneDataset(images[:5], xmls[:5])

    for i in range(5):
        image_tensor, mask_tensor = dataset[i]
        image = image_tensor.numpy().transpose(1, 2, 0)
        mask = mask_tensor.numpy().squeeze()

        plt.figure(figsize=(20, 10))
        plt.subplot(131).imshow(image)
        plt.title("Image")
        plt.subplot(132).imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.subplot(133).imshow(image)
        plt.imshow(mask, alpha=0.3, cmap='jet')
        plt.title("Overlay")
        plt.show()


if __name__ == "__main__":
    Config.show_config()
    visualize_samples(Config.TRAIN_TXT)