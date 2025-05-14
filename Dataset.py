from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import numpy as np

class PineappleDataset(Dataset):
    def __init__(self, train=True, train_ratio=0.8, dataset_path="./FULL_VERTICAL_PINEAPPLE/FULL_UNIFIED/*"):
        # Get all images sorted from the specified folder.
        self.all_images = sorted(glob.glob(dataset_path))
        # Calculate the index at which to split the dataset.
        split_index = int(len(self.all_images) * train_ratio)
        # Partition the images based on the 'train' flag.
        if train:
            self.images = self.all_images[:split_index]
        else:
            self.images = self.all_images[split_index:]
        self.resize_shape = (256, 256)

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        channels = 3
        # Resize the image.
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        # Convert to float32 and normalize.
        image = np.array(image).reshape((image.shape[0], image.shape[1], channels)).astype(np.float32) / 255.0
        # Rearrange the dimensions to (channels, height, width).
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, idx):
        image = self.transform_image(self.images[idx])
        sample = {'image': image, 'idx': idx}
        return sample