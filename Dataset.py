import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
class PineappleDataset(Dataset):
    def __init__(
        self,
        train: bool = True,
        train_ratio: float = 0.8,
        dataset_path: str = "./FULL_VERTICAL_PINEAPPLE/FULL_UNIFIED/*",
        transform=None,            # ← new
    ):
        self.all_images = sorted(glob.glob(dataset_path))
        split_index = int(len(self.all_images) * train_ratio)
        self.images = (
            self.all_images[:split_index] if train
            else self.all_images[split_index:]
        )
        self.resize_shape = (256, 256)
        self.transform = transform  # ← store it

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. load + resize + to numpy H×W×C float32 [0–1]
        img = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resize_shape[1], self.resize_shape[0]))
        img = img.astype(np.float32) / 255.0

        # 2. if the user passed a torchvision transform, apply it
        if self.transform is not None:
            # torchvision transforms expect H×W×C uint8 or PIL.Image,
            # so convert back…
            img = (img * 255).astype(np.uint8)
            # H×W×C → PIL
            
            img = Image.fromarray(img)
            img = self.transform(img)   # now you get a torch.Tensor C×H×W
        else:
            # just H×W×C→C×H×W tensor
            
            img = torch.from_numpy(img.transpose(2, 0, 1))

        return {"image": img, "idx": idx}
