from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import cv2
from transforms import train_transform, val_transform

def read_x_ray(path):
    # Read grayscale image
    x_ray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x_ray = x_ray.astype(np.uint8)  # PIL expects uint8 for images

    x_ray_3ch = cv2.merge([x_ray, x_ray, x_ray])  # HWC
    return x_ray_3ch


class ChestXrayDataset(Dataset):
    def __init__(self, data, train=True, cache=False):
        self.data = data.reset_index(drop=True)
        self.train = train
        self.cache = cache

        self.cached_images = {}
        if self.cache:
            print("[INFO] Caching images in memory...")
            for i in range(len(self.data)):
                path = self.data["Path"].iloc[i]
                self.cached_images[path] = read_x_ray(path)

        self.transform = train_transform if train else val_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data["Path"].iloc[index]
        label = self.data["Label"].iloc[index]

        label_map = {
            "normal": 0,
            "pneumonia": 1
        }
        mapped_label = label_map[label]

        # Load image
        if self.cache and path in self.cached_images:
            image = self.cached_images[path]
        else:
            image = read_x_ray(path)

        # Convert NumPy array to PIL Image
        image = Image.fromarray(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(mapped_label, dtype=torch.float32)
        }
