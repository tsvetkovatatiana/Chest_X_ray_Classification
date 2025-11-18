from torch.utils.data import Dataset
import cv2
import numpy as np
from transforms import train_transform, val_transform

def read_x_ray(path):
    x_ray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x_ray = x_ray.astype(np.float32) / 255

    x_ray_3ch = np.stack([x_ray, x_ray, x_ray], axis=-1)  # HWC

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

        # Load from cache or disk
        if self.cache and path in self.cached_images:
            image = self.cached_images[path]
        else:
            image = read_x_ray(path)

        image = self.transform(image)

        results = {
            "image": image,
            "label": label
        }
        return results
