import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        # Transform only for images
        self.img_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load RGB image
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)  # [3, H, W], float32


        return image

    def __getitem222__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load RGB image
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)  # [3, H, W], float32

        # Load mask (should have 0 or 1 values)
        mask = Image.open(mask_path).convert("L")
        mask = torch.tensor(np.array(mask), dtype=torch.long)  # [H, W]

        return image, mask
