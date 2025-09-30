import os
import sys
import numpy as np
from skimage import io
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import functional as F
from scipy import ndimage


def semantic_to_instance_masks(mask):
    """
    Convert a semantic segmentation mask to instance masks + class labels.

    Args:
        mask (np.ndarray): HxW, each pixel = class ID (0 = background).

    Returns:
        instance_masks: list of binary masks (HxW)
        labels: list of class labels for each instance
    """
    instance_masks = []
    labels = []
    class_ids = class_ids[(class_ids != 0) & (class_ids != 1)]
    for cid in class_ids:
        # Extract all pixels for this class
        class_mask = (mask == cid).astype(np.uint8)
        # Connected components = individual instances
        labeled_mask, num_objs = ndimage.label(class_mask)
        for obj_id in range(1, num_objs + 1):
            inst_mask = (labeled_mask == obj_id)
            if inst_mask.sum() == 0:
                continue
            instance_masks.append(inst_mask)
            labels.append(cid)  # keep original class ID
    return instance_masks, labels

## this script is for the mask rcnn model
class CellDataset(Dataset):

    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".tif")])
        self.mask_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".tif")])
        self.transforms = transforms
        # Build class ID mapping (skip 0 & 1, remap to contiguous)
        all_masks = [io.imread(p) for p in self.mask_paths]
        unique_ids = np.unique(np.concatenate([np.unique(m) for m in all_masks]))
        unique_ids = unique_ids[(unique_ids != 0)]
        self.class_id_map = {cid: idx + 1 for idx, cid in enumerate(unique_ids)}
        # Example: {2:1, 3:2, 4:3} → contiguous labels for training

    
    def __getitem__(self, idx):
        # Load image & mask
        img = io.imread(self.image_paths[idx])
        mask = io.imread(self.mask_paths[idx])

        # Convert grayscale -> RGB
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        # Normalize image to [0,1] float32
        img = img.astype(np.float32)
        if img.max() > 0:
            img /= img.max()

        instance_masks = []
        labels = []
        boxes = []
        # Process each valid class
        class_ids = np.unique(mask)
        class_ids = class_ids[(class_ids != 0)]
        for cid in class_ids:
            # Binary mask for this class
            class_mask = (mask == cid).astype(np.uint8)

            # Split into connected components
            labeled_mask, num_objs = ndimage.label(class_mask)

            for obj_id in range(1, num_objs + 1):
                inst_mask = (labeled_mask == obj_id)
                if inst_mask.sum() == 0:
                    continue

                # Bounding box
                pos = np.where(inst_mask)
                xmin, xmax = pos[1].min(), pos[1].max()
                ymin, ymax = pos[0].min(), pos[0].max()
                # Ensure box is valid (positive width & height)
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])

                # Store instance
                instance_masks.append(inst_mask)
                labels.append(self.class_id_map[cid])  # remapped class id

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(instance_masks), dtype=torch.uint8) if instance_masks else torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        img = F.to_tensor(img)

        target = {"boxes": boxes, "labels": labels, "masks": masks}
        return img, target
        # # Get unique object IDs
        # obj_ids = np.unique(mask)
        # obj_ids = obj_ids[obj_ids != 0]  # skip background


        # # Binary masks
        # masks = mask == obj_ids[:, None, None]

        # # Bounding boxes
        # boxes = []
        # for m in masks:
        #     pos = np.where(m)
        #     xmin, xmax = pos[1].min(), pos[1].max()
        #     ymin, ymax = pos[0].min(), pos[0].max()
        #     boxes.append([xmin, ymin, xmax, ymax])
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # # Labels (for now treat everything as "cell" class = 1)
        # labels = torch.ones((len(obj_ids),), dtype=torch.int64)
        # labels = torch.as_tensor(obj_ids, dtype=torch.int64)

        # # Convert
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        # img = F.to_tensor(img)

        # target = {"boxes": boxes, "labels": labels, "masks": masks}
        # return img, target

    def __len__(self):
        return len(self.image_paths)

class CellDataset_new(Dataset):

    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_paths = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
        self.class_folders = sorted([d for d in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, d))])
        self.transforms = transforms
        # Build class_id_map (e.g. "1"→1, "2"→2, ...)
        self.class_id_map = {int(c): int(c) for c in self.class_folders}
        self.reverse_class_id_map = {v: k for k, v in self.class_id_map.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ----- Load image -----
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        img = io.imread(img_path)
        # Convert grayscale → RGB
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img.astype(np.float32)
        if img.max() > 0:
            img /= img.max()
        img = F.to_tensor(img)
        
        instance_masks, labels, boxes = [], [], []
        # ----- Load label masks per class -----
        for class_name in self.class_folders:
            class_id = int(class_name)
            mask_path = os.path.join(self.label_dir, class_name, self.image_paths[idx])
            if not os.path.exists(mask_path):
                continue
            mask = io.imread(mask_path)
            # Each pixel value > 0 = instance id
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]
            for inst_id in instance_ids:
                inst_mask = (mask == inst_id)
                if inst_mask.sum() == 0:
                    continue
                # Bounding box
                pos = np.where(inst_mask)
                xmin, xmax = pos[1].min(), pos[1].max()
                ymin, ymax = pos[0].min(), pos[0].max()
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
                instance_masks.append(inst_mask)
                labels.append(self.class_id_map[class_id])  # use class id directly
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        if instance_masks:
            masks = torch.as_tensor(np.stack(instance_masks), dtype=torch.uint8)
        else:
            masks = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)
        target = {"boxes": boxes, "labels": labels, "masks": masks}
        return img, target




def build_dataset_dataloader(data_path, batch_size=2, val_split = 1.0, num_workers=0):

    image_dir = os.path.join(data_path, "Images")
    label_dir = os.path.join(data_path, "Labels")
    dataset = CellDataset_new(image_dir, label_dir)
    #print(dataset.class_id_map)
    #print(len(dataset))
    
    # Split
    if val_split < 1.0:
        train_size = int(val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset = dataset
        val_dataset = None

    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
    #     num_workers=num_workers,
    #     collate_fn=lambda x: tuple(zip(*x)))
    # Analyze labels to decide num_classes
    # all_ids = set()
    # for mask_path in dataset.mask_paths:
    #     mask = io.imread(mask_path)
    #     ids = np.unique(mask)
    #     ids = ids[ids != 0]  # skip background
    #     all_ids.update(ids.tolist())
    # num_classes = len(all_ids) + 1  # background + unique labels
    # num_classes = background (0) + all valid class folders
    num_classes = len(dataset.class_id_map) + 1  

    # Train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    # Val dataloader
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )

    return train_dataset, val_dataset, train_loader, val_loader, num_classes


if __name__ == "__main__":
    

    exit()

    data_path = '/home/cli74/Desktop/cbi/public/Biohackathon_2025/Datasets/mask_rcnn_new/'
    #image_dir = os.path.join(data_path, "Images")
    #label_dir = os.path.join(data_path, "Labels_new")
    train_dataset, val_dataset, train_loader, val_loader, num_classes = build_dataset_dataloader(data_path = data_path, val_split=0.8)
    print('traindataset size:', len(train_dataset))
    if val_dataset is not None:
        print('valdataset size:', len(val_dataset))
    print("Number of classes:", num_classes)
    img, target = train_dataset[0]
    print(img.shape, img.dtype)
    print(target["boxes"].shape, target["labels"], target["masks"].shape)
    print("Image range:", img.min().item(), img.max().item())
    print("Boxes:", target["boxes"])
    print("Labels:", target["labels"])
    print("Mask values:", target["masks"].unique())




