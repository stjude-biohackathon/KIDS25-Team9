from models.base_model import BaseModel
from statistics import mean
import monai
import random
import numpy as np
import tifffile
import torch
from patchify import patchify
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
from datasets import Dataset as HFDataset
from transformers import (
    Sam2Model,
    Sam2Processor
)
import json

# # select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))

  bbox_64 = np.array([x_min, y_min, x_max, y_max], dtype=np.int64) # SAMV2 expexts int32
  bbox = bbox_64.astype(np.int32).tolist()

  return bbox


class SAM2Dataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)
    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

class SAM2(BaseModel):
    def __init__(self, config):
        self.config = config
        self.processor = Sam2Processor.from_pretrained(self.config["model_name"])
        self.model = Sam2Model.from_pretrained(self.config["model_name"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def architecture(self):
        return self.model

    def train(self, dataset):

        print("Starting training...")
        train_dataset = SAM2Dataset(dataset=dataset, processor=self.processor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True, drop_last=False)

        batch = next(iter(train_dataloader))
        for k,v in batch.items():
            print(k,v.shape)

        print(batch["ground_truth_mask"].shape)


        optimizer = Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        self.model.train()

        # make sure we only compute gradients for mask decoder
        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

        for epoch in range(self.config['epochs']):
            epoch_losses = []
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config['epochs']}"):
                # inputs = {k: v.to(self.device) for k, v in batch.items() if k != "ground_truth_mask"}
                ground_truth_mask = batch["ground_truth_mask"].float().to(self.device)
                # print()
                # print(ground_truth_mask.shape)
                outputs = self.model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)
                pred_masks = outputs.pred_masks.squeeze(1)
                # ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                loss = loss_fn(pred_masks, ground_truth_mask.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                epoch_losses.append(loss.item())

            print(f"Epoch {epoch+1} Loss: {mean(epoch_losses)}")

    def infer(self, input_data):

        self.model.eval()

        with torch.no_grad():
            ground_truth_mask = np.array(dataset[idx]["label"])
            prompt = get_bounding_box(ground_truth_mask)
            inputs = self.processor(input_data, input_boxes=[[prompt]], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, multimask_output=False)
            pred_prob = torch.sigmoid(outputs.pred_masks.squeeze(1)).to(torch.float32).cpu().numpy().squeeze()
            pred_masks = (pred_prob > 0.5).astype(np.uint8)

        return pred_masks, pred_prob
    


if __name__ == "__main__":
    large_images = tifffile.imread(r"Images\training.tif")
    large_masks = tifffile.imread(r"Labels\training_groundtruth.tif")
    config_file = r'config.json'

    with open(config_file, 'r') as f:
        config = json.load(f)

        patch_size = 256
        step = 256

        all_img_patches = []

        for img in range(large_images.shape[0]):
            large_image = large_images[img]
            patches_img = patchify(large_image, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap

            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):

                    single_patch_img = patches_img[i,j,:,:]
                    all_img_patches.append(single_patch_img)

        images = np.array(all_img_patches)

        all_mask_patches = []
        for img in range(large_masks.shape[0]):
            large_mask = large_masks[img]
            patches_mask = patchify(large_mask, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap

            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):

                    single_patch_mask = patches_mask[i,j,:,:]
                    single_patch_mask = (single_patch_mask / 255.).astype(np.uint8)
                    all_mask_patches.append(single_patch_mask)

        masks = np.array(all_mask_patches)

        # Create a list to store the indices of non-empty masks
        valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
        # Filter the image and mask arrays to keep only the non-empty pairs
        filtered_images = images[valid_indices]
        filtered_masks = masks[valid_indices]
        print("Image shape:", filtered_images.shape)  # e.g., (num_frames, height, width, num_channels)
        print("Mask shape:", filtered_masks.shape)
        print(f"Image data type: {filtered_images.dtype}")
        print(f"Mask data type: {filtered_masks.dtype}")

        # Convert the NumPy arrays to Pillow images and store them in a dictionary
        dataset_dict = {
            "image": [Image.fromarray(img) for img in filtered_images],
            "label": [Image.fromarray(mask) for mask in filtered_masks],
        }

        # Create the dataset using the datasets.Dataset class
        dataset = HFDataset.from_dict(dataset_dict)

        # processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
        # train_dataset = SAM2Dataset(dataset=dataset, processor=processor)
        # train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)
        # print("Outside the loop")
        # batch = next(iter(train_dataloader))
        # for k,v in batch.items():
        #     print(k,v.shape)

        # print(batch["ground_truth_mask"].shape)

        # print("-----_-------")

        custom_model = SAM2(config=config)
        
        custom_model.train(dataset)
        # torch.save(custom_model.state_dict(), "mito_model_checkpoint.pth")

        idx = random.randint(0, filtered_images.shape[0]-1)
        test_image = dataset[idx]["image"]

        pred_masks, pred_prob = custom_model.infer(test_image)

        # tifffile.imwrite("test_image.tif", np.array(test_image))
        # tifffile.imwrite("pred_mask.tif", pred_masks)
        # tifffile.imwrite("probability.tif", pred_prob)


